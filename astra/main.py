"""Astra controller + CLI. State machine: BOOTING -> READY -> RUNNING <-> WAITING_FOR_USER -> POST_TASK_WAIT -> (READY | SHUTTING_DOWN -> STOPPED)."""
from __future__ import annotations
import argparse, contextlib, json, os, queue, sys, threading, time
from .core.config import Config, Secrets, ModelRegistry, on_kaggle
from .core.logging_utils import get_logger, ExecLog
from .core.router import Router
from .core.events import EventBus, AnswerBroker
from .core.task import Task
from .core.state import TaskState
from .core import task as T


class IdleGuard:
    """60s post-task countdown. Fires ONLY in POST_TASK_WAIT with nothing running/waiting/persisting."""
    def __init__(self, astra):
        self.a, self.timer, self.deadline = astra, None, None

    def start(self, seconds):
        self.cancel()
        self.deadline = time.time() + seconds
        self.timer = threading.Timer(seconds, self._fire)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()
        self.timer, self.deadline = None, None

    def remaining(self):
        return max(0, round(self.deadline - time.time(), 1)) if self.deadline else None

    def _fire(self):
        a = self.a
        with a._lock:
            if a.state != "POST_TASK_WAIT":
                return
            blocked = a._busy > 0 or a.broker.pending() or not a._q.empty()
        if blocked:
            self.start(10)  # something is in flight (e.g. persistence): re-check shortly, never cut it off
            return
        a.log.info("idle timeout reached -> controlled shutdown")
        a.shutdown("idle_timeout")


class Astra:
    def __init__(self, overrides=None):
        self.cfg = Config(overrides=overrides)
        self.secrets = Secrets()
        self.secrets.prime()
        self.log = get_logger("core", self.cfg.logs_dir)
        self.reg = ModelRegistry(self.cfg)
        self.router = Router(self.reg, self.log)
        self.bus, self.execlog = EventBus(), ExecLog(self.cfg.logs_dir)
        self.broker = AnswerBroker(self.bus)
        self.state = "BOOTING"
        self._lock = threading.RLock()
        self._busy = 0
        self._q: queue.Queue = queue.Queue()
        self._exec = None
        self._accepting = True
        self.jobs: dict[str, tuple] = {}
        self.idle = IdleGuard(self)
        self.report = None
        self.interrupted, self.server = [], None
        self.rm = self.db = self.scheduler = self.tasks = self.mcp = self.store = self.sessions = self.ckpt = None

    # ------------------------------------------------------------ state
    @contextlib.contextmanager
    def busy(self, why=""):
        with self._lock:
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1

    def set_state(self, s):
        with self._lock:
            if self.state not in ("SHUTTING_DOWN", "STOPPED"):
                self.state = s

    def boot(self, upto="full", **kw):
        from . import bootstrap
        self.report = bootstrap.run(self, upto, **kw)
        if upto == "full":
            self.set_state("READY")
        return self.report

    def enable_console(self):
        def p(tid, e):
            t = e["type"]
            if t == "step_started": print(f"  > {e['step']} [{e['agent']}] on {e['model']}")
            elif t == "step_finished": print(f"  < {e['step']} {e['status']}")
            elif t == "checkpoint" and not e["safe"]: print(f"  ! checkpoint '{e['label']}' NOT safely persisted")
            elif t == "checkpoint_warning": print("  ! " + e["message"])
            elif t == "task_done": print(f"\n[{e['status'].upper()}] {e.get('response') or ''}\nTask completed. Do you want me to continue?")
        self.bus.listeners.append(p)
        self.broker.console = True

    # ------------------------------------------------------------ tasks
    def _need_ready(self):
        if not self._accepting or self.state in ("SHUTTING_DOWN", "STOPPED"):
            raise RuntimeError(f"Astra is {self.state}; not accepting tasks")
        if self.scheduler is None:
            raise RuntimeError("boot(upto='full') first")

    def submit(self, goal: str, project_id: str | None = None) -> str:
        self._need_ready()
        self.idle.cancel()
        t = Task(goal=goal, project_id=project_id)
        st = TaskState(task_id=t.task_id, goal=goal)
        self.jobs[t.task_id] = (t, st)
        with self.busy("submit"):
            self.tasks.save_task(t)
            self.sessions.add_message(self.session_id, "user", goal, t.task_id)
        self._enqueue(t.task_id)
        return t.task_id

    def resume(self, task_id: str) -> str:
        """Calling this IS the user's confirmation to continue an interrupted task."""
        self._need_ready()
        self.idle.cancel()
        cp = self.ckpt.latest(task_id)
        if not cp:
            raise KeyError(f"no checkpoint for {task_id}")
        t, st = T.Task.from_dict(cp["task"]), TaskState.from_dict(cp["state"])
        for s in (t.plan.steps if t.plan else []):
            if s.status == T.SKIPPED:
                s.status = T.PENDING
        t.status = T.INTERRUPTED
        self.jobs[task_id] = (t, st)
        self._enqueue(task_id)
        return task_id

    def answer(self, task_id, text) -> bool:
        self.idle.cancel()
        return self.broker.answer(task_id, text)

    def _enqueue(self, tid):
        self._q.put(tid)
        with self._lock:
            if self._exec is None or not self._exec.is_alive():
                self._exec = threading.Thread(target=self._loop, daemon=True, name="astra-executor")
                self._exec.start()

    def _loop(self):
        while True:
            try:
                tid = self._q.get(timeout=1)
            except queue.Empty:
                if self.state in ("STOPPED", "SHUTTING_DOWN"):
                    return
                continue
            t, st = self.jobs[tid]
            self.set_state("RUNNING")
            with self.busy("task"):
                try:
                    self.scheduler.run_task(t, st)
                except Exception as e:
                    self.log.error(f"executor error: {e}")
                    t.status = T.FAILED
            with contextlib.suppress(Exception):
                resp = next((s.output.get("response") for s in reversed(t.plan.steps) if t.plan and s.agent == "assistant" and s.output), None)
                self.sessions.add_message(self.session_id, "assistant", resp or f"task {t.status}", tid)
            if self._q.empty() and self.state == "RUNNING":
                self.set_state("POST_TASK_WAIT")
                self.idle.start(self.cfg["idle"]["post_task_wait_s"])   # spec: 60s explicit post-task wait

    def run(self, goal, console=True, project_id=None, timeout=None):
        if console:
            self.enable_console() if not self.bus.listeners else None
        tid = self.submit(goal, project_id)
        end = time.time() + timeout if timeout else None
        for e in self.bus.stream(tid):
            if e is None and end and time.time() > end:
                raise TimeoutError
            if e and e["type"] == "task_done":
                break
        return self.jobs[tid][0]

    def wait(self, tid):
        for e in self.bus.stream(tid):
            if e and e["type"] == "task_done":
                return e

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        d = {"state": self.state, "busy_ops": self._busy, "queued": self._q.qsize(), "idle_countdown_s": self.idle.remaining(),
             "pending_question": self.broker.pending()}
        if self.rm:
            d["resources"] = self.rm.state()
        if self.db:
            d["persistence"] = {"db": self.db.kind, "durable": self.db.durable, "object_store": self.store.kind,
                                "last_checkpoint_safe": self.scheduler.last_checkpoint_ok if self.scheduler else None}
            d["tasks"] = self.tasks.list(limit=10)
        if self.mcp:
            d["mcp"] = self.mcp.status()
        return d

    # ------------------------------------------------------------ shutdown
    def shutdown(self, reason="manual", force=False) -> dict:
        with self._lock:
            if self.state in ("SHUTTING_DOWN", "STOPPED"):
                return {"ok": True, "note": f"already {self.state}"}
            if not force and (self._busy > 0 or self.state in ("RUNNING", "WAITING_FOR_USER")):
                return {"ok": False, "refused": "a task/tool/persistence op is active or waiting for user input (use force=True to override)"}
            prev, self.state, self._accepting = self.state, "SHUTTING_DOWN", False   # 1: stop accepting new work
        self.idle.cancel()
        log, steps = self.log, []

        def mark(name, ok, detail=""):
            steps.append({"step": name, "ok": ok, "detail": detail})
            (log.info if ok else log.error)(f"shutdown: {name} {'ok' if ok else 'FAILED'} {detail}")

        active = [(t, s) for t, s in self.jobs.values() if t.status not in (T.COMPLETED, T.FAILED)]
        persisted = True
        if self.db:
            for t, s in active:                                                      # 2-4: conversation/task/checkpoints
                if t.status in (T.RUNNING, T.WAITING):
                    t.status = T.INTERRUPTED
                r = self.ckpt.save(t, s, "shutdown")
                persisted &= r.safe
            mark("save task state + checkpoints", persisted, f"{len(active)} active task(s)")
            for name, rel in (("execution metrics/logs", self.cfg.logs_dir / "executions.jsonl"),):   # 5-7: metrics, MCP refs
                try:
                    if rel.exists():
                        self.artifacts.put("_logs", f"{self.session_id}_{rel.name}", rel.read_bytes())
                    mark(name, True)
                except Exception as e:
                    mark(name, False, str(e)[:150]); persisted = False
            mark("mcp registry (persistent by design)", self.db.ping())
            # 8: verify persistence really succeeded
            ok = self.db.ping() and persisted and self.db.durable
            mark("verify persistence", ok, "" if ok else ("database is NOT durable (SQLite fallback)" if not self.db.durable else "checkpoint/log write failed"))
            if not ok and not force:
                self.state, self._accepting = prev if prev != "SHUTTING_DOWN" else "READY", True
                msg = "Checkpoint persistence failed. I will not shut down until the state is safely persisted."
                log.error(msg)
                return {"ok": False, "refused": msg, "steps": steps}
        with contextlib.suppress(Exception):
            self.mcp and self.mcp.stop_all()                                         # 9
        mark("stop MCP processes", True)
        if self.rm:
            try:
                self.rm.release()                                                    # 10-11
                self.rm.assert_clean()
                mark("terminate model worker + release GPU", True)
            except Exception as e:
                mark("terminate model worker + release GPU", False, str(e))
        with contextlib.suppress(Exception):
            self.sessions.end(self.session_id, reason)                               # 12
        if self.server:
            self.server.should_exit = True
        self.state = "STOPPED"
        plat = platform_shutdown(self.cfg, log)                                      # 13
        return {"ok": True, "reason": reason, "persisted_safely": bool(self.db and self.db.durable and persisted), "steps": steps, "platform": plat}

    def serve(self, block=False):
        import uvicorn
        from .gateway import create_app
        token = self.secrets.get("ASTRA_API_TOKEN")
        if not token:
            raise RuntimeError("ASTRA_API_TOKEN is required to start the API (set it as a Kaggle Secret / env var).")
        cfg = uvicorn.Config(create_app(self, token), host=self.cfg["server"]["host"], port=self.cfg["server"]["port"], log_level="warning")
        self.server = uvicorn.Server(cfg)
        th = threading.Thread(target=self.server.run, daemon=True, name="astra-gateway")
        th.start()
        if block:
            th.join()
        return th


def platform_shutdown(cfg, log) -> dict:
    """HONEST platform shutdown. Kaggle exposes NO supported call to terminate an interactive session's VM from inside the notebook."""
    info = {"platform": "kaggle" if on_kaggle() else "local", "vm_terminated": False, "processes_stopped": True}
    if on_kaggle():
        info["note"] = ("All Astra processes/MCP/model workers are stopped and GPU memory is released. The Kaggle VM itself is NOT terminated by "
                        "this code: interactive sessions end only via 'Stop session' in the UI or Kaggle's idle/time limits; in Save&Run (commit) mode "
                        "the session ends when the notebook's last cell returns.")
        if cfg["shutdown"]["kill_kernel"]:
            info["kernel_exit"] = "os._exit(0) scheduled (kills the Python kernel; still does not prove the VM stops)"
            threading.Timer(1.0, lambda: os._exit(0)).start()
    else:
        info["note"] = "not on Kaggle: nothing to terminate beyond Astra processes"
    log.warning(info["note"])
    return info


# ------------------------------------------------------------ CLI
def cli(argv=None):
    ap = argparse.ArgumentParser(prog="astra")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("status", "models", "mcp", "memory", "tasks", "shutdown", "serve", "selftest"):
        sub.add_parser(c)
    r = sub.add_parser("run"); r.add_argument("goal")
    rs = sub.add_parser("resume"); rs.add_argument("task_id", nargs="?")
    sm = sub.add_parser("smoke"); sm.add_argument("model")
    b = sub.add_parser("benchmark"); b.add_argument("--models"); b.add_argument("--sweep", action="store_true"); b.add_argument("--layers"); b.add_argument("--ctx", type=int)
    a = ap.parse_args(argv)
    from . import bootstrap
    astra = Astra()
    upto = "diagnostics" if a.cmd in ("models", "smoke", "benchmark") else "full"
    if a.cmd == "benchmark":
        upto = "persistence"
    rep = astra.boot(upto)
    if a.cmd == "status":
        print(bootstrap.format_report(rep)); print(json.dumps(astra.status(), indent=2, default=str))
    elif a.cmd == "models":
        print(json.dumps(rep["models"], indent=2, default=str))
    elif a.cmd == "mcp":
        print(json.dumps(astra.mcp.status(), indent=2))
    elif a.cmd == "memory":
        print(astra.mem.compact_context(6000) or "(empty)")
    elif a.cmd == "tasks":
        for t in astra.tasks.list(limit=30):
            print(t["task_id"], t["status"], t["current_step"], "-", t["goal"][:70])
    elif a.cmd == "run":
        astra.enable_console(); astra.run(a.goal); astra.shutdown("cli_done", force=True)
    elif a.cmd == "resume":
        tid = a.task_id or (astra.interrupted[0]["task_id"] if astra.interrupted else None)
        if not tid:
            print("nothing to resume"); return
        if input(f"Resume {tid}? [y/N] ").lower() == "y":
            astra.enable_console(); astra.resume(tid); astra.wait(tid); astra.shutdown("cli_done", force=True)
    elif a.cmd == "shutdown":
        print(json.dumps(astra.shutdown("cli", force=True), indent=2, default=str))
    elif a.cmd == "smoke":
        from .selftest import smoke
        print(json.dumps(smoke(astra, a.model), indent=2, default=str))
    elif a.cmd == "selftest":
        from .selftest import swap_test
        print(json.dumps(swap_test(astra, list(astra.reg.keys())), indent=2, default=str))
    elif a.cmd == "benchmark":
        from .benchmarks.benchmark import run_benchmark
        run_benchmark(astra, a.models.split(",") if a.models else None, sweep=a.sweep,
                      layers=[int(x) for x in a.layers.split(",")] if a.layers else None, ctx=a.ctx)
    elif a.cmd == "serve":
        print(bootstrap.format_report(rep)); astra.serve(block=True)


if __name__ == "__main__":
    cli()
