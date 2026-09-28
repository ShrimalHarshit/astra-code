"""Scheduler: the ONLY component (with ResourceManager) that touches model lifecycle. Executes the plan, checkpoints after every step."""
from __future__ import annotations
import contextlib, json, time, traceback
from .task import Task, Plan, Step, COMPLETED, FAILED, RUNNING, PENDING, WAITING, SKIPPED, INTERRUPTED
from .state import TaskState
from ..agents import AgentContext
from ..agents.orchestrator import Orchestrator
from ..agents.engineer import EngineerAgent
from ..agents.logic import LogicAgent
from ..agents.assistant import AssistantAgent
from ..runtime.process_manager import ModelLoadError

AGENT_CLASSES = {"engineer": EngineerAgent, "logic": LogicAgent, "assistant": AssistantAgent}


class ModelExecutionError(RuntimeError):
    def __init__(self, result: dict, model: str):
        super().__init__(result.get("error") or result.get("status"))
        self.result, self.model = result, model


class Scheduler:
    def __init__(self, cfg, registry, router, rm, tasks, ckpt, mem_store, compiler, toolbox, bus, broker, execlog, log,
                 busy=None, on_state=None):
        self.cfg, self.reg, self.router, self.rm, self.tasks = cfg, registry, router, rm, tasks
        self.ckpt, self.mem, self.compiler, self.tools = ckpt, mem_store, compiler, toolbox
        self.bus, self.broker, self.execlog, self.log = bus, broker, execlog, log
        self.busy = busy or (lambda why: contextlib.nullcontext())
        self.on_state = on_state or (lambda s: None)
        self.overrides: dict = {}
        self.last_checkpoint_ok = True

    # ------------------------------------------------------------ helpers
    def _ckpt(self, task, state, label):
        with self.busy("checkpoint"):
            res = self.ckpt.save(task, state, label)
            try:
                self.tasks.save_task(task)
            except Exception as e:
                self.log.error(f"task row save failed: {e}")
                res["verified"] = False
        self.last_checkpoint_ok = res.safe
        self.bus.publish(task.task_id, {"type": "checkpoint", "label": label, "safe": res.safe, "seq": res["seq"]})
        return res

    def _agent(self, model: str):
        cls = AGENT_CLASSES.get(self.reg.spec(model).get("agent", model))
        if cls is None:
            raise KeyError(f"no agent implementation for '{model}'")
        return cls()

    def _ctx(self, task, state, step, model):
        spec = self.reg.spec(model, self.overrides.get(model))
        try:
            mem = self.mem.compact_context(2500)
        except Exception:
            mem = ""
        return AgentContext(task=task, state=state, step=step, model_key=model, spec=spec, tools=self.tools, memory_text=mem,
                            call_model=lambda msgs, **kw: self.call_model(model, msgs, task=task, step_id=step.id, **kw),
                            checkpoint=lambda label: self._ckpt(task, state, f"{step.id}:{label}"),
                            max_iters=self.cfg["agent"]["max_tool_iterations"])

    # ------------------------------------------------------------ the one place a model is called
    def call_model(self, model, messages, *, task, step_id, max_tokens=None, temperature=None, json_mode=False, stop=None,
                   stream=True, purpose="step") -> dict:
        retries = 0
        while True:
            reused = self.rm.current_model() == model
            rec = {"task_id": task.task_id, "step_id": step_id, "model": model, "purpose": purpose, "start_ts": time.time(), "reused_worker": reused}
            ov = dict(self.overrides.get(model, {}))
            spec = self.reg.spec(model, ov)
            try:
                h = self.rm.acquire(model, ov)
                spec = h.spec
                res = h.generate({"messages": messages, "max_tokens": max_tokens or spec.get("max_tokens", 1024),
                                  "temperature": spec.get("temperature", 0.2) if temperature is None else temperature,
                                  "stop": stop, "json_mode": json_mode, "stream": stream},
                                 self.cfg["timeouts"]["generate_s"],
                                 (lambda t: self.bus.publish(task.task_id, {"type": "token", "step": step_id, "text": t})) if stream else None)
                rec["load_ms"] = 0 if reused else h.ready.get("load_ms")
            except ModelLoadError as e:
                res, rec["load_ms"] = dict(e.result), None
            us = self.rm.sample()
            pt, ot = res.get("prompt_tokens") or 0, res.get("output_tokens") or 0
            rec.update(status=res["status"], prompt_tokens=pt, output_tokens=ot, prompt_ms=res.get("prompt_ms"), gen_ms=res.get("gen_ms"),
                       prompt_tok_s=round(pt / (res["prompt_ms"] / 1000), 1) if res.get("prompt_ms") else None,
                       gen_tok_s=round(ot / (res["gen_ms"] / 1000), 1) if res.get("gen_ms") else None,
                       vram_mb=us["vram_mb"], ram_mb=us["ram_mb"], gpu_layers=spec.get("n_gpu_layers"), context=spec.get("n_ctx"),
                       error=res.get("error"), wall_ms=None)
            self.execlog.write(rec)
            try:
                self.tasks.record_execution(rec)
            except Exception as e:
                self.log.error(f"execution_history write failed: {e}")
            self.log.info(self.execlog.summary(rec))
            if res["status"] == "success":
                task.last_model = model
                return res
            self.log.error(f"model {model} -> {res['status']}: {res.get('error')}")
            self.rm.release()  # dead/failed worker never lingers
            if res["status"] in ("oom", "insufficient_vram") and retries < self.cfg["oom_retry"]["max_retries"]:
                retries += 1
                new = self.rm.degrade(model, ov)
                self.log.warning(f"OOM: retrying {model} with {new} ({retries}/{self.cfg['oom_retry']['max_retries']}); no second model is loaded")
                self.overrides[model] = new
                with contextlib.suppress(Exception):
                    self.tasks.config_set("degraded_overrides", {k: {"gpu_layers": v.get("n_gpu_layers"), "context": v.get("n_ctx")} for k, v in self.overrides.items()})
                continue
            raise ModelExecutionError(res, model)

    # ------------------------------------------------------------ plan execution
    def run_task(self, task: Task, state: TaskState) -> Task:
        task.status = RUNNING
        self._ckpt(task, state, "task_started")
        try:
            if task.plan is None and not self._plan(task, state):
                return self._finish(task, state, FAILED)
            while True:
                ready = [s for s in task.plan.ready()]
                if not ready:
                    break
                cur = self.rm.current_model()
                step = next((s for s in ready if self.router.resolve(s) == cur), ready[0])  # fewer model swaps
                self._run_step(task, state, step)
            for s in task.plan.steps:  # unreachable steps (failed dependency) are marked, never silently dropped
                if s.status in (PENDING, RUNNING, WAITING):
                    s.status = SKIPPED
            ok = all(s.status == COMPLETED for s in task.plan.steps)
            return self._finish(task, state, COMPLETED if ok else FAILED)
        except Exception as e:  # scheduler bug: persist, never lose state
            self.log.error(f"scheduler error: {traceback.format_exc()}")
            state.error(task.current_step or "-", None, "scheduler_error", str(e))
            return self._finish(task, state, FAILED)

    def _plan(self, task, state) -> bool:
        step = Step(id="plan", agent="orchestrator", action="plan", capability="planning")
        model = self.router.resolve(step)
        task.current_step, state.current_step = "plan", "plan"
        try:
            res = Orchestrator(self.reg, self.router, self.log).run(self._ctx(task, state, step, model))
        except ModelExecutionError as e:
            state.error("plan", e.model, e.result["status"], str(e))
            return False
        task.plan = res["_plan"]
        state.pending_steps = [s.id for s in task.plan.steps]
        state.note("decisions", f"plan: {[(s.id, s.agent, s.action) for s in task.plan.steps]}", "plan")
        self.bus.publish(task.task_id, {"type": "plan", "plan": task.plan.to_dict()})
        self._ckpt(task, state, "planner_complete")
        return True

    def _run_step(self, task, state, step):
        model = self.router.resolve(step)
        resuming = step.status in (RUNNING, WAITING)
        step.status = RUNNING
        if not resuming:
            step.attempts += 1
        step.model_used, step.started_at = model, step.started_at or time.time()
        task.current_step = state.current_step = step.id
        self.bus.publish(task.task_id, {"type": "step_started", "step": step.id, "agent": step.agent, "model": model})
        self._ckpt(task, state, f"{step.id}:started")
        while True:
            res = self._run_agent(task, state, step, model)
            st = res["status"]
            if st == "success":
                step.output = res["output"]
                state.step_outputs[step.id] = res["output"]
                step.status, step.finished_at = COMPLETED, time.time()
                state.completed_steps.append(step.id)
                state.pending_steps = [s.id for s in task.plan.steps if s.status not in (COMPLETED, SKIPPED)]
                state.agent_sessions.pop(step.id, None)  # externalized loop state no longer needed
                break
            if st == "needs_user":
                ans = self._ask_user(task, state, step, res)
                state.agent_sessions[step.id]["pending_result"] = {"name": "ask_user", "content": ans}
                continue
            if st == "needs_specialist":
                try:
                    state.agent_sessions[step.id]["pending_result"] = self._delegate(task, state, step, res["request"], model)
                except Exception as e:
                    state.agent_sessions[step.id]["pending_result"] = {"name": "call_specialist", "content": f"delegation failed: {e}"}
                continue
            # failure
            if step.attempts < self.cfg["step_max_attempts"] and st in ("timeout", "crashed", "failed") and res.get("retryable", True):
                step.status = PENDING
                self.log.warning(f"{step.id} failed ({st}); will retry ({step.attempts}/{self.cfg['step_max_attempts']})")
            else:
                step.status, step.finished_at = FAILED, time.time()
            break
        self.bus.publish(task.task_id, {"type": "step_finished", "step": step.id, "status": step.status})
        self._ckpt(task, state, f"{step.id}:{step.status}")

    def _run_agent(self, task, state, step, model) -> dict:
        try:
            return self._agent(model).run(self._ctx(task, state, step, model))
        except ModelExecutionError as e:
            state.error(step.id, e.model, e.result["status"], str(e))
            return {"status": e.result["status"] if e.result["status"] in ("timeout", "crashed") else "failed", "error": str(e),
                    "retryable": e.result["status"] in ("timeout", "crashed")}
        except Exception as e:
            self.log.error(f"agent error in {step.id}: {traceback.format_exc()}")
            state.error(step.id, model, "agent_error", f"{type(e).__name__}: {e}")
            return {"status": "failed", "error": str(e), "retryable": False}

    def _ask_user(self, task, state, step, res) -> str:
        q = {"step": step.id, "question": res["question"], "options": res.get("options")}
        state.waiting_question, task.status, step.status = q, WAITING, WAITING
        self.on_state("WAITING_FOR_USER")
        self._ckpt(task, state, f"{step.id}:waiting_for_user")
        ans = self.broker.ask(task.task_id, res["question"], res.get("options"))   # blocks; idle shutdown is disabled meanwhile
        state.waiting_question, task.status, step.status = None, RUNNING, RUNNING
        state.note("decisions", f"user answered '{res['question'][:120]}' -> {ans[:300]}", step.id)
        with contextlib.suppress(Exception):
            self.tasks.add_decision(task.task_id, step.id, f"{res['question']} -> {ans}")
        self.on_state("RUNNING")
        self._ckpt(task, state, f"{step.id}:user_answered")
        return ans

    def _delegate(self, task, state, step, req, model) -> dict:
        n = state.delegations.get(step.id, 0) + 1
        if n > self.cfg["agent"]["max_delegations"]:
            raise RuntimeError("delegation limit reached")
        state.delegations[step.id] = n
        target = self.router.for_capability(req.get("capability", ""))
        if target == model:
            raise RuntimeError("cannot delegate to self")
        sub = Step(id=f"{step.id}.d{n}", agent=target, capability=req.get("capability"), action="delegated", description=req.get("task", ""),
                   depends_on=[])
        self.log.info(f"{step.id}: engineer state externalized; swapping {model} -> {target} for delegated work")
        self._ckpt(task, state, f"{sub.id}:delegating")  # Hermes/agent state is in state.agent_sessions => safe to unload
        r = self._run_agent(task, state, sub, target)
        if r["status"] != "success":
            return {"name": "call_specialist", "content": f"specialist failed: {r.get('error') or r['status']}"}
        state.step_outputs[sub.id] = r["output"]
        self._ckpt(task, state, f"{sub.id}:done")
        return {"name": "call_specialist", "content": json.dumps(r["output"], default=str)[:4000]}

    def _finish(self, task, state, status) -> Task:
        task.status, task.finished_at = status, time.time()
        state.current_step = None
        try:  # memory compile (heuristic + optional assistant-model extraction). The assistant model may already be resident.
            def cm(msgs, **kw):
                return self.call_model("assistant", msgs, task=task, step_id="memory", purpose="memory_compile", **kw)
            use_llm = status == COMPLETED and self.cfg.runtime.get("memory_llm_pass", True) and "assistant" in self.cfg.models
            self.compiler.compile(task, state, call_model=cm if use_llm else None)
        except Exception as e:
            self.log.warning(f"memory compile failed: {e}")
        self.rm.release()  # worker exits after the task; GPU is empty while we wait for the user
        self._ckpt(task, state, "final")
        final = None
        for s in reversed(task.plan.steps if task.plan else []):
            if s.agent == "assistant" and s.output:
                final = s.output.get("response")
                break
        self.bus.publish(task.task_id, {"type": "task_done", "status": status, "response": final,
                                        "persisted_safely": self.last_checkpoint_ok})
        return task
