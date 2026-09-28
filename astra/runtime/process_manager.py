"""Subprocess-per-model lifecycle: spawn, talk JSON-lines, enforce timeouts, kill whole process group."""
from __future__ import annotations
import json, os, pathlib, queue, signal, subprocess, sys, threading, time, uuid
import psutil
from ..core.config import ROOT, scrubbed_env

OOM_PAT = ("out of memory", "cudamalloc failed", "failed to allocate", "cuda error", "cublas_status_alloc_failed",
           "unable to allocate", "std::bad_alloc", "memoryerror", "ggml_backend_cuda_buffer_type_alloc_buffer")
ARCH_PAT = ("unknown model architecture", "unsupported architecture", "unknown architecture")


class ModelLoadError(RuntimeError):
    def __init__(self, result: dict):
        super().__init__(result.get("error", "model load failed"))
        self.result = result


class WorkerHandle:
    def __init__(self, spec: dict, run_dir: pathlib.Path, cuda_visible: str | None):
        self.spec = spec
        self.run_dir = run_dir
        self.cuda_visible = cuda_visible
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.log_path = run_dir / f"worker_{spec['key']}_{int(time.time())}.log"
        self.ready: dict = {}

    # ------------------------------------------------------------ helpers
    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def stderr_tail(self, n=4000) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - n))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def classify(self, err: str = "") -> str:
        blob = (err + "\n" + self.stderr_tail()).lower()
        rc = self.proc.poll() if self.proc else None
        if any(p in blob for p in OOM_PAT) or rc in (-9, 137):
            return "oom"
        if any(p in blob for p in ARCH_PAT):
            return "unsupported_arch"
        return "failed"

    def _reader(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        finally:
            self.q.put(None)

    def _get(self, timeout: float):
        """Next event, or ('timeout'|'eof')."""
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                return "timeout"
            try:
                ev = self.q.get(timeout=min(left, 1.0))
            except queue.Empty:
                if not self.alive() and self.q.empty():
                    return "eof"
                continue
            if ev is None:
                return "eof"
            return ev

    # ------------------------------------------------------------ lifecycle
    def start(self, load_timeout: float) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        spec_file = self.run_dir / f"spec_{self.spec['key']}_{uuid.uuid4().hex[:6]}.json"
        spec_file.write_text(json.dumps(self.spec))  # contains no secrets
        env = scrubbed_env({"PYTHONPATH": str(ROOT.parent), "PYTHONUNBUFFERED": "1", "ASTRA_WORKER": "1"})
        if self.cuda_visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible
        with open(self.log_path, "wb") as logf:
            self.proc = subprocess.Popen([sys.executable, "-m", "astra.runtime.worker", "--spec-file", str(spec_file)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logf, text=True, bufsize=1,
                                         env=env, start_new_session=True)
        threading.Thread(target=self._reader, daemon=True).start()
        ev = self._get(load_timeout)
        if ev == "timeout":
            self.kill()
            return {"status": "timeout", "error": f"load exceeded {load_timeout}s"}
        if ev == "eof":
            st = self.classify()
            self.kill()
            return {"status": st, "error": "worker died during load", "stderr": self.stderr_tail(1500)}
        if ev.get("event") == "load_failed":
            st = self.classify(ev.get("error", ""))
            self.kill()
            return {"status": st, "error": ev.get("error"), "stderr": self.stderr_tail(1500)}
        self.ready = ev
        return {"status": "success", **ev}

    def generate(self, req: dict, timeout: float, on_token=None) -> dict:
        rid = uuid.uuid4().hex[:8]
        if not self.alive():
            return {"status": "crashed", "error": "worker not alive"}
        try:
            self.proc.stdin.write(json.dumps({"cmd": "generate", "id": rid, **req}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return {"status": "crashed", "error": "broken pipe to worker"}
        end = time.time() + timeout
        while True:
            ev = self._get(max(0.1, end - time.time()))
            if ev == "timeout":
                self.kill()
                return {"status": "timeout", "error": f"generation exceeded {timeout}s"}
            if ev == "eof":
                st = "oom" if self.classify() == "oom" else "crashed"
                return {"status": st, "error": f"worker exited rc={self.proc.poll()}", "stderr": self.stderr_tail(1500)}
            if ev.get("id") != rid:
                continue
            if ev["event"] == "token":
                if on_token:
                    on_token(ev["text"])
                continue
            if ev["event"] == "result":
                if ev["status"] != "success":
                    ev["status"] = self.classify(ev.get("error", "")) if self.classify(ev.get("error", "")) in ("oom",) else ev["status"]
                    if "context" in (ev.get("error") or "").lower() and "exceed" in (ev.get("error") or "").lower():
                        ev["status"] = "context_overflow"
                return ev

    def stop(self, grace: float = 8.0):
        if not self.proc:
            return
        if self.alive():
            try:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        self.kill()

    def kill(self):
        """SIGTERM then SIGKILL the whole process group (covers llama-server children)."""
        if not self.proc:
            return
        pgid = None
        try:
            pgid = os.getpgid(self.proc.pid)
        except Exception:
            pass
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            if self.proc.poll() is not None and pgid is None:
                break
            try:
                if pgid:
                    os.killpg(pgid, sig)
                else:
                    self.proc.send_signal(sig)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        for p in (self.proc.stdin, self.proc.stdout):
            try:
                p and p.close()
            except Exception:
                pass


def find_worker_processes(exclude_pid: int | None = None) -> list[psutil.Process]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            # exact argv element match (a substring match could hit an unrelated shell whose command text mentions it)
            if "astra.runtime.worker" in cl and cl[:1] and "python" in os.path.basename(cl[0]) and p.info["pid"] != exclude_pid and p.info["pid"] != os.getpid():
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out
