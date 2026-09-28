"""Enforces: at most ONE model worker alive. acquire/release/current_model/assert_clean."""
from __future__ import annotations
import glob, os, pathlib, shutil, threading, time
import psutil
from . import hardware
from ..runtime.process_manager import WorkerHandle, ModelLoadError, find_worker_processes


class ResourceLeak(RuntimeError):
    pass


class ResourceManager:
    def __init__(self, cfg, registry, log):
        self.cfg, self.reg, self.log = cfg, registry, log
        self._h: WorkerHandle | None = None
        self._model: str | None = None
        self._ov: dict = {}
        self._lock = threading.RLock()
        self.cuda = cfg["cuda_visible_devices"]
        self.baseline_vram = hardware.gpu_used_mb(self.cuda)  # what the GPU looks like with NO worker of ours
        self.run_dir = cfg.work_dir / "run"

    # ------------------------------------------------------------ public API
    def current_model(self) -> str | None:
        with self._lock:
            return self._model if self._h and self._h.alive() else None

    def worker_pid(self):
        return self._h.pid if self._h else None

    def acquire(self, model: str, overrides: dict | None = None) -> WorkerHandle:
        with self._lock:
            overrides = overrides or {}
            if self._h and self._h.alive() and self._model == model and self._ov == overrides and self.cfg["reuse_same_model"]:
                return self._h
            if self._h:
                self.release(self._model)          # 1-3: terminate previous worker, free memory
            self.assert_clean()                    # 4: verify memory state BEFORE loading anything
            spec = self.reg.spec(model, overrides)
            if spec["backend"] == "llama_server":
                spec["llama_server_bin"] = self.find_llama_server()
            spec["load_timeout_s"] = self.cfg["timeouts"]["load_s"]
            self._preflight(spec)
            h = WorkerHandle(spec, self.run_dir, self.cuda)
            self.log.info(f"loading {model} ctx={spec['n_ctx']} ngl={spec['n_gpu_layers']} backend={spec['backend']}")
            res = h.start(self.cfg["timeouts"]["load_s"])   # 5-6
            if res["status"] != "success":
                h.kill()
                self._h, self._model = None, None
                raise ModelLoadError({**res, "model": model})
            self._h, self._model, self._ov = h, model, dict(overrides)
            return h

    def release(self, model: str | None = None):
        with self._lock:
            if not self._h:
                return
            h, m = self._h, self._model
            self.log.info(f"releasing {m} (pid {h.pid})")
            h.stop()
            self._h, self._model, self._ov = None, None, {}
            self._wait_vram_release()

    def assert_clean(self, wait: bool = True):
        with self._lock:
            live = self._h.pid if (self._h and self._h.alive()) else None
            orphans = find_worker_processes(exclude_pid=live)
            if orphans:
                self.log.warning(f"orphan worker processes: {[p.pid for p in orphans]} - killing")
                self.kill_orphans(orphans)
                orphans = find_worker_processes(exclude_pid=live)
                if orphans:
                    raise ResourceLeak(f"orphan workers survive: {[p.pid for p in orphans]}")
            if self._h is None and self.baseline_vram is not None:
                if wait:
                    self._wait_vram_release()
                used = hardware.gpu_used_mb(self.cuda)
                if used is not None and used > self.baseline_vram + self.cfg["vram_release_tolerance_mb"]:
                    raise ResourceLeak(f"VRAM not released: {used} MB used vs baseline {self.baseline_vram} MB "
                                       f"(tolerance {self.cfg['vram_release_tolerance_mb']} MB)")
            return True

    def state(self) -> dict:
        used = hardware.gpu_used_mb(self.cuda)
        return {"current_model": self.current_model(), "worker_pid": self.worker_pid(), "vram_used_mb": used,
                "vram_baseline_mb": self.baseline_vram, "ram": hardware.ram_info(),
                "orphans": [p.pid for p in find_worker_processes(exclude_pid=self.worker_pid())]}

    def sample(self) -> dict:
        used = hardware.gpu_used_mb(self.cuda)
        return {"vram_mb": (used - self.baseline_vram) if (used is not None and self.baseline_vram is not None) else used,
                "ram_mb": hardware.tree_rss_mb(self.worker_pid()) if self.worker_pid() else 0}

    def degrade(self, model: str, ov: dict) -> dict:
        """OOM fallback: halve context first, then cut GPU layers. Never loads a second model."""
        c = self.cfg["oom_retry"]
        s = self.reg.spec(model, ov)
        new = dict(ov)
        if s["n_ctx"] > c["min_ctx"] * 1.01:
            new["n_ctx"] = max(c["min_ctx"], int(s["n_ctx"] * c["ctx_factor"]))
        else:
            cur = s["n_gpu_layers"]
            new["n_gpu_layers"] = 32 if cur < 0 else max(0, int(cur * c["layers_factor"]))
        return new

    # ------------------------------------------------------------ internals
    def kill_orphans(self, procs=None):
        for p in (procs or find_worker_processes(self._h.pid if self._h else None)):
            try:
                for ch in p.children(recursive=True):
                    ch.kill()
                p.kill()
            except psutil.NoSuchProcess:
                pass
        time.sleep(0.5)

    def _wait_vram_release(self):
        if self.baseline_vram is None:
            return
        end = time.time() + self.cfg["vram_release_timeout_s"]
        while time.time() < end:
            used = hardware.gpu_used_mb(self.cuda)
            if used is None or used <= self.baseline_vram + self.cfg["vram_release_tolerance_mb"]:
                return
            time.sleep(1)

    def _preflight(self, spec):
        if spec["backend"] == "mock":
            return
        p = pathlib.Path(spec["path"])
        if not p.exists():
            raise ModelLoadError({"status": "missing", "error": f"model file missing: {p}", "model": spec["key"]})
        size = p.stat().st_size // (1 << 20)
        ram = hardware.ram_info()
        if size > ram["total_mb"] - self.cfg["reserved"]["ram_mb"] and spec["n_gpu_layers"] == 0:
            raise ModelLoadError({"status": "insufficient_ram", "error": f"{size} MB model > usable RAM", "model": spec["key"]})
        gpus = hardware.gpu_info(self.cuda)
        if gpus and spec["n_gpu_layers"] != 0:
            free = sum(g["vram_total_mb"] for g in gpus) - (hardware.gpu_used_mb(self.cuda) or 0)
            if free < 1500:
                raise ModelLoadError({"status": "insufficient_vram", "error": f"only {free} MB VRAM free before load", "model": spec["key"]})

    def find_llama_server(self) -> str | None:
        v = os.environ.get("LLAMA_SERVER_BIN") or self.cfg["llama_server_bin"]
        if v and v != "auto" and os.path.exists(v):
            return v
        cands = [shutil.which("llama-server"), str(self.cfg.work_dir / "llama.cpp/build/bin/llama-server")]
        cands += glob.glob("/kaggle/input/*/llama-server") + glob.glob("/kaggle/input/*/*/llama-server")
        return next((c for c in cands if c and os.path.exists(c)), None)
