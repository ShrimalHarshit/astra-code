"""Machine-readable JSON logs (secrets redacted) + human summaries."""
from __future__ import annotations
import json, logging, pathlib, threading, time
from .config import Secrets

_lock = threading.Lock()


class _Json(logging.Formatter):
    def format(self, r):
        d = {"ts": round(r.created, 3), "level": r.levelname, "logger": r.name, "msg": r.getMessage()}
        if r.exc_info:
            d["exc"] = self.formatException(r.exc_info)
        extra = getattr(r, "extra_fields", None)
        if extra:
            d.update(extra)
        return Secrets.redact(json.dumps(d, default=str))


class _Human(logging.Formatter):
    def format(self, r):
        return Secrets.redact(f"[{time.strftime('%H:%M:%S', time.localtime(r.created))}] {r.levelname:<7} {r.name}: {r.getMessage()}")


def get_logger(name: str, logs_dir: pathlib.Path | None = None, console=True) -> logging.Logger:
    lg = logging.getLogger(f"astra.{name}")
    if getattr(lg, "_astra", False):
        return lg
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if console:
        h = logging.StreamHandler()
        h.setFormatter(_Human())
        lg.addHandler(h)
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_dir / "astra.jsonl")
        fh.setFormatter(_Json())
        lg.addHandler(fh)
    lg._astra = True
    return lg


class ExecLog:
    """One JSON line per model execution + human summary."""
    def __init__(self, logs_dir: pathlib.Path):
        self.path = logs_dir / "executions.jsonl"
        logs_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict):
        with _lock, open(self.path, "a") as f:
            f.write(Secrets.redact(json.dumps(rec, default=str)) + "\n")

    @staticmethod
    def summary(r: dict) -> str:
        f = lambda k, d=0: r.get(k) if r.get(k) is not None else d
        return (f"{r.get('model')} step={r.get('step_id')} status={r.get('status')} load={f('load_ms')}ms "
                f"prompt={f('prompt_tokens')}t@{f('prompt_tok_s')}t/s gen={f('output_tokens')}t@{f('gen_tok_s')}t/s "
                f"ngl={r.get('gpu_layers')} ctx={r.get('context')} vram={r.get('vram_mb')}MB ram={r.get('ram_mb')}MB"
                + (f" error={r.get('error')}" if r.get("error") else ""))
