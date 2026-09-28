"""Hardware detection and memory measurement (nvidia-smi + psutil)."""
from __future__ import annotations
import os, shutil, subprocess
import psutil


def _smi(query: str):
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return [[c.strip() for c in ln.split(",")] for ln in out.stdout.strip().splitlines() if ln.strip()]
    except Exception:
        return None


def visible_indices(cuda_visible: str | None = None) -> list[int] | None:
    v = cuda_visible if cuda_visible is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if not v:
        return None
    try:
        return [int(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:
        return None


def gpu_info(cuda_visible: str | None = None) -> list[dict]:
    rows = _smi("index,name,memory.total,memory.used,driver_version")
    if not rows:
        return []
    keep = visible_indices(cuda_visible)
    out = []
    for r in rows:
        try:
            i = int(r[0])
            if keep is not None and i not in keep:
                continue
            out.append({"index": i, "name": r[1], "vram_total_mb": int(float(r[2])), "vram_used_mb": int(float(r[3])), "driver": r[4]})
        except (ValueError, IndexError):
            continue
    return out


def gpu_used_mb(cuda_visible: str | None = None) -> int | None:
    g = gpu_info(cuda_visible)
    return sum(x["vram_used_mb"] for x in g) if g else None


def ram_info() -> dict:
    vm = psutil.virtual_memory()
    return {"total_mb": vm.total // (1 << 20), "available_mb": vm.available // (1 << 20), "used_mb": vm.used // (1 << 20)}


def disk_free_mb(path) -> int:
    p = str(path)
    while not os.path.exists(p) and p != "/":
        p = os.path.dirname(p) or "/"
    return shutil.disk_usage(p).free // (1 << 20)


def tree_rss_mb(pid: int) -> int:
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(x.memory_info().rss for x in procs if x.is_running()) // (1 << 20)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def detect(cuda_visible: str | None = None) -> dict:
    return {"gpus": gpu_info(cuda_visible), "ram": ram_info(), "cpu_count": os.cpu_count(),
            "nvidia_smi": bool(shutil.which("nvidia-smi"))}
