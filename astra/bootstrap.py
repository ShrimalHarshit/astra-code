"""Kaggle session bootstrap. Idempotent: safe to run twice. Steps mirror the spec (1-15); every step is timed."""
from __future__ import annotations
import json, os, pathlib, shutil, subprocess, sys, time
from .core import hardware
from .core.config import on_kaggle
from .core.resource_manager import ResourceManager

CHECK = ("import json,llama_cpp as l;print(json.dumps({'version':getattr(l,'__version__','?'),"
         "'gpu':bool(l.llama_supports_gpu_offload())}))")
WHEEL_INDEXES = ["cu124", "cu122", "cu121", "cu125"]


def _run(cmd, log, timeout=3600, env=None):
    log.info("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0:
        log.warning((p.stdout + p.stderr)[-800:])
    return p.returncode == 0


def check_llama_cpp() -> dict:
    """Checked in a SUBPROCESS: the parent must never import llama_cpp (would create a CUDA context that survives worker exit)."""
    p = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return {"installed": False, "error": (p.stderr.strip().splitlines() or ["import failed"])[-1]}
    return {"installed": True, **json.loads(p.stdout.strip().splitlines()[-1])}


def ensure_llama_cpp(log, install=True, source_build=False, need_gpu=True) -> dict:
    info = check_llama_cpp()
    if info.get("installed") and (info["gpu"] or not need_gpu):
        return {**info, "action": "already_ok"}
    if not install:
        return {**info, "action": "none", "warning": "llama-cpp-python missing or CPU-only"}
    for idx in WHEEL_INDEXES:  # prebuilt CUDA wheels: fast if one exists for this python/CUDA combo
        if _run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "--only-binary=:all:",
                 "llama-cpp-python", "--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{idx}"], log, 900):
            info = check_llama_cpp()
            if info.get("installed") and info["gpu"]:
                return {**info, "action": f"wheel:{idx}"}
    if source_build:
        env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75", "FORCE_CMAKE": "1"}
        if _run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-cache-dir", "llama-cpp-python"], log, 3600, env):
            info = check_llama_cpp()
            return {**info, "action": "source_build"}
    return {**info, "action": "failed", "warning": "no CUDA-enabled llama-cpp-python. Re-run with source_build=True (10-20 min) or use backend: llama_server"}


def build_llama_server(cfg, log) -> str | None:
    """Builds llama.cpp's llama-server with CUDA for sm_75 (T4). ~10-15 min. Save the result as a Kaggle Dataset to skip this next time."""
    src = cfg.work_dir / "llama.cpp"
    binp = src / "build/bin/llama-server"
    if binp.exists():
        return str(binp)
    if not src.exists() and not _run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src)], log, 600):
        return None
    ok = _run(["cmake", "-S", str(src), "-B", str(src / "build"), "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
               "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], log, 900)
    ok = ok and _run(["cmake", "--build", str(src / "build"), "--target", "llama-server", "-j", str(os.cpu_count() or 4)], log, 3600)
    return str(binp) if ok and binp.exists() else None


def run(astra, upto="full", install_runtime=True, source_build=False, verify_checksums=False) -> dict:
    """upto: 'diagnostics' (steps 1-6) | 'persistence' (adds 7-11) | 'full' (adds 12-13)"""
    from .persistence.database import connect_database
    from .persistence.artifacts import connect_object_store, ArtifactStore
    from .persistence.sessions import SessionStore
    from .persistence.tasks import TaskStore
    from .persistence.memory import MemoryStore, MemoryCompiler
    from .core.checkpoint import CheckpointManager
    from .core.scheduler import Scheduler
    from .tools import ToolBox
    from .tools.mcp import MCPManager

    cfg, log, R = astra.cfg, astra.log, {}
    T, t_all = {}, time.time()

    def step(name):
        class _T:
            def __enter__(s): s.t = time.time()
            def __exit__(s, *a): T[name] = round(time.time() - s.t, 2)
        return _T()

    with step("hardware"):
        hw = hardware.detect(cfg["cuda_visible_devices"])
        R["compute"] = hw
        for g in hw["gpus"]:
            log.info(f"GPU {g['index']}: {g['name']} {g['vram_total_mb']} MB total, {g['vram_used_mb']} used, driver {g['driver']}")
        if not hw["gpus"] and cfg["backend"] != "mock":
            log.warning("NO GPU detected. On Kaggle: Settings -> Accelerator -> GPU T4. CPU inference will be extremely slow.")
        log.info(f"RAM {hw['ram']['total_mb']} MB total, {hw['ram']['available_mb']} MB available; disk free {hardware.disk_free_mb(cfg.work_dir)} MB")
    with step("model_dataset"):
        R["models_dir"] = str(cfg.models_dir)
        vals = {k: astra.reg.validate(k, verify_checksums) for k in astra.reg.keys()}
        best = json.loads((cfg.results_dir / "best.json").read_text()) if (cfg.results_dir / "best.json").exists() else {}
        R["models"] = {"available": [k for k, v in vals.items() if v["status"] == "ok"],
                       "missing": [k for k, v in vals.items() if v["status"] == "missing"],
                       "invalid": {k: v for k, v in vals.items() if v["status"] not in ("ok", "missing")},
                       "benchmarked": [k for k in vals if k in best], "detail": vals}
    with step("runtime"):
        backends = {astra.reg.spec(k)["backend"] for k in astra.reg.keys()}
        R["runtime"] = {"backends": sorted(backends)}
        if "llama_cpp" in backends:
            R["runtime"]["llama_cpp"] = ensure_llama_cpp(log, install_runtime, source_build, need_gpu=bool(hw["gpus"]))
        if "llama_server" in backends:
            probe = ResourceManager.find_llama_server(type("P", (), {"cfg": cfg})())
            R["runtime"]["llama_server"] = {"path": probe, "action": "found" if probe else "missing (run bootstrap.build_llama_server or set LLAMA_SERVER_BIN)"}
    astra.rm = ResourceManager(cfg, astra.reg, log)
    R["cuda_baseline_vram_mb"] = astra.rm.baseline_vram
    if upto == "diagnostics":
        return _finish(astra, R, T, t_all)

    with step("persistence"):
        astra.db = connect_database(cfg, astra.secrets, log)
        applied = astra.db.migrate()
        astra.store = connect_object_store(cfg, astra.secrets, log)
        R["persistence"] = {"database": astra.db.kind, "database_connected": astra.db.ping(), "database_durable": astra.db.durable,
                            "object_storage": astra.store.kind, "object_storage_connected": bool(astra.store.ping()),
                            "object_storage_durable": astra.store.durable, "migrations_applied": applied}
        astra.tasks = TaskStore(astra.db)
        astra.sessions = SessionStore(astra.db)
        astra.artifacts = ArtifactStore(astra.db, astra.store)
        astra.mem = MemoryStore(astra.db)
        astra.compiler = MemoryCompiler(astra.mem, log)
        astra.ckpt = CheckpointManager(astra.db, cfg, log, astra.bus)
    with step("restore_config"):
        if not astra.db.query_one("SELECT id FROM astra_identity LIMIT 1"):
            astra.db.upsert("astra_identity", ["id"], {"id": "astra", "name": "Astra", "created_at": time.time(), "meta": "{}"})
        cfg.apply_overlay(astra.tasks.config_get("benchmarked_models", {}))
        cfg.apply_overlay(astra.tasks.config_get("degraded_overrides", {}))
    with step("restore_tasks_session"):
        astra.interrupted = astra.tasks.find_interrupted()
        last = astra.sessions.last_session()
        astra.session_id = astra.sessions.start(hw, {"prev_session": last["session_id"] if last else None})
        astra.recent_messages = astra.sessions.recent(10)
        R["tasks"] = {"interrupted": astra.interrupted, "completed": len(astra.tasks.list(["completed"], 1000)), "active": 0}
    astra.mcp = MCPManager(cfg, astra.db, astra.secrets, log)
    astra.toolbox = ToolBox(cfg, astra.secrets, astra.mcp)
    with step("mcp_registry"):
        astra.mcp.load_registry()
    astra.scheduler = Scheduler(cfg, astra.reg, astra.router, astra.rm, astra.tasks, astra.ckpt, astra.mem, astra.compiler, astra.toolbox,
                                astra.bus, astra.broker, astra.execlog, log, busy=astra.busy, on_state=astra.set_state)
    if upto == "persistence":
        return _finish(astra, R, T, t_all)

    with step("mcp_start_health"):
        R["mcp_eager"] = astra.mcp.start_eager()
    R["mcp"] = astra.mcp.status()
    return _finish(astra, R, T, t_all)


def _finish(astra, R, T, t_all):
    R["timings_s"], R["startup_s"] = T, round(time.time() - t_all, 2)
    R["kaggle"] = on_kaggle()
    return R


def format_report(R: dict) -> str:
    L = ["ASTRA READY", "", "Compute:"]
    for g in R["compute"]["gpus"]:
        L.append(f"  GPU: {g['name']}  VRAM: {g['vram_total_mb']} MB ({g['vram_used_mb']} used)")
    if not R["compute"]["gpus"]:
        L.append("  GPU: none detected")
    L.append(f"  RAM: {R['compute']['ram']['total_mb']} MB total / {R['compute']['ram']['available_mb']} MB available")
    m = R["models"]
    L += ["", "Models:", f"  available: {m['available'] or '-'}", f"  missing: {m['missing'] or '-'}",
          f"  invalid: {list(m['invalid']) or '-'}", f"  benchmarked: {m['benchmarked'] or '-'}"]
    if "persistence" in R:
        p = R["persistence"]
        L += ["", "Persistence:", f"  database: {p['database']} connected={p['database_connected']} durable={p['database_durable']}",
              f"  object storage: {p['object_storage']} connected={p['object_storage_connected']} durable={p['object_storage_durable']}"]
        if not p["database_durable"]:
            L.append("  !! DATABASE NOT DURABLE: state will be lost when this Kaggle VM ends. Set DATABASE_URL.")
    if "mcp" in R:
        L += ["", "MCP:"] + [f"  {n}: enabled={s['enabled']} status={s['status']} {s['detail']}" for n, s in R["mcp"].items()]
    if "tasks" in R:
        t = R["tasks"]
        L += ["", "Tasks:", f"  active: {t['active']}  interrupted: {len(t['interrupted'])}  completed: {t['completed']}"]
    L += ["", f"startup: {R['startup_s']}s  {R['timings_s']}", ""]
    if R.get("tasks", {}).get("interrupted"):
        for t in R["tasks"]["interrupted"][:3]:
            L.append(f"An interrupted task was found: '{t['goal'][:80]}' (id {t['task_id']}, step {t['current_step']}, last model {t['last_model']}). Resume it?")
    else:
        L.append("Ready for task.")
    return "\n".join(L)
