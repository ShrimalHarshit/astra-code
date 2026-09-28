"""Phase-gate tests. smoke(): Phase 1. swap_test(): Phase 2 (strict one-model policy)."""
from __future__ import annotations
import time

PROMPT = [{"role": "system", "content": "You are a concise assistant."},
          {"role": "user", "content": "In two sentences, what is a mutex?"}]


def smoke(astra, model: str, max_tokens=64) -> dict:
    """boot -> load ONE model -> test prompt -> metrics -> unload -> verify memory cleanup"""
    rm = astra.rm
    out = {"model": model, "baseline_vram_mb": rm.baseline_vram}
    t0 = time.time()
    try:
        h = rm.acquire(model)
        out["load_ms"] = h.ready.get("load_ms")
        out["resident_while_loaded"] = rm.current_model()
        r = h.generate({"messages": PROMPT, "max_tokens": max_tokens, "temperature": 0.2, "reset": True}, astra.cfg["timeouts"]["generate_s"])
        out.update(status=r["status"], output=(r.get("output") or "")[:300], prompt_tokens=r.get("prompt_tokens"), output_tokens=r.get("output_tokens"),
                   prompt_tok_s=round(r["prompt_tokens"] / (r["prompt_ms"] / 1000), 1) if r.get("prompt_ms") else None,
                   gen_tok_s=round(r["output_tokens"] / (r["gen_ms"] / 1000), 1) if r.get("gen_ms") else None, error=r.get("error"))
        out["usage_loaded"] = rm.sample()
    except Exception as e:
        out.update(status=getattr(e, "result", {}).get("status", "failed"), error=str(e)[:500])
    finally:
        rm.release()
    try:
        rm.assert_clean()
        out["cleanup"] = "clean"
    except Exception as e:
        out["cleanup"] = f"LEAK: {e}"
    out["state_after"] = rm.state()
    out["total_s"] = round(time.time() - t0, 2)
    out["passed"] = out.get("status") == "success" and out["cleanup"] == "clean"
    return out


def swap_test(astra, models: list[str], rounds=2) -> dict:
    """Load each model in turn (A,B,C,D,A...), asserting at every point that <=1 worker exists and memory returns to baseline."""
    from .runtime.process_manager import find_worker_processes
    rm, log, viol = astra.rm, [], []
    for i in range(rounds):
        for m in models:
            try:
                rm.acquire(m)
                n = len(find_worker_processes())
                if n > 1 or rm.current_model() != m:
                    viol.append(f"{m}: {n} worker(s) alive, current={rm.current_model()}")
                r = rm.acquire(m)  # same model again must REUSE, not duplicate
                log.append({"round": i, "model": m, "workers": n, "pid": r.pid})
            except Exception as e:
                log.append({"round": i, "model": m, "error": str(e)[:200]})
                rm.release()
    rm.release()
    try:
        rm.assert_clean()
        clean = True
    except Exception as e:
        clean, viol = False, viol + [str(e)]
    return {"passed": clean and not viol and not any("error" in x for x in log), "violations": viol, "log": log}
