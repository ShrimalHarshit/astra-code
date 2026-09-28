"""astra benchmark: each model INDIVIDUALLY (never concurrently): load -> warm up -> standard prompt -> measure -> save -> terminate -> verify cleanup."""
from __future__ import annotations
import json, threading, time, uuid
from ..core import hardware

STD_TEXT = ("Distributed systems fail in partial, surprising ways. A scheduler decides which task runs next, a resource manager decides what may be "
            "resident in memory, and a persistence layer decides what survives a crash. ") * 12
BENCH = [{"role": "system", "content": "You are a precise technical writer."},
         {"role": "user", "content": STD_TEXT + "\n\nSummarize the passage in exactly five bullet points."}]
WARM = [{"role": "user", "content": "Say OK."}]
SWEEP = [-1, 40, 32, 24, 16]


class Sampler(threading.Thread):
    """Polls VRAM (nvidia-smi) and worker-tree RSS every 0.5s while a model loads/runs; keeps peaks."""
    def __init__(self, rm):
        super().__init__(daemon=True)
        self.rm, self.stop_ev, self.vram, self.ram = rm, threading.Event(), 0, 0

    def run(self):
        while not self.stop_ev.is_set():
            s = self.rm.sample()
            self.vram = max(self.vram, s["vram_mb"] or 0)
            self.ram = max(self.ram, s["ram_mb"] or 0)
            self.stop_ev.wait(0.5)


def one_run(astra, key, gpu_layers, ctx) -> dict:
    rm = astra.rm
    spec = astra.reg.spec(key)
    ov = {"n_gpu_layers": gpu_layers if gpu_layers is not None else spec["n_gpu_layers"], "n_ctx": ctx or spec["n_ctx"]}
    row = {"model": key, "quantization": spec.get("quant"), "context": ov["n_ctx"], "gpu_layers": ov["n_gpu_layers"], "load_ms": None,
           "prompt_tokens": None, "output_tokens": None, "prompt_tok_s": None, "generation_tok_s": None, "vram_peak_mb": None,
           "ram_peak_mb": None, "total_ms": None, "status": "failed", "error": None}
    rm.release(); rm.assert_clean()
    smp = Sampler(rm); smp.start()
    t0 = time.perf_counter()
    try:
        h = rm.acquire(key, ov)
        row["load_ms"] = h.ready.get("load_ms")
        h.generate({"messages": WARM, "max_tokens": 8, "reset": True}, astra.cfg["timeouts"]["generate_s"])            # warm-up
        r = h.generate({"messages": BENCH, "max_tokens": 160, "temperature": 0.2, "reset": True}, astra.cfg["timeouts"]["generate_s"])
        row["status"] = r["status"]
        if r["status"] == "success":
            row.update(prompt_tokens=r["prompt_tokens"], output_tokens=r["output_tokens"],
                       prompt_tok_s=round(r["prompt_tokens"] / max(r["prompt_ms"], 1) * 1000, 1),
                       generation_tok_s=round(r["output_tokens"] / max(r["gen_ms"], 1) * 1000, 1))
        else:
            row["error"] = r.get("error")
    except Exception as e:
        row["status"], row["error"] = getattr(e, "result", {}).get("status", "failed"), str(e)[:300]
    finally:
        row["total_ms"] = int((time.perf_counter() - t0) * 1000)
        smp.stop_ev.set(); smp.join(2)
        row["vram_peak_mb"], row["ram_peak_mb"] = smp.vram or None, smp.ram or None
        rm.release()
    try:
        rm.assert_clean()
        row["cleanup"] = "clean"
    except Exception as e:
        row["cleanup"] = f"LEAK: {e}"
        row["status"] = "leak"
    return row


def save(astra, row):
    d = astra.cfg.results_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{row['model']}_ngl{row['gpu_layers']}_ctx{row['context']}_{int(time.time())}.json").write_text(json.dumps(row, indent=1))
    if getattr(astra, "tasks", None):
        try:
            astra.tasks.record_metric({k: row[k] for k in ("model", "context", "gpu_layers", "load_ms", "prompt_tokens", "output_tokens",
                                                            "prompt_tok_s", "total_ms", "status")} |
                                      {"quant": row["quantization"], "generation_tok_s": row["generation_tok_s"],
                                       "vram_peak_mb": row["vram_peak_mb"], "ram_peak_mb": row["ram_peak_mb"]})
        except Exception as e:
            astra.log.warning(f"metric DB write failed: {e}")


def choose_best(rows, vram_total_mb, reserved_mb):
    """Fastest generation among OK runs whose VRAM peak leaves the reserved headroom."""
    ok = [r for r in rows if r["status"] == "success" and (r["vram_peak_mb"] is None or not vram_total_mb or r["vram_peak_mb"] <= vram_total_mb - reserved_mb)]
    return max(ok, key=lambda r: r["generation_tok_s"] or 0) if ok else None


def run_benchmark(astra, models=None, sweep=False, layers=None, ctx=None) -> dict:
    models = models or astra.reg.keys()
    gpus = hardware.gpu_info(astra.cfg["cuda_visible_devices"])
    total = sum(g["vram_total_mb"] for g in gpus) if gpus else None
    best, table = {}, []
    for key in models:
        cand = layers or (SWEEP if sweep else [None])
        rows = []
        for gl in cand:
            print(f"== {key} ngl={gl if gl is not None else 'registry'} ctx={ctx or 'registry'}")
            row = one_run(astra, key, gl, ctx)
            save(astra, row)
            rows.append(row); table.append(row)
            print(f"   {row['status']:<9} load={row['load_ms']}ms prompt={row['prompt_tok_s']} t/s gen={row['generation_tok_s']} t/s "
                  f"vram={row['vram_peak_mb']}MB ram={row['ram_peak_mb']}MB {row.get('error') or ''}")
            if row["status"] == "leak":
                raise RuntimeError("memory leak detected after worker exit - aborting benchmark")
        b = choose_best(rows, total, astra.cfg["reserved"]["vram_mb"])
        if b:
            best[key] = {"gpu_layers": b["gpu_layers"], "context": b["context"], "generation_tok_s": b["generation_tok_s"], "vram_peak_mb": b["vram_peak_mb"]}
    if best:
        p = astra.cfg.results_dir / "best.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        p.write_text(json.dumps({**old, **best}, indent=1))
        if getattr(astra, "tasks", None):
            try:
                astra.tasks.config_set("benchmarked_models", {**old, **best})
            except Exception as e:
                astra.log.warning(f"could not persist benchmark overlay: {e}")
        astra.cfg.apply_overlay(best)
    print("\nmodel        load_ms  vram_MB  ram_MB  prompt t/s  gen t/s  ngl  ctx    status")
    for r in table:
        print(f"{r['model']:<12} {r['load_ms']!s:<8} {r['vram_peak_mb']!s:<8} {r['ram_peak_mb']!s:<7} {r['prompt_tok_s']!s:<11} {r['generation_tok_s']!s:<8} {r['gpu_layers']!s:<4} {r['context']!s:<6} {r['status']}")
    return {"rows": table, "best": best}
