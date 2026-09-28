# Astra — ephemeral Kaggle local-AI agent server

Kaggle is disposable compute. **State** lives in Supabase/Postgres + object storage, **models** in a Kaggle Dataset, **MCP servers and model workers** are recreatable processes. Exactly one LLM is resident at a time; each model runs in its own subprocess that exits (SIGTERM→SIGKILL of the whole process group) before the next loads, and VRAM is checked back to baseline before every load.

```
astra/  config/{models,runtime,mcp}.yaml   core/{scheduler,router,task,state,resource_manager,checkpoint,events,config,hardware,logging_utils}.py
        runtime/{worker,llama_runner,process_manager}.py   agents/{hermes_agent,orchestrator,engineer,logic,assistant}.py
        tools/{filesystem,shell,git,mcp}.py   persistence/{database,memory,sessions,tasks,artifacts}.py + migrations/
        benchmarks/benchmark.py   prompts/*.txt   bootstrap.py  main.py (controller+CLI)  gateway.py (HTTP)  selftest.py
astra_kaggle_server.ipynb   requirements.txt   .env.example   tests/
```

## Quick start (Kaggle)
1. Upload this folder as a Kaggle Dataset (`astra-code`) or push to GitHub (`ASTRA_REPO_URL`). Upload GGUFs as a second Dataset and edit `config/models.yaml` filenames to match.
2. New notebook → import `astra_kaggle_server.ipynb` → GPU T4, Internet on, attach both datasets.
3. Add-ons → Secrets: `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `ASTRA_API_TOKEN`, optional `GITHUB_TOKEN`. Attach them to the notebook (Kaggle secrets are per-notebook and must be attached manually; they cannot be shipped in a dataset).
4. Run top to bottom. §7 (Phase 1) and §8 (Phase 2) gate the agent layer.

Supabase: use the **pooler** connection string (Kaggle is IPv4-only; the direct `db.<ref>.supabase.co` host is IPv6-only). Migrations run automatically (`persistence/migrations`; RLS enabled on all tables).

CLI: `python -m astra.main status|models|mcp|memory|tasks|run "<goal>"|resume [id]|shutdown|benchmark [--sweep]|serve|smoke <model>|selftest`

## Status — read this
| Verified here (sandbox, **mock backend**, real subprocess workers, SQLite) | NOT verified (needs your T4) |
|---|---|
| one-model invariant + orphan kill, swap/reuse, timeout kill, OOM→degrade→fail without losing state, plan validation/routing, checkpoint write+read-back+resume, WAITING_FOR_USER blocks idle shutdown, shutdown refusal when not durable, sandboxed fs/shell, secret redaction, benchmark flow | real llama.cpp loading, CUDA/VRAM numbers, OOM classification from real stderr, llama-cpp-python wheel/build, llama-server build, Supabase Postgres/Storage, MCP servers (npx packages), HTTP gateway over a tunnel, model output quality (planner JSON, Hermes tool-call reliability) |

Only phases 1–2 are meant to be *run and trusted first* on Kaggle; phases 3–11 are implemented and unit-tested with the mock but have never touched a GPU.

## Things you must know
- **"Qwen 2.5 27B" does not exist** (Qwen2.5 dense: 0.5/1.5/3/7/14/32/72B). Search results point to Qwen3.x 27B dense releases. Their architecture is new, so `orchestrator` defaults to `backend: llama_server` (llama.cpp master) instead of llama-cpp-python, whose wheels can lag. If any model fails with "unknown model architecture", set `backend: llama_server` for it. Any Devstral/Mistral Small version: same rule.
- **27B at Q4 (~16–17 GB) does not fit a 15 GB T4.** Expect partial offload (rest in RAM, much slower) or a smaller quant (Q3_K_M / IQ4_XS). `astra benchmark --sweep` measures this; the result overrides `gpu_layers` in the registry via `best.json` + DB. Load time for 15–20 GB from `/kaggle/input` is minutes per swap — that is the accepted trade-off.
- llama-server has no prebuilt Linux CUDA binary, so it is built from source (`BUILD_LLAMA_SERVER=True`, ~10–15 min). Save `llama.cpp/build/bin/llama-server` into a Dataset to skip this later (`llama_server_bin: auto` scans `/kaggle/input`).
- `T4 x2` sessions: Astra pins `CUDA_VISIBLE_DEVICES=0`. Set `"0,1"` in `runtime.yaml` to use both (~30 GB VRAM, fits 27B Q4 fully) — this deviates from the single-16 GB design.
- **Shutdown is honest:** step 13 cannot terminate the Kaggle VM — there is no supported call. Astra stops MCP + workers, frees the GPU, persists, and reports `vm_terminated: false`. End the session with *Stop session*, Save & Run All, or Kaggle limits. `shutdown.kill_kernel: true` only kills the Python kernel.
- **Persistence policy:** without `DATABASE_URL` Astra falls back to local SQLite, prints a loud warning, marks every checkpoint `safe=False`, and `shutdown()` refuses unless `force=True`. Same for object storage (local fallback).
- Hermes: implemented as the Hermes `<tool_call>/<tool_response>` protocol run by the resident model; loop state is stored in `TaskState.agent_sessions`, so any swap (e.g. engineer → logic via `call_specialist`) serializes state, unloads, runs the specialist, reloads, and resumes. Prompt-level Hermes works with any chat template but small models may emit malformed calls; invalid calls are reported back to the model, not crashed on.
- Security: no shell; argv allowlist; `git push/remote/config/clean/reset` blocked; paths sandboxed (symlink escapes rejected); secrets never enter prompts, logs (redaction), or child-process env (scrubbed). With `allow_python_scripts: true` the engineer can run arbitrary Python — the real boundary is then the Kaggle VM. The HTTP API requires `ASTRA_API_TOKEN`; expose it only through an authenticated tunnel.
- MCP package names in `mcp.yaml` are examples — verify current packages (`npx` must exist on the VM).

## Known limitations / next steps
Planner quality decides everything; add plan-repair and per-step verification. No token-exact context accounting (chars/3.2 heuristic). KV-cache/state is not persisted across swaps (prompt is re-processed). Memory compiler's LLM pass reuses the assistant model at task end (one extra load if it wasn't the last model); disable with `memory_llm_pass: false`. Checksums are opt-in (`verify_checksums=True`) because hashing 60 GB is slow.
