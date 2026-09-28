"""Memory system: long-term / session / task memory + a compiler that turns raw sessions into compact context."""
from __future__ import annotations
import hashlib, json, re, time
from ..agents import extract_json


class MemoryStore:
    def __init__(self, db):
        self.db = db

    def add(self, kind: str, content: str, importance=0.5, task_id=None, scope="global"):
        mid = "mem_" + hashlib.sha1(f"{kind}|{scope}|{content}".encode()).hexdigest()[:14]  # content-hash id => dedupe
        now = time.time()
        self.db.upsert("memories", ["memory_id"], {"memory_id": mid, "kind": kind, "scope": scope, "content": content[:2000], "importance": importance,
                                                    "source_task_id": task_id, "status": "active", "created_at": now, "updated_at": now})
        return mid

    def list(self, kind=None, limit=50):
        p = self.db.ph
        if kind:
            return self.db.query(f"SELECT * FROM memories WHERE kind={p} AND status='active' ORDER BY importance DESC, updated_at DESC LIMIT {int(limit)}", (kind,))
        return self.db.query(f"SELECT * FROM memories WHERE status='active' ORDER BY importance DESC, updated_at DESC LIMIT {int(limit)}")

    def compact_context(self, budget_chars=3000) -> str:
        out, used = [], 0
        for m in self.list(limit=80):
            line = f"- [{m['kind']}] {m['content']}"
            if used + len(line) > budget_chars:
                break
            out.append(line)
            used += len(line)
        return "\n".join(out)


EXTRACT_PROMPT = ("Extract durable memory from this finished task. Reply ONLY JSON: "
                  '{"facts": [..], "decisions": [..], "active_tasks": [..], "unresolved": [..]}. Max 5 short items per list.\n\n')


class MemoryCompiler:
    """raw session -> facts/decisions/active tasks/unresolved -> persistent memory -> compact context.
    Heuristic extraction always runs; an LLM pass (assistant model) is optional and only used when a model call is supplied."""
    def __init__(self, store: MemoryStore, log):
        self.store, self.log = store, log

    def compile(self, task, state, messages: list[dict] | None = None, call_model=None) -> dict:
        found = {"facts": [], "decisions": [], "active_tasks": [], "unresolved": []}
        for d in state.decisions[-10:]:
            found["decisions"].append(d["text"])
        for e in state.errors[-5:]:
            found["unresolved"].append(f"{e['step']}: {e['error'][:200]}")
        for k, v in list(state.project_context.items())[:10]:
            found["facts"].append(f"{k}: {json.dumps(v, default=str)[:200]}")
        if task.status != "completed":
            found["active_tasks"].append(f"{task.goal} (status {task.status}, at {task.current_step})")
        if call_model:
            try:
                summ = "\n".join(f"{s.id}:{s.status}:{json.dumps(state.step_outputs.get(s.id), default=str)[:300]}" for s in (task.plan.steps if task.plan else []))
                r = call_model([{"role": "system", "content": "You compile agent memory."},
                                {"role": "user", "content": EXTRACT_PROMPT + f"GOAL: {task.goal}\nSTEPS:\n{summ}"}], max_tokens=600, json_mode=True)
                j = extract_json(r["output"]) or {}
                for k in found:
                    found[k] += [str(x) for x in (j.get(k) or [])][:5]
            except Exception as e:
                self.log.warning(f"LLM memory extraction skipped: {e}")
        weights = {"facts": ("long_term", 0.6), "decisions": ("long_term", 0.7), "active_tasks": ("task", 0.8), "unresolved": ("task", 0.7)}
        n = 0
        for k, items in found.items():
            kind, w = weights[k]
            for it in items:
                self.store.add(kind, f"{k[:-1] if k.endswith('s') else k}: {it}", w, task.task_id, scope=task.project_id or "global")
                n += 1
        return {"stored": n, **{k: len(v) for k, v in found.items()}}
