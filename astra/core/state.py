"""Persistent structured task state. Specialists never see raw history - only context_for()."""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field, asdict


def _cut(s, n):
    s = s if isinstance(s, str) else json.dumps(s, default=str)
    return s if len(s) <= n else s[:n] + f"...[+{len(s)-n} chars]"


@dataclass
class TaskState:
    task_id: str
    goal: str
    current_step: str | None = None
    completed_steps: list = field(default_factory=list)
    pending_steps: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)      # [{name, storage, key, sha256, size}]
    tool_results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    project_context: dict = field(default_factory=dict)
    step_outputs: dict = field(default_factory=dict)   # step_id -> contract output
    agent_sessions: dict = field(default_factory=dict) # step_id -> externalized Hermes loop state (survives model swaps)
    delegations: dict = field(default_factory=dict)
    waiting_question: dict | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def note(self, kind, text, step=None):
        getattr(self, kind).append({"t": round(time.time(), 1), "step": step, "text": _cut(text, 1500)})
        del getattr(self, kind)[:-100]

    def error(self, step, model, status, msg):
        self.errors.append({"t": round(time.time(), 1), "step": step, "model": model, "status": status, "error": _cut(msg, 1500)})

    def record_tool(self, name, args, ok, out):
        self.tool_results.append({"t": round(time.time(), 1), "tool": name, "args": _cut(args, 300), "ok": ok, "out": _cut(out, 300)})
        del self.tool_results[:-200]

    def context_for(self, step, budget_chars: int, memory_text: str = "") -> str:
        parts = [f"GOAL: {self.goal}", f"CURRENT STEP: {step.id} [{step.agent}/{step.action}] {step.description}"]
        deps = [f"- {d}: {_cut(self.step_outputs.get(d, '(no output)'), 3000)}" for d in step.depends_on]
        if deps:
            parts.append("RESULTS OF PREREQUISITE STEPS:\n" + "\n".join(deps))
        if self.decisions:
            parts.append("DECISIONS:\n" + "\n".join(f"- {_cut(d['text'], 300)}" for d in self.decisions[-8:]))
        if self.observations:
            parts.append("OBSERVATIONS:\n" + "\n".join(f"- {_cut(o['text'], 400)}" for o in self.observations[-8:]))
        if self.errors:
            parts.append("RECENT ERRORS:\n" + "\n".join(f"- {e['step']}/{e['status']}: {_cut(e['error'], 300)}" for e in self.errors[-3:]))
        if self.project_context:
            parts.append("PROJECT CONTEXT: " + _cut(self.project_context, 1500))
        if memory_text:
            parts.append("LONG-TERM MEMORY:\n" + _cut(memory_text, max(400, budget_chars // 4)))
        return _cut("\n\n".join(parts), budget_chars)
