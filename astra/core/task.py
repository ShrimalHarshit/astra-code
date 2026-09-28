from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field, asdict

PENDING, RUNNING, WAITING, COMPLETED, FAILED, INTERRUPTED, SKIPPED = (
    "pending", "running", "waiting_for_user", "completed", "failed", "interrupted", "skipped")


class PlanError(ValueError):
    pass


@dataclass
class Step:
    id: str
    agent: str
    action: str
    capability: str | None = None
    description: str = ""
    depends_on: list = field(default_factory=list)
    status: str = PENDING
    attempts: int = 0
    output: dict | None = None
    model_used: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    goal: str
    steps: list

    def to_dict(self):
        return {"goal": self.goal, "steps": [asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["goal"], [Step.from_dict(s) for s in d["steps"]])

    @classmethod
    def from_llm(cls, d: dict, goal: str):
        steps = []
        for i, s in enumerate(d.get("steps") or []):
            if not isinstance(s, dict) or "agent" not in s:
                raise PlanError(f"step {i} malformed")
            steps.append(Step(id=str(s.get("id") or f"step_{i+1}"), agent=s["agent"], action=s.get("action", "work"),
                              capability=s.get("capability"), description=s.get("description", ""),
                              depends_on=list(s.get("depends_on") or [])))
        return cls(d.get("goal") or goal, steps)

    def validate(self, router):
        if not self.steps:
            raise PlanError("empty plan")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate step ids")
        for s in self.steps:
            for d in s.depends_on:
                if d not in ids:
                    raise PlanError(f"{s.id} depends on unknown {d}")
            router.resolve(s)  # raises RoutingError
        seen, remaining = set(), {s.id: set(s.depends_on) for s in self.steps}
        while remaining:  # Kahn cycle check
            ready = [i for i, deps in remaining.items() if deps <= seen]
            if not ready:
                raise PlanError("dependency cycle")
            for i in ready:
                seen.add(i)
                del remaining[i]

    def step(self, sid):
        return next(s for s in self.steps if s.id == sid)

    def ready(self):
        done = {s.id for s in self.steps if s.status == COMPLETED}
        return [s for s in self.steps if s.status in (PENDING, RUNNING, WAITING) and set(s.depends_on) <= done]


@dataclass
class Task:
    goal: str
    task_id: str = field(default_factory=lambda: "t_" + uuid.uuid4().hex[:10])
    status: str = PENDING
    project_id: str | None = None
    plan: Plan | None = None
    last_model: str | None = None
    current_step: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self):
        d = {k: getattr(self, k) for k in ("goal", "task_id", "status", "project_id", "last_model", "current_step",
                                             "created_at", "updated_at", "finished_at")}
        d["plan"] = self.plan.to_dict() if self.plan else None
        return d

    @classmethod
    def from_dict(cls, d):
        t = cls(goal=d["goal"], task_id=d["task_id"])
        for k in ("status", "project_id", "last_model", "current_step", "created_at", "updated_at", "finished_at"):
            setattr(t, k, d.get(k))
        t.plan = Plan.from_dict(d["plan"]) if d.get("plan") else None
        return t
