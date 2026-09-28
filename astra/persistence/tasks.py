from __future__ import annotations
import json, time, uuid
from dataclasses import asdict
from .database import j
from ..core.task import Task, RUNNING, WAITING, PENDING, INTERRUPTED


class TaskStore:
    def __init__(self, db):
        self.db = db

    def save_task(self, t: Task):
        t.updated_at = time.time()
        self.db.upsert("tasks", ["task_id"], {"task_id": t.task_id, "goal": t.goal, "status": t.status, "project_id": t.project_id,
                                              "plan": j(t.plan.to_dict()) if t.plan else None, "last_model": t.last_model,
                                              "current_step": t.current_step, "created_at": t.created_at, "updated_at": t.updated_at,
                                              "finished_at": t.finished_at})
        if t.plan:
            for s in t.plan.steps:
                self.db.upsert("task_steps", ["task_id", "step_id"], {"task_id": t.task_id, "step_id": s.id, "agent": s.agent,
                                                                       "capability": s.capability, "action": s.action, "depends_on": j(s.depends_on),
                                                                       "status": s.status, "attempts": s.attempts, "output": j(s.output),
                                                                       "model_used": s.model_used, "started_at": s.started_at, "finished_at": s.finished_at})

    def list(self, statuses=None, limit=50) -> list[dict]:
        if statuses:
            ph = ",".join([self.db.ph] * len(statuses))
            return self.db.query(f"SELECT task_id, goal, status, current_step, last_model, updated_at FROM tasks WHERE status IN ({ph}) ORDER BY updated_at DESC LIMIT {int(limit)}", tuple(statuses))
        return self.db.query(f"SELECT task_id, goal, status, current_step, last_model, updated_at FROM tasks ORDER BY updated_at DESC LIMIT {int(limit)}")

    def find_interrupted(self) -> list[dict]:
        return self.list([RUNNING, WAITING, PENDING, INTERRUPTED])

    def get(self, task_id):
        return self.db.query_one(f"SELECT * FROM tasks WHERE task_id={self.db.ph}", (task_id,))

    def add_decision(self, task_id, step_id, decision, rationale=""):
        self.db.upsert("decisions", ["decision_id"], {"decision_id": "d_" + uuid.uuid4().hex[:10], "task_id": task_id, "step_id": step_id,
                                                       "decision": decision, "rationale": rationale, "created_at": time.time()})

    def record_execution(self, rec: dict):
        self.db.upsert("execution_history", ["exec_id"], {"exec_id": "e_" + uuid.uuid4().hex[:10], "task_id": rec.get("task_id"),
                                                           "step_id": rec.get("step_id"), "model": rec.get("model"), "status": rec.get("status"),
                                                           "record": j(rec), "created_at": time.time()})

    def record_metric(self, m: dict):
        row = {"run_id": "r_" + uuid.uuid4().hex[:10], "created_at": time.time(), **m}
        self.db.upsert("model_metrics", ["run_id"], row)

    def config_get(self, key, default=None):
        r = self.db.query_one(f"SELECT value FROM astra_config WHERE key={self.db.ph}", (key,))
        return json.loads(r["value"]) if r else default

    def config_set(self, key, value):
        self.db.upsert("astra_config", ["key"], {"key": key, "value": j(value), "updated_at": time.time()})
