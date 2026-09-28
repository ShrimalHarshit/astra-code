from __future__ import annotations
import time, uuid
from .database import j


class SessionStore:
    def __init__(self, db):
        self.db = db

    def start(self, hardware: dict, boot_report: dict | None = None) -> str:
        sid = "s_" + uuid.uuid4().hex[:10]
        self.db.upsert("sessions", ["session_id"], {"session_id": sid, "started_at": time.time(), "ended_at": None,
                                                     "hardware": j(hardware), "boot_report": j(boot_report or {}), "status": "active"})
        return sid

    def end(self, sid: str, status="ended"):
        self.db.execute(f"UPDATE sessions SET ended_at={self.db.ph}, status={self.db.ph} WHERE session_id={self.db.ph}", (time.time(), status, sid))

    def add_message(self, sid, role, content, task_id=None):
        self.db.upsert("conversations", ["msg_id"], {"msg_id": "m_" + uuid.uuid4().hex[:10], "session_id": sid, "task_id": task_id,
                                                      "role": role, "content": content, "created_at": time.time()})

    def recent(self, n=20, sid=None) -> list[dict]:
        p = self.db.ph
        if sid:
            rows = self.db.query(f"SELECT role, content, task_id, created_at FROM conversations WHERE session_id={p} ORDER BY created_at DESC LIMIT {int(n)}", (sid,))
        else:
            rows = self.db.query(f"SELECT role, content, task_id, created_at FROM conversations ORDER BY created_at DESC LIMIT {int(n)}")
        return rows[::-1]

    def last_session(self):
        return self.db.query_one("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1")
