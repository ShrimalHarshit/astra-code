"""Checkpoints: written to the persistent DB, read back and hash-verified. Never claims 'safe' unless durable AND verified."""
from __future__ import annotations
import hashlib, json, pathlib, time, uuid


class CheckpointResult(dict):
    @property
    def safe(self) -> bool:
        return bool(self.get("verified") and self.get("durable"))


class CheckpointManager:
    def __init__(self, db, cfg, log, bus=None):
        self.db, self.cfg, self.log, self.bus = db, cfg, log, bus
        self.mirror = cfg.work_dir / "checkpoints"
        self.mirror.mkdir(parents=True, exist_ok=True)
        self._seq: dict[str, int] = {}

    def save(self, task, state, label: str) -> CheckpointResult:
        blob = json.dumps({"task": task.to_dict(), "state": state.to_dict()}, default=str)
        sha = hashlib.sha256(blob.encode()).hexdigest()
        seq = self._seq.get(task.task_id)
        if seq is None:
            r = self.db.query_one(f"SELECT MAX(seq) AS m FROM checkpoints WHERE task_id={self.db.ph}", (task.task_id,)) if self.db.ping() else None
            seq = (r["m"] or 0) if r else 0
        seq += 1
        self._seq[task.task_id] = seq
        cid = f"c_{task.task_id}_{seq}_{uuid.uuid4().hex[:4]}"
        (self.mirror / f"{task.task_id}.json").write_text(blob)  # local mirror: fast, NOT authoritative
        res = CheckpointResult(checkpoint_id=cid, seq=seq, label=label, sha256=sha, verified=False, durable=self.db.durable, attempts=0)
        p = self.cfg["persistence"]
        for attempt in range(1, p["checkpoint_retries"] + 1):
            res["attempts"] = attempt
            try:
                self.db.upsert("checkpoints", ["checkpoint_id"], {"checkpoint_id": cid, "task_id": task.task_id, "seq": seq, "label": label,
                                                                   "state": blob, "sha256": sha, "size_bytes": len(blob), "created_at": time.time()})
                back = self.db.query_one(f"SELECT sha256, state FROM checkpoints WHERE checkpoint_id={self.db.ph}", (cid,))
                if back and back["sha256"] == sha and hashlib.sha256(back["state"].encode()).hexdigest() == sha:
                    res["verified"] = True
                    break
                res["error"] = "read-back mismatch"
            except Exception as e:
                res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            self.log.warning(f"checkpoint '{label}' attempt {attempt} failed: {res.get('error')}")
            time.sleep(p["checkpoint_backoff_s"] * attempt)
        if not res.safe:
            why = res.get("error") or "database is not durable (local SQLite fallback)"
            self.log.error(f"Checkpoint persistence failed ({why}). State is NOT safely persisted.")
            if self.bus:
                self.bus.publish(task.task_id, {"type": "checkpoint_warning", "label": label, "detail": why,
                                                "message": "Checkpoint persistence failed. I will not shut down until the state is safely persisted."})
        return res

    def latest(self, task_id: str):
        r = None
        try:
            r = self.db.query_one(f"SELECT state, label, created_at, seq FROM checkpoints WHERE task_id={self.db.ph} ORDER BY seq DESC LIMIT 1", (task_id,))
        except Exception as e:
            self.log.error(f"checkpoint read failed: {e}")
        if r:
            d = json.loads(r["state"])
            d["_meta"] = {"label": r["label"], "created_at": r["created_at"], "seq": r["seq"]}
            return d
        p = self.mirror / f"{task_id}.json"  # last resort: same-VM mirror
        return json.loads(p.read_text()) if p.exists() else None
