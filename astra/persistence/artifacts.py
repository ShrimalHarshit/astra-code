"""Object storage for large artifacts/logs/backups. Supabase Storage (REST) or local-dir fallback (non-durable)."""
from __future__ import annotations
import hashlib, pathlib, time, uuid
import requests


class ObjectStoreUnavailable(RuntimeError):
    pass


class LocalStore:
    kind, durable = "local", False

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key, data: bytes):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key) -> bytes:
        return (self.root / key).read_bytes()

    def ping(self):
        return self.root.exists()


class SupabaseStore:
    kind, durable = "supabase", True

    def __init__(self, url, key, bucket):
        self.base, self.bucket = url.rstrip("/") + "/storage/v1", bucket
        self.h = {"Authorization": f"Bearer {key}", "apikey": key}
        try:  # create private bucket if absent (400/409 = already exists)
            requests.post(f"{self.base}/bucket", headers=self.h, json={"id": bucket, "name": bucket, "public": False}, timeout=15)
        except requests.RequestException as e:
            raise ObjectStoreUnavailable(str(e)[:200])

    def put(self, key, data: bytes):
        try:
            r = requests.post(f"{self.base}/object/{self.bucket}/{key}", headers={**self.h, "x-upsert": "true"}, data=data, timeout=120)
        except requests.RequestException as e:
            raise ObjectStoreUnavailable(str(e)[:200])
        if r.status_code >= 300:
            raise ObjectStoreUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")

    def get(self, key) -> bytes:
        r = requests.get(f"{self.base}/object/{self.bucket}/{key}", headers=self.h, timeout=120)
        if r.status_code >= 300:
            raise ObjectStoreUnavailable(f"HTTP {r.status_code}")
        return r.content

    def ping(self):
        try:
            return requests.get(f"{self.base}/bucket/{self.bucket}", headers=self.h, timeout=10).status_code < 300
        except requests.RequestException:
            return False


def connect_object_store(cfg, secrets, log):
    url, key = secrets.get("SUPABASE_URL"), secrets.get("SUPABASE_SERVICE_KEY")
    if url and key:
        try:
            s = SupabaseStore(url, key, cfg["persistence"]["object_bucket"])
            if s.ping():
                log.info("object storage: supabase connected")
                return s
        except ObjectStoreUnavailable as e:
            log.error(f"object storage unavailable: {e}")
    log.warning("object storage: local fallback (NOT DURABLE)")
    return LocalStore(cfg.work_dir / "objects")


class ArtifactStore:
    def __init__(self, db, store):
        self.db, self.store = db, store

    def put(self, task_id: str, name: str, data: bytes) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        key = f"{task_id}/{int(time.time())}_{name}"
        self.store.put(key, data)
        rec = {"artifact_id": "a_" + uuid.uuid4().hex[:10], "task_id": task_id, "name": name, "storage": self.store.kind,
               "key": key, "size_bytes": len(data), "sha256": sha, "created_at": time.time()}
        self.db.upsert("artifacts", ["artifact_id"], rec)
        return {k: rec[k] for k in ("name", "storage", "key", "sha256", "size_bytes")}

    def get(self, key) -> bytes:
        return self.store.get(key)
