"""Replaceable DB layer. Postgres (Supabase pooler) is durable. SQLite is a NON-DURABLE fallback: Astra will not claim safety with it."""
from __future__ import annotations
import json, pathlib, sqlite3, threading, time

MIG_DIR = pathlib.Path(__file__).parent / "migrations"


class DatabaseUnavailable(RuntimeError):
    pass


class Database:
    kind = "abstract"
    durable = False
    ph = "?"

    def execute(self, sql, params=()):
        raise NotImplementedError

    def query(self, sql, params=()) -> list[dict]:
        raise NotImplementedError

    def query_one(self, sql, params=()):
        r = self.query(sql, params)
        return r[0] if r else None

    def ping(self) -> bool:
        try:
            return self.query_one("SELECT 1 AS ok")["ok"] == 1
        except Exception:
            return False

    def upsert(self, table, keys: list[str], row: dict):
        cols = list(row)
        ph = ",".join([self.ph] * len(cols))
        upd = ",".join(f"{c}=excluded.{c}" for c in cols if c not in keys) or f"{keys[0]}=excluded.{keys[0]}"
        self.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph}) ON CONFLICT ({','.join(keys)}) DO UPDATE SET {upd}",
                     tuple(row[c] for c in cols))

    def migrate(self):
        self.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at DOUBLE PRECISION)")
        done = {r["version"] for r in self.query("SELECT version FROM schema_migrations")}
        applied = []
        for f in sorted(MIG_DIR.glob("*.sql")):
            if f.name.endswith(".pg.sql") and self.kind != "postgres":
                continue
            if f.name in done:
                continue
            for stmt in [s.strip() for s in f.read_text().split(";\n") if s.strip() and not s.strip().startswith("--")]:
                self.execute(stmt.rstrip(";"))
            self.execute(f"INSERT INTO schema_migrations (version, applied_at) VALUES ({self.ph},{self.ph})", (f.name, time.time()))
            applied.append(f.name)
        return applied

    def close(self):
        pass


class SqliteDB(Database):
    kind = "sqlite"
    durable = False  # /kaggle/working is wiped with the VM

    def __init__(self, path):
        self.path = str(path)
        self._c = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._c.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def execute(self, sql, params=()):
        with self._lock:
            self._c.execute(sql, params)

    def query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._c.execute(sql, params).fetchall()]

    def close(self):
        self._c.close()


class PostgresDB(Database):
    kind = "postgres"
    durable = True
    ph = "%s"

    def __init__(self, url: str):
        self.url = url
        self._lock = threading.RLock()
        self._c = None
        self._connect()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
            # prepare_threshold=None: required for Supabase transaction-mode pooler (port 6543)
            self._c = psycopg.connect(self.url, autocommit=True, row_factory=dict_row, prepare_threshold=None, connect_timeout=15)
        except Exception as e:
            raise DatabaseUnavailable(f"postgres connect failed: {type(e).__name__}: {str(e)[:200]}")

    def _run(self, fn):
        with self._lock:
            for attempt in (0, 1):
                try:
                    return fn()
                except Exception as e:
                    if attempt == 0 and "psycopg" in type(e).__module__ and type(e).__name__ in ("OperationalError", "InterfaceError"):
                        self._connect()
                        continue
                    raise

    def execute(self, sql, params=()):
        self._run(lambda: self._c.execute(sql.replace("?", "%s"), params))

    def query(self, sql, params=()):
        return self._run(lambda: self._c.execute(sql.replace("?", "%s"), params).fetchall())

    def close(self):
        try:
            self._c.close()
        except Exception:
            pass


def connect_database(cfg, secrets, log) -> Database:
    url = secrets.get("DATABASE_URL")
    if url:
        try:
            db = PostgresDB(url)
            log.info("database: postgres connected (durable)")
            return db
        except DatabaseUnavailable as e:
            log.error(f"database unavailable: {e}")
            if cfg.runtime.get("require_durable_db"):
                raise
    else:
        log.warning("DATABASE_URL not set")
    p = cfg.work_dir / "astra_local.sqlite"
    log.warning(f"FALLBACK: local SQLite at {p} - NOT DURABLE (lost when the Kaggle VM ends). Astra will not report state as safely persisted.")
    return SqliteDB(p)


def j(o) -> str:
    return json.dumps(o, default=str)
