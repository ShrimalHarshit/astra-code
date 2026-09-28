"""MCP manager. CONFIG = persistent (DB). PROCESS = ephemeral, recreated on boot or lazily on first use.
Minimal stdio JSON-RPC MCP client: initialize -> tools/list -> tools/call. Secrets are resolved only at process start and never logged."""
from __future__ import annotations
import json, os, pathlib, subprocess, threading, time
from ..core.config import scrubbed_env
from ..persistence.database import j

AUTH_HINTS = ("401", "403", "unauthorized", "bad credentials", "forbidden", "invalid token", "authentication")


class MCPError(RuntimeError):
    pass


class MCPServer:
    def __init__(self, name, cfg, secrets, log_dir, workspace):
        self.name, self.cfg, self.secrets, self.workspace = name, cfg, secrets, str(workspace)
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"
        self.detail = ""
        self.tools: list[dict] = []
        self.last_used = time.time()
        self._id = 0
        self._resp: dict = {}
        self._cv = threading.Condition()
        self.errlog = pathlib.Path(log_dir) / f"mcp_{name}.stderr.log"

    def _resolve(self, s: str) -> str:
        return self.secrets.resolve(s, {"WORKSPACE_ROOT": self.workspace})

    def start(self, timeout=90) -> str:
        missing = [n for n in self.cfg.get("required_secrets", []) if not self.secrets.get(n)]
        if missing:
            self.status, self.detail = "auth_failed", f"missing secrets: {missing}"
            return self.status
        try:
            cmd = [self._resolve(self.cfg["command"])] + [self._resolve(a) for a in self.cfg.get("args", [])]
            env = scrubbed_env({k: self._resolve(v) for k, v in (self.cfg.get("env") or {}).items()})
        except KeyError as e:
            self.status, self.detail = "auth_failed", f"unresolved variable {e}"
            return self.status
        self.status = "starting"
        try:
            with open(self.errlog, "wb") as ef:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ef, text=True, bufsize=1,
                                             env=env, cwd=self.workspace, start_new_session=True)
        except FileNotFoundError:
            self.status, self.detail = "failed", f"command not found: {cmd[0]} (is node/npx installed?)"
            return self.status
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "astra", "version": "0.1"}}, timeout)
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self.tools = self._rpc("tools/list", {}, 30).get("tools", [])
            self.status, self.detail = "healthy", f"{len(self.tools)} tools"
        except Exception as e:
            err = self._stderr_tail().lower() + str(e).lower()
            self.status = "auth_failed" if any(h in err for h in AUTH_HINTS) else "failed"
            self.detail = self.secrets.redact(f"{e} | {self._stderr_tail(300)}")
            self.stop()
        return self.status

    def _stderr_tail(self, n=1000):
        try:
            return self.secrets.redact(self.errlog.read_text(errors="replace")[-n:])
        except OSError:
            return ""

    def _read(self):
        for line in self.proc.stdout:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in m:
                with self._cv:
                    self._resp[m["id"]] = m
                    self._cv.notify_all()

    def _send(self, o):
        self.proc.stdin.write(json.dumps(o) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params, timeout):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        end = time.time() + timeout
        with self._cv:
            while rid not in self._resp:
                if self.proc.poll() is not None:
                    raise MCPError(f"server exited rc={self.proc.returncode}")
                left = end - time.time()
                if left <= 0:
                    raise MCPError(f"timeout on {method}")
                self._cv.wait(min(left, 1))
            m = self._resp.pop(rid)
        if "error" in m:
            raise MCPError(str(m["error"])[:500])
        return m.get("result", {})

    def call(self, tool, args, timeout=120):
        self.last_used = time.time()
        r = self._rpc("tools/call", {"name": tool, "arguments": args or {}}, timeout)
        return "\n".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text") or json.dumps(r)[:4000]

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), 9)
                except Exception:
                    pass
        if self.status in ("healthy", "starting"):
            self.status = "stopped"


class MCPManager:
    def __init__(self, cfg, db, secrets, log):
        self.cfg, self.db, self.secrets, self.log = cfg, db, secrets, log
        self.servers: dict[str, MCPServer] = {}
        self._lock = threading.RLock()
        self._reaper = None

    # ---- persistent config
    def load_registry(self) -> dict:
        """Seed DB from mcp.yaml for names not yet present; DB is authoritative afterwards."""
        have = {r["name"] for r in self.db.query("SELECT name FROM mcp_registry")}
        for n, c in self.cfg.mcp.items():
            if n not in have:
                self.db.upsert("mcp_registry", ["name"], {"name": n, "enabled": 1 if c.get("enabled") else 0, "config": j(c), "updated_at": time.time()})
        return {r["name"]: {**json.loads(r["config"]), "enabled": bool(r["enabled"])} for r in self.db.query("SELECT * FROM mcp_registry")}

    def start_eager(self) -> dict:
        reg = self.load_registry()
        out = {}
        for n, c in reg.items():
            if c["enabled"] and not c.get("lazy", True):
                out[n] = self.ensure(n)
        return out

    def ensure(self, name: str) -> MCPServer:
        with self._lock:
            reg = self.load_registry()
            if name not in reg or not reg[name]["enabled"]:
                raise MCPError(f"MCP '{name}' is not registered/enabled")
            s = self.servers.get(name)
            if s and s.status == "healthy" and s.proc.poll() is None:
                return s
            s = MCPServer(name, reg[name], self.secrets, self.cfg.logs_dir, self.cfg.workspace_root)
            self.servers[name] = s
            self.log.info(f"MCP start: {name}")
            st = s.start()
            if st != "healthy":
                self.log.error(f"MCP {name} -> {st}: {s.detail}")
                raise MCPError(f"MCP '{name}' {st}: {s.detail}")
            self._ensure_reaper()
            return s

    def call(self, name, tool, args):
        return self.ensure(name).call(tool, args)

    def list_tools(self, name):
        return [{"name": t["name"], "description": t.get("description", "")[:200]} for t in self.ensure(name).tools]

    def status(self) -> dict:
        reg = self.load_registry() if self.db.ping() else {n: {**c} for n, c in self.cfg.mcp.items()}
        return {n: {"enabled": c.get("enabled"), "lazy": c.get("lazy", True),
                    "status": self.servers[n].status if n in self.servers else "not_started",
                    "detail": self.servers[n].detail if n in self.servers else ""} for n, c in reg.items()}

    def _ensure_reaper(self):
        if self._reaper:
            return
        def loop():
            while True:
                time.sleep(15)
                for s in list(self.servers.values()):
                    if s.status == "healthy" and time.time() - s.last_used > self.cfg["idle"]["mcp_idle_s"]:
                        self.log.info(f"MCP {s.name} idle -> stopping")
                        s.stop()
        self._reaper = threading.Thread(target=loop, daemon=True)
        self._reaper.start()

    def stop_all(self):
        for s in self.servers.values():
            s.stop()
