from __future__ import annotations
import base64, subprocess
from ..core.config import scrubbed_env


class Git:
    def __init__(self, shell, secrets, root):
        self.sh, self.secrets, self.root = shell, secrets, root

    def status(self):
        return self.sh.run(["git", "status", "--short", "--branch"])["output"]

    def diff(self, path=None):
        return self.sh.run(["git", "diff", "--stat"] + ([path] if path else []))["output"]

    def commit(self, message: str):
        self.sh.run(["git", "add", "-A"])
        return self.sh.run(["git", "-c", "user.name=Astra", "-c", "user.email=astra@localhost", "commit", "-m", message])["output"]

    def clone(self, url: str, dest: str = "repo") -> str:
        """Token is injected through env (http.extraheader), never in argv, logs, or prompts."""
        env = scrubbed_env({"GIT_TERMINAL_PROMPT": "0"})
        tok = self.secrets.get("GITHUB_TOKEN")
        if tok and url.startswith("https://github.com/"):
            hdr = "AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{tok}".encode()).decode()
            env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader", "GIT_CONFIG_VALUE_0": hdr})
        p = subprocess.run(["git", "clone", "--depth", "50", url, dest], cwd=self.root, env=env, capture_output=True, text=True, timeout=300)
        return self.secrets.redact(p.stdout + p.stderr)[-2000:]
