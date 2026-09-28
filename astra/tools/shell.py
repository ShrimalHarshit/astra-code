"""Restricted command runner: argv allowlist (prefix match), no shell, cwd pinned to workspace, timeout, output cap, scrubbed env."""
from __future__ import annotations
import os, pathlib, shlex, subprocess
from ..core.config import scrubbed_env


class CommandDenied(PermissionError):
    pass


class Shell:
    def __init__(self, root, allow: list[list[str]], timeout_s=180, max_output=20000, allow_python_scripts=False):
        self.root = pathlib.Path(root).resolve()
        self.allow = [list(a) for a in allow]
        self.timeout, self.max_out = timeout_s, max_output
        if allow_python_scripts:
            self.allow.append(["python"])

    def check(self, argv: list[str]):
        if not argv:
            raise CommandDenied("empty command")
        if not any(argv[: len(a)] == a for a in self.allow):
            raise CommandDenied(f"command not allowlisted: {argv[:3]}")
        for a in argv[1:]:
            if a.startswith("-"):
                continue
            if ".." in pathlib.PurePath(a).parts:
                raise CommandDenied(f"'..' not allowed in argument: {a}")
            if os.path.isabs(a):
                rp = pathlib.Path(a).resolve()
                if rp != self.root and self.root not in rp.parents:
                    raise CommandDenied(f"absolute path outside workspace: {a}")
        if argv[0] == "git" and len(argv) > 1 and argv[1] in ("push", "remote", "config", "credential", "clean", "reset"):
            raise CommandDenied(f"git {argv[1]} is blocked")
        if argv[0] == "find" and any(x in argv for x in ("-exec", "-execdir", "-delete", "-ok")):
            raise CommandDenied("find with -exec/-delete is blocked")

    def run(self, cmd: str | list[str], timeout: int | None = None) -> dict:
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        self.check(argv)
        try:
            p = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, timeout=timeout or self.timeout,
                               env=scrubbed_env({"HOME": str(self.root), "GIT_TERMINAL_PROMPT": "0"}), stdin=subprocess.DEVNULL)
            out = (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else ""))
            rc, to = p.returncode, False
        except subprocess.TimeoutExpired as e:
            out, rc, to = ((e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n[timed out]", -1, True
        except FileNotFoundError:
            raise CommandDenied(f"executable not found: {argv[0]}")
        if len(out) > self.max_out:
            out = out[: self.max_out // 2] + f"\n...[truncated {len(out)-self.max_out} chars]...\n" + out[-self.max_out // 2:]
        return {"returncode": rc, "output": out, "timed_out": to}
