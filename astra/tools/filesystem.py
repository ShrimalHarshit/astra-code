"""Sandboxed filesystem: every path is resolved and must stay inside workspace_root (symlink escapes rejected)."""
from __future__ import annotations
import os, pathlib, re


class SandboxError(PermissionError):
    pass


class SandboxFS:
    def __init__(self, root, max_file_bytes=1_000_000, max_output=20000):
        self.root = pathlib.Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file, self.max_out = max_file_bytes, max_output

    def path(self, p: str, must_exist=False) -> pathlib.Path:
        if "\x00" in p:
            raise SandboxError("bad path")
        full = (self.root / p).resolve() if not os.path.isabs(p) else pathlib.Path(p).resolve()
        if full != self.root and self.root not in full.parents:
            raise SandboxError(f"path escapes workspace: {p}")
        if ".git" in full.relative_to(self.root).parts[:1] and False:
            pass
        if must_exist and not full.exists():
            raise FileNotFoundError(p)
        return full

    def _trunc(self, s: str) -> str:
        return s if len(s) <= self.max_out else s[: self.max_out] + f"\n...[truncated {len(s)-self.max_out} chars]"

    def read(self, p, start=1, end=None) -> str:
        f = self.path(p, True)
        if f.stat().st_size > self.max_file:
            raise SandboxError(f"file too large ({f.stat().st_size} > {self.max_file})")
        lines = f.read_text(errors="replace").splitlines()
        sel = lines[max(0, start - 1): end]
        return self._trunc("\n".join(f"{i}: {l}" for i, l in enumerate(sel, start=max(1, start))))

    def write(self, p, content: str) -> str:
        if len(content.encode()) > self.max_file:
            raise SandboxError("content exceeds max file size")
        f = self.path(p)
        if ".git" in f.relative_to(self.root).parts:
            raise SandboxError("writing inside .git is not allowed")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"wrote {len(content)} chars to {f.relative_to(self.root)}"

    def edit(self, p, old: str, new: str) -> str:
        f = self.path(p, True)
        t = f.read_text()
        n = t.count(old)
        if n != 1:
            raise ValueError(f"'old' must match exactly once (matches: {n})")
        self.write(p, t.replace(old, new))
        return f"edited {p}"

    def list(self, p=".", depth=2) -> str:
        base = self.path(p, True)
        out = []
        for dp, dn, fn in os.walk(base):
            rel = pathlib.Path(dp).relative_to(base)
            if len(rel.parts) >= depth:
                dn[:] = []
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv")]
            for f in sorted(fn):
                out.append(str((rel / f)))
            if len(out) > 400:
                out.append("...[more files]")
                break
        return self._trunc("\n".join(out) or "(empty)")

    def search(self, pattern: str, p=".", glob="*") -> str:
        rx = re.compile(pattern)
        base, hits = self.path(p, True), []
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv")]
            for f in fn:
                if not pathlib.PurePath(f).match(glob):
                    continue
                fp = pathlib.Path(dp) / f
                try:
                    if fp.stat().st_size > self.max_file:
                        continue
                    for i, line in enumerate(fp.read_text(errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{fp.relative_to(self.root)}:{i}: {line.strip()[:200]}")
                            if len(hits) >= 100:
                                return self._trunc("\n".join(hits) + "\n...[100 hit limit]")
                except OSError:
                    continue
        return self._trunc("\n".join(hits) or "no matches")
