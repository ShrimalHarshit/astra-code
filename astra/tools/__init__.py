"""ToolBox: the tool surface Hermes may request. Tools run in the PARENT process (never inside model workers)."""
from __future__ import annotations
import json
from .filesystem import SandboxFS
from .shell import Shell
from .git import Git

S = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}
STR = {"type": "string"}
SCHEMAS = {
    "list_dir": ("List files under a workspace directory (depth 2).", S({"path": STR})),
    "read_file": ("Read a file with line numbers (optionally start/end lines).", S({"path": STR, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"])),
    "search_repo": ("Regex search across workspace files.", S({"pattern": STR, "path": STR, "glob": STR}, ["pattern"])),
    "write_file": ("Create/overwrite a file.", S({"path": STR, "content": STR}, ["path", "content"])),
    "edit_file": ("Replace exactly one occurrence of `old` with `new` in a file.", S({"path": STR, "old": STR, "new": STR}, ["path", "old", "new"])),
    "run_command": ("Run an allowlisted command (no shell, workspace cwd).", S({"command": STR}, ["command"])),
    "run_tests": ("Run the test suite (default: pytest -x -q).", S({"command": STR})),
    "git_status": ("git status.", S({})),
    "git_diff": ("git diff --stat.", S({"path": STR})),
    "mcp_list_tools": ("List tools of a registered MCP server (starts it lazily).", S({"server": STR}, ["server"])),
    "mcp_call": ("Call an MCP tool.", S({"server": STR, "tool": STR, "arguments": {"type": "object"}}, ["server", "tool"])),
    "inspect_task_state": ("Show a compact view of the current task state.", S({})),
    "ask_user": ("Ask the human a blocking question; task pauses in WAITING_FOR_USER.", S({"question": STR, "options": {"type": "array", "items": STR}}, ["question"])),
    "call_specialist": ("Delegate to another specialist by capability (mathematics|verification|conversation). You will be resumed with its result.",
                        S({"capability": STR, "task": STR}, ["capability", "task"])),
}


class ToolBox:
    def __init__(self, cfg, secrets, mcp=None):
        t = cfg["tools"]
        self.fs = SandboxFS(cfg.workspace_root, t["max_file_bytes"], t["max_output_bytes"])
        self.sh = Shell(cfg.workspace_root, t["shell_allow"], t["shell_timeout_s"], t["max_output_bytes"], t["allow_python_scripts"])
        self.git = Git(self.sh, secrets, self.fs.root)
        self.mcp = mcp
        self.changes: list[str] = []
        self.tests: list[dict] = []

    def begin_step(self):
        self.changes, self.tests = [], []

    def schemas(self, names=None):
        return [{"name": n, "description": d, "parameters": p} for n, (d, p) in SCHEMAS.items() if not names or n in names]

    def execute(self, name: str, a: dict, state=None) -> str:
        try:
            out = self._exec(name, a, state)
            ok = True
        except Exception as e:  # tool errors go back to the model as text; they never crash the loop
            out, ok = f"ERROR {type(e).__name__}: {str(e)[:500]}", False
        if state is not None:
            state.record_tool(name, {k: v for k, v in a.items() if k != "content"}, ok, out)
        return out if len(out) <= 12000 else out[:12000] + "...[truncated]"

    def _exec(self, n, a, state):
        if n == "list_dir":
            return self.fs.list(a.get("path", "."))
        if n == "read_file":
            return self.fs.read(a["path"], int(a.get("start", 1)), a.get("end"))
        if n == "search_repo":
            return self.fs.search(a["pattern"], a.get("path", "."), a.get("glob", "*"))
        if n == "write_file":
            r = self.fs.write(a["path"], a["content"]); self.changes.append(a["path"]); return r
        if n == "edit_file":
            r = self.fs.edit(a["path"], a["old"], a["new"]); self.changes.append(a["path"]); return r
        if n == "run_command":
            r = self.sh.run(a["command"]); return f"rc={r['returncode']}\n{r['output']}"
        if n == "run_tests":
            r = self.sh.run(a.get("command") or "pytest -x -q")
            self.tests.append({"command": a.get("command") or "pytest -x -q", "passed": r["returncode"] == 0, "summary": r["output"][-300:]})
            return f"rc={r['returncode']}\n{r['output']}"
        if n == "git_status":
            return self.git.status()
        if n == "git_diff":
            return self.git.diff(a.get("path"))
        if n == "mcp_list_tools":
            return json.dumps(self.mcp.list_tools(a["server"]))
        if n == "mcp_call":
            return self.mcp.call(a["server"], a["tool"], a.get("arguments") or {})
        if n == "inspect_task_state":
            return json.dumps({"goal": state.goal, "current_step": state.current_step, "completed": state.completed_steps,
                               "pending": state.pending_steps, "errors": state.errors[-3:]}, default=str)[:3000]
        raise ValueError(f"unknown tool '{n}'")
