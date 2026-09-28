from __future__ import annotations
from . import load_prompt
from .hermes_agent import HermesLoop


class EngineerAgent:
    """INPUT: task/relevant files/constraints/previous results (via context + tools). OUTPUT: {status, changes[], tests[], notes[]}"""
    TOOLS = ["list_dir", "read_file", "search_repo", "write_file", "edit_file", "run_command", "run_tests", "git_status", "git_diff",
             "mcp_list_tools", "mcp_call", "inspect_task_state", "ask_user", "call_specialist"]

    def run(self, ctx):
        ctx.tools.begin_step()
        res = HermesLoop(ctx, load_prompt("engineer"), self.TOOLS).run()
        if res["status"] == "success":
            o = res["output"]
            res["output"] = {"status": o.get("status", "success"),
                             "changes": o.get("changes") or ctx.tools.changes,     # harness-observed facts win over model claims
                             "tests": o.get("tests") or ctx.tools.tests, "notes": o.get("notes", [])}
            if ctx.tools.tests and any(not t["passed"] for t in ctx.tools.tests[-1:]):
                res["output"]["status"] = "failed"
                res["output"]["notes"].append("last test run FAILED (observed by harness)")
        return res
