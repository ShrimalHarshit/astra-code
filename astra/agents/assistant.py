from __future__ import annotations
from . import load_prompt, extract_json, strip_think


class AssistantAgent:
    """INPUT: user goal / execution summary / artifacts. OUTPUT: {response}"""
    def run(self, ctx):
        summary = []
        for s in (ctx.task.plan.steps if ctx.task.plan else []):
            summary.append(f"{s.id} [{s.agent}] {s.action}: {s.status}")
        body = ctx.render_context() + "\n\nEXECUTION SUMMARY:\n" + "\n".join(summary)
        if ctx.state.errors:
            body += f"\nERRORS OBSERVED: {len(ctx.state.errors)} (latest: {ctx.state.errors[-1]['error'][:200]})"
        if ctx.state.artifacts:
            body += "\nARTIFACTS: " + ", ".join(a["name"] for a in ctx.state.artifacts)
        r = ctx.call_model([{"role": "system", "content": load_prompt("assistant")}, {"role": "user", "content": body}],
                           max_tokens=ctx.spec.get("max_tokens", 2048), temperature=ctx.spec.get("temperature", 0.4))
        d = extract_json(r["output"])
        return {"status": "success", "output": {"response": (d or {}).get("response") or strip_think(r["output"])}}
