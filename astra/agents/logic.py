from __future__ import annotations
from . import load_prompt, extract_json, strip_think


class LogicAgent:
    """INPUT: problem/constraints/relevant info. OUTPUT: {answer, algorithm, complexity, risks[]}"""
    def run(self, ctx):
        msgs = [{"role": "system", "content": load_prompt("logic")}, {"role": "user", "content": ctx.render_context()}]
        r = ctx.call_model(msgs, max_tokens=ctx.spec.get("max_tokens", 4096), temperature=ctx.spec.get("temperature", 0.6))
        d = extract_json(r["output"])
        text = strip_think(r["output"])
        out = {"answer": (d or {}).get("answer", text[:3000]), "algorithm": (d or {}).get("algorithm", ""),
               "complexity": (d or {}).get("complexity", ""), "risks": (d or {}).get("risks", []) if d else ["output was not valid JSON"]}
        return {"status": "success", "output": out}
