"""Hermes-style tool-calling loop. Astra keeps ownership of lifecycle: the loop's entire state lives in TaskState.agent_sessions,
so the model can be unloaded/swapped at ANY tool boundary and the loop resumed later (serialize -> unload -> run specialist -> reload -> restore)."""
from __future__ import annotations
import json, re
from . import strip_think, extract_json, load_prompt

TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
CONTROL = ("ask_user", "call_specialist")


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TC_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
            calls.append({"name": d["name"], "arguments": d.get("arguments") or {}})
        except (json.JSONDecodeError, KeyError, TypeError):
            calls.append({"name": "_invalid", "arguments": {"raw": m.group(1)[:200]}})
    return calls


def _tokens(msgs):
    return int(sum(len(m["content"]) for m in msgs) / 3.2)


class HermesLoop:
    def __init__(self, ctx, system_prompt: str, tool_names: list[str] | None = None):
        self.ctx = ctx
        tools = ctx.tools.schemas(tool_names)
        self.system = system_prompt + "\n\n" + load_prompt("hermes") + f"\n<tools>\n{json.dumps(tools)}\n</tools>"

    def _fit(self, sess):
        limit = int(self.ctx.spec["n_ctx"] * 0.7)
        msgs = sess["messages"]
        if _tokens(msgs) <= limit:
            return
        for m in msgs[2:-3]:  # 1) shrink old tool responses
            if m["role"] == "user" and "<tool_response>" in m["content"] and len(m["content"]) > 400:
                m["content"] = m["content"][:350] + "...[compacted]</tool_response>"
        while _tokens(msgs) > limit and len(msgs) > 6:  # 2) drop oldest assistant/user pairs (keeps role alternation)
            del msgs[2:4]
            msgs[1]["content"] += "\n[note: earlier turns were dropped to fit context]" if "dropped" not in msgs[1]["content"] else ""

    def run(self) -> dict:
        c = self.ctx
        sess = c.state.agent_sessions.get(c.step.id)
        if sess is None:
            sess = {"messages": [{"role": "system", "content": self.system},
                                 {"role": "user", "content": c.render_context() + "\n\nBegin."}], "iters": 0}
            c.state.agent_sessions[c.step.id] = sess
        pend = sess.pop("pending_result", None)  # answer to ask_user / result of call_specialist
        if pend:
            sess["messages"].append({"role": "user", "content": f"<tool_response>{json.dumps(pend)}</tool_response>"})
        while sess["iters"] < c.max_iters:
            sess["iters"] += 1
            self._fit(sess)
            res = c.call_model(sess["messages"], max_tokens=c.spec.get("max_tokens", 2048), temperature=c.spec.get("temperature", 0.2))
            text = strip_think(res["output"])
            sess["messages"].append({"role": "assistant", "content": text})
            calls = parse_tool_calls(text)
            if not calls:
                final = extract_json(text)
                return {"status": "success", "output": final if isinstance(final, dict) else {"notes": [text[:2000]]}}
            out = []
            for call in calls:
                n, a = call["name"], call["arguments"]
                if n == "ask_user":
                    c.checkpoint("agent_waiting_user")
                    return {"status": "needs_user", "question": a.get("question", "?"), "options": a.get("options")}
                if n == "call_specialist":
                    c.checkpoint("agent_delegating")
                    return {"status": "needs_specialist", "request": a}
                out.append({"name": n, "content": c.tools.execute(n, a, c.state)})
            sess["messages"].append({"role": "user", "content": "\n".join(f"<tool_response>{json.dumps(o)}</tool_response>" for o in out)})
            c.checkpoint(f"agent_iter_{sess['iters']}")
        return {"status": "failed", "error": f"exceeded {c.max_iters} tool iterations"}
