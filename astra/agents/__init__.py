from __future__ import annotations
import json, pathlib, re
from dataclasses import dataclass, field
from typing import Any, Callable

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def strip_think(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def extract_json(text: str):
    """First balanced JSON object in text (handles ```json fences, prose around it). None if absent."""
    text = strip_think(text or "")
    text = re.sub(r"```(?:json)?", "", text)
    for m in re.finditer(r"\{", text):
        depth, in_s, esc = 0, False, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if in_s:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_s = False
                continue
            if c == '"':
                in_s = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
    return None


@dataclass
class AgentContext:
    task: Any
    state: Any
    step: Any
    model_key: str
    spec: dict
    call_model: Callable          # (messages, **gen) -> result dict ; bound to task/step by the scheduler
    tools: Any = None
    checkpoint: Callable = lambda label: None
    memory_text: str = ""
    max_iters: int = 12

    def render_context(self) -> str:
        budget = int(self.spec["n_ctx"] * 3 * 0.35)
        return self.state.context_for(self.step, budget, self.memory_text)
