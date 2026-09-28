"""Capability-based routing: planner picks a capability (and an agent hint); router maps it to a registry model."""
from __future__ import annotations
ALIASES = {"code": "coding", "programming": "coding", "math": "mathematics", "reasoning": "formal_reasoning",
           "summarize": "summarization", "chat": "conversation", "plan": "planning", "repo": "repository_analysis"}


class RoutingError(ValueError):
    pass


class Router:
    def __init__(self, registry, log=None):
        self.reg, self.log = registry, log

    def resolve(self, step) -> str:
        cap = ALIASES.get(step.capability, step.capability) if step.capability else None
        by_agent = step.agent if step.agent in self.reg.keys() else None
        if cap:
            c = self.reg.by_capability(cap)
            if c:
                pick = by_agent if by_agent in c else c[0]
                if by_agent and pick != by_agent and self.log:
                    self.log.warning(f"{step.id}: capability '{cap}' overrides agent hint '{by_agent}' -> {pick}")
                return pick
        if by_agent:
            return by_agent
        raise RoutingError(f"cannot route step {step.id}: agent={step.agent!r} capability={step.capability!r}")

    def for_capability(self, cap: str) -> str:
        c = self.reg.by_capability(ALIASES.get(cap, cap))
        if not c:
            raise RoutingError(f"no model has capability '{cap}'")
        return c[0]
