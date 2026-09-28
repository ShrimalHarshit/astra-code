from __future__ import annotations
import json
from . import load_prompt, extract_json
from ..core.task import Plan, Step, PlanError
from ..core.router import RoutingError


class Orchestrator:
    """Produces the structured execution plan. Does not execute anything."""
    def __init__(self, registry, router, log):
        self.reg, self.router, self.log = registry, router, log

    def plan(self, ctx, goal: str, project_context: dict | None = None) -> Plan:
        agents = "\n".join(f"- {k}: {', '.join(caps)}" for k, caps in self.reg.capabilities().items() if k != ctx.model_key)
        sysm = load_prompt("orchestrator").replace("{agents}", agents)
        user = f"GOAL: {goal}\nPROJECT CONTEXT: {json.dumps(project_context or {}, default=str)[:1500]}\nMEMORY:\n{ctx.memory_text[:1500]}"
        msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
        err = ""
        for attempt in range(2):
            if err:
                msgs.append({"role": "user", "content": f"Your plan was invalid ({err}). Reply with ONLY corrected JSON."})
            r = ctx.call_model(msgs, max_tokens=ctx.spec.get("max_tokens", 2048), temperature=0.1, json_mode=True)
            d = extract_json(r["output"])
            msgs.append({"role": "assistant", "content": r["output"]})
            try:
                if not d:
                    raise PlanError("no JSON object found")
                plan = Plan.from_llm(d, goal)
                plan.validate(self.router)
                return plan
            except (PlanError, RoutingError, KeyError) as e:
                err = str(e)
                self.log.warning(f"plan attempt {attempt+1} invalid: {err}")
        # Explicit, logged fallback - never silent.
        self.log.error(f"planner failed twice ({err}); using single-step assistant fallback plan")
        ctx.state.error(ctx.step.id, ctx.model_key, "plan_fallback", err)
        return Plan(goal, [Step(id="step_1", agent="assistant", capability="conversation", action="answer_directly", description=goal)])

    def run(self, ctx):  # agent contract for scheduler
        plan = self.plan(ctx, ctx.task.goal, ctx.state.project_context)
        return {"status": "success", "output": {"plan": plan.to_dict()}, "_plan": plan}
