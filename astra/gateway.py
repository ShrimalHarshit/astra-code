"""HTTP gateway. It only talks to the Astra controller; it can NOT load models (only Scheduler/ResourceManager can)."""
from __future__ import annotations
import hmac, json
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class Goal(BaseModel):
    goal: str
    project_id: str | None = None


class Answer(BaseModel):
    answer: str


def create_app(astra, token: str) -> FastAPI:
    app = FastAPI(title="Astra Gateway", docs_url=None, redoc_url=None)

    def auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), token):
            raise HTTPException(401, "bad token")

    @app.get("/status", dependencies=[Depends(auth)])
    def status():
        return astra.status()

    @app.post("/tasks", dependencies=[Depends(auth)])
    def submit(g: Goal):
        try:
            return {"task_id": astra.submit(g.goal, g.project_id)}
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/tasks", dependencies=[Depends(auth)])
    def tasks():
        return astra.tasks.list(limit=50)

    @app.get("/tasks/{tid}", dependencies=[Depends(auth)])
    def get_task(tid: str):
        if tid in astra.jobs:
            t, s = astra.jobs[tid]
            return {**t.to_dict(), "question": astra.broker.pending(tid), "errors": s.errors[-5:], "outputs": s.step_outputs}
        r = astra.tasks.get(tid)
        if not r:
            raise HTTPException(404)
        return r

    @app.get("/tasks/{tid}/stream", dependencies=[Depends(auth)])
    def stream(tid: str, since: int = 0):
        def gen():
            for e in astra.bus.stream(tid, since):
                yield ": hb\n\n" if e is None else f"data: {json.dumps(e, default=str)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/tasks/{tid}/answer", dependencies=[Depends(auth)])
    def answer(tid: str, a: Answer):
        if not astra.answer(tid, a.answer):
            raise HTTPException(409, "no pending question for this task")
        return {"ok": True}

    @app.post("/tasks/{tid}/resume", dependencies=[Depends(auth)])
    def resume(tid: str):
        try:
            return {"task_id": astra.resume(tid)}
        except (KeyError, RuntimeError) as e:
            raise HTTPException(409, str(e))

    @app.post("/shutdown", dependencies=[Depends(auth)])
    def shutdown(force: bool = False):
        return astra.shutdown("api", force=force)

    return app
