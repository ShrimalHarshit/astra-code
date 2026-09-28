from __future__ import annotations
import threading, time


class EventBus:
    """In-process pub/sub with per-task history so late subscribers (SSE clients, notebook) can replay."""
    def __init__(self, cap=5000):
        self._h: dict[str, list] = {}
        self._cv = threading.Condition()
        self.cap = cap
        self.listeners = []  # callables(task_id, event) - console printing etc.

    def publish(self, task_id: str, ev: dict):
        ev = {"ts": round(time.time(), 2), "task_id": task_id, **ev}
        with self._cv:
            h = self._h.setdefault(task_id, [])
            h.append(ev)
            del h[:-self.cap]
            self._cv.notify_all()
        for l in list(self.listeners):
            try:
                l(task_id, ev)
            except Exception:
                pass

    def history(self, task_id, since=0):
        with self._cv:
            return list(self._h.get(task_id, []))[since:]

    def stream(self, task_id, since=0, heartbeat=15.0):
        i = since
        while True:
            with self._cv:
                while len(self._h.get(task_id, [])) <= i:
                    if not self._cv.wait(timeout=heartbeat):
                        yield None  # heartbeat
                evs = self._h[task_id][i:]
            for e in evs:
                i += 1
                yield e
                if e["type"] == "task_done":
                    return


class AnswerBroker:
    """Blocks the scheduler thread until the user answers (via API, notebook, or console). No timeout: WAITING_FOR_USER never auto-shuts-down."""
    def __init__(self, bus: EventBus):
        self.bus = bus
        self._pending: dict[str, dict] = {}
        self._cv = threading.Condition()
        self.console = False

    def ask(self, task_id, question, options=None) -> str:
        with self._cv:
            self._pending[task_id] = {"question": question, "options": options, "answer": None}
        self.bus.publish(task_id, {"type": "question", "question": question, "options": options})
        if self.console:
            try:
                ans = input(f"\n[ASTRA asks] {question}\n{('options: ' + str(options)) if options else ''}\n> ")
                self.answer(task_id, ans)
            except EOFError:
                pass
        with self._cv:
            while self._pending[task_id]["answer"] is None:
                self._cv.wait(timeout=5)
            return self._pending.pop(task_id)["answer"]

    def answer(self, task_id, text: str) -> bool:
        with self._cv:
            if task_id not in self._pending or self._pending[task_id]["answer"] is not None:
                return False
            self._pending[task_id]["answer"] = text
            self._cv.notify_all()
        self.bus.publish(task_id, {"type": "answer_received"})
        return True

    def pending(self, task_id=None):
        with self._cv:
            if task_id:
                p = self._pending.get(task_id)
                return {"question": p["question"], "options": p["options"]} if p and p["answer"] is None else None
            return {t: p["question"] for t, p in self._pending.items() if p["answer"] is None}
