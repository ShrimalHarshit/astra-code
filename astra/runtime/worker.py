"""One model per process. Protocol: JSON lines. stdin <- commands, stdout(dup'ed) -> events. All native noise goes to stderr.
Run:  python -m astra.runtime.worker --spec-file spec.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss // (1 << 20)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-file", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)  # keep the real stdout for the protocol
    os.dup2(2, 1)                                    # anything (llama.cpp printf) written to fd 1 goes to stderr

    def emit(o):
        proto.write(json.dumps(o) + "\n")
        proto.flush()

    spec = json.load(open(a.spec_file))
    from astra.runtime.llama_runner import make_runner
    t0 = time.perf_counter()
    try:
        runner = make_runner(spec)
    except BaseException as e:  # noqa
        emit({"event": "load_failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        sys.exit(3)
    emit({"event": "ready", "load_ms": int((time.perf_counter() - t0) * 1000), "rss_mb": _rss_mb(), "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        if cmd["cmd"] == "shutdown":
            break
        if cmd["cmd"] == "generate":
            rid = cmd["id"]

            def on_token(t, rid=rid):
                if cmd.get("stream"):
                    emit({"event": "token", "id": rid, "text": t})
            try:
                r = runner.generate(cmd["messages"], cmd.get("max_tokens", 512), cmd.get("temperature", 0.2), cmd.get("stop"),
                                    cmd.get("json_mode", False), on_token, cmd.get("reset", False))
                r.update({"event": "result", "id": rid, "status": "success", "rss_mb": _rss_mb()})
                emit(r)
            except BaseException as e:  # noqa
                emit({"event": "result", "id": rid, "status": "failed", "error": f"{type(e).__name__}: {e}"})
                if isinstance(e, (MemoryError, KeyboardInterrupt)):
                    break
    try:
        runner.close()
    finally:
        emit({"event": "bye"})


if __name__ == "__main__":
    main()
