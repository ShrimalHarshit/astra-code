"""Sandbox tests with the MOCK backend (real subprocess workers, no GPU). They verify lifecycle/orchestration logic - NOT model quality or CUDA behavior."""
import json, os, pathlib, time
import pytest

os.environ["ASTRA_BACKEND"] = "mock"
from astra.main import Astra
from astra.core.task import Plan, PlanError, Step
from astra.core.router import Router, RoutingError
from astra.tools.filesystem import SandboxFS, SandboxError
from astra.tools.shell import Shell, CommandDenied
from astra.runtime.process_manager import find_worker_processes
from astra.agents import extract_json
from astra.agents.hermes_agent import parse_tool_calls


@pytest.fixture()
def astra(tmp_path, monkeypatch):
    for k in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ASTRA_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("ASTRA_RESULTS_DIR", str(tmp_path / "results"))
    a = Astra({"checkpoint_backoff_s": 0, "persistence": {"checkpoint_retries": 1, "checkpoint_backoff_s": 0}})
    a.boot("full", install_runtime=False)
    yield a
    a.rm.release()
    a.idle.cancel()


def test_extract_and_toolcalls():
    assert extract_json('noise ```json\n{"a": {"b": 1}}\n``` tail') == {"a": {"b": 1}}
    assert extract_json("<think>{\"x\":1}</think>{\"y\":2}") == {"y": 2}
    c = parse_tool_calls('<tool_call>\n{"name":"read_file","arguments":{"path":"a"}}\n</tool_call>')
    assert c == [{"name": "read_file", "arguments": {"path": "a"}}]
    assert parse_tool_calls("<tool_call>{bad</tool_call>")[0]["name"] == "_invalid"


def test_plan_validation(astra):
    r = astra.router
    ok = Plan("g", [Step("a", "engineer", "x", "coding"), Step("b", "assistant", "y", depends_on=["a"])])
    ok.validate(r)
    with pytest.raises(PlanError):
        Plan("g", [Step("a", "engineer", "x", depends_on=["b"]), Step("b", "engineer", "y", depends_on=["a"])]).validate(r)
    with pytest.raises(PlanError):
        Plan("g", [Step("a", "engineer", "x", depends_on=["zzz"])]).validate(r)
    with pytest.raises(RoutingError):
        Plan("g", [Step("a", "nobody", "x")]).validate(r)
    assert r.resolve(Step("a", "engineer", "x", "mathematics")) == "logic"   # capability beats agent hint


def test_one_model_at_a_time(astra):
    rm = astra.rm
    for m in ["orchestrator", "engineer", "logic", "assistant", "engineer"]:
        rm.acquire(m)
        assert rm.current_model() == m
        assert len(find_worker_processes()) == 1
    h1 = rm.acquire("engineer"); h2 = rm.acquire("engineer")
    assert h1 is h2                                   # reuse, not duplicate
    rm.release()
    assert len(find_worker_processes()) == 0 and rm.current_model() is None
    rm.assert_clean()


def test_orphan_detection_and_kill(astra):
    import subprocess, sys
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).parents[1]))
    p = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)", "astra.runtime.worker"], env=env)
    time.sleep(0.3)
    assert len(find_worker_processes()) == 1
    astra.rm.assert_clean()                            # kills orphans
    assert len(find_worker_processes()) == 0
    p.wait(timeout=5)


def test_smoke_and_swap(astra):
    from astra.selftest import smoke, swap_test
    r = smoke(astra, "logic")
    assert r["passed"], r
    assert swap_test(astra, list(astra.reg.keys()), rounds=1)["passed"]


def wait_state(astra, want, t=10):
    end = time.time() + t
    while time.time() < end and astra.state != want:
        time.sleep(0.05)
    return astra.state


def test_end_to_end_task_mock(astra):
    t = astra.run("Please do math on the repo", console=False, timeout=60)
    assert t.status == "completed", astra.jobs[t.task_id][1].errors
    ids = [s.id for s in t.plan.steps]
    assert [s.agent for s in t.plan.steps] == ["engineer", "logic", "assistant"], ids
    assert astra.rm.current_model() is None            # worker exited after task
    cp = astra.ckpt.latest(t.task_id)
    assert cp["_meta"]["label"] == "final" and cp["task"]["status"] == "completed"
    labels = [r["label"] for r in astra.db.query("SELECT label FROM checkpoints WHERE task_id=?", (t.task_id,))]
    assert {"task_started", "planner_complete", "step_1:completed", "final"} <= set(labels)
    assert wait_state(astra, "POST_TASK_WAIT") == "POST_TASK_WAIT"
    assert astra.status()["idle_countdown_s"] is not None


def test_oom_degrades_and_never_loads_two(astra, monkeypatch):
    monkeypatch.setenv("ASTRA_MOCK_OOM", "logic")
    t = astra.run("math please", console=False, timeout=90)
    assert t.plan.step("step_2").status == "failed"    # persistent OOM -> failed, recorded, not lost
    assert any(e["status"] == "oom" for e in astra.jobs[t.task_id][1].errors)
    assert len(find_worker_processes()) == 0
    assert astra.scheduler.overrides["logic"]["n_ctx"] < 8192   # context was reduced before giving up
    assert t.status == "failed"


def test_generate_timeout_kills_worker(astra, monkeypatch):
    monkeypatch.setenv("ASTRA_MOCK_HANG", "assistant")
    astra.cfg.runtime["timeouts"]["generate_s"] = 2
    h = astra.rm.acquire("assistant")
    r = h.generate({"messages": [{"role": "user", "content": "x"}], "max_tokens": 5}, 2)
    assert r["status"] == "timeout" and not h.alive()
    astra.rm.release(); astra.rm.assert_clean()


def test_resume_from_checkpoint(astra):
    from astra.core.task import Task, RUNNING
    from astra.core.state import TaskState
    t = Task(goal="resume me")
    st = TaskState(task_id=t.task_id, goal=t.goal)
    t.status = RUNNING
    astra.jobs[t.task_id] = (t, st)
    astra.ckpt.save(t, st, "task_started"); astra.tasks.save_task(t)
    assert t.task_id in [x["task_id"] for x in astra.tasks.find_interrupted()]
    astra.resume(t.task_id)
    e = astra.wait(t.task_id)
    assert e["status"] in ("completed", "failed")


def test_waiting_for_user_blocks_idle_shutdown(astra, monkeypatch):
    monkeypatch.setenv("ASTRA_MOCK_ASK", "engineer")   # worker is a subprocess: control the mock through env
    tid = astra.submit("do something ambiguous")
    for _ in range(100):
        if astra.broker.pending(tid):
            break
        time.sleep(0.2)
    assert astra.broker.pending(tid)["question"] == "A or B?"
    assert astra.state == "WAITING_FOR_USER"
    astra.idle.start(0.1); time.sleep(0.5)             # idle timer must NOT shut anything down now
    assert astra.state == "WAITING_FOR_USER"
    assert astra.answer(tid, "A")
    e = astra.wait(tid)
    assert e["status"] == "completed"


def test_shutdown_refuses_when_not_durable_then_force(astra):
    astra.run("hello", console=False, timeout=60)
    wait_state(astra, "POST_TASK_WAIT")
    r = astra.shutdown("test")                          # SQLite fallback => not durable => must refuse
    assert r["ok"] is False and "not shut down" in r["refused"].lower()
    assert astra.state in ("POST_TASK_WAIT", "READY")
    r = astra.shutdown("test", force=True)
    assert r["ok"] and r["persisted_safely"] is False and r["platform"]["vm_terminated"] is False
    assert astra.state == "STOPPED"


def test_shutdown_refused_while_busy(astra):
    with astra.busy("x"):
        assert astra.shutdown("t")["ok"] is False


def test_sandbox_fs(tmp_path):
    fs = SandboxFS(tmp_path / "ws")
    fs.write("a/b.txt", "hi")
    assert "hi" in fs.read("a/b.txt")
    for bad in ("../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(SandboxError):
            fs.path(bad)
    (tmp_path / "ws" / "link").symlink_to("/etc")
    with pytest.raises(SandboxError):
        fs.path("link/passwd")
    with pytest.raises(SandboxError):
        fs.write(".git/config", "x")


def test_shell_allowlist(tmp_path):
    sh = Shell(tmp_path, [["ls"], ["git"], ["pytest"]])
    assert sh.run("ls")["returncode"] == 0
    for bad in ("rm -rf /", "curl http://x", "ls ../..", "ls /etc", "git push", "python -c 'print(1)'", "bash -c ls"):
        with pytest.raises(CommandDenied):
            sh.run(bad)


def test_secrets_redacted_and_scrubbed(monkeypatch):
    from astra.core.config import Secrets, scrubbed_env
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_supersecrettoken123")
    s = Secrets()
    assert s.get("GITHUB_TOKEN") == "ghp_supersecrettoken123"
    assert "ghp_supersecrettoken123" not in s.redact("x ghp_supersecrettoken123 y")
    assert "GITHUB_TOKEN" not in scrubbed_env()


def test_benchmark_mock(astra):
    from astra.benchmarks.benchmark import run_benchmark
    out = run_benchmark(astra, ["logic", "engineer"], sweep=False)
    assert all(r["status"] == "success" and r["cleanup"] == "clean" for r in out["rows"])
    assert set(out["best"]) == {"logic", "engineer"}
