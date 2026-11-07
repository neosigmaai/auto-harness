"""Unit tests that need no database: the agent policy gate and the failure classifier."""
from service.worker.digest import build_digest, classify, _steps
from service.worker.executors.base import TaskResult
from service.worker.job_runner import TEMPLATE_PATH, validate_agent_source


def test_template_passes_policy():
    ok, why = validate_agent_source(TEMPLATE_PATH.read_text())
    assert ok, why


def test_policy_rejects_forbidden_imports_and_env():
    base = TEMPLATE_PATH.read_text()
    assert not validate_agent_source(base + "\nimport subprocess\n")[0]
    assert not validate_agent_source(base + "\nimport socket\n")[0]
    assert not validate_agent_source(base.replace('os.environ.get("AGENT_MODEL", "gpt-5.4")', 'os.environ["OPENAI_API_KEY"]'))[0]
    assert not validate_agent_source("def nothing(): pass")[0]
    assert not validate_agent_source("class HarnessAgent:\n  def run(self): pass")[0]
    assert not validate_agent_source("this is not python")[0]


def test_policy_allows_prompt_and_loop_changes():
    src = TEMPLATE_PATH.read_text().replace("MAX_STEPS = 80", "MAX_STEPS = 60").replace(
        "Work step by step.", "Plan first, then act, then verify.")
    assert validate_agent_source(src)[0]


def _tr(status, trace, steps=10, exit_reason="agent_declared_done", fc=None):
    return TaskResult("t", status, 1.0 if status == "passed" else 0.0, fc, 10.0, steps, exit_reason, "out", trace)


def _trace(cmds):
    msgs = []
    for i, c in enumerate(cmds):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "%s"}' % c}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    msgs.append({"role": "assistant", "content": "done"})
    return msgs


def test_classify_heuristics():
    assert classify(_tr("passed", _trace(["ls"])), []) is None
    assert classify(_tr("timeout", _trace(["ls"]), exit_reason="agent_timeout"), []) == "timeout"
    loop = _trace(["cat a"] * 8)
    assert classify(_tr("failed", loop), _steps(loop)) == "loop_detected"
    short = _trace(["ls", "cat x"])
    assert classify(_tr("failed", short), _steps(short)) == "gave_up_early"
    many = _trace([f"cmd{i}" for i in range(12)])
    assert classify(_tr("failed", many, steps=80), _steps(many)) == "step_limit"
    assert classify(_tr("failed", many), _steps(many)) == "wrong_output"


def test_digest_shape():
    tr = _trace([f"cmd{i}" for i in range(12)])
    d = build_digest(_tr("failed", tr), instruction="Do the thing")
    assert d["stats"]["failure_class"] == "wrong_output"
    assert len(d["last_steps"]) == 8 and d["last_steps"][-1]["command"] == "cmd11"
    assert d["final_message"] == "done" and d["instruction"] == "Do the thing"
