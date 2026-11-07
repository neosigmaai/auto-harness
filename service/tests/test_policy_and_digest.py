"""Structural compatibility and digest tests; AST checks are not a security boundary."""
from service.worker.digest import build_digest, classify, _steps, dossier_from_row
from service.worker.executors.base import TaskResult
from service.worker.job_runner import TEMPLATE_PATH, validate_agent_source


def test_template_passes_policy():
    ok, why = validate_agent_source(TEMPLATE_PATH.read_text())
    assert ok, why


def test_validation_is_structural_and_does_not_pretend_to_sandbox_python():
    base = TEMPLATE_PATH.read_text()
    # These are valid Python. Process/container isolation must enforce host and
    # credential access restrictions; an import denylist cannot do that.
    assert validate_agent_source(base + "\nimport subprocess\n")[0]
    assert validate_agent_source(base + "\nimport socket\n")[0]
    assert validate_agent_source(base + '\ncredential = os.environ.get("OPENAI_API_KEY")\n')[0]
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


def test_digest_handles_malformed_untrusted_trace_and_empty_stdout():
    trace = [None, 4, "not a message", {"role": "assistant", "tool_calls": [None, {"id": [], "function": []}]},
             {"role": "tool", "tool_call_id": [], "content": "ok"},
             {"role": "assistant", "tool_calls": "bad"}]
    result = _tr("failed", trace)
    result.test_stdout = "  \n"
    digest = build_digest(result)
    assert "test: n/a" in digest["summary"]


def test_digest_preserves_trusted_executor_failure_cause():
    infra = _tr("error", [], exit_reason="DockerError", fc="infra_error")
    agent = _tr("error", [], exit_reason="RuntimeError", fc="agent_error")
    assert build_digest(infra)["stats"]["failure_class"] == "infra_error"
    assert build_digest(agent)["stats"]["failure_class"] == "agent_error"


def test_instruction_excerpt_preserves_both_specification_and_ending_constraints():
    instruction = "FIRST requirement " + "x" * 2000 + " FINAL constraint"
    result = _tr("failed", [{"role": "user", "content": "Task:\n" + instruction}])
    excerpt = build_digest(result)["instruction"]
    assert excerpt.startswith("FIRST requirement") and excerpt.endswith("FINAL constraint")
    assert excerpt == instruction
    assert build_digest(result)["instruction_complete"]
    result.trace[0]["content"] = "Task:\n" + "FIRST " + "x" * 9000 + " FINAL constraint"
    excerpt = build_digest(result)["instruction"]
    assert len(excerpt) <= 8192 and excerpt.endswith("FINAL constraint")
    assert not build_digest(result)["instruction_complete"]
    result.instruction = "trusted task manifest"
    assert build_digest(result)["instruction"] == "trusted task manifest"


def test_middle_truncation_marker_survives_as_structural_evidence_not_raw_output():
    trace = _trace(["grep -R pattern repository"])
    trace[1]["content"] = "head " * 900 + "\n... [13000 chars truncated] ...\n" + "tail " * 900
    digest = build_digest(_tr("failed", trace, fc="wrong_output"))
    observation = digest["structural_observations"]
    assert "truncated" not in digest["last_steps"][0]["output"]
    assert len(digest["last_steps"][0]["output"]) <= 800
    assert observation["outputs_with_truncation_marker"] == 1
    assert observation["truncation_examples"][0]["reported_omitted_chars"] == 13000
    assert observation["truncation_examples"][0]["step"] == 1
    assert observation["causal_failure_claim"] is False
    assert digest["stats"]["failure_class"] == "wrong_output"


def test_exact_repetition_uses_full_commands_and_is_not_automatically_a_failure():
    prefix = "x" * 300
    trace = _trace([prefix + str(i) for i in range(8)])
    digest = build_digest(_tr("failed", trace))
    assert digest["structural_observations"]["repeated_exact_command_groups"] == 0
    assert digest["stats"]["failure_class"] == "wrong_output"
    repeated = build_digest(_tr("passed", _trace(["check", "inspect", "check"])))
    assert repeated["structural_observations"]["repeated_command_examples"] == [
        {"command_excerpt":"check", "count":2, "step_indices":[1,3]}]
    assert repeated["stats"]["failure_class"] is None


def test_missing_trace_is_unknown_and_structural_examples_remain_bounded():
    unknown = build_digest(_tr("error", None, fc="infra_error"))["structural_observations"]
    assert unknown["evidence_available"] is False
    assert unknown["outputs_with_truncation_marker"] is None
    assert unknown["repeated_exact_command_groups"] is None
    trace = _trace([f"command{i}" for i in range(20)] * 2)
    for message in trace:
        if message["role"] == "tool":
            message["content"] = "[5 chars truncated]"
    bounded = build_digest(_tr("failed", trace))["structural_observations"]
    assert bounded["outputs_with_truncation_marker"] == 40
    assert bounded["repeated_exact_command_groups"] == 20
    assert len(bounded["truncation_examples"]) == len(bounded["repeated_command_examples"]) == 6
    assert bounded["coverage_guaranteed"] is False


def test_long_command_retains_heredoc_redirection_and_error_origin():
    import json
    command = "python3 - <<'PY'\n" + "# large script\n" * 100 + "print('result')\nPY\n> /tmp/result.json\ncat /tmp/result.json"
    trace = [{"role": "assistant", "tool_calls": [{"id": "call", "function": {
        "arguments": json.dumps({"command": command})}}]},
        {"role": "tool", "tool_call_id": "call", "content": "Traceback: input was empty\n" + "filler " * 300 + "\n[exit code: 1]"}]
    result = _tr("failed", trace)
    result.test_stdout = "installation noise\n" * 300 + "=== FAILURES ===\njson.load(open('/tmp/result.json'))\n" + "stack frame\n" * 200 + "JSONDecodeError: s=''\nFAILED test_output"
    digest = build_digest(result)
    step = digest["last_steps"][0]
    assert step["command"].startswith("python3 - <<'PY'")
    assert "PY\n> /tmp/result.json\ncat /tmp/result.json" in step["command"]
    assert step["command_original_chars"] == len(command) and len(step["command"]) <= 800
    assert step["output"].startswith("Traceback: input was empty")
    assert step["output"].endswith("[exit code: 1]")
    assert "json.load(open('/tmp/result.json'))" in digest["test_failure_excerpt"]
    assert "JSONDecodeError: s=''" in digest["test_failure_excerpt"]
    assert len(digest["test_failure_excerpt"]) <= 2400


def test_rebuilding_legacy_dossier_recovers_full_instruction_from_available_trace():
    instruction = "first requirement " + "complete details " * 200 + " final constraint"
    row = {"task_name": "feedback", "status": "failed", "reward": 0.,
           "trace": [{"role": "user", "content": "Task:\n" + instruction}],
           "digest": {"instruction": "first ... [instruction excerpt] ... final"}}
    assert dossier_from_row(row)["instruction"] == instruction
    assert dossier_from_row(row)["instruction_complete"]
    row["trace"] = None
    assert not dossier_from_row(row)["instruction_complete"]
    row["instruction"] = "trusted instruction"
    assert dossier_from_row(row)["instruction"] == "trusted instruction"
