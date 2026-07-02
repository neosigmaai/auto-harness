import sys
from types import SimpleNamespace

import pytest
from autoharness_service.models import TaskResultRecord
from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_reward_result,
)
from autoharness_service.optimizer import (
    OptimizationProposal,
    Optimizer,
    build_optimizer_prompt,
    parse_optimizer_json,
)


def test_parse_optimizer_json_accepts_required_fields():
    payload = """
    {
      "hypothesis": "The agent stops before verifying artifacts.",
      "new_agent_instruction": "A valid long replacement instruction...",
      "expected_effect": "The agent verifies output files before finishing.",
      "risks": "The agent may spend extra time verifying."
    }
    """

    proposal = parse_optimizer_json(payload)

    assert proposal == OptimizationProposal(
        hypothesis="The agent stops before verifying artifacts.",
        new_agent_instruction="A valid long replacement instruction...",
        expected_effect="The agent verifies output files before finishing.",
        risks="The agent may spend extra time verifying.",
    )


def test_parse_optimizer_json_rejects_free_text():
    with pytest.raises(ValueError):
        parse_optimizer_json("hypothesis: maybe this helps")


def test_parse_optimizer_json_rejects_extra_fields():
    payload = """
    {
      "hypothesis": "The agent skips verification.",
      "new_agent_instruction": "Replacement instruction",
      "expected_effect": "The agent verifies files.",
      "risks": "Runs take longer.",
      "extra": "unexpected"
    }
    """

    with pytest.raises(ValueError, match="exactly"):
        parse_optimizer_json(payload)


def test_build_optimizer_prompt_includes_current_instruction_and_artifact_paths():
    results = [
        TaskResultRecord(
            task_id="task-fail",
            status="failed",
            reward=0.0,
            failure_type="agent_failed",
            error_summary="Verifier reward below pass threshold",
            trace_path="/tmp/task-fail/trace.json",
            result_path="/tmp/task-fail/result.json",
            metadata={
                "artifacts": [
                    "/tmp/task-fail/stdout.txt",
                    "/tmp/task-fail/report.md",
                ]
            },
        )
    ]
    summary = build_failure_summary(results)

    prompt = build_optimizer_prompt(
        results,
        summary,
        current_instruction="Always verify generated files before finishing.",
    )

    assert "Always verify generated files before finishing." in prompt
    assert "task-fail" in prompt
    assert "agent_failed" in prompt
    assert "Verifier reward below pass threshold" in prompt
    assert "/tmp/task-fail/trace.json" in prompt
    assert "/tmp/task-fail/result.json" in prompt
    assert "/tmp/task-fail/stdout.txt" in prompt
    assert "/tmp/task-fail/report.md" in prompt
    assert "Return JSON only" in prompt
    assert "no Markdown" in prompt


def test_optimizer_propose_instruction_patch_raises_without_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    results = [normalize_reward_result("task-fail", 0.0)]
    summary = build_failure_summary(results)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        Optimizer().propose_instruction_patch(
            results,
            summary,
            model="test-model",
            current_instruction="Current instruction text.",
        )


def test_optimizer_propose_instruction_patch_parses_single_json_object(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self) -> None:
            self.responses = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text="""
                {
                  "hypothesis": "The agent exits before checking artifacts.",
                  "new_agent_instruction": "Before finishing, inspect artifact paths and confirm expected files exist.",
                  "expected_effect": "Fewer runs end without validating outputs.",
                  "risks": "Runs may take slightly longer."
                }
                """
            )

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda: FakeClient())
    )

    results = [
        TaskResultRecord(
            task_id="task-fail",
            status="failed",
            reward=0.0,
            failure_type="agent_failed",
            error_summary="Verifier reward below pass threshold",
            trace_path="/tmp/task-fail/trace.json",
            result_path="/tmp/task-fail/result.json",
            metadata={"artifacts": ["/tmp/task-fail/report.md"]},
        )
    ]
    summary = build_failure_summary(results)

    proposal = Optimizer().propose_instruction_patch(
        results,
        summary,
        model="test-model",
        current_instruction="Current instruction text.",
    )

    assert proposal == OptimizationProposal(
        hypothesis="The agent exits before checking artifacts.",
        new_agent_instruction="Before finishing, inspect artifact paths and confirm expected files exist.",
        expected_effect="Fewer runs end without validating outputs.",
        risks="Runs may take slightly longer.",
    )
    assert captured["model"] == "test-model"
    assert isinstance(captured["input"], list)
    user_prompt = captured["input"][1]["content"]
    assert "Current instruction:\nCurrent instruction text." in user_prompt
    assert "Return JSON only as a single object" in user_prompt
    assert "exactly these string fields" in user_prompt
    assert "Do not propose multiple candidates" in user_prompt
    assert "/tmp/task-fail/report.md" in user_prompt


def test_optimizer_propose_reads_current_instruction_and_serializes_json(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgentPatchService:
        def read_instruction(self) -> str:
            captured["read_instruction_called"] = True
            return "Instruction from agent patch service."

    def fake_propose_instruction_patch(
        self,
        task_results,
        failure_summary,
        *,
        model,
        current_instruction,
    ):
        captured["model"] = model
        captured["current_instruction"] = current_instruction
        return OptimizationProposal(
            hypothesis="Need a stronger verification step.",
            new_agent_instruction="Always inspect the produced artifact before exiting.",
            expected_effect="More runs verify the expected file.",
            risks="Slightly longer runs.",
        )

    monkeypatch.setattr(
        "autoharness_service.optimizer.AgentPatchService",
        FakeAgentPatchService,
        raising=False,
    )
    monkeypatch.setattr(
        Optimizer,
        "propose_instruction_patch",
        fake_propose_instruction_patch,
    )

    results = [normalize_reward_result("task-fail", 0.0)]
    summary = build_failure_summary(results)

    proposal_json = Optimizer().propose(results, summary, model="test-model")

    assert captured == {
        "read_instruction_called": True,
        "model": "test-model",
        "current_instruction": "Instruction from agent patch service.",
    }
    assert proposal_json == (
        '{"hypothesis": "Need a stronger verification step.", '
        '"new_agent_instruction": "Always inspect the produced artifact before '
        'exiting.", "expected_effect": "More runs verify the expected file.", '
        '"risks": "Slightly longer runs."}'
    )
