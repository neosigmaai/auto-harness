import pytest
from autoharness_service.models import TaskResultRecord
from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_missing_result,
    normalize_reward_result,
)


def test_normalize_reward_threshold_behavior():
    failed = normalize_reward_result("task-fail", 0.1)
    passed = normalize_reward_result("task-pass", 0.5)

    assert failed.status == "failed"
    assert failed.failure_type == "agent_failed"
    assert passed.status == "passed"
    assert passed.reward == 0.5


def test_normalize_reward_failed():
    result = normalize_reward_result("task-fail", 0.0)

    assert result.status == "failed"
    assert result.failure_type == "agent_failed"
    assert result.error_summary == "Verifier reward below pass threshold"


def test_normalize_reward_none_as_missing_result():
    result = normalize_reward_result("task-missing", None)

    assert result.status == "infra_failed"
    assert result.reward is None
    assert result.failure_type == "missing_result"


def test_normalize_missing_result_as_infra_failure():
    result = normalize_missing_result(
        "task-missing",
        "Trial result.json missing",
        trace_path="/tmp/trace.json",
        result_path="/tmp/result.json",
        metadata={"source": "trial"},
    )

    assert result.status == "infra_failed"
    assert result.reward is None
    assert result.failure_type == "missing_result"
    assert result.error_summary == "Trial result.json missing"
    assert result.trace_path == "/tmp/trace.json"
    assert result.result_path == "/tmp/result.json"
    assert result.metadata == {"source": "trial"}


def test_normalize_non_finite_reward_as_invalid_result():
    result = normalize_reward_result("task-nan", float("nan"))

    assert result.status == "infra_failed"
    assert result.reward is None
    assert result.failure_type == "invalid_result"


def test_build_failure_summary_counts_failure_types():
    results = [
        normalize_reward_result("task-pass", 1.0),
        normalize_reward_result("task-fail", 0.0),
        normalize_missing_result("task-missing", "Trial result.json missing"),
        TaskResultRecord(
            task_id="task-timeout",
            status="timed_out",
            reward=None,
            failure_type=None,
            error_summary=None,
            metadata={},
        ),
    ]

    summary = build_failure_summary(results)

    assert summary.agent_failures == 1
    assert summary.infra_failures == 2
    assert summary.tasks_passed == 1
    assert summary.tasks_infra_failed == 2
    assert summary.tasks_total == 4
    assert summary.top_failure_modes == ["agent_failed", "missing_result"]


def test_run_create_request_rejects_unsafe_task_ids():
    from autoharness_service.schemas import RunCreateRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunCreateRequest(task_ids=["../secret"], mode="simulated")


def test_run_create_request_validates_mode_provider_pair():
    from autoharness_service.schemas import RunCreateRequest
    from pydantic import ValidationError

    assert (
        RunCreateRequest(
            task_ids=["task-pass"],
            mode="simulated",
            sandbox_provider="simulated",
        ).sandbox_provider
        == "simulated"
    )

    with pytest.raises(ValidationError):
        RunCreateRequest(
            task_ids=["task-pass"],
            mode="simulated",
            sandbox_provider="daytona",
        )

    with pytest.raises(ValidationError):
        RunCreateRequest(
            task_ids=["break-filter-js-from-html"],
            mode="real",
            sandbox_provider="simulated",
        )
