import pytest

from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_missing_result,
    normalize_reward_result,
)


def test_normalize_reward_passed():
    result = normalize_reward_result("task-pass", 1.0)

    assert result.task_id == "task-pass"
    assert result.status == "passed"
    assert result.reward == 1.0
    assert result.failure_type is None


def test_normalize_reward_failed():
    result = normalize_reward_result("task-fail", 0.0)

    assert result.status == "failed"
    assert result.failure_type == "agent_failed"
    assert result.error_summary == "Verifier reward below pass threshold"


def test_normalize_missing_result_as_infra_failure():
    result = normalize_missing_result("task-missing", "Trial result.json missing")

    assert result.status == "infra_failed"
    assert result.reward is None
    assert result.failure_type == "missing_result"
    assert result.error_summary == "Trial result.json missing"


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
    ]

    summary = build_failure_summary(results)

    assert summary.agent_failures == 1
    assert summary.infra_failures == 1
    assert summary.tasks_passed == 1
    assert summary.tasks_total == 3
    assert summary.top_failure_modes == ["agent_failed", "missing_result"]


def test_run_create_request_rejects_unsafe_task_ids():
    from pydantic import ValidationError

    from autoharness_service.schemas import RunCreateRequest

    with pytest.raises(ValidationError):
        RunCreateRequest(task_ids=["../secret"], mode="simulated")


def test_run_create_request_validates_mode_provider_pair():
    from pydantic import ValidationError

    from autoharness_service.schemas import RunCreateRequest

    assert RunCreateRequest(
        task_ids=["task-pass"],
        mode="simulated",
        sandbox_provider="simulated",
    ).sandbox_provider == "simulated"

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
