"""Pure schema tests for the job API models (no database required)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.agent_spec import baseline_spec
from api.schemas import (
    AgentSpecView,
    CreateJobRequest,
    IterationView,
    JobConfigEcho,
    JobResponse,
    RunStatus,
)


def test_create_job_request_accepts_empty_body_and_applies_no_defaults() -> None:
    body = CreateJobRequest()
    assert body.task_ids is None
    assert body.agent_model is None
    assert body.improver_model is None
    assert body.max_iterations is None
    assert body.patience is None
    assert body.min_delta is None


def test_create_job_request_accepts_full_body() -> None:
    body = CreateJobRequest(
        task_ids=["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=3,
        patience=1,
        min_delta=0.0,
    )
    assert body.task_ids == ["fix-git", "regex-log"]
    assert body.max_iterations == 3
    assert body.patience == 1
    assert body.min_delta == 0.0


def test_create_job_request_rejects_empty_task_ids() -> None:
    with pytest.raises(ValidationError) as exc:
        CreateJobRequest(task_ids=[])
    assert "non-empty" in str(exc.value)


def test_create_job_request_allows_null_task_ids() -> None:
    assert CreateJobRequest(task_ids=None).task_ids is None


@pytest.mark.parametrize("value", [0, -1, 51])
def test_create_job_request_rejects_out_of_range_max_iterations(value: int) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(max_iterations=value)


@pytest.mark.parametrize("value", [0, 11])
def test_create_job_request_rejects_out_of_range_patience(value: int) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(patience=value)


@pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
def test_create_job_request_rejects_out_of_range_min_delta(value: float) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(min_delta=value)


def test_job_response_serializes_with_empty_iterations() -> None:
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    response = JobResponse(
        job_id="11111111-1111-1111-1111-111111111111",
        status=RunStatus.queued,
        created_at=now,
        config=JobConfigEcho(
            task_ids=["fix-git"],
            agent_model="gpt-4.1-mini",
            improver_model="gpt-5.4",
            max_iterations=5,
            patience=2,
            min_iterations=3,
            min_delta=0.01,
        ),
        current_iteration=0,
    )
    dumped = response.model_dump()
    assert dumped["iterations"] == []
    assert dumped["best"] is None
    assert dumped["stop_reason"] is None
    assert dumped["started_at"] is None
    assert dumped["finished_at"] is None
    assert dumped["error"] is None
    assert dumped["config"]["task_ids"] == ["fix-git"]


def test_agent_spec_view_round_trips_every_agent_spec_field() -> None:
    spec = baseline_spec("gpt-4.1-mini")
    view = AgentSpecView(**spec.model_dump())
    assert view.model_dump() == spec.model_dump()
    assert set(view.model_dump()) == {
        "system_prompt",
        "agent_model",
        "max_steps",
        "max_output_chars",
        "exec_timeout_sec",
    }
    assert view.agent_model == "gpt-4.1-mini"
    assert view.system_prompt == spec.system_prompt


def test_create_job_request_min_iterations_bounds() -> None:
    assert CreateJobRequest(min_iterations=1).min_iterations == 1
    with pytest.raises(ValidationError):
        CreateJobRequest(min_iterations=0)


def test_iteration_view_movement_defaults_empty() -> None:
    view = IterationView(
        iteration=0,
        agent_version_id="v",
        version=0,
        run_id=None,
        score=None,
        improved=None,
        summary=None,
        proposal=None,
    )
    assert view.fixed_tasks == [] and view.regressed_tasks == []
