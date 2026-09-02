"""PostgresJobStore tests: queue claiming, transitions and history."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from api.agent_spec import baseline_spec
from api.config import clear_config_cache
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import (
    EvaluateOutcome,
    ImproveOutcome,
    PostgresJobStore,
)
from api.models import AgentVersionRow, JobRow, StepRow

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)


@pytest.fixture()
def job_store() -> PostgresJobStore:
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    store = PostgresJobStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


def _create_job(
    store: PostgresJobStore,
    *,
    task_ids: list[str] | None = None,
    max_iterations: int = 5,
    patience: int = 2,
    min_iterations: int = 3,
    min_delta: float = 0.01,
    max_job_duration_sec: int = 21600,
    evaluate_stale_after_sec: int = 3600,
):
    return store.create_job(
        task_ids=task_ids or ["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=max_iterations,
        patience=patience,
        min_iterations=min_iterations,
        min_delta=min_delta,
        max_job_duration_sec=max_job_duration_sec,
        evaluate_stale_after_sec=evaluate_stale_after_sec,
    )


def test_create_job_inserts_v0_and_one_queued_evaluate_step(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store)

    assert job.status == "queued"
    assert job.task_ids == ["fix-git", "regex-log"]
    assert job.agent_model == "gpt-4.1-mini"
    assert job.improver_model == "gpt-5.4"
    assert job.current_iteration == 0
    assert job.best_agent_version_id is None
    assert job.best_score is None
    assert job.stop_reason is None
    assert job.started_at is None and job.finished_at is None

    # Exactly one iteration record, from the queued evaluate step.
    assert len(job.iterations) == 1
    it0 = job.iterations[0]
    assert it0.iteration == 0
    assert it0.version == 0
    assert it0.status == "queued"
    assert it0.score is None
    assert it0.improved is None
    assert it0.rationale is None
    assert it0.changed_fields == []

    # v0 spec is the baseline spec for the requested model.
    version = job_store.get_agent_version(it0.agent_version_id)
    assert version is not None
    assert version.version == 0
    assert version.parent_version_id is None
    assert version.created_by == "baseline"
    assert version.rationale == "baseline"
    assert version.spec == baseline_spec("gpt-4.1-mini")

    # Row-level shape: one job, one version, one queued evaluate step.
    session = get_session_factory()()
    try:
        assert len(list(session.scalars(select(JobRow)))) == 1
        assert len(list(session.scalars(select(AgentVersionRow)))) == 1
        steps = list(session.scalars(select(StepRow)))
        assert len(steps) == 1
        assert steps[0].type == "evaluate"
        assert steps[0].status == "queued"
        assert steps[0].iteration == 0
        assert steps[0].stale_after_sec == 3600
        assert steps[0].run_id is None
    finally:
        session.close()


def test_get_job_returns_none_for_unknown_or_malformed_id(
    job_store: PostgresJobStore,
) -> None:
    assert job_store.get_job(str(uuid.uuid4())) is None
    assert job_store.get_job("not-a-uuid") is None
    assert job_store.get_agent_version("not-a-uuid") is None


def test_claim_next_step_marks_step_running_and_job_running(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store)

    step = job_store.claim_next_step("w1")
    assert step is not None
    assert step.job_id == job.job_id
    assert step.type == "evaluate"
    assert step.iteration == 0
    assert step.version == 0
    assert step.task_ids == ["fix-git", "regex-log"]
    assert step.agent_model == "gpt-4.1-mini"
    assert step.improver_model == "gpt-5.4"
    assert step.run_id is None
    assert step.stale_after_sec == 3600
    assert step.spec == baseline_spec("gpt-4.1-mini")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.started_at is not None
    assert refreshed.iterations[0].status == "running"

    # Nothing else is queued.
    assert job_store.claim_next_step("w2") is None

    session = get_session_factory()()
    try:
        row = session.scalars(select(StepRow)).one()
        assert row.status == "running"
        assert row.worker_id == "w1"
        assert row.claimed_at is not None
        assert row.started_at is not None
    finally:
        session.close()


def test_two_workers_contend_for_one_step(job_store: PostgresJobStore) -> None:
    """Genuine contention: ONE job, ONE queued step, TWO threads race for it.

    This is the real test of FOR UPDATE SKIP LOCKED: with two separate jobs (one
    step each) two threads would each get a distinct row even under materially
    weaker locking (or none at all), so that shape proves nothing about mutual
    exclusion. Here both threads target the same row; exactly one must win it.
    """
    job = _create_job(job_store, task_ids=["fix-git"])

    def claim(worker_id: str):
        return job_store.claim_next_step(worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(claim, "w1")
        f2 = pool.submit(claim, "w2")
        claimed = [f1.result(), f2.result()]

    winners = [c for c in claimed if c is not None]
    losers = [c for c in claimed if c is None]
    assert len(winners) == 1, "exactly one worker must claim the single queued step"
    assert len(losers) == 1
    assert winners[0].job_id == job.job_id

    # Third claim finds nothing queued.
    assert job_store.claim_next_step("w3") is None

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"


def test_two_workers_claim_two_distinct_steps_in_same_job(
    job_store: PostgresJobStore,
) -> None:
    """Two genuinely queued steps in the SAME job: two threads must not collide."""
    job = _create_job(job_store, task_ids=["fix-git"])
    v0 = job.iterations[0].agent_version_id

    # A second, independently queued step in the same job (as if an earlier
    # evaluate step had already completed and enqueued an improve step).
    session = get_session_factory()()
    try:
        session.add(
            StepRow(
                id=uuid.uuid4(),
                job_id=uuid.UUID(job.job_id),
                type="improve",
                status="queued",
                iteration=1,
                agent_version_id=uuid.UUID(v0),
                stale_after_sec=1800,
            )
        )
        session.commit()
    finally:
        session.close()

    def claim(worker_id: str):
        return job_store.claim_next_step(worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(claim, "w1")
        f2 = pool.submit(claim, "w2")
        claimed = [f1.result(), f2.result()]

    assert None not in claimed
    assert len({c.step_id for c in claimed}) == 2
    assert {c.job_id for c in claimed} == {job.job_id}

    # Nothing left queued in this job.
    assert job_store.claim_next_step("w3") is None


def test_stale_running_step_is_requeued_and_reclaimable(
    job_store: PostgresJobStore,
) -> None:
    _create_job(job_store, evaluate_stale_after_sec=0)

    first = job_store.claim_next_step("w1")
    assert first is not None
    assert first.stale_after_sec == 0

    # With stale_after_sec=0 the row is stale as soon as now() moves past claimed_at.
    time.sleep(0.05)

    second = job_store.claim_next_step("w2")
    assert second is not None
    assert second.step_id == first.step_id

    session = get_session_factory()()
    try:
        row = session.scalars(select(StepRow)).one()
        assert row.status == "running"
        assert row.worker_id == "w2"
    finally:
        session.close()


def test_claim_next_step_returns_none_when_no_jobs(job_store: PostgresJobStore) -> None:
    assert job_store.claim_next_step("w1") is None


def _complete_evaluate(
    job_store: PostgresJobStore,
    worker_id: str = "w1",
    *,
    score: float | None = 0.5,
    error_code: str | None = None,
    error_message: str | None = None,
) -> str:
    """Claim the next (evaluate) step and complete it. Returns the step id."""
    step = job_store.claim_next_step(worker_id)
    assert step is not None and step.type == "evaluate"
    job_store.complete_step_and_advance(
        step.step_id,
        EvaluateOutcome(
            run_id=str(uuid.uuid4()),
            score=score,
            error_code=error_code,
            error_message=error_message,
        ),
    )
    return step.step_id


def test_evaluate_improvement_enqueues_improve_step(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    step = job_store.claim_next_step("w1")
    assert step is not None
    run_id = str(uuid.uuid4())

    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=run_id, score=0.5)
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.finished_at is None
    assert refreshed.stop_reason is None
    assert refreshed.best_score == pytest.approx(0.5)
    assert refreshed.best_agent_version_id == step.agent_version_id
    assert refreshed.best_version == 0
    assert refreshed.iterations[0].status == "completed"
    assert refreshed.iterations[0].score == pytest.approx(0.5)
    assert refreshed.iterations[0].improved is True
    assert refreshed.iterations[0].run_id == run_id

    # An improve step for the same iteration is now claimable.
    improve = job_store.claim_next_step("w1")
    assert improve is not None
    assert improve.type == "improve"
    assert improve.iteration == 0
    assert improve.agent_version_id == step.agent_version_id
    assert improve.stale_after_sec == 1800


def test_evaluate_hitting_max_iterations_completes_job(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=1)
    _complete_evaluate(job_store, score=0.25)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "max_iterations"
    assert refreshed.finished_at is not None
    assert refreshed.best_score == pytest.approx(0.25)

    # No successor step was enqueued.
    assert job_store.claim_next_step("w1") is None
    session = get_session_factory()()
    try:
        assert len(list(session.scalars(select(StepRow)))) == 1
    finally:
        session.close()


def test_improve_ok_inserts_version_one_and_next_evaluate(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)

    improve = job_store.claim_next_step("w1")
    assert improve is not None and improve.type == "improve"
    new_spec = improve.spec.model_copy(
        update={"system_prompt": "Verify every command before finishing.", "max_steps": 90}
    )
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(spec=new_spec, rationale="Add an explicit verification step."),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.current_iteration == 1
    assert refreshed.stop_reason is None

    next_step = job_store.claim_next_step("w1")
    assert next_step is not None
    assert next_step.type == "evaluate"
    assert next_step.iteration == 1
    assert next_step.version == 1
    assert next_step.spec == new_spec
    assert next_step.stale_after_sec == 3600

    version = job_store.get_agent_version(next_step.agent_version_id)
    assert version is not None
    assert version.version == 1
    assert version.parent_version_id == improve.agent_version_id
    assert version.created_by == "improver"
    assert version.rationale == "Add an explicit verification step."


def test_improve_error_with_existing_best_completes_failed_improve(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=None,
            error_code="improver_error",
            error_message="LLM returned invalid JSON twice",
        ),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "failed_improve"
    assert refreshed.finished_at is not None
    # The best-so-far agent is still a valid answer, so no job-level error.
    assert refreshed.error_code is None
    assert refreshed.best_agent_version_id is not None
    assert job_store.claim_next_step("w1") is None

    session = get_session_factory()()
    try:
        improve_row = session.scalars(
            select(StepRow).where(StepRow.type == "improve")
        ).one()
        assert improve_row.status == "failed"
        assert improve_row.error_code == "improver_error"
    finally:
        session.close()


def test_improve_error_without_best_fails_job(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)
    improve = job_store.claim_next_step("w1")
    assert improve is not None

    # Force the "no best yet" branch: clear the best pointer the baseline set.
    session = get_session_factory()()
    try:
        row = session.get(JobRow, uuid.UUID(job.job_id))
        assert row is not None
        row.best_agent_version_id = None
        row.best_score = None
        session.commit()
    finally:
        session.close()

    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=None, error_code="improver_error", error_message="no usable proposal"
        ),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.stop_reason == "failed"
    assert refreshed.error_code == "improver_error"
    assert refreshed.error_message == "no usable proposal"
    assert refreshed.finished_at is not None


def test_evaluate_error_fails_job_with_copied_error(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(
        job_store,
        score=None,
        error_code="execution_unavailable",
        error_message="harbor CLI not found",
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_code == "execution_unavailable"
    assert refreshed.error_message == "harbor CLI not found"
    assert refreshed.finished_at is not None
    assert refreshed.best_agent_version_id is None
    assert refreshed.best_score is None
    assert refreshed.stop_reason is None
    assert refreshed.iterations[0].status == "failed"
    assert refreshed.iterations[0].score is None
    assert refreshed.iterations[0].improved is None
    assert job_store.claim_next_step("w1") is None


def test_fail_step_fails_step_and_job(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store)
    step = job_store.claim_next_step("w1")
    assert step is not None

    job_store.fail_step(step.step_id, error_code="internal_error", error_message="boom")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_code == "internal_error"
    assert refreshed.error_message == "boom"
    assert refreshed.iterations[0].status == "failed"
    assert job_store.claim_next_step("w1") is None


def test_get_job_iterations_improved_flags_and_changed_fields(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=4, min_delta=0.01)

    # Iteration 0: baseline scores 0.50.
    _complete_evaluate(job_store, score=0.50)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    proposal = improve.spec.model_copy(
        update={"system_prompt": "Check your work.", "max_steps": 100}
    )
    job_store.complete_step_and_advance(
        improve.step_id, ImproveOutcome(spec=proposal, rationale="Verify before exit.")
    )

    # Iteration 1: improved to 0.70.
    _complete_evaluate(job_store, score=0.70)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert len(refreshed.iterations) == 2

    it0, it1 = refreshed.iterations
    assert it0.iteration == 0
    assert it0.version == 0
    assert it0.improved is True
    assert it0.rationale is None
    assert it0.changed_fields == []
    assert it0.score == pytest.approx(0.50)

    assert it1.iteration == 1
    assert it1.version == 1
    assert it1.improved is True
    assert it1.rationale == "Verify before exit."
    assert it1.changed_fields == ["max_steps", "system_prompt"]
    assert it1.score == pytest.approx(0.70)

    assert refreshed.best_score == pytest.approx(0.70)
    assert refreshed.best_version == 1
    assert refreshed.best_agent_version_id == it1.agent_version_id
    assert refreshed.current_iteration == 1


def test_non_improving_iteration_reports_improved_false_and_keeps_best(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=4, patience=2, min_delta=0.01)

    _complete_evaluate(job_store, score=0.60)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=improve.spec.model_copy(update={"max_steps": 120}),
            rationale="More steps.",
        ),
    )

    # Iteration 1 lands exactly on the min_delta boundary -> NOT an improvement.
    _complete_evaluate(job_store, score=0.61)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.iterations[1].improved is False
    assert refreshed.iterations[1].changed_fields == ["max_steps"]
    assert refreshed.best_score == pytest.approx(0.60)
    assert refreshed.best_version == 0

    # Streak is 1 of patience 2, so the loop continues with another improve step.
    nxt = job_store.claim_next_step("w1")
    assert nxt is not None and nxt.type == "improve" and nxt.iteration == 1


def test_patience_exhausted_stops_with_no_improvement(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=10, patience=2, min_delta=0.01)

    _complete_evaluate(job_store, score=0.60)
    for max_steps in (110, 120):
        improve = job_store.claim_next_step("w1")
        assert improve is not None and improve.type == "improve"
        job_store.complete_step_and_advance(
            improve.step_id,
            ImproveOutcome(
                spec=improve.spec.model_copy(update={"max_steps": max_steps}),
                rationale=f"Try {max_steps} steps.",
            ),
        )
        _complete_evaluate(job_store, score=0.40)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "no_improvement"
    assert refreshed.best_score == pytest.approx(0.60)
    assert refreshed.best_version == 0
    assert len(refreshed.iterations) == 3
    assert [i.improved for i in refreshed.iterations] == [True, False, False]
    assert job_store.claim_next_step("w1") is None


def test_improve_step_after_regression_is_based_on_best_version(job_store) -> None:
    """The §8.3 invariant: a regressed iteration must not become the next parent."""
    job = _create_job(job_store, max_iterations=9, patience=9, min_iterations=9)

    # iteration 0 scores 0.8 and becomes best
    step0 = job_store.claim_next_step("w")
    run0 = "aaaaaaaa-0000-0000-0000-000000000000"
    job_store.complete_step_and_advance(
        step0.step_id,
        EvaluateOutcome(run_id=run0, score=0.8, task_rewards={"a": 0.8}),
    )
    best_after_0 = job_store.get_job(job.job_id).best_agent_version_id

    # improve -> version 1
    improve0 = job_store.claim_next_step("w")
    assert improve0.type == "improve"
    assert improve0.agent_version_id == best_after_0
    worse = improve0.spec.model_copy(update={"max_steps": 11})
    job_store.complete_step_and_advance(
        improve0.step_id, ImproveOutcome(spec=worse, rationale="try more steps")
    )

    # iteration 1 REGRESSES to 0.2 — best must stay at version 0
    step1 = job_store.claim_next_step("w")
    job_store.complete_step_and_advance(
        step1.step_id,
        EvaluateOutcome(
            run_id="aaaaaaaa-1111-0000-0000-000000000000",
            score=0.2,
            task_rewards={"a": 0.2},
        ),
    )
    refreshed = job_store.get_job(job.job_id)
    assert refreshed.best_score == 0.8
    assert refreshed.best_agent_version_id == best_after_0

    # the NEXT improve step must be based on version 0, not the regressed version 1
    improve1 = job_store.claim_next_step("w")
    assert improve1.type == "improve"
    assert improve1.agent_version_id == best_after_0, (
        "improve step must backtrack to the best version, not build on the regression"
    )
    assert improve1.version == 0
    assert improve1.spec.max_steps != 11, "must be editing the best spec, not the worse one"

    # and the resulting version records version 0 as its parent
    job_store.complete_step_and_advance(
        improve1.step_id,
        ImproveOutcome(
            spec=improve1.spec.model_copy(update={"max_steps": 42}),
            rationale="retry from best",
        ),
    )
    history = job_store.get_job(job.job_id).iterations
    assert history[-1].based_on_version == 0


def test_iteration_history_reports_per_task_movement(job_store) -> None:
    """Same mean, different distribution — the case a mean score cannot express."""
    job = _create_job(
        job_store, task_ids=["a", "b"], max_iterations=9, patience=9, min_iterations=9
    )

    step0 = job_store.claim_next_step("w")
    job_store.complete_step_and_advance(
        step0.step_id,
        EvaluateOutcome(
            run_id="bbbbbbbb-0000-0000-0000-000000000000",
            score=0.5,
            task_rewards={"a": 1.0, "b": 0.0},
        ),
    )
    improve0 = job_store.claim_next_step("w")
    job_store.complete_step_and_advance(
        improve0.step_id,
        ImproveOutcome(spec=improve0.spec, rationale="swap emphasis"),
    )
    step1 = job_store.claim_next_step("w")
    job_store.complete_step_and_advance(
        step1.step_id,
        EvaluateOutcome(
            run_id="bbbbbbbb-1111-0000-0000-000000000000",
            score=0.5,
            task_rewards={"a": 0.0, "b": 1.0},
        ),
    )

    history = job_store.get_job(job.job_id).iterations
    assert history[0].fixed_tasks == [] and history[0].regressed_tasks == []
    assert history[1].score == history[0].score          # identical mean
    assert history[1].improved is False                  # so it is not promoted
    assert history[1].fixed_tasks == ["b"]               # but the movement is visible
    assert history[1].regressed_tasks == ["a"]


def test_complete_step_and_advance_is_idempotent_on_double_call(
    job_store: PostgresJobStore,
) -> None:
    """A stale-requeued step can legitimately be completed twice (by two workers).

    The second call must be a silent no-op: no second successor step, no second
    advance of current_iteration, no change to best_score.
    """
    job = _create_job(job_store, max_iterations=5)
    step = job_store.claim_next_step("w1")
    assert step is not None

    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.5)
    )
    after_first = job_store.get_job(job.job_id)
    assert after_first is not None
    session = get_session_factory()()
    try:
        step_count_after_first = len(list(session.scalars(select(StepRow))))
    finally:
        session.close()
    assert step_count_after_first == 2  # original evaluate + enqueued improve

    # Second call, same step_id, DIFFERENT outcome: must be a no-op.
    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.9)
    )

    after_second = job_store.get_job(job.job_id)
    assert after_second is not None
    assert after_second.best_score == after_first.best_score == pytest.approx(0.5)
    assert after_second.current_iteration == after_first.current_iteration
    session = get_session_factory()()
    try:
        step_count_after_second = len(list(session.scalars(select(StepRow))))
    finally:
        session.close()
    assert step_count_after_second == step_count_after_first


def test_fail_step_is_idempotent_on_double_call(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store)
    step = job_store.claim_next_step("w1")
    assert step is not None

    job_store.fail_step(step.step_id, error_code="internal_error", error_message="boom")
    after_first = job_store.get_job(job.job_id)
    assert after_first is not None
    assert after_first.status == "failed"

    # Second call with a different error must not overwrite the first.
    job_store.fail_step(step.step_id, error_code="other_error", error_message="different")

    after_second = job_store.get_job(job.job_id)
    assert after_second is not None
    assert after_second.error_code == "internal_error"
    assert after_second.error_message == "boom"


def test_complete_step_and_advance_on_already_completed_job_is_noop(
    job_store: PostgresJobStore,
) -> None:
    """Simulates the exact race: a second worker's late completion call against a
    job/step that a first, faster worker already carried to a terminal state."""
    job = _create_job(job_store, max_iterations=1)
    step = job_store.claim_next_step("w1")
    assert step is not None

    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.25)
    )
    completed = job_store.get_job(job.job_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.stop_reason == "max_iterations"

    # A late duplicate call for the same (already-terminal) step/job must not
    # mutate anything.
    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.99)
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.best_score == pytest.approx(0.25)
    assert refreshed.finished_at == completed.finished_at


def test_fail_step_on_already_terminal_job_leaves_it_untouched(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=1)
    step = job_store.claim_next_step("w1")
    assert step is not None
    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.25)
    )
    completed = job_store.get_job(job.job_id)
    assert completed is not None
    assert completed.status == "completed"

    # fail_step against the now-completed step/job must not flip it to failed.
    job_store.fail_step(step.step_id, error_code="internal_error", error_message="boom")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.error_code is None
    assert refreshed.error_message is None


def test_improve_outcome_missing_spec_normalizes_to_invalid_proposal(
    job_store: PostgresJobStore,
) -> None:
    """Appendix A4: ImproveOutcome(spec=None) with no error_code is normalized."""
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)

    improve = job_store.claim_next_step("w1")
    assert improve is not None

    job_store.complete_step_and_advance(improve.step_id, ImproveOutcome(spec=None))

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "failed_improve"

    session = get_session_factory()()
    try:
        improve_row = session.scalars(
            select(StepRow).where(StepRow.type == "improve")
        ).one()
        assert improve_row.status == "failed"
        assert improve_row.error_code == "invalid_proposal"
        assert improve_row.error_message
    finally:
        session.close()


def test_complete_step_and_advance_malformed_step_id_is_noop(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store)
    # Must not raise, must not touch anything.
    job_store.complete_step_and_advance(
        "not-a-uuid", EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.5)
    )
    job_store.complete_step_and_advance(
        str(uuid.uuid4()), EvaluateOutcome(run_id=str(uuid.uuid4()), score=0.5)
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "queued"
    assert refreshed.iterations[0].status == "queued"


def test_fail_step_malformed_step_id_is_noop(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store)
    job_store.fail_step("not-a-uuid", error_code="x", error_message="y")
    job_store.fail_step(str(uuid.uuid4()), error_code="x", error_message="y")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "queued"
