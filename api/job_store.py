"""Postgres-backed store for iterative improvement jobs and their step queue.

Mirrors ``api/store.py``: sessions come from ``self._factory()()`` and are always
closed in a ``finally`` block. The steps table is a queue claimed with
``SELECT ... FOR UPDATE SKIP LOCKED``, exactly like ``PostgresRunStore.claim_next``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select, text, update

from api.agent_spec import AgentSpec, baseline_spec, changed_fields
from api.db import get_session_factory
from api.models import AgentVersionRow, JobRow, StepRow
from api.schemas import RunStatus
from api.services.scoring import compute_stop, task_movement
from api.store import _utcnow

STEP_EVALUATE = "evaluate"
STEP_IMPROVE = "improve"

#: Improve steps are a single LLM call; the run default is plenty.
IMPROVE_STALE_AFTER_SEC = 1800

CREATED_BY_BASELINE = "baseline"
CREATED_BY_IMPROVER = "improver"

STOP_FAILED_IMPROVE = "failed_improve"
STOP_FAILED = "failed"

_ACTIVE_JOB_STATUSES = frozenset({RunStatus.queued.value, RunStatus.running.value})

#: Per-row staleness predicate. The threshold lives in the row itself
#: (``steps.stale_after_sec``) because evaluate steps can legitimately run for hours,
#: so no Python-side timedelta can express it. The table name must be spelled out:
#: a bulk UPDATE does not alias the target table.
_STALE_STEP_PREDICATE = text(
    "steps.claimed_at < now() - make_interval(secs => steps.stale_after_sec)"
)


@dataclass(frozen=True)
class StepRecord:
    """Everything a worker needs to execute one claimed step."""

    step_id: str
    job_id: str
    type: str
    iteration: int
    agent_version_id: str
    version: int
    spec: AgentSpec
    task_ids: list[str]
    agent_model: str
    improver_model: str
    run_id: str | None
    stale_after_sec: int


@dataclass(frozen=True)
class AgentVersionRecord:
    version_id: str
    job_id: str
    version: int
    parent_version_id: str | None
    spec: AgentSpec
    rationale: str
    created_by: str
    created_at: datetime


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    agent_version_id: str
    version: int
    run_id: str | None
    score: float | None
    improved: bool | None
    rationale: str | None
    changed_fields: list[str]
    status: str
    # Per-task movement of this iteration against the best iteration BEFORE it.
    # A mean score cannot distinguish "fixed A, broke B" from "changed nothing";
    # these two lists are that missing signal, and they are fed to the improver.
    task_rewards: dict[str, float | None] = field(default_factory=dict)
    fixed_tasks: list[str] = field(default_factory=list)
    regressed_tasks: list[str] = field(default_factory=list)
    # Version this iteration's spec was derived from (None for the baseline).
    based_on_version: int | None = None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: str
    task_ids: list[str]
    agent_model: str
    improver_model: str
    max_iterations: int
    patience: int
    min_iterations: int
    min_delta: float
    current_iteration: int
    best_agent_version_id: str | None
    best_version: int | None
    best_score: float | None
    stop_reason: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_code: str | None
    error_message: str | None
    iterations: list[IterationRecord] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluateOutcome:
    """Result of an evaluate step. ``score`` is the mean reward for the iteration.

    ``task_rewards`` is the per-task snapshot ``{task_id: reward | None}`` that
    makes per-task movement derivable later; pass ``{}`` on the failure paths.
    """

    run_id: str
    score: float | None
    task_rewards: dict[str, float | None] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ImproveOutcome:
    """Result of an improve step. ``spec is None`` means the proposal was unusable."""

    spec: AgentSpec | None
    rationale: str = ""
    error_code: str | None = None
    error_message: str | None = None


def _uuid_or_none(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _version_to_record(row: AgentVersionRow) -> AgentVersionRecord:
    return AgentVersionRecord(
        version_id=str(row.id),
        job_id=str(row.job_id),
        version=row.version,
        parent_version_id=(
            str(row.parent_version_id) if row.parent_version_id is not None else None
        ),
        spec=AgentSpec.model_validate(row.spec),
        rationale=row.rationale,
        created_by=row.created_by,
        created_at=row.created_at,
    )


def _build_iterations(
    steps: list[StepRow],
    versions: dict[UUID, AgentVersionRow],
    min_delta: float,
) -> list[IterationRecord]:
    """One IterationRecord per evaluate step, ordered by iteration.

    ``improved`` is derived here rather than stored: an iteration improved when its
    score beats the best score of all *strictly earlier* evaluate steps by more than
    ``min_delta``. The first scored iteration always improved. Steps with no score yet
    (queued/running/failed) report ``improved=None``.

    ``fixed_tasks``/``regressed_tasks`` compare this step's task-reward snapshot
    against the best-so-far snapshot (not the immediately preceding one), matching
    ``best_score_so_far``/``improved``.
    """
    evaluates = sorted(
        (s for s in steps if s.type == STEP_EVALUATE),
        key=lambda s: (s.iteration, s.created_at),
    )
    records: list[IterationRecord] = []
    best_score_so_far: float | None = None
    best_rewards_so_far: dict[str, float | None] | None = None

    for step in evaluates:
        version = versions.get(step.agent_version_id)
        rationale: str | None = None
        changed: list[str] = []
        based_on_version: int | None = None
        if version is not None and version.parent_version_id is not None:
            rationale = version.rationale or None
            parent = versions.get(version.parent_version_id)
            if parent is not None:
                based_on_version = parent.version
                changed = changed_fields(
                    AgentSpec.model_validate(parent.spec),
                    AgentSpec.model_validate(version.spec),
                )

        improved: bool | None = None
        if step.score is not None:
            improved = (
                best_score_so_far is None or step.score > best_score_so_far + min_delta
            )

        movement = task_movement(best_rewards_so_far, step.task_rewards)

        records.append(
            IterationRecord(
                iteration=step.iteration,
                agent_version_id=str(step.agent_version_id),
                version=version.version if version is not None else -1,
                run_id=str(step.run_id) if step.run_id is not None else None,
                score=step.score,
                improved=improved,
                rationale=rationale,
                changed_fields=changed,
                status=step.status,
                task_rewards=dict(step.task_rewards or {}),
                fixed_tasks=movement.fixed,
                regressed_tasks=movement.regressed,
                based_on_version=based_on_version,
            )
        )

        if improved:
            best_score_so_far = step.score
            best_rewards_so_far = step.task_rewards

    return records


def _elapsed_sec(job: JobRow, now: datetime) -> float:
    started = job.started_at or job.created_at
    return (now - started).total_seconds()


def _apply_evaluate_outcome(
    session,
    step: StepRow,
    job: JobRow,
    outcome: EvaluateOutcome,
    now: datetime,
) -> None:
    step.run_id = _uuid_or_none(outcome.run_id)
    step.finished_at = now

    if outcome.error_code:
        # Infra failure: never counted as "no improvement".
        step.status = RunStatus.failed.value
        step.error_code = outcome.error_code
        step.error_message = outcome.error_message
        job.status = RunStatus.failed.value
        job.error_code = outcome.error_code
        job.error_message = outcome.error_message
        job.finished_at = now
        return

    score = 0.0 if outcome.score is None else float(outcome.score)
    step.status = RunStatus.completed.value
    step.score = score
    step.task_rewards = dict(outcome.task_rewards or {})

    decision = compute_stop(
        iteration=step.iteration,
        score=score,
        best_score=job.best_score,
        prior_non_improving_streak=job.non_improving_streak,
        max_iterations=job.max_iterations,
        patience=job.patience,
        min_iterations=job.min_iterations,
        min_delta=job.min_delta,
        elapsed_sec=_elapsed_sec(job, now),
        max_job_duration_sec=job.max_job_duration_sec,
    )

    job.non_improving_streak = decision.non_improving_streak
    if decision.improved:
        job.best_score = score
        job.best_agent_version_id = step.agent_version_id

    if decision.should_stop:
        job.status = RunStatus.completed.value
        job.stop_reason = decision.stop_reason
        job.finished_at = now
        return

    # The improve step edits the BEST spec so far, never a version that regressed.
    # When this iteration improved, best_agent_version_id was just set to
    # step.agent_version_id above, so the two coincide; when it regressed, this is
    # what backtracks the loop instead of compounding a bad proposal. The rejected
    # attempt stays visible in the iteration history the improver reads.
    session.add(
        StepRow(
            id=uuid.uuid4(),
            job_id=job.id,
            type=STEP_IMPROVE,
            status=RunStatus.queued.value,
            iteration=step.iteration,
            agent_version_id=job.best_agent_version_id or step.agent_version_id,
            stale_after_sec=IMPROVE_STALE_AFTER_SEC,
            created_at=now,
        )
    )


def _apply_improve_outcome(
    session,
    step: StepRow,
    job: JobRow,
    outcome: ImproveOutcome,
    now: datetime,
) -> None:
    step.finished_at = now

    if outcome.error_code or outcome.spec is None:
        error_code = outcome.error_code or "invalid_proposal"
        error_message = outcome.error_message or "Improver returned no valid AgentSpec"
        step.status = RunStatus.failed.value
        step.error_code = error_code
        step.error_message = error_message
        job.finished_at = now
        if job.best_agent_version_id is not None:
            # A best-so-far agent is still a valid answer for the job.
            job.status = RunStatus.completed.value
            job.stop_reason = STOP_FAILED_IMPROVE
        else:
            job.status = RunStatus.failed.value
            job.stop_reason = STOP_FAILED
            job.error_code = error_code
            job.error_message = error_message
        return

    step.status = RunStatus.completed.value

    next_version_number = (
        session.scalar(
            select(func.max(AgentVersionRow.version)).where(
                AgentVersionRow.job_id == job.id
            )
        )
        or 0
    ) + 1
    next_iteration = step.iteration + 1
    new_version_id = uuid.uuid4()

    session.add(
        AgentVersionRow(
            id=new_version_id,
            job_id=job.id,
            version=next_version_number,
            parent_version_id=step.agent_version_id,
            spec=outcome.spec.model_dump(),
            rationale=outcome.rationale,
            created_by=CREATED_BY_IMPROVER,
            created_at=now,
        )
    )
    session.flush()

    job.current_iteration = next_iteration
    session.add(
        StepRow(
            id=uuid.uuid4(),
            job_id=job.id,
            type=STEP_EVALUATE,
            status=RunStatus.queued.value,
            iteration=next_iteration,
            agent_version_id=new_version_id,
            stale_after_sec=job.evaluate_stale_after_sec,
            created_at=now,
        )
    )


class PostgresJobStore:
    """Job/agent-version/step store; also the step queue."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _factory(self):
        return self._session_factory or get_session_factory()

    def create_job(
        self,
        *,
        task_ids: list[str],
        agent_model: str,
        improver_model: str,
        max_iterations: int,
        patience: int,
        min_iterations: int,
        min_delta: float,
        max_job_duration_sec: int,
        evaluate_stale_after_sec: int,
    ) -> JobRecord:
        """Insert the job, agent version 0 and the iteration-0 evaluate step.

        All three inserts happen in ONE transaction. The explicit ``flush()`` calls
        are required: with no ORM relationships between these tables, SQLAlchemy would
        otherwise order the INSERTs by mapper sort key and try ``agent_versions``
        before ``jobs``, violating the foreign key.
        """
        now = _utcnow()
        job_id = uuid.uuid4()
        version_id = uuid.uuid4()
        spec = baseline_spec(agent_model)

        session = self._factory()()
        try:
            session.add(
                JobRow(
                    id=job_id,
                    status=RunStatus.queued.value,
                    task_ids=list(task_ids),
                    agent_model=agent_model,
                    improver_model=improver_model,
                    max_iterations=max_iterations,
                    patience=patience,
                    min_iterations=min_iterations,
                    min_delta=min_delta,
                    max_job_duration_sec=max_job_duration_sec,
                    evaluate_stale_after_sec=evaluate_stale_after_sec,
                    current_iteration=0,
                    non_improving_streak=0,
                    created_at=now,
                )
            )
            session.flush()
            session.add(
                AgentVersionRow(
                    id=version_id,
                    job_id=job_id,
                    version=0,
                    parent_version_id=None,
                    spec=spec.model_dump(),
                    rationale=CREATED_BY_BASELINE,
                    created_by=CREATED_BY_BASELINE,
                    created_at=now,
                )
            )
            session.flush()
            session.add(
                StepRow(
                    id=uuid.uuid4(),
                    job_id=job_id,
                    type=STEP_EVALUATE,
                    status=RunStatus.queued.value,
                    iteration=0,
                    agent_version_id=version_id,
                    stale_after_sec=evaluate_stale_after_sec,
                    created_at=now,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        record = self.get_job(str(job_id))
        assert record is not None
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        uid = _uuid_or_none(job_id)
        if uid is None:
            return None
        session = self._factory()()
        try:
            job = session.get(JobRow, uid)
            if job is None:
                return None

            versions = {
                row.id: row
                for row in session.scalars(
                    select(AgentVersionRow)
                    .where(AgentVersionRow.job_id == uid)
                    .order_by(AgentVersionRow.version)
                )
            }
            steps = list(
                session.scalars(
                    select(StepRow)
                    .where(StepRow.job_id == uid)
                    .order_by(StepRow.iteration, StepRow.created_at)
                )
            )

            best_version: int | None = None
            if job.best_agent_version_id is not None:
                best_row = versions.get(job.best_agent_version_id)
                best_version = best_row.version if best_row is not None else None

            return JobRecord(
                job_id=str(job.id),
                status=job.status,
                task_ids=list(job.task_ids),
                agent_model=job.agent_model,
                improver_model=job.improver_model,
                max_iterations=job.max_iterations,
                patience=job.patience,
                min_iterations=job.min_iterations,
                min_delta=job.min_delta,
                current_iteration=job.current_iteration,
                best_agent_version_id=(
                    str(job.best_agent_version_id)
                    if job.best_agent_version_id is not None
                    else None
                ),
                best_version=best_version,
                best_score=job.best_score,
                stop_reason=job.stop_reason,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                error_code=job.error_code,
                error_message=job.error_message,
                iterations=_build_iterations(steps, versions, job.min_delta),
            )
        finally:
            session.close()

    def get_agent_version(self, version_id: str) -> AgentVersionRecord | None:
        uid = _uuid_or_none(version_id)
        if uid is None:
            return None
        session = self._factory()()
        try:
            row = session.get(AgentVersionRow, uid)
            if row is None:
                return None
            return _version_to_record(row)
        finally:
            session.close()

    def claim_next_step(self, worker_id: str) -> StepRecord | None:
        """Atomically claim the next queued step (or requeue-and-claim a stale one).

        Uses SELECT ... FOR UPDATE SKIP LOCKED so concurrent workers never claim the
        same step. Locks the step row first and the job row second — every method in
        this class uses that order to avoid deadlocks.
        """
        session = self._factory()()
        try:
            now = _utcnow()

            # Requeue steps whose own stale_after_sec has elapsed (best-effort).
            session.execute(
                update(StepRow)
                .where(
                    StepRow.status == RunStatus.running.value,
                    StepRow.claimed_at.is_not(None),
                    _STALE_STEP_PREDICATE,
                )
                .values(
                    status=RunStatus.queued.value,
                    worker_id=None,
                    claimed_at=None,
                    started_at=None,
                )
            )

            step = session.scalar(
                select(StepRow)
                .where(StepRow.status == RunStatus.queued.value)
                .order_by(StepRow.created_at, StepRow.iteration)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if step is None:
                session.commit()
                return None

            step.status = RunStatus.running.value
            step.worker_id = worker_id
            step.claimed_at = now
            step.started_at = now

            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is None:
                # Job vanished under us; drop the claim rather than run an orphan.
                session.rollback()
                return None
            if job.status == RunStatus.queued.value:
                job.status = RunStatus.running.value
                if job.started_at is None:
                    job.started_at = now

            version = session.get(AgentVersionRow, step.agent_version_id)
            if version is None:
                session.rollback()
                return None

            record = StepRecord(
                step_id=str(step.id),
                job_id=str(step.job_id),
                type=step.type,
                iteration=step.iteration,
                agent_version_id=str(step.agent_version_id),
                version=version.version,
                spec=AgentSpec.model_validate(version.spec),
                task_ids=list(job.task_ids),
                agent_model=job.agent_model,
                improver_model=job.improver_model,
                run_id=str(step.run_id) if step.run_id is not None else None,
                stale_after_sec=step.stale_after_sec,
            )
            session.commit()
            return record
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def complete_step_and_advance(
        self, step_id: str, outcome: EvaluateOutcome | ImproveOutcome
    ) -> None:
        """Close a step and, in the SAME transaction, advance the job.

        Either the successor step is enqueued or the job reaches a terminal state, so
        there is never a live job with nothing queued. A crash before commit leaves the
        step ``running`` until stale-requeue picks it up again.
        """
        uid = _uuid_or_none(step_id)
        if uid is None:
            return

        session = self._factory()()
        try:
            step = session.get(StepRow, uid, with_for_update=True)
            if step is None:
                session.commit()
                return
            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is None:
                session.commit()
                return

            now = _utcnow()
            if isinstance(outcome, EvaluateOutcome):
                _apply_evaluate_outcome(session, step, job, outcome, now)
            else:
                _apply_improve_outcome(session, step, job, outcome, now)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def fail_step(self, step_id: str, *, error_code: str, error_message: str) -> None:
        """Fail a step and its job (worker-level unexpected failures)."""
        uid = _uuid_or_none(step_id)
        if uid is None:
            return

        session = self._factory()()
        try:
            step = session.get(StepRow, uid, with_for_update=True)
            if step is None:
                session.commit()
                return
            now = _utcnow()
            step.status = RunStatus.failed.value
            step.error_code = error_code
            step.error_message = error_message
            step.finished_at = now

            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is not None and job.status in _ACTIVE_JOB_STATUSES:
                job.status = RunStatus.failed.value
                job.error_code = error_code
                job.error_message = error_message
                job.finished_at = now

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def clear(self) -> None:
        """Delete all job data. Order matters: children before parents."""
        session = self._factory()()
        try:
            session.execute(delete(StepRow))
            session.execute(delete(AgentVersionRow))
            session.execute(delete(JobRow))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Default process-wide store (API and worker construct their own as needed).
job_store = PostgresJobStore()
