"""Job repository: create, fetch, list, claim (SKIP LOCKED), and persist results.

All DB access for the job lifecycle lives here so the worker and API routes stay
thin. Domain objects (Trajectory/Iteration/…) are mapped to ORM rows on write.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from harness_service.constants import ExecutorKind, IterationDecision, JobMode, JobStatus
from harness_service.domain import Iteration as DomainIteration
from harness_service.domain import Trajectory
from harness_service.db.models import Iteration, Job, TaskResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class JobContext:
    """Detached snapshot of a claimed job — safe to use outside a DB session."""

    id: UUID
    mode: JobMode
    executor: ExecutorKind
    subset: list[str]
    config: dict
    max_iterations: int
    patience: int

    @classmethod
    def from_orm(cls, job: Job) -> "JobContext":
        return cls(
            id=job.id,
            mode=JobMode(job.mode),
            executor=ExecutorKind(job.executor),
            subset=list(job.subset),
            config=dict(job.config or {}),
            max_iterations=job.max_iterations,
            patience=job.patience,
        )


async def create_job(
    session: AsyncSession,
    *,
    org_id: UUID,
    user_id: UUID,
    mode: JobMode,
    executor: ExecutorKind,
    subset: list[str],
    config: dict,
    max_iterations: int,
    patience: int,
) -> Job:
    job = Job(
        org_id=org_id,
        user_id=user_id,
        mode=mode,
        executor=executor,
        status=JobStatus.QUEUED,
        subset=subset,
        config=config,
        max_iterations=max_iterations,
        patience=patience,
    )
    session.add(job)
    await session.flush()
    return job


async def get_job(session: AsyncSession, job_id: UUID) -> Job | None:
    stmt = (
        select(Job)
        .where(Job.id == job_id)
        .options(selectinload(Job.iterations).selectinload(Iteration.task_results))
    )
    return (await session.execute(stmt)).scalars().first()


async def list_jobs(
    session: AsyncSession, *, org_id: UUID, user_id: UUID | None = None, limit: int = 100
) -> list[Job]:
    stmt = (
        select(Job)
        .where(Job.org_id == org_id)
        .options(selectinload(Job.iterations))
        .order_by(Job.created_at.desc())
        .limit(limit)
    )
    if user_id is not None:  # M5 uses this to scope members to their own jobs
        stmt = stmt.where(Job.user_id == user_id)
    return list((await session.execute(stmt)).scalars().all())


async def claim_next_job(session: AsyncSession) -> Job | None:
    """Atomically claim one queued job and mark it RUNNING.

    ``FOR UPDATE SKIP LOCKED`` lets multiple worker loops claim distinct jobs
    without blocking each other. The session_scope commit releases the row lock
    with status already flipped to RUNNING, so no other worker re-claims it.
    """
    stmt = (
        select(Job)
        .where(Job.status == JobStatus.QUEUED)
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = (await session.execute(stmt)).scalars().first()
    if job is None:
        return None
    job.status = JobStatus.RUNNING
    await session.flush()
    return job


def _to_orm_iteration(job_id: UUID, di: DomainIteration) -> Iteration:
    imp = di.improvement
    orm = Iteration(
        job_id=job_id,
        idx=di.idx,
        agent_source=di.agent_state.source,
        agent_params=di.agent_state.params,
        agent_hash=di.agent_state.content_hash,
        proposer=(imp.proposer if imp else None),
        proposal_rationale=(imp.rationale if imp else None),
        proposal_diff=(imp.diff_summary if imp else None),
        llm_request=(imp.llm_request if imp else None),
        llm_response=(imp.llm_response if imp else None),
        val_score=di.result.val_score,
        n_passed=di.result.n_passed,
        n_failed=di.result.n_failed,
        decision=di.decision,
        decision_reason=di.decision_reason,
        context_snapshot=di.context_snapshot,
    )
    orm.task_results = [
        TaskResult(
            task_id=o.task_id,
            reward=o.reward,
            passed=o.passed,
            duration_s=o.duration_s,
            trace_excerpt=o.trace_excerpt,
            failure_reason=o.failure_reason,
        )
        for o in di.result.outcomes
    ]
    return orm


async def persist_success(session: AsyncSession, job_id: UUID, outcome) -> None:
    """Write every iteration + task result, set best-so-far + optimize outcome, mark SUCCEEDED.

    ``outcome`` is a services.runner.RunOutcome (trajectory + optional train/test split
    and held-out test result).
    """
    job = await session.get(Job, job_id)
    if job is None:
        raise RuntimeError(f"job {job_id} vanished before persistence")

    trajectory = outcome.trajectory
    best_score = -1.0
    best_orm_id: UUID | None = None
    # Only accepted/baseline iterations are eligible to be "best" (ERROR iters score 0).
    for di in trajectory.iterations:
        orm_it = _to_orm_iteration(job_id, di)
        session.add(orm_it)
        await session.flush()  # assigns orm_it.id
        if di.decision != IterationDecision.ERROR and di.result.val_score > best_score:
            best_score = di.result.val_score
            best_orm_id = orm_it.id

    job.best_val_score = best_score if best_score >= 0 else None
    job.best_iteration_id = best_orm_id
    job.accumulated_context = trajectory.build_context()
    job.baseline_val_score = outcome.baseline_val_score
    job.train_subset = outcome.train_subset or None
    job.test_subset = outcome.test_subset or None
    if outcome.test_result is not None:
        tr = outcome.test_result
        job.test_val_score = tr.val_score
        job.test_results = {
            "val_score": tr.val_score,
            "n_passed": tr.n_passed,
            "n_failed": tr.n_failed,
            "tasks": [
                {"task_id": o.task_id, "reward": o.reward, "passed": o.passed}
                for o in tr.outcomes
            ],
        }
    job.status = JobStatus.SUCCEEDED
    job.finished_at = _now()


async def mark_failed(session: AsyncSession, job_id: UUID, error: str) -> None:
    job = await session.get(Job, job_id)
    if job is None:
        return
    job.status = JobStatus.FAILED
    job.error = error[:4000]
    job.finished_at = _now()
