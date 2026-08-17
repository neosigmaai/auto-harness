"""Job endpoints: submit, fetch status/summary, list, and iteration history."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from harness_service.api.deps import Principal, get_principal
from harness_service.api.schemas import (
    IterationRead,
    JobCreate,
    JobListItem,
    JobRead,
    JobSummary,
    TaskResultRead,
)
from harness_service.config import get_settings
from harness_service.constants import DEFAULT_MAX_ITERATIONS, DEFAULT_PATIENCE, JobMode
from harness_service.db import get_session
from harness_service.db.models import Iteration, Job
from harness_service.services import jobs as jobs_svc
from harness_service.tasks import resolve_subset

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _build_summary(job: Job) -> JobSummary | None:
    """Structured result from the job's best iteration (baseline in M1)."""
    if not job.iterations:
        return None
    best = max(job.iterations, key=lambda it: it.val_score)
    passed = [t.task_id for t in best.task_results if t.passed]
    failed = [t.task_id for t in best.task_results if not t.passed]
    failures = [
        {"task_id": t.task_id, "failure_reason": t.failure_reason}
        for t in best.task_results
        if not t.passed
    ]
    return JobSummary(
        val_score=best.val_score,
        n_passed=best.n_passed,
        n_failed=best.n_failed,
        passed_tasks=passed,
        failed_tasks=failed,
        failures=failures,
    )


def _job_read(job: Job, *, with_summary: bool = True) -> JobRead:
    improvement = (
        job.best_val_score - job.baseline_val_score
        if job.best_val_score is not None and job.baseline_val_score is not None
        else None
    )
    return JobRead(
        id=job.id,
        org_id=job.org_id,
        user_id=job.user_id,
        mode=JobMode(job.mode),
        executor=job.executor,
        status=job.status,
        subset=list(job.subset),
        max_iterations=job.max_iterations,
        patience=job.patience,
        n_iterations=len(job.iterations),
        best_val_score=job.best_val_score,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
        summary=_build_summary(job) if with_summary else None,
        baseline_val_score=job.baseline_val_score,
        train_subset=job.train_subset,
        test_subset=job.test_subset,
        test_val_score=job.test_val_score,
        improvement=improvement,
    )


def _iteration_read(it: Iteration) -> IterationRead:
    return IterationRead(
        idx=it.idx,
        decision=it.decision,
        decision_reason=it.decision_reason,
        val_score=it.val_score,
        n_passed=it.n_passed,
        n_failed=it.n_failed,
        agent_hash=it.agent_hash,
        agent_params=it.agent_params or {},
        proposer=it.proposer,
        proposal_rationale=it.proposal_rationale,
        proposal_diff=it.proposal_diff,
        created_at=it.created_at,
        agent_source=it.agent_source,
        task_results=[
            TaskResultRead(
                task_id=t.task_id,
                reward=t.reward,
                passed=t.passed,
                duration_s=t.duration_s,
                failure_reason=t.failure_reason,
                trace_excerpt=t.trace_excerpt,
            )
            for t in it.task_results
        ],
    )


@router.post("", response_model=JobRead, status_code=status.HTTP_201_CREATED)
async def submit_job(
    body: JobCreate,
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_session),
) -> JobRead:
    settings = get_settings()
    try:
        task_ids = resolve_subset(body.subset)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    executor = body.executor or settings.default_executor
    max_iter = body.max_iterations or (
        DEFAULT_MAX_ITERATIONS if body.mode == JobMode.OPTIMIZE else 1
    )
    patience = body.patience or DEFAULT_PATIENCE

    job = await jobs_svc.create_job(
        session,
        org_id=principal.org_id,
        user_id=principal.user_id,
        mode=body.mode,
        executor=executor,
        subset=task_ids,
        config=body.config,
        max_iterations=max_iter,
        patience=patience,
    )
    await session.commit()
    await session.refresh(job, attribute_names=["iterations"])
    return _job_read(job)


@router.get("", response_model=list[JobListItem])
async def list_jobs(
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_session),
) -> list[JobListItem]:
    jobs = await jobs_svc.list_jobs(session, org_id=principal.org_id)
    return [
        JobListItem(
            id=j.id,
            mode=JobMode(j.mode),
            executor=j.executor,
            status=j.status,
            best_val_score=j.best_val_score,
            n_iterations=len(j.iterations),
            created_at=j.created_at,
        )
        for j in jobs
    ]


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: UUID,
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_session),
) -> JobRead:
    job = await jobs_svc.get_job(session, job_id)
    if job is None or job.org_id != principal.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _job_read(job)


@router.get("/{job_id}/iterations", response_model=list[IterationRead])
async def get_iterations(
    job_id: UUID,
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_session),
) -> list[IterationRead]:
    job = await jobs_svc.get_job(session, job_id)
    if job is None or job.org_id != principal.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return [_iteration_read(it) for it in sorted(job.iterations, key=lambda i: i.idx)]
