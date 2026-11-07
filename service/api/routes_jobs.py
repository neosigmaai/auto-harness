from __future__ import annotations

import difflib
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response

from ..auth import Principal, get_principal
from ..config import settings
from ..db import get_pool
from ..errors import ApiError
from ..models import (AgentOut, EventOut, IterationDetail, IterationSummary, JobAccepted, JobCreate, JobList,
                      JobOut, ResultSummary, TaskRunDetail, TaskRunOut)
from .. import repo

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_out(row, progress=None, best_index=None) -> JobOut:
    return JobOut(
        job_id=row["id"], status=row["status"], created_by=row["created_by"], config=row["config"],
        current_iteration=row["current_iteration"], baseline_score=row["baseline_score"], best_score=row["best_score"],
        best_iteration_index=best_index, stop_reason=row["stop_reason"], error=row["error"],
        cancel_requested=row["cancel_requested"], progress=progress, created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
    )


def _summary(it) -> ResultSummary | None:
    if it["n_pass"] is None:
        return None
    n = (it["n_pass"] or 0) + (it["n_fail"] or 0) + (it["n_error"] or 0)
    return ResultSummary(n_tasks=n, passed=it["n_pass"] or 0, failed=it["n_fail"] or 0, errored=it["n_error"] or 0, score=it["score"])


def _iter_summary(it) -> IterationSummary:
    prop = it["proposal"] or {}
    return IterationSummary(
        index=it["index"], phase=it["phase"], status=it["status"], decision=it["decision"],
        decision_reason=it["decision_reason"], score=it["score"], summary=_summary(it),
        hypothesis=prop.get("hypothesis"), agent_version_id=it["agent_version_id"], agent_sha256=it["agent_sha256"],
        started_at=it["started_at"], finished_at=it["finished_at"],
    )


async def _owned_job(conn, p: Principal, job_id: UUID):
    job = await repo.get_job(conn, p, job_id)
    if job is None:
        raise ApiError(404, "Not found", "Job not found.")  # same answer for 'other tenant' and 'does not exist'
    return job


@router.post("", response_model=JobAccepted, status_code=202, summary="Submit an optimisation job (async)")
async def submit_job(body: JobCreate, response: Response, p: Principal = Depends(get_principal),
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    pool = await get_pool()
    async with pool.acquire() as conn:
        org = await repo.get_org(conn, p.org_id)
        if body.max_iterations > org["max_iterations"]:
            raise ApiError(422, "Quota exceeded", f"max_iterations may not exceed {org['max_iterations']} for this organisation.")
        config = body.model_dump()
        config["tasks"] = body.resolved_tasks()
        config["agent_model"] = body.agent_model or settings.agent_model
        config["optimizer_model"] = body.optimizer_model or settings.optimizer_model
        job, created = await repo.create_job(conn, p, config, idempotency_key)
    response.status_code = 202 if created else 200
    response.headers["Location"] = f"/v1/jobs/{job['id']}"
    return JobAccepted(job_id=job["id"], status=job["status"], status_url=f"/v1/jobs/{job['id']}")


@router.get("", response_model=JobList, summary="List jobs visible to the caller")
async def list_jobs(p: Principal = Depends(get_principal), status: str | None = Query(default=None),
                    limit: int = Query(default=50, ge=1, le=200), before: str | None = Query(default=None)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await repo.list_jobs(conn, p, status, limit, before)
    items = [_job_out(r) for r in rows]
    next_cursor = rows[-1]["created_at"].isoformat() if len(rows) == limit else None
    return JobList(items=items, next_cursor=next_cursor)


@router.get("/{job_id}", response_model=JobOut, summary="Job status (cheap; poll this)")
async def get_job(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await _owned_job(conn, p, job_id)
        progress = await repo.job_progress(conn, job_id)
        best_index = None
        if job["best_iteration_id"]:
            best_index = await conn.fetchval("SELECT index FROM iterations WHERE id=$1", job["best_iteration_id"])
    return _job_out(job, progress, best_index)


@router.post("/{job_id}/cancel", response_model=JobOut, summary="Request cancellation")
async def cancel_job(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await repo.request_cancel(conn, p, job_id)
        if job is None:
            raise ApiError(404, "Not found", "Job not found.")
        progress = await repo.job_progress(conn, job_id)
    return _job_out(job, progress)


@router.get("/{job_id}/iterations", response_model=list[IterationSummary], summary="Iteration history")
async def list_iterations(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        its = await repo.list_iterations(conn, job_id)
    return [_iter_summary(i) for i in its]


@router.get("/{job_id}/iterations/{index}", response_model=IterationDetail, summary="One iteration in full")
async def get_iteration(job_id: UUID, index: int, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        it = await repo.get_iteration(conn, job_id, index)
        if it is None:
            raise ApiError(404, "Not found", "Iteration not found.")
        runs = await repo.list_task_runs(conn, it["id"])
        parent_src = None
        if it["parent_iteration_id"]:
            parent = await conn.fetchrow(
                "SELECT a.content FROM iterations i JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.id=$1",
                it["parent_iteration_id"])
            parent_src = parent["content"] if parent else None
        digests = await conn.fetch(
            "SELECT task_name, failure_class, digest FROM task_runs WHERE iteration_id=$1 AND status IN ('failed','timeout','error') ORDER BY task_name",
            it["id"])
    diff = None
    if parent_src is not None:
        diff = "".join(difflib.unified_diff(parent_src.splitlines(True), it["agent_source"].splitlines(True),
                                            fromfile=f"agent.py@iter{index - 1}", tofile=f"agent.py@iter{index}"))
    base = _iter_summary(it).model_dump()
    return IterationDetail(
        **base, proposal=it["proposal"], learnings=it["learnings"], agent_source=it["agent_source"], diff_vs_parent=diff,
        tasks=[TaskRunOut(**dict(r)) for r in runs],
        failure_summary=[{"task_name": d["task_name"], "failure_class": d["failure_class"],
                          "summary": (d["digest"] or {}).get("summary")} for d in digests],
    )


@router.get("/{job_id}/iterations/{index}/tasks/{task}", response_model=TaskRunDetail, summary="Task run detail (trace, test output)")
async def get_task_run(job_id: UUID, index: int, task: str, p: Principal = Depends(get_principal),
                       include_trace: bool = Query(default=False)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        it = await repo.get_iteration(conn, job_id, index)
        if it is None:
            raise ApiError(404, "Not found", "Iteration not found.")
        tr = await repo.get_task_run(conn, it["id"], task)
        if tr is None:
            raise ApiError(404, "Not found", "Task run not found.")
    d = dict(tr)
    return TaskRunDetail(task_name=d["task_name"], status=d["status"], reward=d["reward"], failure_class=d["failure_class"],
                         duration_s=d["duration_s"], n_steps=d["n_steps"], exit_reason=d["exit_reason"],
                         test_stdout=d["test_stdout"], digest=d["digest"], trace=d["trace"] if include_trace else None)


@router.get("/{job_id}/agent", response_model=AgentOut, summary="Best agent source + diff vs baseline")
async def get_agent(job_id: UUID, p: Principal = Depends(get_principal)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        job = await _owned_job(conn, p, job_id)
        base = await repo.get_iteration(conn, job_id, 0)
        if base is None:
            raise ApiError(409, "Not ready", "Baseline iteration has not started yet.")
        best = None
        if job["best_iteration_id"]:
            best = await conn.fetchrow(
                "SELECT i.index, a.sha256, a.content FROM iterations i JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.id=$1",
                job["best_iteration_id"])
    src = best["content"] if best else base["agent_source"]
    diff = None
    if best and best["sha256"] != base["agent_sha256"]:
        diff = "".join(difflib.unified_diff(base["agent_source"].splitlines(True), src.splitlines(True),
                                            fromfile="agent.py@baseline", tofile=f"agent.py@iter{best['index']}"))
    return AgentOut(job_id=job_id, best_iteration_index=best["index"] if best else None, baseline_sha256=base["agent_sha256"],
                    best_sha256=best["sha256"] if best else None, source=src, diff_vs_baseline=diff)


@router.get("/{job_id}/events", response_model=list[EventOut], summary="Event timeline")
async def list_events(job_id: UUID, p: Principal = Depends(get_principal), after: int = Query(default=0, ge=0),
                      limit: int = Query(default=200, ge=1, le=1000)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await _owned_job(conn, p, job_id)
        rows = await repo.list_events(conn, job_id, after, limit)
    return [EventOut(**dict(r)) for r in rows]
