"""Iterative-improvement job submission and status endpoints."""

from __future__ import annotations

import math

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.agent_spec import AgentSpec
from api.config import BenchmarkConfig, load_config
from api.db import ping_db
from api.job_store import JobRecord, PostgresJobStore
from api.schemas import (
    AgentSpecView,
    BestAgentResponse,
    CreateJobRequest,
    CreateJobResponse,
    ErrorDetail,
    ErrorResponse,
    IterationView,
    JobBest,
    JobConfigEcho,
    JobResponse,
    ProposalView,
    RunError,
    RunStatus,
)
from api.store import PostgresRunStore, compute_summary

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_job_store(request: Request) -> PostgresJobStore:
    return request.app.state.job_store


def _get_run_store(request: Request) -> PostgresRunStore:
    return request.app.state.store


def evaluate_stale_after_sec(task_count: int, cfg: BenchmarkConfig) -> int:
    """Worst-case wall clock for one evaluate step, plus 20% slack.

    Tasks run `max_concurrency` at a time, each bounded by `per_task_timeout`.
    """
    waves = math.ceil(task_count / cfg.max_concurrency)
    return int(waves * cfg.per_task_timeout * 1.2)


def spec_view(spec: AgentSpec) -> AgentSpecView:
    return AgentSpecView(**spec.model_dump())


def _job_to_response(
    record: JobRecord,
    run_store: PostgresRunStore | None = None,
) -> JobResponse:
    """Map a JobRecord onto the wire shape.

    When `run_store` is supplied, each iteration's `summary` is enriched from
    the run row that iteration produced; without it `summary` stays None.
    """
    iterations: list[IterationView] = []
    for it in record.iterations:
        summary = None
        if run_store is not None and it.run_id:
            run = run_store.get(it.run_id)
            if run is not None:
                summary = compute_summary(run.tasks)

        proposal = None
        if it.rationale is not None:
            proposal = ProposalView(
                rationale=it.rationale,
                changed_fields=list(it.changed_fields),
                based_on_version=it.based_on_version,
            )

        iterations.append(
            IterationView(
                iteration=it.iteration,
                agent_version_id=it.agent_version_id,
                version=it.version,
                run_id=it.run_id,
                score=it.score,
                improved=it.improved,
                summary=summary,
                proposal=proposal,
                fixed_tasks=list(it.fixed_tasks),
                regressed_tasks=list(it.regressed_tasks),
            )
        )

    best = None
    if record.best_agent_version_id is not None:
        best = JobBest(
            agent_version_id=record.best_agent_version_id,
            version=record.best_version if record.best_version is not None else 0,
            score=record.best_score,
        )

    error = None
    if record.error_code:
        error = RunError(code=record.error_code, message=record.error_message or "")

    return JobResponse(
        job_id=record.job_id,
        status=RunStatus(record.status),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        config=JobConfigEcho(
            task_ids=list(record.task_ids),
            agent_model=record.agent_model,
            improver_model=record.improver_model,
            max_iterations=record.max_iterations,
            patience=record.patience,
            min_iterations=record.min_iterations,
            min_delta=record.min_delta,
        ),
        current_iteration=record.current_iteration,
        best=best,
        stop_reason=record.stop_reason,
        iterations=iterations,
        error=error,
    )


@router.post(
    "",
    response_model=CreateJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_job(
    body: CreateJobRequest,
    request: Request,
) -> CreateJobResponse | JSONResponse:
    cfg = load_config()
    job_store = _get_job_store(request)

    if body.task_ids is None:
        task_ids = list(cfg.default_task_ids)
    else:
        unknown = [tid for tid in body.task_ids if tid not in cfg.known_task_ids]
        if unknown:
            return _error(
                400,
                "unknown_task_ids",
                f"Unknown task_ids: {unknown}",
                details={"unknown": unknown},
            )
        task_ids = list(body.task_ids)

    agent_model = body.agent_model or cfg.default_agent_model
    improver_model = body.improver_model or cfg.improver_model
    # `is None` (not `or`): an explicit min_delta=0.0 must survive.
    max_iterations = cfg.max_iterations if body.max_iterations is None else body.max_iterations
    patience = cfg.patience if body.patience is None else body.patience
    min_iterations = (
        cfg.min_iterations if body.min_iterations is None else body.min_iterations
    )
    min_delta = cfg.min_delta if body.min_delta is None else body.min_delta

    if not ping_db():
        return _error(
            503,
            "execution_unavailable",
            "Database is unavailable; cannot enqueue job",
        )

    try:
        record = job_store.create_job(
            task_ids=task_ids,
            agent_model=agent_model,
            improver_model=improver_model,
            max_iterations=max_iterations,
            patience=patience,
            min_iterations=min_iterations,
            min_delta=min_delta,
            max_job_duration_sec=cfg.max_job_duration_sec,
            evaluate_stale_after_sec=evaluate_stale_after_sec(len(task_ids), cfg),
        )
    except Exception as exc:  # noqa: BLE001
        return _error(503, "execution_unavailable", f"Failed to enqueue job: {exc}")

    return CreateJobResponse(
        job_id=record.job_id,
        status=RunStatus.queued,
        created_at=record.created_at,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job(job_id: str, request: Request) -> JobResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_job(job_id)
    if record is None:
        return _error(404, "job_not_found", f"No job found with id {job_id}")
    return _job_to_response(record, run_store=_get_run_store(request))


@router.get(
    "/{job_id}/best",
    response_model=BestAgentResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def get_best_agent(job_id: str, request: Request) -> BestAgentResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_job(job_id)
    if record is None:
        return _error(404, "job_not_found", f"No job found with id {job_id}")
    if record.best_agent_version_id is None:
        return _error(
            409,
            "no_evaluation_yet",
            f"Job {job_id} has no completed evaluation yet",
        )

    version = job_store.get_agent_version(record.best_agent_version_id)
    if version is None:
        return _error(
            404,
            "agent_version_not_found",
            f"No agent version found with id {record.best_agent_version_id}",
        )

    return BestAgentResponse(
        job_id=record.job_id,
        agent_version_id=version.version_id,
        version=version.version,
        score=record.best_score,
        rationale=version.rationale,
        spec=spec_view(version.spec),
    )
