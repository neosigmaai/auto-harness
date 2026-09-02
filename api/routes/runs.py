"""Run submission and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.config import load_config
from api.db import ping_db
from api.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    ErrorDetail,
    ErrorResponse,
    RunResponse,
    RunStatus,
)
from api.store import PostgresRunStore

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_store(request: Request) -> PostgresRunStore:
    return request.app.state.store


@router.post(
    "",
    response_model=CreateRunResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_run(
    body: CreateRunRequest,
    request: Request,
) -> CreateRunResponse | JSONResponse:
    cfg = load_config()
    store = _get_store(request)

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

    if not ping_db():
        return _error(
            503,
            "execution_unavailable",
            "Database is unavailable; cannot enqueue run",
        )

    try:
        record = store.create(task_ids=task_ids, agent_model=agent_model)
    except Exception as exc:  # noqa: BLE001
        return _error(503, "execution_unavailable", f"Failed to enqueue run: {exc}")

    return CreateRunResponse(
        run_id=record.run_id,
        status=RunStatus.queued,
        created_at=record.created_at,
    )


@router.get(
    "/{run_id}",
    response_model=RunResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_run(run_id: str, request: Request) -> RunResponse | JSONResponse:
    store = _get_store(request)
    record = store.get(run_id)
    if record is None:
        return _error(404, "run_not_found", f"No run found with id {run_id}")
    return record.to_response()
