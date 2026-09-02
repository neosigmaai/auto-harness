"""Agent version lookup endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.job_store import PostgresJobStore
from api.routes.jobs import spec_view
from api.schemas import AgentVersionResponse, ErrorDetail, ErrorResponse

router = APIRouter(prefix="/v1/agent-versions", tags=["agent-versions"])


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_job_store(request: Request) -> PostgresJobStore:
    return request.app.state.job_store


@router.get(
    "/{version_id}",
    response_model=AgentVersionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_agent_version(
    version_id: str, request: Request
) -> AgentVersionResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_agent_version(version_id)
    if record is None:
        # get_agent_version also returns None for a malformed UUID, mirroring
        # PostgresRunStore.get — a bad id is "not found", never a 500.
        return _error(
            404,
            "agent_version_not_found",
            f"No agent version found with id {version_id}",
        )

    return AgentVersionResponse(
        agent_version_id=record.version_id,
        job_id=record.job_id,
        version=record.version,
        parent_version_id=record.parent_version_id,
        rationale=record.rationale,
        created_by=record.created_by,
        created_at=record.created_at,
        spec=spec_view(record.spec),
    )
