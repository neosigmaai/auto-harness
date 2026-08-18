"""Liveness + readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from harness_service.api.schemas import HealthResponse
from harness_service.config import get_settings
from harness_service.db import get_session

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse,
            summary="Liveness + DB/worker readiness")
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # surface DB connectivity plainly
        db_status = f"error: {exc}"
    return HealthResponse(db=db_status, worker_enabled=get_settings().worker_enabled)
