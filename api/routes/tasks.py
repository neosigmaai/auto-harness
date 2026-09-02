"""Task discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.config import load_config
from api.schemas import TaskListResponse

router = APIRouter(tags=["tasks"])


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    cfg = load_config()
    return TaskListResponse(
        default_task_ids=list(cfg.default_task_ids),
        default_agent_model=cfg.default_agent_model,
    )
