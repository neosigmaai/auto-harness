from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import Principal, get_principal
from ..models import TaskCatalogEntry
from ..tasks_catalog import TASKS

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.get("", response_model=list[TaskCatalogEntry], summary="Curated task catalogue")
async def list_tasks(_: Principal = Depends(get_principal)):
    return [TaskCatalogEntry(**t) for t in TASKS]
