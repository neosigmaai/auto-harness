from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
ALLOWED_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-4o", "gpt-4o-mini"}


class RunCreateRequest(BaseModel):
    task_ids: list[str] = Field(min_length=1, max_length=20)
    max_iterations: int = Field(default=0, ge=0, le=1)
    sandbox_provider: Literal["daytona", "simulated"] = "simulated"
    model: str = "gpt-5.4"
    mode: Literal["simulated", "real"] = "simulated"
    requested_concurrency: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def validate_request(self) -> "RunCreateRequest":
        for task_id in self.task_ids:
            if not TASK_ID_RE.fullmatch(task_id):
                raise ValueError("task_ids must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of: {', '.join(sorted(ALLOWED_MODELS))}")
        if self.mode == "simulated" and self.sandbox_provider != "simulated":
            raise ValueError("simulated mode requires sandbox_provider=simulated")
        if self.mode == "real" and self.sandbox_provider != "daytona":
            raise ValueError("real mode requires sandbox_provider=daytona")
        return self


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime | None
    status_url: str
    result_url: str


class RunProgress(BaseModel):
    total: int
    queued: int
    running: int
    completed: int
