from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

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
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        for task_id in self.task_ids:
            if not TASK_ID_RE.fullmatch(task_id):
                raise ValueError(
                    "task_ids must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
                )
        if self.model not in ALLOWED_MODELS:
            raise ValueError(
                f"model must be one of: {', '.join(sorted(ALLOWED_MODELS))}"
            )
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


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    reward: float | None
    failure_type: str | None
    error_summary: str | None
    trace_path: str | None
    result_path: str | None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: RunProgress
    score: float | None
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    task_results: list[TaskResultResponse] = Field(default_factory=list)


class FailureSummaryResponse(BaseModel):
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    agent_failures: int
    infra_failures: int
    top_failure_modes: list[str]


class RunResultsResponse(BaseModel):
    run_id: str
    status: str
    score: float | None
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    task_results: list[TaskResultResponse]
    failure_summary: FailureSummaryResponse


class IterationResponse(BaseModel):
    iteration: int
    agent_version: str
    status: str
    score: float | None
    proposal: str | None
    accepted: bool | None


class IterationsResponse(BaseModel):
    run_id: str
    iterations: list[IterationResponse]


class TaskListResponse(BaseModel):
    tasks: list[str]
