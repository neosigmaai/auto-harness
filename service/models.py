"""API request/response schemas (Pydantic). These are the contract test_client.py relies on."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .tasks_catalog import ALL_TASK_NAMES, DEFAULT_TASKS

JobStatusT = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class Budget(BaseModel):
    max_wall_clock_min: int = Field(default=120, ge=1, le=24 * 60)
    max_llm_usd: float = Field(default=10.0, ge=0)


class JobOptions(BaseModel):
    two_phase_eval: bool = Field(default=True, description="Run failing tasks first; skip the rest if nothing improved.")
    confirm_flips: bool = Field(default=True, description="Re-run tasks whose outcome flipped before accepting.")
    min_improvement: float = Field(default=0.0, ge=0, le=1, description="Accept only if score - best > this.")


class JobCreate(BaseModel):
    benchmark: Literal["terminal-bench-2"] = "terminal-bench-2"
    tasks: list[str] | None = Field(default=None, description="Task names; null = curated default subset.")
    max_iterations: int = Field(default=5, ge=0, le=50, description="0 = baseline only (plain benchmark run).")
    patience: int = Field(default=3, ge=1, le=20)
    agent_model: str | None = None
    optimizer_model: str | None = None
    agent_source: str | None = Field(default=None, description="Optional agent.py override; default = repo template.")
    seed: int | None = None
    budget: Budget = Field(default_factory=Budget)
    options: JobOptions = Field(default_factory=JobOptions)

    @field_validator("tasks")
    @classmethod
    def _known_tasks(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        if not v:
            raise ValueError("tasks must be null or a non-empty list")
        unknown = sorted(set(v) - ALL_TASK_NAMES)
        if unknown:
            raise ValueError(f"unknown task(s): {unknown}; see GET /v1/tasks")
        if len(v) != len(set(v)):
            raise ValueError("tasks contains duplicates")
        return v

    def resolved_tasks(self) -> list[str]:
        return list(self.tasks) if self.tasks else list(DEFAULT_TASKS)


class JobAccepted(BaseModel):
    job_id: UUID
    status: JobStatusT
    status_url: str


class TaskRunOut(BaseModel):
    task_name: str
    status: Literal["pending", "running", "passed", "failed", "timeout", "error"]
    reward: float | None = None
    failure_class: str | None = None
    duration_s: float | None = None
    n_steps: int | None = None
    exit_reason: str | None = None


class TaskRunDetail(TaskRunOut):
    test_stdout: str | None = None
    digest: dict[str, Any] | None = None
    trace: Any | None = None


class ResultSummary(BaseModel):
    n_tasks: int
    passed: int
    failed: int
    errored: int
    score: float | None


class IterationSummary(BaseModel):
    index: int
    phase: str
    status: Literal["running", "completed", "errored"]
    decision: str | None
    decision_reason: str | None = None
    score: float | None
    summary: ResultSummary | None = None
    hypothesis: str | None = None
    agent_version_id: UUID
    agent_sha256: str
    started_at: datetime
    finished_at: datetime | None


class IterationDetail(IterationSummary):
    proposal: dict[str, Any] | None = None
    learnings: str | None = None
    agent_source: str
    diff_vs_parent: str | None = None
    tasks: list[TaskRunOut]
    failure_summary: list[dict[str, Any]]


class JobOut(BaseModel):
    job_id: UUID
    status: JobStatusT
    created_by: UUID
    config: dict[str, Any]
    current_iteration: int | None
    baseline_score: float | None
    best_score: float | None
    best_iteration_index: int | None = None
    stop_reason: str | None
    error: str | None
    cancel_requested: bool
    progress: dict[str, Any] | None = None   # {"iteration": n, "phase": ..., "tasks_done": k, "tasks_total": n}
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobList(BaseModel):
    items: list[JobOut]
    next_cursor: str | None = None


class AgentOut(BaseModel):
    job_id: UUID
    best_iteration_index: int | None
    baseline_sha256: str
    best_sha256: str | None
    source: str
    diff_vs_baseline: str | None


class EventOut(BaseModel):
    id: int
    ts: datetime
    kind: str
    payload: dict[str, Any]


class TaskCatalogEntry(BaseModel):
    name: str
    role: str
    category: str
    difficulty: str
    agent_timeout_sec: int
    verifier_timeout_sec: int
    leaderboard_solve_rate: float | None
    why: str


# ── admin ──────────────────────────────────────────────────────────────────────
class OrgCreate(BaseModel):
    name: str = Field(min_length=2, max_length=64)
    admin_email: str
    max_concurrent_jobs: int = Field(default=2, ge=1, le=50)


class UserCreate(BaseModel):
    email: str
    role: Literal["admin", "member"] = "member"


class CredentialOut(BaseModel):
    org_id: UUID
    org_name: str
    user_id: UUID
    email: str
    role: str
    api_key: str = Field(description="Shown once. Store it; only its hash is persisted.")


class MeOut(BaseModel):
    org_id: UUID
    org_name: str
    user_id: UUID
    email: str
    role: str
