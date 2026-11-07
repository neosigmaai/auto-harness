"""API request/response schemas (Pydantic). These are the contract test_client.py relies on."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from .tasks_catalog import ALL_TASK_NAMES, BY_NAME, DEFAULT_PROFILE, PROFILES

JobStatusT = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class Budget(BaseModel):
    max_wall_clock_min: int = Field(default=120, ge=1, le=24 * 60)
    max_llm_usd: float = Field(default=10.0, ge=0, allow_inf_nan=False)


class JobOptions(BaseModel):
    two_phase_eval: bool = Field(default=True, description="Screen quality proposals on feedback tasks; screened candidates cannot be promoted.")
    confirm_flips: bool = Field(default=True, description="Confirm promising candidates and incumbent on the complete selection set; retained for client compatibility.")
    min_improvement: float = Field(default=0.0, ge=0, le=1, description="Accept only if score - best > this.")
    confirmation_repetitions: int = Field(default=2, ge=1, le=5)
    efficiency_tolerance: float = Field(default=0.10, ge=0, le=1,
        description="Predeclared maximum relative runtime/model-cost increase during a quality improvement.")
    min_efficiency_improvement: float = Field(default=0.10, gt=0, le=1)
    max_agent_steps: int = Field(default=80, ge=1, le=100)
    max_tokens_per_call: int = Field(default=4096, ge=128, le=16384)
    agent_timeout_sec: int = Field(default=900, ge=10, le=1800)
    max_infrastructure_attempts: int = Field(default=2, ge=1, le=3,
        description="Total attempts for an infrastructure failure; failed verifier outcomes are never retried.")


class EvaluationSplit(BaseModel):
    feedback: list[str]
    selection: list[str]
    holdout: list[str]

    @model_validator(mode="after")
    def disjoint_nonempty(self):
        parts = (self.feedback, self.selection, self.holdout)
        flat = [task for part in parts for task in part]
        if any(not part for part in parts) or len(flat) != len(set(flat)):
            raise ValueError("feedback, selection and holdout must be nonempty, disjoint task lists")
        return self


class JobCreate(BaseModel):
    benchmark: Literal["terminal-bench-2"] = "terminal-bench-2"
    profile: Literal["calibrated12", "stress16"] | None = Field(default=None,
        description="Named frozen task/split profile; omitted with no custom tasks resolves to calibrated12. Mutually exclusive with tasks.")
    tasks: list[str] | None = Field(default=None, description="Custom task names; omit with profile, or omit both for calibrated12.")
    max_iterations: int = Field(default=5, ge=0, le=50, description="0 = baseline only (plain benchmark run).")
    patience: int = Field(default=3, ge=1, le=20)
    agent_model: str | None = None
    optimizer_model: str | None = None
    agent_source: str | None = Field(default=None, description="Optional agent.py override; default = repo template.")
    seed: int | None = None
    budget: Budget = Field(default_factory=Budget)
    options: JobOptions = Field(default_factory=JobOptions)
    evaluation_split: EvaluationSplit | None = None
    prior_job_id: UUID | None = Field(default=None,
        description="Authorized prior experiment with the identical frozen task/model/baseline plan; imports completed rejections only.")
    resume_candidate_index: int | None = Field(default=None, ge=1, le=50,
        description="Re-evaluate this prior unconfirmed candidate using fresh evidence, without another optimizer call.")

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
        return list(self.tasks) if self.tasks is not None else list(PROFILES[self.profile or DEFAULT_PROFILE]["tasks"])

    @model_validator(mode="after")
    def freeze_split(self):
        if self.tasks is not None and self.profile is not None:
            raise ValueError("profile and custom tasks are mutually exclusive")
        if self.tasks is None:
            self.profile = self.profile or DEFAULT_PROFILE
        named_split = PROFILES[self.profile]["evaluation_split"] if self.profile else None
        if named_split and self.evaluation_split is not None and self.evaluation_split.model_dump() != named_split:
            raise ValueError("evaluation_split must match the named profile; use custom tasks for a custom split")
        if self.resume_candidate_index is not None and self.prior_job_id is None:
            raise ValueError("resume_candidate_index requires prior_job_id")
        if self.resume_candidate_index is not None and self.max_iterations == 0:
            raise ValueError("resuming a candidate requires max_iterations greater than zero")
        tasks = self.resolved_tasks()
        if self.max_iterations == 0:
            return self
        if len(tasks) < 3:
            raise ValueError("optimization needs at least three tasks for feedback, selection and holdout; use max_iterations=0 for a small benchmark")
        if named_split is not None:
            self.evaluation_split = EvaluationSplit(**{name: list(tasks) for name, tasks in named_split.items()})
        elif self.evaluation_split is None:
            n_feedback = max(1, len(tasks) // 2)
            n_selection = max(1, len(tasks) // 4)
            capacity = {"feedback": n_feedback, "selection": n_selection,
                        "holdout": len(tasks) - n_feedback - n_selection}
            groups = {name: [] for name in capacity}
            # Spread easy/hard proxy ranks across all parts, instead of slicing
            # the catalog (which puts canaries first). This is a transparent
            # sampling heuristic, not a representativeness guarantee.
            ranked = sorted(tasks, key=lambda task: (-(BY_NAME[task].get("leaderboard_solve_rate") or 0), task))
            cycle = ("feedback", "selection", "holdout", "feedback")
            position = 0
            for task in ranked:
                while len(groups[cycle[position % len(cycle)]]) >= capacity[cycle[position % len(cycle)]]:
                    position += 1
                groups[cycle[position % len(cycle)]].append(task)
                position += 1
            self.evaluation_split = EvaluationSplit(**groups)
        split = self.evaluation_split
        if set(split.feedback + split.selection + split.holdout) != set(tasks):
            raise ValueError("evaluation_split must partition exactly the selected tasks")
        return self


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
    input_tokens: int = 0
    output_tokens: int = 0
    model_cost_usd: float | None = None
    sandbox_cost_usd: float | None = None
    cost_usd: float | None = None
    cost_known: bool = False


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
    metrics: dict[str, Any] = Field(default_factory=dict)


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
    deadline_at: datetime | None = None
    final_evaluation: dict[str, Any] | None = None
    spent_usd: float = 0.0
    reserved_usd: float = 0.0


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
    profiles: list[str] = Field(default_factory=list)
    evaluation_roles: dict[str, str] = Field(default_factory=dict)


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
