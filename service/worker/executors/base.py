"""Executor contract: run an agent version against tasks, report one result per task."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID


@dataclass
class RunContext:
    job_id: UUID
    iteration_index: int
    iteration_id: UUID
    org_id: UUID
    agent_model: str
    seed: int
    attempt: int = 1                 # bumped for confirmation re-runs
    work_dir: Path | None = None     # runs/<job>/<iter>/ for executors that need disk
    extra: dict[str, Any] = field(default_factory=dict)
    lease_owner: str = ""
    lease_epoch: int = 0
    deadline_at: datetime | None = None
    evaluation_phase: str = "main"
    repetition: int = 1
    lease: Any = None


@dataclass
class TaskResult:
    task_name: str
    status: str                      # passed | failed | timeout | error
    reward: float | None
    failure_class: str | None = None
    duration_s: float | None = None
    n_steps: int | None = None
    exit_reason: str | None = None
    test_stdout: str | None = None
    trace: Any = None                # list of messages (OpenAI-style) or None
    sandbox_id: str | None = None
    instruction: str | None = None   # task instruction text when the executor knows it
    input_tokens: int = 0
    output_tokens: int = 0
    model_cost_usd: float | None = None
    sandbox_cost_usd: float | None = None
    cost_usd: float | None = None
    cost_known: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


OnResult = Callable[[TaskResult], Awaitable[None]]


class Executor(Protocol):
    name: str

    async def run_benchmark(self, ctx: RunContext, agent_source: str, tasks: list[str],
                            on_result: OnResult, on_start: Callable[[str], Awaitable[None]] | None = None) -> dict[str, TaskResult]:
        """Run every task (in parallel where possible). Must call on_result as each task finishes
        so progress is persisted incrementally. Returns results keyed by task name."""
        ...

    async def cancel(self) -> None:
        """Best-effort: stop any in-flight work (kill runner, destroy sandboxes)."""
        ...
