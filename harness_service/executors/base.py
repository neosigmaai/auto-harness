"""The Executor protocol — the agent-execution isolation boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness_service.constants import ExecutorKind
from harness_service.domain import AgentState, BenchmarkResult


@runtime_checkable
class Executor(Protocol):
    """Runs an agent against a set of tasks and returns per-task outcomes.

    Implementations MUST NOT execute agent code in the caller's process — that is
    the whole point of the boundary. ``SimulatedExecutor`` returns dummy data;
    ``HarborExecutor`` shells out to ``harbor`` which runs the agent in a sandbox.
    """

    kind: ExecutorKind

    async def run(self, agent: AgentState, task_ids: list[str]) -> BenchmarkResult:
        ...
