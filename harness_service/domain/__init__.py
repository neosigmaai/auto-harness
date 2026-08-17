"""Pure in-memory domain model of an optimization run.

No SQLAlchemy, no FastAPI — just the vocabulary the worker and the optimizer
reason over. Persistence (db/) is "this model frozen to disk"; the API layer
serializes it to JSON. See PLAN.md §3a.
"""

from harness_service.domain.models import (
    AgentState,
    BenchmarkResult,
    Improvement,
    Iteration,
    TaskOutcome,
    Trajectory,
)

__all__ = [
    "AgentState",
    "TaskOutcome",
    "BenchmarkResult",
    "Improvement",
    "Iteration",
    "Trajectory",
]
