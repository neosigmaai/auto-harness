"""Pluggable agent execution.

The ``Executor`` protocol is the core isolation boundary (PLAN.md §3): the agent
never runs in the API or worker process. ``SimulatedExecutor`` fakes it for M1;
``HarborExecutor`` (M3) runs the real agent in an E2B sandbox.
"""

from harness_service.executors.base import Executor
from harness_service.executors.registry import get_executor
from harness_service.executors.simulated import SimulatedExecutor

__all__ = ["Executor", "SimulatedExecutor", "get_executor"]
