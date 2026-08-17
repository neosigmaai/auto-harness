"""Executor factory — resolves an ExecutorKind to a concrete implementation."""

from __future__ import annotations

from harness_service.config import Settings
from harness_service.constants import ExecutorKind
from harness_service.executors.base import Executor
from harness_service.executors.simulated import SimulatedExecutor


def get_executor(kind: ExecutorKind, settings: Settings) -> Executor:
    if kind == ExecutorKind.SIMULATED:
        return SimulatedExecutor()
    if kind == ExecutorKind.HARBOR:
        # Imported lazily so the simulated path has zero harbor/e2b dependencies.
        try:
            from harness_service.executors.harbor import HarborExecutor
        except ImportError as exc:  # pragma: no cover - lands in M3
            raise NotImplementedError(
                "HarborExecutor not available yet (M3). Use executor='simulated'."
            ) from exc
        return HarborExecutor(settings=settings)
    raise ValueError(f"Unknown executor kind: {kind}")
