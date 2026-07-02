from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any, Iterable

from autoharness_service.models import FailureSummary, TaskResultRecord


def normalize_reward_result(
    task_id: str,
    reward: float,
    trace_path: str | None = None,
    result_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskResultRecord:
    if not isfinite(reward):
        return TaskResultRecord(
            task_id=task_id,
            status="infra_failed",
            reward=None,
            failure_type="invalid_result",
            error_summary="Reward must be a finite number",
            trace_path=trace_path,
            result_path=result_path,
            metadata={} if metadata is None else dict(metadata),
        )

    if reward > 0:
        return TaskResultRecord(
            task_id=task_id,
            status="passed",
            reward=reward,
            trace_path=trace_path,
            result_path=result_path,
            metadata={} if metadata is None else dict(metadata),
        )

    return TaskResultRecord(
        task_id=task_id,
        status="failed",
        reward=reward,
        failure_type="agent_failed",
        error_summary="Verifier reward below pass threshold",
        trace_path=trace_path,
        result_path=result_path,
        metadata={} if metadata is None else dict(metadata),
    )


def normalize_missing_result(
    task_id: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> TaskResultRecord:
    return TaskResultRecord(
        task_id=task_id,
        status="infra_failed",
        reward=None,
        failure_type="missing_result",
        error_summary=reason,
        metadata={} if metadata is None else dict(metadata),
    )


def build_failure_summary(task_results: Iterable[TaskResultRecord]) -> FailureSummary:
    results = list(task_results)
    failure_counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}

    tasks_passed = 0
    tasks_failed = 0
    tasks_infra_failed = 0
    agent_failures = 0
    infra_failures = 0

    for index, result in enumerate(results):
        if result.status == "passed":
            tasks_passed += 1
        elif result.status == "failed":
            tasks_failed += 1
        elif result.status == "infra_failed":
            tasks_infra_failed += 1

        if result.failure_type:
            failure_counts[result.failure_type] += 1
            first_seen.setdefault(result.failure_type, index)
            if result.failure_type == "agent_failed":
                agent_failures += 1
            else:
                infra_failures += 1

    top_failure_modes = [
        failure_type
        for failure_type, _ in sorted(
            failure_counts.items(),
            key=lambda item: (-item[1], first_seen[item[0]]),
        )
    ]

    return FailureSummary(
        tasks_total=len(results),
        tasks_passed=tasks_passed,
        tasks_failed=tasks_failed,
        tasks_infra_failed=tasks_infra_failed,
        agent_failures=agent_failures,
        infra_failures=infra_failures,
        top_failure_modes=top_failure_modes,
    )
