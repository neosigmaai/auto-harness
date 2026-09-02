"""Scoring and stopping rules for the iterative improvement loop.

Deliberately dependency-free (stdlib only) so it can be unit-tested without a
database, config file or LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

STOP_MAX_ITERATIONS = "max_iterations"
STOP_NO_IMPROVEMENT = "no_improvement"
STOP_BUDGET_EXCEEDED = "budget_exceeded"


def mean_reward(rewards: Iterable[float | None]) -> float:
    """Mean reward across a run's tasks; ``None`` (error/timeout) counts as 0.0.

    An empty iterable scores 0.0 rather than raising, so a run that produced no
    task rows is simply the worst possible score.
    """
    values = [0.0 if reward is None else float(reward) for reward in rewards]
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass(frozen=True)
class StopDecision:
    """Outcome of the stopping check performed after an evaluate step."""

    improved: bool
    should_stop: bool
    stop_reason: str | None
    non_improving_streak: int


def compute_stop(
    *,
    iteration: int,
    score: float,
    best_score: float | None,
    prior_non_improving_streak: int,
    max_iterations: int,
    patience: int,
    min_iterations: int,
    min_delta: float,
    elapsed_sec: float,
    max_job_duration_sec: int,
) -> StopDecision:
    """Decide whether the job improved and whether the loop should stop.

    ``improved`` requires a strict gain of more than ``min_delta`` over the best
    score so far; the first evaluation (``best_score is None``) always improves.

    Stop precedence, first match wins:
      1. ``max_iterations``  — this was the last allowed iteration.
      2. ``no_improvement``  — the non-improving streak reached ``patience``,
         AND at least ``min_iterations`` iterations have completed. The agent is
         stochastic, so an early non-improving run may be variance rather than a
         real plateau; ``min_iterations`` is the noise floor that stops the loop
         from giving up on it. It never overrides ``max_iterations`` or the
         wall-clock budget — a cost ceiling always beats a "keep trying" floor.
      3. ``budget_exceeded`` — wall-clock since job start passed the budget.
    """
    improved = best_score is None or score > best_score + min_delta
    streak = 0 if improved else prior_non_improving_streak + 1

    stop_reason: str | None = None
    if iteration + 1 >= max_iterations:
        stop_reason = STOP_MAX_ITERATIONS
    elif streak >= patience and iteration + 1 >= min_iterations:
        stop_reason = STOP_NO_IMPROVEMENT
    elif elapsed_sec > max_job_duration_sec:
        stop_reason = STOP_BUDGET_EXCEEDED

    return StopDecision(
        improved=improved,
        should_stop=stop_reason is not None,
        stop_reason=stop_reason,
        non_improving_streak=streak,
    )


@dataclass(frozen=True)
class TaskMovement:
    """Per-task deltas between two reward snapshots."""

    fixed: list[str]
    regressed: list[str]


def task_movement(
    baseline: dict[str, float | None] | None,
    current: dict[str, float | None] | None,
) -> TaskMovement:
    """
    Compare two ``{task_id: reward}`` snapshots.

    A ``None`` reward (timeout / infra error) counts as 0.0, matching
    :func:`mean_reward`, so an task that stopped producing a verifier result reads
    as a regression rather than as missing data. Tasks absent from either snapshot
    are ignored: they carry no comparable signal.
    """
    if not baseline or not current:
        return TaskMovement(fixed=[], regressed=[])

    fixed: list[str] = []
    regressed: list[str] = []
    for task_id in sorted(set(baseline) & set(current)):
        before = baseline[task_id] or 0.0
        after = current[task_id] or 0.0
        if after > before:
            fixed.append(task_id)
        elif after < before:
            regressed.append(task_id)
    return TaskMovement(fixed=fixed, regressed=regressed)
