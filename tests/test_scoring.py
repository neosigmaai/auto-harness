"""Pure unit tests for scoring and the stopping rule (no DB, no config)."""

from __future__ import annotations

import pytest

from api.services.scoring import (
    STOP_BUDGET_EXCEEDED,
    STOP_MAX_ITERATIONS,
    STOP_NO_IMPROVEMENT,
    StopDecision,
    TaskMovement,
    compute_stop,
    mean_reward,
    task_movement,
)


def test_mean_reward_empty_is_zero() -> None:
    assert mean_reward([]) == 0.0


def test_mean_reward_all_none_is_zero() -> None:
    assert mean_reward([None, None]) == 0.0


def test_mean_reward_counts_none_as_zero() -> None:
    # 1.0 + 0.0 + 0.5 + 0.0 over 4 tasks
    assert mean_reward([1.0, None, 0.5, 0.0]) == pytest.approx(0.375)


def test_mean_reward_all_passing() -> None:
    assert mean_reward([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_mean_reward_accepts_ints_and_generators() -> None:
    assert mean_reward(r for r in [1, 0, None]) == pytest.approx(1 / 3)


BASE = {
    "iteration": 0,
    "score": 0.5,
    "best_score": None,
    "prior_non_improving_streak": 0,
    "max_iterations": 5,
    "patience": 2,
    "min_iterations": 3,
    "min_delta": 0.01,
    "elapsed_sec": 10.0,
    "max_job_duration_sec": 21600,
}


@pytest.mark.parametrize(
    "name,overrides,expected",
    [
        (
            "first_iteration_always_improves",
            {"iteration": 0, "score": 0.0, "best_score": None},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "improvement_resets_streak",
            {"iteration": 2, "score": 0.60, "best_score": 0.50,
             "prior_non_improving_streak": 1},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "non_improvement_increments_streak",
            {"iteration": 1, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 0},
            StopDecision(
                improved=False, should_stop=False, stop_reason=None, non_improving_streak=1
            ),
        ),
        (
            "patience_reached_stops_with_no_improvement",
            {"iteration": 2, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 1, "patience": 2},
            StopDecision(
                improved=False,
                should_stop=True,
                stop_reason="no_improvement",
                non_improving_streak=2,
            ),
        ),
        (
            "max_iterations_wins_over_no_improvement",
            {"iteration": 4, "max_iterations": 5, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 1, "patience": 2},
            StopDecision(
                improved=False,
                should_stop=True,
                stop_reason="max_iterations",
                non_improving_streak=2,
            ),
        ),
        (
            "budget_exceeded_when_no_other_rule_fires",
            {"iteration": 1, "score": 0.70, "best_score": 0.50,
             "elapsed_sec": 100.0, "max_job_duration_sec": 60},
            StopDecision(
                improved=True,
                should_stop=True,
                stop_reason="budget_exceeded",
                non_improving_streak=0,
            ),
        ),
        (
            "budget_not_exceeded_at_exact_limit",
            {"iteration": 1, "score": 0.70, "best_score": 0.50,
             "elapsed_sec": 60.0, "max_job_duration_sec": 60},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "min_delta_boundary_is_not_an_improvement",
            # 0.5 + 0.01 == 0.51 exactly in IEEE754, and 0.51 > 0.51 is False.
            {"iteration": 1, "score": 0.51, "best_score": 0.50, "min_delta": 0.01},
            StopDecision(
                improved=False, should_stop=False, stop_reason=None, non_improving_streak=1
            ),
        ),
        (
            "just_past_min_delta_is_an_improvement",
            {"iteration": 1, "score": 0.52, "best_score": 0.50, "min_delta": 0.01},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "single_iteration_job_stops_immediately",
            {"iteration": 0, "max_iterations": 1, "score": 0.3, "best_score": None},
            StopDecision(
                improved=True,
                should_stop=True,
                stop_reason="max_iterations",
                non_improving_streak=0,
            ),
        ),
        (
            "patience_reached_but_below_min_iterations_keeps_going",
            {"iteration": 0, "score": 0.5, "best_score": 0.5,
             "prior_non_improving_streak": 0, "patience": 1,
             "min_iterations": 3, "max_iterations": 9},
            StopDecision(improved=False, should_stop=False,
                         stop_reason=None, non_improving_streak=1),
        ),
        (
            "patience_reached_at_min_iterations_stops",
            {"iteration": 2, "score": 0.5, "best_score": 0.5,
             "prior_non_improving_streak": 0, "patience": 1,
             "min_iterations": 3, "max_iterations": 9},
            StopDecision(improved=False, should_stop=True,
                         stop_reason=STOP_NO_IMPROVEMENT, non_improving_streak=1),
        ),
        (
            "min_iterations_never_overrides_max_iterations",
            {"iteration": 1, "score": 0.5, "best_score": 0.5,
             "prior_non_improving_streak": 5, "patience": 1,
             "min_iterations": 99, "max_iterations": 2},
            StopDecision(improved=False, should_stop=True,
                         stop_reason=STOP_MAX_ITERATIONS, non_improving_streak=6),
        ),
        (
            "min_iterations_never_overrides_budget",
            {"iteration": 0, "score": 0.5, "best_score": 0.5,
             "prior_non_improving_streak": 0, "patience": 9,
             "min_iterations": 99, "max_iterations": 9,
             "elapsed_sec": 10_000.0, "max_job_duration_sec": 100},
            StopDecision(improved=False, should_stop=True,
                         stop_reason=STOP_BUDGET_EXCEEDED, non_improving_streak=1),
        ),
    ],
)
def test_compute_stop_table(name: str, overrides: dict, expected: StopDecision) -> None:
    kwargs = {**BASE, **overrides}
    assert compute_stop(**kwargs) == expected, name


def test_stop_decision_is_frozen() -> None:
    decision = compute_stop(**BASE)
    with pytest.raises(Exception):
        decision.improved = False  # type: ignore[misc]


def test_task_movement_detects_fixed_and_regressed() -> None:
    move = task_movement(
        {"a": 0.0, "b": 1.0, "c": 0.5},
        {"a": 1.0, "b": 0.0, "c": 0.5},
    )
    assert move == TaskMovement(fixed=["a"], regressed=["b"])


def test_task_movement_treats_none_as_zero() -> None:
    # a passing task that stopped producing a verifier result is a regression
    assert task_movement({"a": 1.0}, {"a": None}) == TaskMovement(
        fixed=[], regressed=["a"]
    )
    assert task_movement({"a": None}, {"a": 1.0}) == TaskMovement(
        fixed=["a"], regressed=[]
    )


def test_task_movement_ignores_non_overlapping_tasks() -> None:
    assert task_movement({"a": 0.0}, {"b": 1.0}) == TaskMovement(
        fixed=[], regressed=[]
    )


def test_task_movement_empty_or_missing_snapshots() -> None:
    assert task_movement(None, {"a": 1.0}) == TaskMovement(fixed=[], regressed=[])
    assert task_movement({"a": 1.0}, None) == TaskMovement(fixed=[], regressed=[])
    assert task_movement({}, {}) == TaskMovement(fixed=[], regressed=[])


def test_flat_mean_redistribution_is_visible_in_movement() -> None:
    """The exact case the designers flagged: same mean, different distribution."""
    before = {"a": 1.0, "b": 0.0}
    after = {"a": 0.0, "b": 1.0}
    assert mean_reward(before.values()) == mean_reward(after.values())  # 0.5 both
    move = task_movement(before, after)
    assert move.fixed == ["b"] and move.regressed == ["a"]
