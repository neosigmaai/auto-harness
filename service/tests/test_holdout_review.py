"""Final holdout protocol checks. No provider or Docker calls are made."""
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace

import pytest

from scripts.run_final_holdout_review import (
    DIAGNOSTIC_USD, HOLDOUT_TASKS, acquire_controller, aggregate_holdout, compare_holdout,
    freeze_plan, redact_attempt, remaining_budget, run, validate_arguments,
)
from service.config import settings
from service import repo
from service.repo import sha256_text
from service.tests.test_persistence import pool, principal, submit, claim, lease, iteration


def args():
    return SimpleNamespace(allow_paid=True, max_total_usd=3, max_wall_clock_min=90,
        selected_source="baseline", selected_sha256="a" * 64)


def frozen_inputs(qualifies=True):
    sources = {"baseline": "# original source", "candidate": "# proposed source"}
    cfg = {"benchmark_revision": "pinned", "agent_model": "fixed-model",
        "options": {"max_tokens_per_call": 512, "max_infrastructure_attempts": 1},
        "evaluation_split": {"selection": ["a", "b", "c", "d"], "holdout": list(HOLDOUT_TASKS)}}
    proof = {"files": {"service/example.py": "hash"}}
    report = {"status": "COMPLETE", "holdout_executed": False, "source_job_id": "job-id",
        "implementation_sha256": proof, "benchmark_revision": cfg["benchmark_revision"],
        "agent_model": cfg["agent_model"], "tasks": cfg["evaluation_split"]["selection"],
        "options": {**cfg["options"], "max_infrastructure_attempts": 2},
        "source_hashes": {key: sha256_text(value) for key, value in sources.items()},
        "metrics": {key: {"complete": True} for key in sources},
        "observed_policy_decision": {"candidate_qualifies": qualifies},
        "effective_inference_limits": {"max_output_tokens": min(512, settings.agent_max_output_tokens),
            "max_request_bytes": settings.max_inference_body_bytes,
            "max_model_calls_per_trial": settings.max_model_calls_per_trial},
        "fixed_reasoning_policy": {"tool_bearing_agent_requests": "none", "optimizer_calls": 0}}
    label = "candidate" if qualifies else "baseline"
    return [report, cfg, "job-id", sources, proof, label, sha256_text(sources[label])]


@pytest.mark.asyncio
async def test_paid_execution_requires_explicit_gate_before_any_file_or_database_access():
    with pytest.raises(SystemExit, match="No model calls"):
        await run(SimpleNamespace(allow_paid=False))


@pytest.mark.parametrize("amount", [0, -1, 3.000001, 6, float("inf"), float("nan")])
def test_holdout_has_strict_three_dollar_ceiling(amount):
    value = args()
    value.max_total_usd = amount
    with pytest.raises(SystemExit, match="at most"):
        validate_arguments(value)


def test_holdout_uses_aggregate_and_cumulative_budget_with_outstanding_reservations():
    assert remaining_budget("8.5", "9", ".25", 3) == Decimal("2.250000")
    expected = (Decimal(".25") - DIAGNOSTIC_USD).quantize(Decimal(".000001"), rounding=ROUND_DOWN)
    assert remaining_budget("12", "14.5", ".25", 3) == expected
    assert remaining_budget("8", "11", ".01", 3) == 0


def test_final_baseline_choice_reuses_one_four_task_run_without_claiming_improvement():
    values = frozen_inputs(False)
    plan = freeze_plan(*values)
    assert plan["planned_order"] == ["baseline"]
    assert plan["same_source"] is True
    assert plan["comparison_status"] == "same_source_no_improvement"
    assert plan["tasks"] == list(HOLDOUT_TASKS)
    assert plan["options"]["max_infrastructure_attempts"] == 2
    assert values[1]["options"]["max_infrastructure_attempts"] == 1


def test_selected_candidate_is_frozen_before_baseline_first_holdout():
    values = frozen_inputs(True)
    plan = freeze_plan(*values)
    assert plan["planned_order"] == ["baseline", "selected"]
    assert plan["selected_sha256"] == values[-1]
    assert not plan["same_source"]


@pytest.mark.parametrize("mutation", [
    lambda v: v[0].update(status="RUNNING"),
    lambda v: v[0].update(holdout_executed=True),
    lambda v: v[0].update(source_job_id="other-job"),
    lambda v: v[0].update(implementation_sha256={"different": True}),
    lambda v: v[0].update(agent_model="other-model"),
    lambda v: v[0].update(tasks=["other-task"]),
    lambda v: v[0]["options"].update(max_infrastructure_attempts=3),
    lambda v: v[0]["effective_inference_limits"].update(max_output_tokens=123),
    lambda v: v[0]["fixed_reasoning_policy"].update(tool_bearing_agent_requests="low"),
    lambda v: v[1]["evaluation_split"].update(holdout=["a", "b", "c", "d"]),
    lambda v: v[3].update(baseline="# replacement baseline"),
    lambda v: v[0]["metrics"]["candidate"].update(complete=False),
    lambda v: v.__setitem__(6, "0" * 64),
    lambda v: v.__setitem__(5, "baseline"),
])
def test_selection_freeze_rejects_drift_or_incomplete_evidence(mutation):
    values = frozen_inputs(True)
    mutation(values)
    with pytest.raises(ValueError):
        freeze_plan(*values)


def observation(task="task", attempt=1, status="passed", **changes):
    row = {"task_name": task, "evaluation_phase": "main", "repetition": 1,
        "attempt": attempt, "status": status, "reward": float(status == "passed"),
        "duration_s": 3, "model_cost_usd": .1, "sandbox_cost_usd": None, "cost_usd": None,
        "cost_known": True, "input_tokens": 10, "output_tokens": 2,
        "metadata": {"model_cost_known": True, "verifier_validity": {"valid": True,
            "contract": "pinned-pytest-ctrf-v1", "fresh_session_marker": True,
            "counts": {"tests": 2, "passed": 2 if status == "passed" else 1,
                       "failed": 0 if status == "passed" else 1, "skipped": 0, "pending": 0, "other": 0},
            "pytest_exit_code": 0 if status == "passed" else 1}}}
    row.update(changes)
    return row


def test_first_valid_outcome_used_and_all_retry_resources_charged():
    rows = [observation(attempt=1, status="error", reward=None, model_cost_usd=.2),
            observation(attempt=2, status="failed"), observation(attempt=3, status="passed")]
    result = aggregate_holdout(rows, ["task"])
    assert result["comparison_eligible"]
    assert result["verifiers"]["task"]["chosen_attempt"] == 2
    assert result["metrics"]["mean_reward"] == 0
    assert result["metrics"]["total_model_cost_usd"] == pytest.approx(.4)
    assert result["metrics"]["p95_task_duration_s"] == 9
    assert result["metrics"]["input_tokens"] == 30
    assert result["metrics"]["physical_attempts"] == 3
    assert result["infrastructure_retries"] == 2


def test_missing_verifier_or_incomplete_tasks_never_qualify_for_comparison():
    invalid = observation(metadata={"model_cost_known": True})
    result = aggregate_holdout([invalid], ["task"])
    assert result["metrics"]["complete"]  # Raw task status alone is insufficient.
    assert not result["verifier_complete"]
    assert not result["comparison_eligible"]
    missing = aggregate_holdout([observation()], ["task", "missing"])
    assert missing["metrics"]["mean_reward"] is None
    assert not missing["comparison_eligible"]
    assert compare_holdout(result, missing, False)["quality_delta"] is None


def test_holdout_requires_exact_v2_coverage_but_preserves_historical_v1_reader():
    from service.tests.test_paired_review_protocol import v2_attempt
    row = v2_attempt()
    task = row["task_name"]
    assert aggregate_holdout([row], [task])["comparison_eligible"]
    row["metadata"]["verifier_validity"].pop("selected_cases")
    assert not aggregate_holdout([row], [task])["comparison_eligible"]
    assert aggregate_holdout([observation()], ["task"])["comparison_eligible"]


def test_unknown_retry_usage_cannot_become_zero_cost():
    rows = [observation(attempt=1, status="error", model_cost_usd=None, cost_known=False,
                        metadata={"model_cost_known": False}), observation(attempt=2)]
    result = aggregate_holdout(rows, ["task"])
    assert result["metrics"]["total_model_cost_usd"] is None
    assert result["metrics"]["physical_attempts"] == 2


@pytest.mark.parametrize("changes", [{"task_name": "unplanned"}, {"evaluation_phase": "holdout_final"},
                                     {"repetition": 2}])
def test_unplanned_attempt_cannot_enter_summary(changes):
    with pytest.raises(ValueError, match="Unexpected"):
        aggregate_holdout([observation(**changes)], ["task"])


def test_duplicate_attempt_identity_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_holdout([observation(), observation()], ["task"])


def test_redaction_keeps_completeness_counts_but_never_raw_traces_or_error_bodies():
    raw = observation()
    raw.update(trace=["secret"], test_stdout="private test content")
    raw["metadata"].update(error_detail="private provider body", artifact_dir="private path")
    raw["metadata"]["verifier_validity"]["raw_report"] = "private test assertions"
    row = redact_attempt(raw)
    assert row["metadata"]["verifier_validity"]["counts"]["tests"] == 2
    text = str(row)
    assert all(forbidden not in text for forbidden in ("secret", "private", "raw_report", "trace"))


def test_same_source_reuse_is_explicit_and_does_not_reselect():
    summary = aggregate_holdout([observation()], ["task"])
    result = compare_holdout(summary, summary, True)
    assert result["status"] == "same_source_no_improvement"
    assert result["same_source_result_reused"] is True
    assert result["quality_delta"] == 0
    assert result["mean_model_cost_delta_usd"] == 0
    assert not result["selection_changed"]
    assert not result["improvement_demonstrated"]


@pytest.mark.asyncio
async def test_only_one_holdout_controller_can_enter_disposable_database(pool):
    async with pool.acquire() as first, pool.acquire() as second:
        await acquire_controller(first)
        with pytest.raises(RuntimeError, match="Another"):
            await acquire_controller(second)
        await first.execute("SELECT pg_advisory_unlock_all()")
        await acquire_controller(second)
        await second.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_active_evidence_job_blocks_holdout_before_dispatch(pool):
    caller, _ = await principal(pool)
    await submit(pool, caller, "selection-still-running")
    async with pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="Active evidence"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_existing_terminal_holdout_blocks_repeated_inspection(pool):
    caller, _ = await principal(pool)
    job, _ = await submit(pool, caller, "aos_final_holdout_previous-baseline")
    async with pool.acquire() as connection:
        await connection.execute("UPDATE jobs SET status='CANCELLED' WHERE id=$1", job["id"])
        with pytest.raises(RuntimeError, match="already submitted"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_holdout_evidence_from_original_workflow_blocks_inspection_again(pool):
    caller, _ = await principal(pool)
    await submit(pool, caller, "original-optimization")
    job = await claim(pool)
    candidate = await iteration(pool, job)
    async with pool.acquire() as connection:
        await repo.record_task_result(connection, lease(job), candidate["id"], HOLDOUT_TASKS[0],
            status="failed", reward=0, failure_class="assertion_failure", duration_s=1,
            n_steps=1, exit_reason="done", test_stdout=None, trace=None, digest={},
            evaluation_phase="holdout_final", repetition=1, attempt=1)
        await connection.execute("UPDATE jobs SET status='CANCELLED' WHERE id=$1", job["id"])
        with pytest.raises(RuntimeError, match="already observed"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")
