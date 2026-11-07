from decimal import Decimal

import pytest
import httpx
from pydantic import ValidationError

from scripts.run_paired_selection_review import aggregate_source, remaining_budget, cancel_owned_submissions, audit_options
from service import repo
from service.models import JobCreate
from service.api.app import create_app
from service.auth import get_principal
from service.tests.test_persistence import pool, principal, submit, claim, lease


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_job_request_rejects_nonfinite_monetary_budget(value):
    with pytest.raises(ValidationError):
        JobCreate(tasks=["fix-git"], max_iterations=0, budget={"max_llm_usd":value})


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
async def test_nonfinite_json_budget_returns_safe_422_without_dispatch(literal):
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/jobs", headers={"Content-Type":"application/json"},
            content='{"tasks":["fix-git"],"max_iterations":0,"budget":{"max_llm_usd":'+literal+'}}')
    assert response.status_code == 422
    assert response.json()["errors"][0]["type"] == "finite_number"


def test_audit_cap_accounts_for_outstanding_and_never_rounds_up():
    assert remaining_budget("2.69184525", "3.2", ".15", 6) == Decimal("5.341845")
    assert remaining_budget(2, 8, .01, 6) == 0


def test_session_ceiling_is_stricter_when_other_spending_consumes_budget():
    assert remaining_budget(10, "14.5", ".25", 6) == Decimal("0.249064")


def test_regressed_ledger_is_rejected():
    with pytest.raises(ValueError, match="regressed"):
        remaining_budget(3, 2, 0, 6)


def test_retry_override_preserves_frozen_agent_limits_and_source_configuration():
    original = {"max_infrastructure_attempts":1, "agent_timeout_sec":900,
                "max_agent_steps":80, "max_tokens_per_call":4096}
    configured = audit_options(original)
    assert original["max_infrastructure_attempts"] == 1
    assert configured == {**original,"max_infrastructure_attempts":2}
    assert audit_options(original) == configured  # same predeclared settings for both sources


def test_partial_audit_never_yields_a_complete_score():
    observation = dict(task_name="task", evaluation_phase="main", repetition=1, attempt=1, status="passed", reward=1,
        duration_s=2, model_cost_usd=.1, cost_known=True, input_tokens=10, output_tokens=3)
    result = aggregate_source([{"attempts": [observation]}], ["task"])
    assert result["planned_trials"] == 2
    assert result["complete"] is False
    assert result["mean_reward"] is None


def test_audit_uses_first_valid_outcome_and_charges_all_attempts():
    common = dict(task_name="task", evaluation_phase="main", repetition=1,
                  cost_known=True, input_tokens=10, output_tokens=3)
    trials = [dict(common, attempt=1, status="error", reward=None, duration_s=5, model_cost_usd=.2),
              dict(common, attempt=2, status="failed", reward=0, duration_s=2, model_cost_usd=.1),
              dict(common, attempt=3, status="passed", reward=1, duration_s=3, model_cost_usd=.4)]
    for row in trials:
        reward = row["reward"]
        row["metadata"] = {"verifier_validity": {"valid": True, "contract": "pinned-pytest-ctrf-v1",
            "fresh_session_marker": True, "pytest_exit_code": 0 if reward == 1 else 1,
            "counts": {"tests": 1, "passed": int(reward == 1), "failed": int(reward != 1),
                       "skipped": 0, "pending": 0, "other": 0}}}
    result = aggregate_source([{"attempts": trials}, {"attempts": trials}], ["task"])
    assert result["complete"] is True
    assert result["mean_reward"] == 0
    assert result["total_model_cost_usd"] == pytest.approx(1.4)
    assert result["p95_task_duration_s"] == 10
    assert result["physical_attempts"] == 6


@pytest.mark.asyncio
async def test_ambiguous_committed_submission_is_cancelled_by_exact_owned_key(pool):
    caller, key = await principal(pool)
    committed, _ = await submit(pool, caller, "lost-response")
    untouched, _ = await submit(pool, caller, "different-audit")
    foreign, _ = await principal(pool)
    foreign_job, _ = await submit(pool, foreign, "lost-response")
    # The caller received no job ID (POST response lost); only the prewritten key
    # remains. Reconciliation must cancel this submission and nothing else.
    async with pool.acquire() as conn:
        rows = await cancel_owned_submissions(conn, key, ["lost-response", "never-committed"])
        assert [row["job_id"] for row in rows] == [str(committed["id"])]
        assert rows[0]["status"] == "CANCELLED"
        assert (await repo.get_job(conn, caller, untouched["id"]))["status"] == "QUEUED"
        assert (await repo.get_job(conn, foreign, foreign_job["id"]))["status"] == "QUEUED"


@pytest.mark.asyncio
async def test_database_fallback_blocks_new_spend_when_cancel_api_is_unavailable(pool):
    caller, key = await principal(pool)
    await submit(pool, caller, "running-audit")
    job = await claim(pool)
    async with pool.acquire() as conn:
        # No HTTP cancellation is needed for this owned-key fallback.
        rows = await cancel_owned_submissions(conn, key, ["running-audit"])
        assert rows[0]["status"] == "RUNNING"
        assert rows[0]["cancel_requested"] is True
        with pytest.raises(repo.JobCancelled):
            await repo.reserve_spend(conn, lease(job), "should-not-dispatch", .1, "agent")
