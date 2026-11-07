"""Fresh calibrated experiment controller checks; no paid/model/Docker calls."""
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from scripts.run_calibrated_review import (
    BASELINE_SHA256, DIAGNOSTIC_USD, acquire_controller, api_snapshot, build_request,
    cancel_owned_submissions, database_snapshot, remaining_budget, run,
    summarize_iteration, summarize_job, summarize_metrics, validate_arguments,
)
from scripts.run_final_holdout_review import acquire_controller as acquire_holdout_controller
from service import repo
from service.repo import sha256_text
from service.tasks_catalog import CALIBRATED_SPLIT
from service.tests.test_persistence import pool, principal, submit, claim, lease, iteration, record

BASELINE = Path(__file__).resolve().parents[2] / "agent" / "templates" / "terminal_bench.py"


@pytest.mark.asyncio
async def test_no_paid_gate_exits_before_reading_any_files_or_credentials():
    with pytest.raises(SystemExit, match="No model calls"):
        await run(SimpleNamespace(allow_paid=False))


@pytest.mark.parametrize("cap", [0, -1, 6.000001, float("nan"), float("inf")])
def test_controller_cap_cannot_exceed_authorized_six_dollars(cap):
    with pytest.raises(SystemExit, match="at most"):
        validate_arguments(SimpleNamespace(allow_paid=True, max_total_usd=cap, max_wall_clock_min=90))


def test_original_baseline_and_new_frozen_profile_have_no_inherited_state():
    source = BASELINE.read_text()
    assert sha256_text(source) == BASELINE_SHA256
    body = build_request(source, 6)
    assert body["profile"] == "calibrated12"
    assert body["evaluation_split"] == CALIBRATED_SPLIT
    assert body["agent_model"] == "gpt-5.4-mini"
    assert body["optimizer_model"] == "gpt-5.4"
    assert body["max_iterations"] == body["patience"] == 3
    assert body["seed"] == 1234
    assert body["options"]["confirmation_repetitions"] == 2
    assert body["options"]["max_infrastructure_attempts"] == 2
    assert body["budget"] == {"max_llm_usd": 6, "max_wall_clock_min": 90}
    assert not ({"tasks", "prior_job_id", "resume_candidate_index", "optimizer_history_seed",
                 "resume_candidate", "score", "results"} & body.keys())
    with pytest.raises(ValueError, match="original pinned"):
        build_request(source + "\n# different candidate", 6)


def test_prior_history_changes_only_api_linkage_without_evaluation_or_candidate_reuse():
    prior = UUID('253886be-b646-4176-ac23-7e6990e71755')
    fresh = build_request(BASELINE.read_text(), 4, 25)
    inherited = build_request(BASELINE.read_text(), 4, 25, prior_job_id=prior)
    assert inherited == {**fresh, 'prior_job_id': str(prior)}
    assert inherited['options']['confirmation_repetitions'] == 2
    assert inherited['evaluation_split'] == CALIBRATED_SPLIT
    assert inherited['max_iterations'] == inherited['patience'] == 3
    assert inherited['options']['efficiency_tolerance'] == .1
    assert inherited['options']['min_efficiency_improvement'] == .1
    assert not {'resume_candidate_index', 'resume_candidate', 'optimizer_history_seed', 'score', 'results'} & inherited.keys()


def test_prior_history_requires_explicit_repeated_exposure_and_valid_id():
    args = SimpleNamespace(allow_paid=True,max_total_usd=4,max_wall_clock_min=25,
                           prior_job_id='253886be-b646-4176-ac23-7e6990e71755',development_repeat=False)
    with pytest.raises(SystemExit, match='development-repeat'):
        validate_arguments(args)
    args.development_repeat=True
    validate_arguments(args)
    with pytest.raises(ValueError):
        build_request(BASELINE.read_text(),4,25,prior_job_id='not-a-job-id')


def test_remaining_allowance_includes_all_prior_spending_diagnostics_and_reservations():
    expected = (Decimal(".25") - DIAGNOSTIC_USD).quantize(Decimal(".000001"), rounding=ROUND_DOWN)
    assert remaining_budget(10, "14.5", ".25", 6) == expected
    assert remaining_budget(5, 11, .01, 6) == 0
    with pytest.raises(ValueError, match="allowance"):
        build_request(BASELINE.read_text(), 6.1)


def test_numerical_reports_exclude_prompts_code_and_arbitrary_error_bodies():
    raw = {"index": 1, "decision": "rejected", "score": 0, "agent_sha256": "hash",
        "hypothesis": "PRIVATE HYPOTHESIS", "agent_source": "PRIVATE SOURCE", "proposal": {"raw": "PRIVATE"},
        "decision_reason": "PRIVATE MODEL TEXT",
        "metrics": {"complete": True, "mean_reward": 0, "raw_body": "PRIVATE",
            "feedback_summary": {"observed_tasks": 5, "failure_classes": {"PRIVATE": 1}},
            "incumbent": {"mean_reward": .5, "trace": ["PRIVATE"]}}}
    summary = summarize_iteration(raw)
    assert "PRIVATE" not in str(summary)
    assert summary["metrics"]["incumbent"]["mean_reward"] == .5
    assert summary["metrics"]["feedback_summary"]["observed_tasks"] == 5
    nested = summarize_metrics({"preliminary": {"mean_reward": .5, "trace": "PRIVATE"},
                                "confirmation_eligibility": {"eligible": True, "reason": "PRIVATE"}})
    assert nested == {"preliminary": {"mean_reward": .5}, "confirmation_eligibility": {"eligible": True}}
    assert summarize_metrics({"mean_reward": "PRIVATE", "not_a_metric": 1}) == {}
    job = summarize_job({"job_id": "x", "error": "PRIVATE", "spent_usd": Decimal(".25"),
        "reserved_usd": Decimal(".1"), "config": {"agent_source": "PRIVATE"},
        "final_evaluation": {"status": "incomplete", "reason": "budget_exhausted", "used_for_selection": False}})
    assert "PRIVATE" not in str(job)
    assert job["spent_usd"] == .25 and isinstance(job["spent_usd"], float)
    assert job["final_evaluation"]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_api_snapshot_paginates_attempts_without_fetching_traces_or_candidate_source():
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/jobs/j":
            return httpx.Response(200, json={"job_id": "j", "status": "RUNNING", "spent_usd": .1})
        if request.url.path == "/v1/jobs/j/iterations":
            return httpx.Response(200, json=[{"index": 0, "hypothesis": "PRIVATE"}])
        assert request.url.path == "/v1/jobs/j/iterations/0/attempts"
        assert request.url.params["include_trace"] == "false"
        offset = int(request.url.params["offset"])
        rows = [{"task_name": "fix-git", "attempt": n, "trace": ["PRIVATE"],
                 "metadata": {"error_detail": "PRIVATE"}} for n in range(100)] if offset == 0 else [
                     {"task_name": "fix-git", "attempt": 101}]
        return httpx.Response(200, json=rows)
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        job, iterations, attempts = await api_snapshot(client, {"Authorization": "Bearer test-only"}, "j")
    assert len(attempts) == 101
    assert len(requests) == 4
    assert "PRIVATE" not in str((job, iterations, attempts))


@pytest.mark.asyncio
async def test_calibrated_and_holdout_controllers_cannot_overlap(pool):
    async with pool.acquire() as first, pool.acquire() as second:
        await acquire_controller(first)
        with pytest.raises(RuntimeError, match="Another"):
            await acquire_holdout_controller(second)
        await first.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_calibrated_controller_rejects_active_job_and_prior_completed_experiment(pool):
    caller, _ = await principal(pool)
    old, _ = await submit(pool, caller, "aos_calibrated_previous")
    async with pool.acquire() as connection:
        with pytest.raises(RuntimeError, match="Active jobs"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")
        await connection.execute("UPDATE jobs SET status='CANCELLED' WHERE id=$1", old["id"])
        with pytest.raises(RuntimeError, match="already submitted"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_prior_stress_holdout_observation_blocks_fresh_calibrated_inspection(pool):
    caller, _ = await principal(pool)
    await submit(pool, caller, "old-stress-experiment")
    job = await claim(pool)
    baseline = await iteration(pool, job)
    async with pool.acquire() as connection:
        await repo.record_task_result(connection, lease(job), baseline["id"], CALIBRATED_SPLIT["holdout"][0],
            status="failed", reward=0, failure_class="assertion_failure", duration_s=1,
            n_steps=1, exit_reason="done", test_stdout=None, trace=None, digest={},
            evaluation_phase="holdout_final", repetition=1, attempt=1)
        await connection.execute("UPDATE jobs SET status='CANCELLED' WHERE id=$1", job["id"])
        with pytest.raises(RuntimeError, match="holdout already"):
            await acquire_controller(connection)
        await connection.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_explicit_development_repeat_preserves_and_reports_prior_exposure(pool):
    caller, _ = await principal(pool)
    await submit(pool, caller, "aos_calibrated_previous")
    job = await claim(pool)
    baseline = await iteration(pool, job)
    async with pool.acquire() as connection:
        await repo.record_task_result(connection, lease(job), baseline["id"], CALIBRATED_SPLIT["holdout"][0],
            status="failed", reward=0, failure_class="wrong_output", duration_s=1,
            n_steps=1, exit_reason="done", test_stdout=None, trace=None, digest={},
            evaluation_phase="holdout_final", repetition=1, attempt=1)
        with pytest.raises(RuntimeError, match="Active jobs"):
            await acquire_controller(connection, development_repeat=True)
        await connection.execute("SELECT pg_advisory_unlock_all()")
        await connection.execute("UPDATE jobs SET status='SUCCEEDED' WHERE id=$1", job["id"])
        exposure = await acquire_controller(connection, development_repeat=True)
        assert exposure == {"development_repeat": True, "prior_calibrated_jobs": 1,
                            "prior_holdout_observations": 1, "previously_unobserved_holdout": False}
        assert await connection.fetchval("SELECT count(*) FROM task_attempts") == 1
        await connection.execute("SELECT pg_advisory_unlock_all()")


@pytest.mark.asyncio
async def test_lost_submission_response_cleanup_preserves_all_iterations_without_private_payloads(pool):
    caller, key = await principal(pool)
    await submit(pool, caller, "aos_calibrated_ambiguous")
    job = await claim(pool)
    baseline = await iteration(pool, job)
    async with pool.acquire() as connection:
        await record(connection, lease(job), baseline, trace=[{"private": "DO_NOT_COPY"}])
        version = await repo.upsert_agent_version(connection, caller.org_id, "# candidate source data", None)
        candidate = await repo.create_iteration(connection, lease(job), 1, version["id"], baseline["id"],
            {"hypothesis": "DO_NOT_COPY"})
        await record(connection, lease(job), candidate, status="error", phase="confirm_1", trace=["DO_NOT_COPY"])
        reservation = await repo.reserve_spend(connection, lease(job), "test-accounting", .1, "agent")
        await repo.settle_spend(connection, reservation, .02)
        rows = await cancel_owned_submissions(connection, key, ["aos_calibrated_ambiguous"])
        assert len(rows) == 1 and rows[0]["cancel_requested"]
        snapshot = await database_snapshot(connection, rows[0]["job_id"])
        observed_job, iterations, attempts = snapshot
        assert len(iterations) == len(attempts) == 2
        assert observed_job["spent_usd"] == .02
        assert observed_job["reserved_usd"] == 0
        assert "DO_NOT_COPY" not in str(snapshot)
        assert {a["iteration_index"] for a in attempts} == {0, 1}
        with pytest.raises(repo.JobCancelled):
            await repo.reserve_spend(connection, lease(job), "must-not-dispatch", .1, "agent")
