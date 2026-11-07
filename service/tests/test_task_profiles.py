"""Named calibration profiles freeze roles without rewriting historical jobs."""
from copy import deepcopy
import json
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

import test_client
from service import db, repo
from service.api.app import create_app
from service.models import JobCreate
from service.tasks_catalog import (
    CALIBRATED_SPLIT, CALIBRATED_TASKS, CALIBRATION_EXCLUSIONS, DEFAULT_TASKS,
    PROFILES, STRESS_SPLIT, STRESS_TASKS,
)
from service.tests.test_persistence import pool, principal, submit


def test_default_profile_freezes_exact_calibrated_roles():
    request = JobCreate()
    assert request.profile == "calibrated12"
    assert request.resolved_tasks() == CALIBRATED_TASKS == DEFAULT_TASKS
    assert request.evaluation_split.model_dump() == {
        "feedback": ["fix-git", "openssl-selfsigned-cert", "sqlite-db-truncate", "extract-elf", "sanitize-git-repo"],
        "selection": ["nginx-request-logging", "largest-eigenval", "cancel-async-tasks"],
        "holdout": ["regex-log", "sparql-university", "tune-mjcf", "polyglot-c-py"],
    }
    assert len(request.resolved_tasks()) == 12
    assert set(STRESS_TASKS) - set(CALIBRATED_TASKS) == set(CALIBRATION_EXCLUSIONS)


def test_explicit_stress_profile_preserves_original_tasks_and_split():
    request = JobCreate(profile="stress16")
    assert len(request.resolved_tasks()) == 16
    assert request.resolved_tasks() == STRESS_TASKS
    assert request.evaluation_split.model_dump() == STRESS_SPLIT
    # Old requests specified their tasks directly. Their algorithmic split stays
    # identical even though the default of new requests has changed.
    legacy_request = JobCreate(tasks=STRESS_TASKS)
    assert legacy_request.profile is None
    assert legacy_request.evaluation_split.model_dump() == STRESS_SPLIT


def test_custom_tasks_keep_generic_and_explicit_split_support():
    tasks = ["fix-git", "regex-log", "nginx-request-logging"]
    request = JobCreate(tasks=tasks)
    assert request.profile is None
    assert sorted(sum(request.evaluation_split.model_dump().values(), [])) == sorted(tasks)
    custom = {"feedback": ["nginx-request-logging"], "selection": ["regex-log"], "holdout": ["fix-git"]}
    assert JobCreate(tasks=tasks, evaluation_split=custom).evaluation_split.model_dump() == custom


@pytest.mark.parametrize("profile", ["calibrated12", "stress16"])
def test_profile_and_custom_tasks_are_mutually_exclusive(profile):
    with pytest.raises(ValidationError, match="mutually exclusive"):
        JobCreate(profile=profile, tasks=PROFILES[profile]["tasks"])


def test_named_roles_cannot_be_silently_overridden():
    changed = deepcopy(CALIBRATED_SPLIT)
    changed["feedback"][0], changed["selection"][0] = changed["selection"][0], changed["feedback"][0]
    with pytest.raises(ValidationError, match="named profile"):
        JobCreate(profile="calibrated12", evaluation_split=changed)
    # A caller may explicitly pin the same split without changing the profile.
    assert JobCreate(profile="calibrated12", evaluation_split=CALIBRATED_SPLIT).evaluation_split.model_dump() == CALIBRATED_SPLIT


def test_profile_values_are_copied_and_baseline_only_still_works():
    request = JobCreate()
    request.resolved_tasks().clear()
    request.evaluation_split.feedback.clear()
    assert JobCreate().evaluation_split.model_dump() == CALIBRATED_SPLIT
    baseline = JobCreate(profile="calibrated12", max_iterations=0)
    assert baseline.evaluation_split is None
    assert len(baseline.resolved_tasks()) == 12
    assert JobCreate(tasks=["fix-git"], max_iterations=0).evaluation_split is None


def test_unknown_profile_and_nonfinite_budget_are_rejected():
    with pytest.raises(ValidationError):
        JobCreate(profile="unknown")
    for amount in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError, match="finite"):
            JobCreate(budget={"max_llm_usd": amount})


@pytest.mark.parametrize(("arguments", "expected"), [
    ([], {"profile": "calibrated12"}),
    (["--profile", "stress16"], {"profile": "stress16"}),
    (["--tasks", "fix-git", "regex-log", "--max-iterations", "0"], {"tasks": ["fix-git", "regex-log"]}),
])
def test_cli_submission_profile_or_custom_tasks(arguments, expected):
    args = test_client.build_parser().parse_args(["--api-key", "test-only-key", *arguments])
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "test-only-job", "status": "QUEUED"})
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        test_client.submit(client, args)
    body = captured[0]
    assert {key: body[key] for key in ("tasks", "profile") if key in body} == expected
    assert body["budget"]["max_llm_usd"] == 2


def test_cli_rejects_ambiguous_or_empty_custom_tasks():
    parser = test_client.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--profile", "calibrated12", "--tasks", "fix-git"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--tasks"])


def test_cli_forwards_explicit_execution_and_proposal_models():
    args = test_client.build_parser().parse_args([
        "--api-key", "test-only-key", "--agent-model", "gpt-5.4-mini",
        "--optimizer-model", "gpt-5.4", "--no-confirm-flips"])
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "test-only-job", "status": "QUEUED"})
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        test_client.submit(client, args)
    assert captured[0]["agent_model"] == "gpt-5.4-mini"
    assert captured[0]["optimizer_model"] == "gpt-5.4"
    assert captured[0]["profile"] == "calibrated12"
    assert "confirmation is mandatory" in test_client.build_parser().format_help()


@pytest.mark.asyncio
async def test_api_persists_named_profile_and_preserves_historical_config(pool, monkeypatch):
    caller, key = await principal(pool)
    legacy = {"tasks": STRESS_TASKS, "evaluation_split": STRESS_SPLIT,
              "max_iterations": 2, "budget": {"max_llm_usd": 1, "max_wall_clock_min": 10}}
    previous, _ = await submit(pool, caller, "historical-stress", legacy)
    monkeypatch.setattr(db, "_pool", pool)
    headers = {"Authorization": "Bearer " + key}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        for body, profile in (({}, "calibrated12"), ({"profile": "stress16"}, "stress16")):
            response = await client.post("/v1/jobs", json=body, headers=headers)
            assert response.status_code == 202, response.text
            job_id = UUID(response.json()["job_id"])
            async with pool.acquire() as connection:
                job = await repo.get_job(connection, caller, job_id)
            assert job["config"]["profile"] == profile
            assert job["config"]["tasks"] == PROFILES[profile]["tasks"]
            assert job["config"]["evaluation_split"] == PROFILES[profile]["evaluation_split"]
        response = await client.get(f"/v1/jobs/{previous['id']}", headers=headers)
        assert response.status_code == 200
        assert response.json()["config"] == legacy
        response = await client.get("/v1/tasks", headers=headers)
        by_name = {task["name"]: task for task in response.json()}
        assert by_name["fix-git"]["evaluation_roles"]["calibrated12"] == "feedback"
        assert by_name["regex-log"]["evaluation_roles"]["calibrated12"] == "holdout"
        assert by_name["chess-best-move"]["profiles"] == ["stress16"]


@pytest.mark.asyncio
async def test_api_rejects_conflicting_profile_before_creating_job(pool, monkeypatch):
    _, key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.post("/v1/jobs", headers={"Authorization": "Bearer " + key},
            json={"profile": "calibrated12", "tasks": ["fix-git"], "max_iterations": 0})
        assert response.status_code == 422
    async with pool.acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM jobs") == 0
