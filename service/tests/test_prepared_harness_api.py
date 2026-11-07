"""Frozen preparation is part of the experiment, not a mutable worker default."""
from copy import deepcopy
from uuid import UUID

import httpx
import pytest

from service import db, repo
from service.api import routes_jobs
from service.api.app import create_app
from service.tests.test_persistence import pool, principal, _prior_experiment_fixture


@pytest.mark.asyncio
async def test_admission_freezes_preparation_and_idempotency_does_not_silently_change_it(pool, monkeypatch):
    caller, key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    selected = {"mode": "prepared-tools-v1", "manifest_sha256": "first", "entries": {"fix-git": {"image_id": "sha256:first"}}}
    monkeypatch.setattr(routes_jobs, "freeze_harness", lambda *_: deepcopy(selected))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        headers = {"Authorization": "Bearer " + key, "Idempotency-Key": "fixed-images"}
        body = {"tasks": ["fix-git"], "max_iterations": 0}
        response = await client.post("/v1/jobs", json=body, headers=headers)
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        assert (await client.post("/v1/jobs", json=body, headers=headers)).status_code == 200
        selected["entries"]["fix-git"]["image_id"] = "sha256:replacement"
        selected["manifest_sha256"] = "second"
        assert (await client.post("/v1/jobs", json=body, headers=headers)).status_code == 409
        async with pool.acquire() as conn:
            saved = await repo.get_job(conn, caller, job_id)
            assert saved["config"]["sandbox_harness"]["entries"]["fix-git"]["image_id"] == "sha256:first"


@pytest.mark.asyncio
async def test_preparation_mismatch_prevents_old_history_from_crossing_harness_variants(pool, monkeypatch):
    _, key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    monkeypatch.setattr(routes_jobs, "freeze_harness", lambda *_: {"mode": "cold-v1"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        body, old_id = await _prior_experiment_fixture(pool, client, key)
        monkeypatch.setattr(routes_jobs, "freeze_harness", lambda *_: {"mode": "prepared-tools-v1", "manifest_sha256": "new"})
        response = await client.post("/v1/jobs", json={**body, "prior_job_id": str(old_id)},
                                     headers={"Authorization": "Bearer " + key})
        assert response.status_code == 422
        assert "same frozen sandbox preparation" in response.text


@pytest.mark.asyncio
async def test_legacy_job_without_harness_field_can_continue_in_same_cold_environment(pool, monkeypatch):
    caller, key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    monkeypatch.setattr(routes_jobs.settings, "prepared_images_manifest", None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        body, old_id = await _prior_experiment_fixture(pool, client, key)
        async with pool.acquire() as conn:
            # Pre-feature jobs have a frozen benchmark revision but no preparation field.
            await conn.execute("UPDATE jobs SET config=config-'sandbox_harness' WHERE id=$1", old_id)
        response = await client.post("/v1/jobs", json={**body, "prior_job_id": str(old_id),
                                                       "resume_candidate_index": 2},
                                     headers={"Authorization": "Bearer " + key})
        assert response.status_code == 202, response.text
        async with pool.acquire() as conn:
            saved = await repo.get_job(conn, caller, UUID(response.json()["job_id"]))
            assert saved["config"]["sandbox_harness"] == {"mode": "cold-v1"}
            assert saved["config"]["resume_candidate"]["source_iteration_index"] == 2
            assert saved["config"]["agent_source"] == body["agent_source"]
            assert len(saved["config"]["optimizer_history_seed"]) == 1
            assert not await conn.fetchval("SELECT count(*) FROM iterations WHERE job_id=$1", saved["id"])
            old = await repo.get_job(conn, caller, old_id)
            assert "sandbox_harness" not in old["config"]


@pytest.mark.asyncio
async def test_unavailable_preparation_rejects_before_creating_a_job_without_exposing_host_details(pool, monkeypatch):
    _, key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    def invalid(*_):
        raise OSError("PRIVATE_HOST_PATH")
    monkeypatch.setattr(routes_jobs, "freeze_harness", invalid)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.post("/v1/jobs", json={"tasks": ["fix-git"], "max_iterations": 0},
                                     headers={"Authorization": "Bearer " + key})
        assert response.status_code == 422 and "PRIVATE_HOST_PATH" not in response.text
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM jobs") == 0
