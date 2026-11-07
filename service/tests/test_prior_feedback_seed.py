"""Authorized prior experiments copy bounded diagnostic evidence, never evaluation data."""
import json
from uuid import UUID

import httpx
import pytest

from service import db, repo
from service.api import routes_jobs
from service.api.app import create_app
from service.tasks_catalog import CALIBRATED_SPLIT
from service.tests.test_persistence import pool, principal, lease


def trace(marker):
    return [{"role": "assistant", "tool_calls": [{"id": "x", "function": {
        "name": "bash", "arguments": json.dumps({"command": "printf " + marker})}}]},
        {"role": "tool", "tool_call_id": "x", "content": marker},
        {"role": "assistant", "content": "Observed " + marker}]


@pytest.mark.asyncio
async def test_prior_feedback_snapshot_is_bounded_authorized_and_attempt_aware(pool, monkeypatch):
    caller, key = await principal(pool)
    _, foreign_key = await principal(pool)
    monkeypatch.setattr(db, "_pool", pool)
    monkeypatch.setattr(routes_jobs.settings, "prepared_images_manifest", None)
    body = {"profile": "calibrated12", "max_iterations": 5, "agent_source": "# original source"}
    headers = {"Authorization": "Bearer " + key}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.post("/v1/jobs", json=body, headers=headers)
        assert response.status_code == 202
        prior_id = UUID(response.json()["job_id"])
        async with pool.acquire() as conn:
            job = await repo.claim_job(conn, "fixture", 60)
            owner = lease(job)
            version = await repo.upsert_agent_version(conn, caller.org_id, body["agent_source"], None)
            baseline = await repo.create_iteration(conn, owner, 0, version["id"], None, None)
            await repo.complete_iteration(conn, owner, baseline["id"], score=0, n_pass=0, n_fail=3,
                                          n_error=0, decision="baseline", decision_reason="baseline")

            async def record(it, task, marker, *, phase="main", repetition=1, attempt=1, status="failed"):
                await repo.record_task_result(conn, owner, it["id"], task, status=status,
                    reward=None if status == "error" else 0, failure_class="wrong_output",
                    duration_s=1, n_steps=1, exit_reason="done", test_stdout=marker, trace=trace(marker),
                    digest={"summary": "STALE_DIGEST_MUST_BE_REBUILT"}, evaluation_phase=phase,
                    repetition=repetition, attempt=attempt)

            for index in range(1, 5):
                v = await repo.upsert_agent_version(conn, caller.org_id, f"# candidate {index}", version["id"])
                it = await repo.create_iteration(conn, owner, index, v["id"], baseline["id"],
                    {"hypothesis": f"candidate {index}", "targets": CALIBRATED_SPLIT["feedback"]})
                for task in CALIBRATED_SPLIT["feedback"]:
                    if index == 3 and task == CALIBRATED_SPLIT["feedback"][0]:
                        await record(it, task, "INFRA_ATTEMPT_MUST_NOT_COPY", status="error")
                        await record(it, task, "CHOSEN_VALID_FEEDBACK", attempt=2)
                        await record(it, task, "LATER_RETRY_MUST_NOT_COPY", attempt=3)
                    else:
                        await record(it, task, f"FEEDBACK_ITER_{index}")
                await record(it, CALIBRATED_SPLIT["selection"][0], "SELECTION_MUST_NOT_COPY")
                await record(it, CALIBRATED_SPLIT["holdout"][0], "HOLDOUT_MUST_NOT_COPY", phase="holdout_final")
                await record(it, CALIBRATED_SPLIT["feedback"][0], "CONFIRM_MUST_NOT_COPY", phase="confirm_3")
                await record(it, CALIBRATED_SPLIT["feedback"][1], "REPETITION2_MUST_NOT_COPY", repetition=2)
                await repo.complete_iteration(conn, owner, it["id"], score=0, n_pass=0, n_fail=3, n_error=0,
                    decision="rejected", decision_reason="cancelled" if index == 4 else f"observed_rejection_{index}",
                    status="errored" if index == 4 else "completed")
            await repo.finish_job(conn, owner, "CANCELLED", "fixture_done")

        request = {**body, "prior_job_id": str(prior_id)}
        denied = await client.post("/v1/jobs", json=request, headers={"Authorization": "Bearer " + foreign_key})
        assert denied.status_code == 404
        incompatible = await client.post("/v1/jobs", json={**request, "agent_source": "# different"}, headers=headers)
        assert incompatible.status_code == 422
        response = await client.post("/v1/jobs", json=request, headers=headers)
        assert response.status_code == 202, response.text
        async with pool.acquire() as conn:
            saved = await repo.get_job(conn, caller, UUID(response.json()["job_id"]))
            recent = saved["config"]["optimizer_recent_feedback_seed"]
            assert len(recent) == 6
            assert [entry["iteration_index"] for entry in recent] == [2, 2, 2, 3, 3, 3]
            assert all(entry["source_job_id"] == str(prior_id) and entry["decision"] == "rejected" for entry in recent)
            assert all(entry["task_name"] in CALIBRATED_SPLIT["feedback"][:3] for entry in recent)
            assert all(entry["decision_reason"] == f"observed_rejection_{entry['iteration_index']}" for entry in recent)
            encoded = json.dumps(recent)
            assert "CHOSEN_VALID_FEEDBACK" in encoded and "FEEDBACK_ITER_2" in encoded
            for forbidden in ("MUST_NOT_COPY", "STALE_DIGEST", "FEEDBACK_ITER_1", "FEEDBACK_ITER_4"):
                assert forbidden not in encoded
            assert await conn.fetchval("SELECT count(*) FROM iterations WHERE job_id=$1", saved["id"]) == 0
            assert saved["best_score"] is None
