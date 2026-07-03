import json
import threading

from autoharness_service.agent_patch import AgentPatchService
from autoharness_service.api import create_app
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.service import RunService
from fastapi.testclient import TestClient
from tests.service.test_service import (
    FakeOptimizer,
    FakeStore,
    SequencedSimulatedRunner,
    _write_agent_file,
)


def _build_client():
    service = RunService(store=FakeStore(), simulated_runner=SimulatedBenchmarkRunner())
    app = create_app(service=service, start_background=False)
    return client_from_app(app), service


def client_from_app(app):
    return TestClient(app)


def test_api_lists_tasks():
    client, _service = _build_client()

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json()["tasks"] == [
        "break-filter-js-from-html",
        "multi-source-data-merger",
    ]


def test_api_submit_poll_and_read_results():
    client, service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass", "task-fail"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
            "requested_concurrency": 1,
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )

    assert create_response.status_code == 202
    assert create_response.headers["location"].startswith("/runs/")
    assert (
        create_response.json()["status_url"]
        == f"/runs/{create_response.json()['run_id']}"
    )
    assert (
        create_response.json()["result_url"]
        == f"/runs/{create_response.json()['run_id']}/results"
    )

    run_id = create_response.json()["run_id"]

    service.execute_run(run_id, org_id="org-1")

    status_response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )
    results_response = client.get(
        f"/runs/{run_id}/results",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )
    iterations_response = client.get(
        f"/runs/{run_id}/iterations",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"
    assert status_response.json()["progress"] == {
        "total": 2,
        "queued": 0,
        "running": 0,
        "completed": 2,
    }
    assert results_response.status_code == 200
    assert results_response.json()["tasks_passed"] == 1
    assert results_response.json()["tasks_failed"] == 1
    assert iterations_response.status_code == 200
    assert iterations_response.json()["run_id"] == run_id
    assert iterations_response.json()["iterations"] == [
        {
            "iteration": 0,
            "agent_version": "initial",
            "status": "completed",
            "score": 0.5,
            "proposal": None,
            "accepted": None,
        }
    ]


def test_api_iterations_show_completed_optimization_attempt(tmp_path):
    store = FakeStore()
    runner = SequencedSimulatedRunner(
        [
            {"task-fail": 0.0},
            {"task-fail": 1.0},
        ]
    )
    service = RunService(
        store=store,
        simulated_runner=runner,
        optimizer=FakeOptimizer(),
        agent_patcher=AgentPatchService(_write_agent_file(tmp_path)),
        service_run_root=tmp_path / "service_runs",
    )
    app = create_app(service=service, start_background=False)
    client = client_from_app(app)

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-fail"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
            "requested_concurrency": 1,
            "max_iterations": 1,
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    service.execute_run(run_id, org_id="org-1")

    response = client.get(
        f"/runs/{run_id}/iterations",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )

    assert response.status_code == 200
    iterations = response.json()["iterations"]
    assert [iteration["iteration"] for iteration in iterations] == [0, 1]
    assert iterations[0]["status"] == "completed"
    assert iterations[1]["status"] == "completed"
    assert iterations[1]["score"] == 1.0
    assert iterations[1]["accepted"] is True
    proposal = json.loads(iterations[1]["proposal"])
    assert proposal["baseline_score"] == 0.0
    assert proposal["rerun_score"] == 1.0
    assert proposal["decision_reason"] == "rerun score improved baseline score"
    assert proposal["changed_section"] == "AGENT_INSTRUCTION"


def test_api_run_status_includes_per_task_lifecycle_rows():
    client, _service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass", "task-fail"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
            "requested_concurrency": 1,
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    status_response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )

    assert status_response.status_code == 200
    assert sorted(
        (
            {"task_id": task["task_id"], "status": task["status"]}
            for task in status_response.json()["task_results"]
        ),
        key=lambda task: task["task_id"],
    ) == [
        {"task_id": "task-fail", "status": "queued"},
        {"task_id": "task-pass", "status": "queued"},
    ]


def test_api_background_run_reports_running_then_succeeded():
    class BlockingRunner:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def run(
            self, task_ids, *, model, sandbox_provider, requested_concurrency, run_id
        ):
            self.started.set()
            assert self.release.wait(timeout=1)
            return {task_id: 1.0 for task_id in task_ids}

    runner = BlockingRunner()
    service = RunService(store=FakeStore(), terminal_runner=runner)
    app = create_app(service=service, start_background=True)
    client = client_from_app(app)

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "real",
            "sandbox_provider": "daytona",
            "requested_concurrency": 1,
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )

    assert create_response.status_code == 202
    assert create_response.json()["status"] == "queued"

    run_id = create_response.json()["run_id"]
    assert runner.started.wait(timeout=1)

    pending_response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )
    assert pending_response.status_code == 200
    print("\n=== RUN STATUS WHILE EXECUTING ===")
    print(json.dumps(pending_response.json(), indent=2, sort_keys=True))
    assert pending_response.json()["status"] == "running"
    assert pending_response.json()["progress"] == {
        "total": 1,
        "queued": 0,
        "running": 1,
        "completed": 0,
    }
    results_while_running = client.get(
        f"/runs/{run_id}/results",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )
    assert results_while_running.status_code == 409
    assert results_while_running.json()["detail"] == "run is not finished"

    runner.release.set()

    finished_response = _poll_run_status(
        client,
        run_id,
        org_id="org-1",
        terminal_statuses={"succeeded", "failed", "timed_out", "cancelled"},
    )
    print("\n=== RUN STATUS AFTER FINISH ===")
    print(json.dumps(finished_response, indent=2, sort_keys=True))
    assert finished_response["status"] == "succeeded"
    assert finished_response["score"] == 1.0
    assert finished_response["progress"] == {
        "total": 1,
        "queued": 0,
        "running": 0,
        "completed": 1,
    }
    results_after_finish = client.get(
        f"/runs/{run_id}/results",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )
    assert results_after_finish.status_code == 200
    assert results_after_finish.json()["tasks_passed"] == 1


def test_api_startup_poller_resumes_preexisting_queued_run():
    store = FakeStore()
    service = RunService(store=store, simulated_runner=SimulatedBenchmarkRunner())
    run = service.submit_run(
        RunCreateRequest(
            task_ids=["task-pass"],
            mode="simulated",
            sandbox_provider="simulated",
            requested_concurrency=1,
        ),
        org_id="org-1",
        created_by="user-1",
        start_background=False,
    )
    app = create_app(service=service, start_background=True)

    with client_from_app(app) as client:
        finished_response = _poll_run_status(
            client,
            run.run_id,
            org_id="org-1",
            terminal_statuses={"succeeded", "failed", "timed_out", "cancelled"},
        )

    assert finished_response["status"] == "succeeded"
    assert finished_response["progress"] == {
        "total": 1,
        "queued": 0,
        "running": 0,
        "completed": 1,
    }


def test_api_enforces_org_boundary():
    client, _service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-a", "X-User-Id": "user-a", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-b", "X-User-Id": "user-b", "X-Role": "viewer"},
    )

    assert response.status_code == 404


def test_api_forbids_non_admin_reading_another_users_run():
    client, _service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "owner", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "other", "X-Role": "viewer"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "user cannot access this run"


def test_api_allows_admin_reading_another_users_run_in_same_org():
    client, _service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "owner", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "admin", "X-Role": "admin"},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_api_forbids_viewer_create_run():
    client, _service = _build_client()

    response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )

    assert response.status_code == 403


def test_api_requires_x_role_header_for_create_run():
    client, _service = _build_client()

    response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1"},
    )

    assert response.status_code == 422


def test_api_rejects_duplicate_task_ids():
    client, _service = _build_client()

    response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass", "task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )

    assert response.status_code == 422


def test_api_results_returns_409_before_run_finishes():
    client, _service = _build_client()

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass"],
            "mode": "simulated",
            "sandbox_provider": "simulated",
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "runner"},
    )
    run_id = create_response.json()["run_id"]

    response = client.get(
        f"/runs/{run_id}/results",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1", "X-Role": "viewer"},
    )

    assert response.status_code == 409


def _poll_run_status(client, run_id, *, org_id, terminal_statuses):
    for _ in range(20):
        response = client.get(
            f"/runs/{run_id}",
            headers={"X-Org-Id": org_id, "X-User-Id": "user-1", "X-Role": "viewer"},
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in terminal_statuses:
            return payload
    raise AssertionError(f"run {run_id} did not finish")
