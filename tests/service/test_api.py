from autoharness_service.api import create_app
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.service import RunService
from fastapi.testclient import TestClient
from tests.service.test_service import FakeStore


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
        "task-pass",
        "task-fail",
        "task-infra",
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
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1"},
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
        headers={"X-Org-Id": "org-b", "X-User-Id": "user-b"},
    )

    assert response.status_code == 404


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
