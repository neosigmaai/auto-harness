import pytest
from test_client import (
    DEFAULT_MODE,
    DEMO_BATCH_TASKS,
    build_headers,
    build_run_request,
    build_summary,
    poll_run,
)


def test_build_summary_includes_results_and_iterations():
    summary = build_summary(
        status={"run_id": "run-1", "status": "succeeded"},
        results={"score": 0.5, "tasks_passed": 1, "tasks_failed": 1},
        iterations={"iterations": [{"iteration": 0, "status": "completed"}]},
    )

    assert summary["run_id"] == "run-1"
    assert summary["status"] == "succeeded"
    assert summary["score"] == 0.5
    assert summary["iterations"][0]["status"] == "completed"


def test_build_summary_includes_iteration_status_and_rejection_state():
    summary = build_summary(
        status={"run_id": "run-1", "status": "succeeded"},
        results={"score": 0.5, "tasks_passed": 1, "tasks_failed": 1},
        iterations={
            "iterations": [
                {"iteration": 0, "status": "completed", "accepted": None},
                {"iteration": 1, "status": "patch_rejected", "accepted": False},
            ]
        },
    )

    assert summary["iteration_statuses"] == [
        "0:completed",
        "1:patch_rejected",
    ]
    assert summary["optimization_status"] == "rejected"


def test_build_summary_includes_iteration_status_and_acceptance_state():
    summary = build_summary(
        status={"run_id": "run-1", "status": "succeeded"},
        results={"score": 1.0, "tasks_passed": 1, "tasks_failed": 0},
        iterations={
            "iterations": [
                {"iteration": 0, "status": "completed", "accepted": None},
                {"iteration": 1, "status": "completed", "accepted": True},
            ]
        },
    )

    assert summary["iteration_statuses"] == [
        "0:completed",
        "1:completed",
    ]
    assert summary["optimization_status"] == "accepted"


def test_build_headers_sets_demo_identity_and_role():
    headers = build_headers("runner")

    assert headers == {
        "X-Org-Id": "demo-org",
        "X-User-Id": "demo-user",
        "X-Role": "runner",
    }


def test_build_run_request_rejects_simulated_mode_for_manual_tests():
    with pytest.raises(ValueError, match="only supports real mode"):
        build_run_request(
            task_ids=["task-pass"],
            mode="simulated",
            max_iterations=0,
            requested_concurrency=1,
        )


def test_build_run_request_sets_daytona_sandbox_provider_for_real_mode():
    payload = build_run_request(
        task_ids=["break-filter-js-from-html"],
        mode=DEFAULT_MODE,
        max_iterations=0,
        requested_concurrency=1,
    )

    assert payload["mode"] == DEFAULT_MODE
    assert payload["sandbox_provider"] == "daytona"


def test_demo_batch_tasks_are_real_terminal_bench_tasks():
    assert DEMO_BATCH_TASKS == [
        "break-filter-js-from-html",
        "multi-source-data-merger",
    ]


def test_poll_run_reports_every_status_payload_without_sleep(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self):
            self.payloads = [
                {
                    "run_id": "run-1",
                    "status": "queued",
                    "progress": {
                        "total": 4,
                        "queued": 4,
                        "running": 0,
                        "completed": 0,
                    },
                },
                {
                    "run_id": "run-1",
                    "status": "running",
                    "progress": {
                        "total": 4,
                        "queued": 2,
                        "running": 1,
                        "completed": 1,
                    },
                },
                {
                    "run_id": "run-1",
                    "status": "succeeded",
                    "progress": {
                        "total": 4,
                        "queued": 0,
                        "running": 0,
                        "completed": 4,
                    },
                },
            ]

        def get(self, path, headers):
            assert path == "/runs/run-1"
            assert headers["X-Role"] == "viewer"
            return FakeResponse(self.payloads.pop(0))

    statuses = []
    monkeypatch.setattr("test_client.time.sleep", lambda _seconds: None)

    final_status = poll_run(
        FakeClient(),
        "run-1",
        poll_interval_sec=5,
        timeout_sec=60,
        on_status=statuses.append,
    )

    assert final_status["status"] == "succeeded"
    assert [status["status"] for status in statuses] == [
        "queued",
        "running",
        "succeeded",
    ]
