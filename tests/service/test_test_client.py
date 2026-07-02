from test_client import build_headers, build_run_request, build_summary


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


def test_build_headers_sets_demo_identity_and_role():
    headers = build_headers("runner")

    assert headers == {
        "X-Org-Id": "demo-org",
        "X-User-Id": "demo-user",
        "X-Role": "runner",
    }


def test_build_run_request_sets_simulated_sandbox_provider_for_simulated_mode():
    payload = build_run_request(
        task_ids=["task-pass"],
        mode="simulated",
        max_iterations=0,
        requested_concurrency=1,
    )

    assert payload["mode"] == "simulated"
    assert payload["sandbox_provider"] == "simulated"


def test_build_run_request_sets_daytona_sandbox_provider_for_real_mode():
    payload = build_run_request(
        task_ids=["task-pass"],
        mode="real",
        max_iterations=0,
        requested_concurrency=1,
    )

    assert payload["mode"] == "real"
    assert payload["sandbox_provider"] == "daytona"
