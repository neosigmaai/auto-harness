from pathlib import Path

from autoharness_service.runner import SimulatedBenchmarkRunner, TerminalBenchRunnerAdapter


def test_simulated_runner_returns_deterministic_rewards():
    runner = SimulatedBenchmarkRunner()

    results = runner.run(["task-pass", "task-fail", "task-infra"])

    assert results["task-pass"] == 1.0
    assert results["task-fail"] == 0.0
    assert results["task-infra"] is None


def test_terminal_runner_adapter_uses_per_run_jobs_dir_and_caps_concurrency(
    monkeypatch, tmp_path
):
    created = {}

    class FakeTerminalBenchRunner:
        def __init__(
            self,
            *,
            agent_model,
            split,
            env_provider,
            n_concurrent,
            jobs_dir,
            dataset="terminal-bench@2.0",
            agent_import_path="agent.agent:HarnessAgent",
            per_task_timeout=1200,
            reasoning_effort=None,
        ):
            created["agent_model"] = agent_model
            created["split"] = split
            created["env_provider"] = env_provider
            created["n_concurrent"] = n_concurrent
            created["jobs_dir"] = jobs_dir

        def run(self, task_ids):
            created["task_ids"] = list(task_ids)
            return {task_id: 1.0 for task_id in task_ids}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("autoharness_service.runner.TerminalBenchRunner", FakeTerminalBenchRunner)

    runner = TerminalBenchRunnerAdapter(split="train")
    results = runner.run(
        ["task-a", "task-b"],
        model="gpt-5.4",
        sandbox_provider="daytona",
        requested_concurrency=8,
        run_id="run-123",
    )

    assert results == {"task-a": 1.0, "task-b": 1.0}
    assert created["agent_model"] == "gpt-5.4"
    assert created["split"] == "train"
    assert created["env_provider"] == "daytona"
    assert created["n_concurrent"] == 2
    assert Path(created["jobs_dir"]) == Path("workspace/service_runs/run-123/tbench_jobs")
    assert created["task_ids"] == ["task-a", "task-b"]
