import json
import subprocess
from pathlib import Path

import pytest

from benchmark import TerminalBenchRunner


def test_terminal_bench_runner_resumes_pending_harbor_job(monkeypatch, tmp_path):
    calls = []
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "2026-07-02__14-31-29"
    trial_dir = job_dir / "multi-source-data-merger__abc123"

    def write_pending_job():
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.log").write_text("")
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "finished_at": None,
                    "n_total_trials": 1,
                    "stats": {
                        "n_completed_trials": 0,
                        "n_errored_trials": 0,
                        "n_running_trials": 0,
                        "n_pending_trials": 1,
                        "n_cancelled_trials": 0,
                    },
                }
            )
        )

    def write_completed_trial():
        (job_dir / "job.log").write_text("job log")
        (job_dir / "lock.json").write_text("{}")
        (job_dir / "config.json").write_text("{}")
        (trial_dir / "config.json").write_text("{}")
        (trial_dir / "trial.log").write_text("trial log")
        (trial_dir / "agent").mkdir()
        (trial_dir / "agent" / "trace.json").write_text('{"messages": []}')
        (trial_dir / "agent" / "trace.jsonl").write_text('{"event": "tool"}\n')
        (trial_dir / "verifier").mkdir()
        (trial_dir / "verifier" / "reward.txt").write_text("1.0")
        (trial_dir / "verifier" / "test-stdout.txt").write_text("ok")
        (trial_dir / "verifier" / "ctrf.json").write_text("{}")
        (trial_dir / "artifacts").mkdir()
        (trial_dir / "artifacts" / "manifest.json").write_text("{}")
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "multi-source-data-merger",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "finished_at": "2026-07-02T14:32:29Z",
                    "n_total_trials": 1,
                    "stats": {
                        "n_completed_trials": 1,
                        "n_errored_trials": 0,
                        "n_running_trials": 0,
                        "n_pending_trials": 0,
                        "n_cancelled_trials": 0,
                    },
                }
            )
        )

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["harbor", "run"]:
            write_pending_job()
        elif cmd[:3] == ["harbor", "job", "resume"]:
            write_completed_trial()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = TerminalBenchRunner(
        agent_model="gpt-5.4",
        split="test",
        env_provider="daytona",
        n_concurrent=1,
        jobs_dir=str(jobs_dir),
        per_task_timeout=1,
    )

    results = runner.run(task_ids=["multi-source-data-merger"])

    assert results == {"multi-source-data-merger": 1.0}
    artifacts = runner.last_artifacts["multi-source-data-merger"]
    assert artifacts["jsonl_files"] == [
        str(trial_dir / "agent" / "trace.jsonl"),
    ]
    assert str(trial_dir / "agent" / "trace.jsonl") in artifacts["all_files"]
    assert {"all_files", "json_files", "jsonl_files", "log_files"}.issubset(artifacts)
    assert {key: artifacts[key] for key in expected_artifact_keys()} == {
        "artifact_manifest": str(trial_dir / "artifacts" / "manifest.json"),
        "job_config": str(job_dir / "config.json"),
        "job_lock": str(job_dir / "lock.json"),
        "job_log": str(job_dir / "job.log"),
        "job_result": str(job_dir / "result.json"),
        "trace": str(trial_dir / "agent" / "trace.json"),
        "trial_config": str(trial_dir / "config.json"),
        "trial_log": str(trial_dir / "trial.log"),
        "trial_result": str(trial_dir / "result.json"),
        "verifier_ctrf": str(trial_dir / "verifier" / "ctrf.json"),
        "verifier_reward": str(trial_dir / "verifier" / "reward.txt"),
        "verifier_stdout": str(trial_dir / "verifier" / "test-stdout.txt"),
    }
    assert any(call[:3] == ["harbor", "job", "resume"] for call in calls)
    assert any(
        str(job_dir) in call
        for call in calls
        if call[:3] == ["harbor", "job", "resume"]
    )


def test_terminal_bench_runner_reports_resume_stderr(monkeypatch, tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "2026-07-02__14-43-33"
    trial_dir = job_dir / "multi-source-data-merger__abc123"

    def write_pending_job():
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.log").write_text("")
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "finished_at": None,
                    "n_total_trials": 1,
                    "stats": {
                        "n_completed_trials": 0,
                        "n_errored_trials": 0,
                        "n_running_trials": 0,
                        "n_pending_trials": 1,
                        "n_cancelled_trials": 0,
                    },
                }
            )
        )

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["harbor", "run"]:
            write_pending_job()
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["harbor", "job", "resume"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="Module 'agent.agent' has no class 'HarnessAgent'",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = TerminalBenchRunner(
        agent_model="gpt-5.4",
        split="test",
        env_provider="daytona",
        n_concurrent=1,
        jobs_dir=str(jobs_dir),
        per_task_timeout=1,
    )

    with pytest.raises(RuntimeError, match="HarnessAgent"):
        runner.run(task_ids=["multi-source-data-merger"])


def expected_artifact_keys():
    return {
        "artifact_manifest",
        "job_config",
        "job_lock",
        "job_log",
        "job_result",
        "trace",
        "trial_config",
        "trial_log",
        "trial_result",
        "verifier_ctrf",
        "verifier_reward",
        "verifier_stdout",
    }
