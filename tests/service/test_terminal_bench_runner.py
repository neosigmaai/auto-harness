import json
import subprocess
from pathlib import Path

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
    assert any(call[:3] == ["harbor", "job", "resume"] for call in calls)
    assert any(
        str(job_dir) in call
        for call in calls
        if call[:3] == ["harbor", "job", "resume"]
    )
