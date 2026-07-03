from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_smoke_module():
    script_path = Path(__file__).parents[2] / "scripts" / "optimize_smoke_check.py"
    spec = importlib.util.spec_from_file_location("optimize_smoke_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_guided_order_echoes_all_major_steps():
    smoke = load_smoke_module()

    output = smoke.format_guided_order(["break-filter-js-from-html"])

    assert "1 Submit" in output
    assert "2 Baseline" in output
    assert "3 Collect" in output
    assert "4 Optimize" in output
    assert "5 Rerun" in output
    assert "6 Decide" in output
    assert "7 FinalSummary" in output
    assert "tasks=break-filter-js-from-html" in output


def test_format_step_status_line_includes_phase_tasks_and_optimization_state():
    smoke = load_smoke_module()

    line = smoke.format_step_status_line(
        {
            "status": "running",
            "progress": {
                "total": 1,
                "queued": 0,
                "running": 1,
                "completed": 0,
            },
            "score": None,
            "task_results": [
                {
                    "task_id": "break-filter-js-from-html",
                    "status": "running",
                }
            ],
        },
        {
            "iterations": [
                {"iteration": 0, "status": "completed", "score": 0.0},
                {
                    "iteration": 1,
                    "status": "rerun_running",
                    "accepted": None,
                    "score": None,
                },
            ]
        },
        elapsed_sec=12.3,
    )

    assert "phase=5_rerun_after_patch" in line
    assert "run=running" in line
    assert "progress=0/1" in line
    assert "tasks=[break-filter-js-from-html=running]" in line
    assert "iterations=[0:completed,1:rerun_running]" in line
    assert "optimize=rerun_running accepted=None" in line


def test_format_step_status_line_reports_rejected_final_state():
    smoke = load_smoke_module()

    line = smoke.format_step_status_line(
        {
            "status": "succeeded",
            "progress": {
                "total": 1,
                "queued": 0,
                "running": 0,
                "completed": 1,
            },
            "score": 0.0,
            "task_results": [
                {
                    "task_id": "break-filter-js-from-html",
                    "status": "failed",
                }
            ],
        },
        {
            "iterations": [
                {"iteration": 0, "status": "completed", "score": 0.0},
                {
                    "iteration": 1,
                    "status": "patch_rejected",
                    "accepted": False,
                    "score": 0.0,
                },
            ]
        },
        elapsed_sec=80.7,
    )

    assert "phase=7_final_summary" in line
    assert "run=succeeded" in line
    assert "score=0.0" in line
    assert "optimize=patch_rejected accepted=False" in line


def test_build_attempt_timelines_reads_harbor_daytona_artifacts(tmp_path):
    smoke = load_smoke_module()
    baseline_result = _write_attempt_artifacts(
        tmp_path,
        attempt="baseline",
        task_id="break-filter-js-from-html",
        started_at="2026-07-03T00:00:01Z",
        finished_at="2026-07-03T00:00:31Z",
    )
    rerun_result = _write_attempt_artifacts(
        tmp_path,
        attempt="proposal-1",
        task_id="break-filter-js-from-html",
        started_at="2026-07-03T00:01:01Z",
        finished_at="2026-07-03T00:01:26Z",
    )

    timelines = smoke.build_attempt_timelines(
        {
            "iterations": [
                {
                    "iteration": 1,
                    "proposal": json.dumps(
                        {
                            "baseline_tasks": [
                                {
                                    "task_id": "break-filter-js-from-html",
                                    "result_path": str(baseline_result),
                                }
                            ],
                            "rerun_tasks": [
                                {
                                    "task_id": "break-filter-js-from-html",
                                    "result_path": str(rerun_result),
                                }
                            ],
                        }
                    ),
                }
            ]
        }
    )

    assert timelines[0]["attempt"] == "baseline"
    assert timelines[0]["task_id"] == "break-filter-js-from-html"
    assert timelines[0]["daytona_strategy"] == "_DaytonaDirect"
    assert timelines[0]["daytona_sandbox_create_signal"] == (
        "Creating new AsyncDaytona client"
    )
    assert timelines[0]["daytona_sandbox_finished_at"] == "2026-07-03T00:00:31Z"
    assert timelines[1]["attempt"] == "proposal-1"

    output = smoke.format_attempt_timeline(timelines)

    assert "[attempt-timeline] baseline task=break-filter-js-from-html" in output
    assert "harbor_job_started_at=2026-07-03T00:00:01Z" in output
    assert "daytona_strategy=_DaytonaDirect" in output
    assert "proposal-1 task=break-filter-js-from-html" in output


def _write_attempt_artifacts(
    tmp_path: Path,
    *,
    attempt: str,
    task_id: str,
    started_at: str,
    finished_at: str,
) -> Path:
    job_dir = tmp_path / attempt / "2026-07-02__17-11-51"
    trial_dir = job_dir / f"{task_id}__abc123"
    trial_dir.mkdir(parents=True)
    (job_dir / "job.log").write_text(
        "\n".join(
            [
                "Selected strategy: _DaytonaDirect",
                "Creating new AsyncDaytona client",
                "Using prebuilt image: alexgshaw/break-filter-js-from-html:20251031",
            ]
        ),
        encoding="utf-8",
    )
    (job_dir / "result.json").write_text(
        json.dumps({"started_at": started_at, "finished_at": finished_at}),
        encoding="utf-8",
    )
    trial_result = trial_dir / "result.json"
    trial_result.write_text(
        json.dumps(
            {
                "task_name": task_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    return trial_result
