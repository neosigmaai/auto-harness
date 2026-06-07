"""API of auto-harness benchmark.

Design (single process, runs on the host that has the repo + harbor + keys):

    POST /auto-harness -> start prepare.py and return a job id
    GET  /auto-harness/{job_id} -> return benchmark job status/result
    GET  /health  -> liveness
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sys
import threading
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TSV = os.path.join(REPO_ROOT, "workspace", "results.tsv")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import prepare


def _load_env_file(path: str) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_env_file(os.path.join(REPO_ROOT, ".env"))

app = FastAPI(title="auto-harness orchestrator", version="0.1.0")

_job_lock = threading.Lock()
_current_job: dict | None = None  # For local setup, only the latest job is retained.


class PrepareRequest(BaseModel):
    tasks: list[str]


def _normalize_tasks(tasks: list[str]) -> list[str]:
    normalized = [task.strip() for task in tasks if task and task.strip()]
    if not normalized:
        raise HTTPException(status_code=422, detail="tasks must include at least one task id")
    return normalized


def _reset_prepare_state() -> None:
    os.makedirs(os.path.join(REPO_ROOT, "workspace"), exist_ok=True)
    with open(RESULTS_TSV, "w") as f:
        f.write("iteration\tval_score\tcommit\tevals_passed\tevals_total\ttimestamp\n")

    split_file = os.path.join(REPO_ROOT, prepare.SPLIT_FILE)
    if os.path.exists(split_file):
        os.remove(split_file)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_active() -> bool:
    with _job_lock:
        return _current_job is not None and _current_job["status"] == "running"


def _set_job_finished(job_id: str, fields: dict) -> None:
    with _job_lock:
        if _current_job is None or _current_job["job_id"] != job_id:
            return
        _current_job.update(fields)
        _current_job["finished_at"] = _now_iso()


def _run_prepare_job(job_id: str, tasks: list[str]) -> None:
    original_cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        _reset_prepare_state()
        result = prepare.main(task_ids=tasks)
        if not result:
            _set_job_finished(
                job_id,
                {
                    "status": "failed",
                    "error": "prepare.py returned no result",
                },
            )
            return
        _set_job_finished(job_id, {"status": "completed", "result": result})
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code in (0, None):
            error = "prepare.py returned no result"
        else:
            error = f"prepare.py exited with {code}"
        _set_job_finished(job_id, {"status": "failed", "error": error})
    except Exception as exc:
        _set_job_finished(job_id, {"status": "failed", "error": f"prepare failed: {exc!r}"})
    finally:
        os.chdir(original_cwd)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "run_active": _run_active(),
    }


@app.post("/auto-harness")
def create_auto_harness(req: PrepareRequest) -> dict:
    global _current_job

    tasks = _normalize_tasks(req.tasks)

    with _job_lock:
        if _current_job is not None and _current_job["status"] == "running":
            raise HTTPException(status_code=409, detail="a prepare job is already active")

        job_id = str(uuid.uuid4())
        _current_job = {
            "job_id": job_id,
            "status": "running",
            "tasks": tasks,
            "started_at": _now_iso(),
        }
        job = dict(_current_job)

    thread = threading.Thread(target=_run_prepare_job, args=(job_id, tasks), daemon=True)
    thread.start()

    return dict(job)


@app.get("/auto-harness/{job_id}")
def get_auto_harness_job(job_id: str) -> dict:
    with _job_lock:
        if _current_job is None or _current_job["job_id"] != job_id:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(_current_job)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8800"))
    uvicorn.run(app, host=host, port=port)
