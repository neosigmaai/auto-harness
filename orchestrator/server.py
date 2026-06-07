"""API of auto-harness benchmark.

Design (single process, runs on the host that has the repo + harbor + keys):

    POST /auto-harness -> run prepare.py, queue Cursor agent, and return benchmark results
    GET  /health  -> liveness + active job status
"""

from __future__ import annotations

import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TSV = os.path.join(REPO_ROOT, "workspace", "results.tsv")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import prepare
from orchestrator import agent_runner


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

_ACTIVE_LOCK = threading.Lock()  # Requests share this repo checkout and workspace files.


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


def _run_active() -> bool:
    """True if prepare.py is currently running."""
    acquired = _ACTIVE_LOCK.acquire(blocking=False)
    if acquired:
        _ACTIVE_LOCK.release()
        return False
    return True


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "run_active": _run_active(),
        "agent": agent_runner.get_state_dict(),
    }


@app.post("/auto-harness")
def create_auto_harness(req: PrepareRequest) -> dict:
    tasks = _normalize_tasks(req.tasks)

    if not _ACTIVE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a prepare job is already active")

    original_cwd = os.getcwd()
    try:
        if agent_runner.is_active():
            raise HTTPException(status_code=409, detail="an optimization agent is already running")

        os.chdir(REPO_ROOT)
        _reset_prepare_state()
        result = prepare.main(task_ids=tasks)
        if not result:
            raise HTTPException(status_code=500, detail="prepare.py returned no result")
        agent_info = agent_runner.start_optimization_agent(REPO_ROOT)
        return {**result, "agent": agent_info}
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code in (0, None):
            raise HTTPException(status_code=500, detail="prepare.py returned no result") from exc
        raise HTTPException(status_code=500, detail=f"prepare.py exited with {code}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prepare failed: {exc!r}") from exc
    finally:
        os.chdir(original_cwd)
        _ACTIVE_LOCK.release()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8800"))
    uvicorn.run(app, host=host, port=port)
