"""Public orchestration API for auto-harness.

The orchestrator stays outside the sandbox. It accepts public requests, builds
instructions for the worker coding-agent API, and proxies job status back to
clients.
"""

from __future__ import annotations

import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKER_BASE_URL = os.environ.get("WORKER_BASE_URL", "http://worker:8810").rstrip("/")
WORKER_TIMEOUT = float(os.environ.get("WORKER_HTTP_TIMEOUT", "10"))

app = FastAPI(title="auto-harness orchestrator", version="0.1.0")


class PrepareRequest(BaseModel):
    tasks: list[str]


def _normalize_tasks(tasks: list[str]) -> list[str]:
    normalized = [task.strip() for task in tasks if task and task.strip()]
    if not normalized:
        raise HTTPException(status_code=422, detail="tasks must include at least one task id")
    return normalized


def _proxy_response(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="worker returned non-JSON response")

    if resp.status_code >= 400:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return body


def _request_worker(method: str, path: str, **kwargs) -> dict:
    try:
        with httpx.Client(timeout=WORKER_TIMEOUT) as client:
            resp = client.request(method, f"{WORKER_BASE_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"worker unavailable: {exc!r}")
    return _proxy_response(resp)


def _build_prepare_instruction(tasks: list[str]) -> str:
    task_ids_json = json.dumps(tasks)
    return f"""You are operating inside the auto-harness repo.

    Run prepare.py for these task IDs:
    {task_ids_json}

    Write the final response as valid JSON to workspace/coding_agent_result.json.
    Expected success shape:
    {{
    "results": [{{"task_id": "string", "status": "passed | failed"}}],
    "summary": {{"val_score": number, "passed": number, "failed": number, "total": number}}
    }}

    If the run fails, write a JSON object with an "error" field to the same file.
"""


def _parse_auto_harness_response(worker_job: dict) -> dict:
    body = {
        "job_id": worker_job.get("job_id"),
        "status": worker_job.get("status"),
        "started_at": worker_job.get("started_at"),
    }

    if "finished_at" in worker_job:
        body["finished_at"] = worker_job["finished_at"]
    if "error" in worker_job:
        body["error"] = worker_job["error"]

    result = worker_job.get("result")
    if isinstance(result, dict):
        body["results"] = result.get("results", [])
        body["summary"] = result.get("summary", {})

    return body


@app.get("/health")
def health() -> dict:
    worker_health = _request_worker("GET", "/health")
    return {"status": "ok", "worker": worker_health}


@app.post("/auto-harness")
def create_auto_harness(req: PrepareRequest) -> dict:
    tasks = _normalize_tasks(req.tasks)
    instruction = _build_prepare_instruction(tasks)
    worker_job = _request_worker("POST", "/coding_agent", json={"instruction": instruction})

    return _parse_auto_harness_response(worker_job)


@app.get("/auto-harness/{job_id}")
def get_auto_harness_job(job_id: str) -> dict:
    worker_job = _request_worker("GET", f"/coding_agent/{job_id}")
    return _parse_auto_harness_response(worker_job)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8800"))
    uvicorn.run(app, host=host, port=port)
