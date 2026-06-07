"""Worker API for running Cursor coding-agent instructions inside the sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import textwrap
import threading
import tarfile
import uuid

from cursor_sdk import CursorAgentError
from e2b import Sandbox
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_FILE = os.path.join(REPO_ROOT, "workspace", "coding_agent_result.json")
E2B_REPO_ROOT = "/tmp/auto-harness"
E2B_ARCHIVE_PATH = "/tmp/auto-harness.tar.gz"
E2B_RUNNER_PATH = "/tmp/run_cursor_agent.py"
E2B_COMMAND_TIMEOUT = int(os.environ.get("E2B_COMMAND_TIMEOUT", "3600"))
E2B_SANDBOX_TIMEOUT = int(os.environ.get("E2B_SANDBOX_TIMEOUT", "3600"))
REPO_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tau2_data",
    "tbench_data",
    "workspace/tbench_jobs",
    "workspace/traces",
    "workspace/coding_agent_result.json",
)

app = FastAPI(title="auto-harness worker", version="0.1.0")

_job_lock = threading.Lock()
_current_job: dict | None = None


class CodingAgentRequest(BaseModel):
    instruction: str
    model: str | None = None


@dataclass
class E2BAgentRun:
    final: object
    meta: dict

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


def _read_result_file() -> dict | list | None:
    try:
        with open(RESULT_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _remove_stale_result_file() -> None:
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    try:
        os.remove(RESULT_FILE)
    except FileNotFoundError:
        pass


def _is_excluded(path: str) -> bool:
    return any(path == exclude or path.startswith(f"{exclude}/") for exclude in REPO_EXCLUDES)


def _build_repo_archive() -> bytes:
    buffer = io.BytesIO()
    root = Path(REPO_ROOT)
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in root.rglob("*"):
            rel = path.relative_to(root).as_posix()
            if _is_excluded(rel):
                continue
            tar.add(path, arcname=rel, recursive=False)
    return buffer.getvalue()


def _sandbox_envs(model: str | None) -> dict[str, str]:
    keys = (
        "CURSOR_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "E2B_API_KEY",
        "DAYTONA_API_KEY",
        "AGENT_MODEL",
    )
    envs = {key: os.environ[key] for key in keys if os.getenv(key)}
    envs["CURSOR_AGENT_MODEL"] = model or os.getenv("CURSOR_AGENT_MODEL", "composer-2.5")
    return envs


def _command_failed(result: object) -> bool:
    for attr in ("exit_code", "return_code"):
        code = getattr(result, attr, None)
        if isinstance(code, int):
            return code != 0
    return bool(getattr(result, "error", None))


def _command_error(prefix: str, result: object) -> RuntimeError:
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    error = getattr(result, "error", "")
    return RuntimeError(f"{prefix} failed: {error or stderr or stdout}")


def _run_sandbox_command(sandbox: Sandbox, cmd: str, *, cwd: str | None = None) -> object:
    result = sandbox.commands.run(cmd, cwd=cwd, timeout=E2B_COMMAND_TIMEOUT)
    if _command_failed(result):
        raise _command_error(cmd, result)
    return result


def _upload_repo_to_e2b(sandbox: Sandbox) -> None:
    sandbox.files.write(E2B_ARCHIVE_PATH, _build_repo_archive())
    _run_sandbox_command(
        sandbox,
        f"rm -rf {E2B_REPO_ROOT} && mkdir -p {E2B_REPO_ROOT} && "
        f"tar -xzf {E2B_ARCHIVE_PATH} -C {E2B_REPO_ROOT}",
    )


def _cursor_runner_script(instruction: str, model: str | None) -> str:
    return textwrap.dedent(
        f"""
        import os
        import sys

        from cursor_sdk import Agent, LocalAgentOptions

        instruction = {instruction!r}
        model = {model!r} or os.getenv("CURSOR_AGENT_MODEL", "composer-2.5")

        with Agent.create(
            api_key=os.environ["CURSOR_API_KEY"],
            model=model,
            local=LocalAgentOptions(cwd="{E2B_REPO_ROOT}"),
        ) as agent:
            run = agent.send(instruction)
            result = run.wait()
            print("agent_id=" + str(getattr(agent, "agent_id", "")))
            print("run_id=" + str(getattr(run, "id", "")))
            if getattr(result, "status", None) == "error":
                sys.exit(2)
        """
    ).strip()


def _bootstrap_e2b_repo(sandbox: Sandbox) -> None:
    _run_sandbox_command(sandbox, "python3 -m pip install --no-cache-dir uv")
    _run_sandbox_command(sandbox, "uv tool install harbor")
    _run_sandbox_command(sandbox, "uv sync --no-dev --no-install-project", cwd=E2B_REPO_ROOT)


def _download_result_from_e2b(sandbox: Sandbox) -> None:
    result = sandbox.files.read(f"{E2B_REPO_ROOT}/workspace/coding_agent_result.json")
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    mode = "wb" if isinstance(result, (bytes, bytearray)) else "w"
    with open(RESULT_FILE, mode) as f:
        f.write(result)


def _fail_job(job_id: str, error: str, **fields) -> None:
    _set_job_finished(job_id, {"status": "failed", "error": error, **fields})


def _update_running_job(job_id: str, fields: dict) -> None:
    with _job_lock:
        if _current_job is not None and _current_job["job_id"] == job_id:
            _current_job.update(fields)


def _parse_agent_meta(result: object) -> dict:
    meta = {"agent_id": None, "run_id": None}
    for line in str(getattr(result, "stdout", "")).splitlines():
        if line.startswith("agent_id="):
            meta["agent_id"] = line.removeprefix("agent_id=") or None
        elif line.startswith("run_id="):
            meta["run_id"] = line.removeprefix("run_id=") or None
    return meta


def _run_cursor_agent(instruction: str, model: str | None, job_id: str) -> E2BAgentRun:
    with Sandbox.create(timeout=E2B_SANDBOX_TIMEOUT, envs=_sandbox_envs(model)) as sandbox:
        sandbox_meta = {"sandbox_id": getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)}
        _update_running_job(job_id, sandbox_meta)

        _upload_repo_to_e2b(sandbox)
        _bootstrap_e2b_repo(sandbox)
        sandbox.files.write(E2B_RUNNER_PATH, _cursor_runner_script(instruction, model))

        result = _run_sandbox_command(
            sandbox,
            f"{E2B_REPO_ROOT}/.venv/bin/python {E2B_RUNNER_PATH}",
            cwd=E2B_REPO_ROOT,
        )
        agent_meta = {**sandbox_meta, **_parse_agent_meta(result)}
        _update_running_job(job_id, agent_meta)
        _download_result_from_e2b(sandbox)
        return E2BAgentRun(final=result, meta=agent_meta)


def _finish_from_result_file(job_id: str, agent_meta: dict) -> None:
    result = _read_result_file()

    if result is None:
        _fail_job(job_id, "agent did not produce a valid result file", **agent_meta)
        return

    if isinstance(result, dict) and "error" in result:
        _fail_job(job_id, str(result["error"]), **agent_meta, result=result)
        return

    _set_job_finished(job_id, {"status": "completed", "result": result, **agent_meta})


def _run_agent_job(job_id: str, instruction: str, model: str | None) -> None:
    _remove_stale_result_file()
    agent_meta = {"agent_id": None, "run_id": None}

    try:
        agent_run = _run_cursor_agent(instruction, model, job_id)
        agent_meta = agent_run.meta
        if _command_failed(agent_run.final):
            run_id = agent_meta["run_id"]
            _fail_job(job_id, f"run failed: {run_id}", **agent_meta)
            return

        _finish_from_result_file(job_id, agent_meta)
    except KeyError:
        _fail_job(job_id, "CURSOR_API_KEY is required for coding agent jobs")
    except CursorAgentError as exc:
        _fail_job(job_id, f"agent startup failed: {exc!r}", **agent_meta)
    except Exception as exc:
        _fail_job(job_id, f"coding agent failed: {exc!r}", **agent_meta)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "run_active": _run_active(),
    }


@app.post("/coding_agent")
def create_coding_agent_job(req: CodingAgentRequest) -> dict:
    global _current_job

    instruction = req.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="instruction must not be empty")

    with _job_lock:
        if _current_job is not None and _current_job["status"] == "running":
            raise HTTPException(status_code=409, detail="a coding agent job is already active")

        job_id = str(uuid.uuid4())
        _current_job = {
            "job_id": job_id,
            "status": "running",
            "instruction": instruction,
            "started_at": _now_iso(),
        }
        if req.model:
            _current_job["model"] = req.model
        job = dict(_current_job)

    thread = threading.Thread(
        target=_run_agent_job,
        args=(job_id, instruction, req.model),
        daemon=True,
    )
    thread.start()

    return job


@app.get("/coding_agent/{job_id}")
def get_coding_agent_job(job_id: str) -> dict:
    with _job_lock:
        if _current_job is None or _current_job["job_id"] != job_id:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(_current_job)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("WORKER_PORT", "8810"))
    uvicorn.run(app, host=host, port=port)
