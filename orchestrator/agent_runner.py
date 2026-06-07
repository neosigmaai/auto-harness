"""Background Cursor SDK runner for the auto-harness optimization loop."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

AGENT_PROMPT = "Refer to PROGRAM.md and perform the tasks"
ACTIVE_STATUSES = {"queued", "running"}


@dataclass
class AgentState:
    status: str = "idle"
    run_id: str | None = None
    prompt: str = AGENT_PROMPT
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


_STATE_LOCK = threading.Lock()
_STATE = AgentState()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_state(**updates: object) -> dict:
    with _STATE_LOCK:
        for key, value in updates.items():
            setattr(_STATE, key, value)
        return asdict(_STATE)


def get_state_dict() -> dict:
    with _STATE_LOCK:
        return asdict(_STATE)


def is_active() -> bool:
    return get_state_dict()["status"] in ACTIVE_STATUSES


def start_optimization_agent(repo_root: str) -> dict:
    """Start the PROGRAM.md optimization loop in a background thread."""
    with _STATE_LOCK:
        if _STATE.status in ACTIVE_STATUSES:
            return asdict(_STATE)

        api_key = os.environ.get("CURSOR_API_KEY")
        if not api_key:
            _STATE.status = "failed_to_start"
            _STATE.run_id = None
            _STATE.prompt = AGENT_PROMPT
            _STATE.error = "CURSOR_API_KEY is not set"
            _STATE.started_at = None
            _STATE.finished_at = _now_iso()
            return asdict(_STATE)

        _STATE.status = "queued"
        _STATE.run_id = None
        _STATE.prompt = AGENT_PROMPT
        _STATE.error = None
        _STATE.started_at = _now_iso()
        _STATE.finished_at = None
        snapshot = asdict(_STATE)

    thread = threading.Thread(target=_run_agent, args=(repo_root, api_key), daemon=True)
    thread.start()
    return snapshot


def _run_agent(repo_root: str, api_key: str) -> None:
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except Exception as exc:
        _set_state(status="failed_to_start", error=f"failed to import cursor_sdk: {exc!r}", finished_at=_now_iso())
        return

    try:
        model = os.environ.get("ORCHESTRATOR_AGENT_MODEL", "composer-2.5")
        with Agent.create(
            AgentOptions(
                api_key=api_key,
                model=model,
                local=LocalAgentOptions(cwd=repo_root),
            )
        ) as agent:
            _set_state(status="running")
            run = agent.send(AGENT_PROMPT)
            _set_state(run_id=run.id)
            result = run.wait()
            _set_state(status=str(result.status), finished_at=_now_iso())
    except CursorAgentError as exc:
        _set_state(status="error", error=str(exc), finished_at=_now_iso())
    except Exception as exc:
        _set_state(status="error", error=repr(exc), finished_at=_now_iso())
