"""Turn a raw task result into a compact, LLM-ready failure dossier and a failure class.

Heuristic first (cheap, deterministic). Works for both simulated and real Harbor traces
(OpenAI-style message lists written by agent.py)."""
from __future__ import annotations

from typing import Any

from .executors.base import TaskResult

MAX_CMD = 240
MAX_OUT = 400
LAST_STEPS = 8


def _steps(trace: Any) -> list[dict]:
    """Collapse an OpenAI-style message list into [{command, output}] pairs."""
    if not isinstance(trace, list):
        return []
    steps, pending = [], {}
    for m in trace:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args = tc.get("function", {}).get("arguments", "")
                cmd = args
                try:
                    import json
                    cmd = json.loads(args).get("command", args)
                except Exception:
                    pass
                pending[tc.get("id")] = {"command": str(cmd)[:MAX_CMD], "output": ""}
        elif m.get("role") == "tool":
            s = pending.pop(m.get("tool_call_id"), {"command": "?", "output": ""})
            s["output"] = str(m.get("content") or "")[-MAX_OUT:]
            steps.append(s)
    return steps


def classify(r: TaskResult, steps: list[dict], max_steps: int = 80) -> str | None:
    if r.passed:
        return None
    if r.status == "error":
        return "infra_error" if (r.exit_reason or "").startswith("infra") else "agent_exception"
    if r.status == "timeout" or r.exit_reason == "agent_timeout":
        return "timeout"
    if r.failure_class:                       # executor already knows (simulated / harbor exception info)
        return r.failure_class
    cmds = [s["command"] for s in steps]
    if len(cmds) >= 6 and len(set(cmds[-6:])) <= 2:
        return "loop_detected"
    if r.n_steps is not None and r.n_steps >= max_steps:
        return "step_limit"
    if len(cmds) <= 3:
        return "gave_up_early"
    outs = " ".join(s["output"].lower() for s in steps[-4:])
    if "command not found" in outs or "no module named" in outs or "permission denied" in outs:
        return "env_setup_failure"
    return "wrong_output"


def build_digest(r: TaskResult, instruction: str | None = None, max_steps: int = 80) -> dict:
    steps = _steps(r.trace)
    fc = classify(r, steps, max_steps)
    final_msg = None
    if isinstance(r.trace, list):
        for m in reversed(r.trace):
            if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
                final_msg = str(m["content"])[:400]
                break
    summary = {
        "passed": r.passed, "failure_class": fc, "n_steps": r.n_steps, "duration_s": r.duration_s,
        "exit_reason": r.exit_reason,
    }
    text = (f"{'PASS' if r.passed else 'FAIL'} [{fc or '-'}] steps={r.n_steps} dur={r.duration_s}s; "
            f"test: {(r.test_stdout or '').strip().splitlines()[-1][:160] if r.test_stdout else 'n/a'}")
    return {
        "summary": text,
        "stats": summary,
        "instruction": (instruction or "")[:1200] or None,
        "last_steps": steps[-LAST_STEPS:],
        "final_message": final_msg,
        "test_stdout_tail": (r.test_stdout or "")[-1500:] or None,
    }
