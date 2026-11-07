"""Turn a raw task result into a compact, LLM-ready failure dossier and a failure class.

Heuristic first (cheap, deterministic). Works for both simulated and real Harbor traces
(OpenAI-style message lists written by agent.py)."""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .executors.base import TaskResult

MAX_CMD = 800
MAX_OUT = 800
MAX_INSTRUCTION = 8192
LAST_STEPS = 8
MAX_OBSERVATIONS = 6
TRUNCATION_MARKER = re.compile(r"\[(?:(\d{1,12})\s+chars\s+)?truncated\]", re.IGNORECASE)


def head_tail(text: str, limit: int) -> str:
    """Keep command endings and explicitly mark evidence that was omitted."""
    if len(text) <= limit:
        return text
    marker = f"\n... [{len(text)} original chars; middle omitted] ...\n"
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def dossier_from_row(row: dict) -> dict:
    """Rebuild a bounded dossier from persisted evidence, never run its code."""
    from dataclasses import fields
    data = {field.name: row[field.name] for field in fields(TaskResult) if field.name in row}
    trace = data.get("trace")
    has_task_message = isinstance(trace, list) and any(isinstance(message, dict)
        and message.get("role") == "user" and isinstance(message.get("content"), str)
        and message["content"].startswith("Task:\n") for message in trace)
    legacy = row.get("digest") or {}
    legacy_fallback = not data.get("instruction") and not has_task_message
    if legacy_fallback:
        data["instruction"] = legacy.get("instruction")
    result = build_digest(TaskResult(**data))
    if legacy_fallback and (legacy.get("instruction_complete") is False or
                           "[instruction excerpt]" in (legacy.get("instruction") or "")):
        result["instruction_complete"] = False
    return result


def _steps(trace: Any) -> list[dict]:
    """Collapse an OpenAI-style message list into [{command, output}] pairs."""
    if not isinstance(trace, list):
        return []
    steps, pending = [], {}
    for m in trace:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"] if isinstance(m["tool_calls"], list) else []
            for tc in calls:
                if not isinstance(tc, dict):
                    continue
                function = tc.get("function") or {}
                args = function.get("arguments", "") if isinstance(function, dict) else ""
                cmd = args
                try:
                    import json
                    cmd = json.loads(args).get("command", args)
                except Exception:
                    pass
                pending[str(tc.get("id"))] = {"command": str(cmd), "output": ""}
        elif m.get("role") == "tool":
            s = pending.pop(str(m.get("tool_call_id")), {"command": "?", "output": ""})
            s["output"] = str(m.get("content") or "")
            steps.append(s)
    return steps


def structural_observations(trace: Any, steps: list[dict]) -> dict:
    """Bounded observations from available logs, not claims about failure causes.

    Inspect before making the 400-character dossier preview: the original
    harness's omission marker normally sits in the middle of a tool result.
    Repeated commands can be intentional checks; a zero marker count means only
    that no marker was observed in this available, potentially partial trace.
    """
    available = isinstance(trace, list)
    truncations = []
    for number, step in enumerate(steps, 1):
        matches = list(TRUNCATION_MARKER.finditer(step["output"]))
        if matches:
            omitted = [int(match.group(1)) for match in matches if match.group(1) is not None]
            truncations.append({"step": number, "command_excerpt": step["command"][:MAX_CMD],
                "visible_output_chars": len(step["output"]),
                "reported_omitted_chars": sum(omitted) if omitted else None,
                "markers_observed": len(matches)})
    commands = [step["command"] for step in steps if step["command"] != "?"]
    counts = Counter(commands)
    repeated = [{"command_excerpt": command[:MAX_CMD], "count": count,
                 "step_indices": [i for i, step in enumerate(steps, 1) if step["command"] == command][:8]}
                for command, count in counts.most_common() if count > 1]
    return {"scope": "available_untrusted_trace", "evidence_available": available,
        "coverage_guaranteed": False, "causal_failure_claim": False,
        "observed_tool_results": len(steps),
        "outputs_with_truncation_marker": len(truncations) if available else None,
        "truncation_examples": truncations[:MAX_OBSERVATIONS],
        "repeated_exact_command_groups": len(repeated) if available else None,
        "repeated_command_examples": repeated[:MAX_OBSERVATIONS],
        "interpretation": "Markers and exact repeats are observations, not proof of harmful truncation or a loop; absent markers do not prove complete logs."}


def classify(r: TaskResult, steps: list[dict], max_steps: int = 80) -> str | None:
    if r.passed:
        return None
    if r.status == "error":
        if r.failure_class:
            return r.failure_class
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
    instruction = instruction or r.instruction
    if not instruction and isinstance(r.trace, list):
        # Fallback evidence only: trace text is untrusted data, never executable
        # instructions for the optimizer. The trusted task manifest wins above.
        for message in r.trace:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.startswith("Task:\n"):
                    instruction = content[len("Task:\n"):]
                break
    instruction_chars = len(instruction) if instruction else 0
    if instruction:
        instruction = head_tail(instruction, MAX_INSTRUCTION)
    steps = _steps(r.trace)
    fc = classify(r, steps, max_steps)
    final_msg = None
    if isinstance(r.trace, list):
        for m in reversed(r.trace):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
                final_msg = str(m["content"])[:400]
                break
    summary = {
        "passed": r.passed, "failure_class": fc, "n_steps": r.n_steps, "duration_s": r.duration_s,
        "exit_reason": r.exit_reason,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        "model_cost_usd": r.model_cost_usd, "cost_known": r.cost_known,
    }
    output_lines = (r.test_stdout or "").strip().splitlines()
    test_output = r.test_stdout or ""
    failure_start = re.search(r"(?m)^.*(?:={3,} FAILURES|Traceback \(most recent call last\)|E\s+\w*(?:Error|Exception)).*$", test_output)
    failure_excerpt = head_tail(test_output[failure_start.start():], 2400) if failure_start else None
    text = (f"{'PASS' if r.passed else 'FAIL'} [{fc or '-'}] steps={r.n_steps} dur={r.duration_s}s; "
            f"test: {output_lines[-1][:160] if output_lines else 'n/a'}")
    return {
        "summary": text,
        "stats": summary,
        "instruction": instruction or None,
        "instruction_original_chars": instruction_chars,
        "instruction_complete": instruction_chars <= MAX_INSTRUCTION,
        "last_steps": [{"command": head_tail(step["command"], MAX_CMD),
                        "output": head_tail(step["output"], MAX_OUT),
                        "command_original_chars": len(step["command"]),
                        "output_original_chars": len(step["output"])}
                       for step in steps[-LAST_STEPS:]],
        "structural_observations": structural_observations(r.trace, steps),
        "final_message": final_msg,
        "test_stdout_tail": (r.test_stdout or "")[-1500:] or None,
        "test_failure_excerpt": failure_excerpt,
    }
