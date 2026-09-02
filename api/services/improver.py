"""Improver: assembles the optimization prompt and proposes the next AgentSpec."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from api.agent_spec import AgentSpec
from api.job_store import IterationRecord

# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskOutcome:
    """One benchmark task's result, flattened for prompt rendering."""

    task_id: str
    status: str  # "passed" | "failed" | "error"
    reward: float | None
    remarks: str | None


@dataclass(frozen=True)
class EvaluationSummary:
    """The latest evaluation: aggregate score, per-task results, and trace text."""

    score: float
    tasks: list[TaskOutcome]
    traces: dict[str, str]  # task_id -> trace text (already read from artifacts)
    # Per-task movement vs the best iteration before this one (see scoring.task_movement).
    # A mean score cannot distinguish "fixed A, broke B" from "changed nothing", so
    # this is surfaced to the improver explicitly.
    fixed_tasks: list[str] = field(default_factory=list)
    regressed_tasks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Proposal:
    spec: AgentSpec
    rationale: str


class ImproverError(Exception):
    """Raised when the improver cannot produce a valid proposal."""


class Improver(Protocol):
    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal: ...


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #

SPEC_HEADER = "## CURRENT AGENT SPEC (JSON)"
HISTORY_HEADER = "## ITERATION HISTORY (oldest first)"
HISTORY_COLUMNS = "iteration | version | score | improved | changed_fields | rationale"
TASKS_COLUMNS = "task_id | status | reward | remarks"
FAILURES_HEADER = "## FAILURE DETAILS (worst tasks first)"

_TRACE_TAIL_CHARS = 4_000
_TRACE_TAIL_MESSAGES = 12
_MESSAGE_CHARS = 600


def _flat(text: str | None, limit: int = 200) -> str:
    """Collapse a value onto one length-capped line (tables must stay row-shaped)."""
    if text is None:
        return "-"
    one_line = " ".join(str(text).split())
    if not one_line:
        return "-"
    if len(one_line) > limit:
        one_line = one_line[: limit - 1] + "…"
    return one_line


def _fmt_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


def _fmt_flag(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _render_trace(text: str) -> str:
    """Render the tail of a trace: last N messages, each output truncated."""
    if not text or not text.strip():
        return "(no trace captured)"

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None

    rendered = text
    if isinstance(data, list):
        tail = data[-_TRACE_TAIL_MESSAGES:]
        lines: list[str] = []
        if len(data) > len(tail):
            lines.append(f"...[{len(data) - len(tail)} earlier messages omitted]...")
        for message in tail:
            if not isinstance(message, dict):
                lines.append(str(message)[:_MESSAGE_CHARS])
                continue
            role = str(message.get("role", "?"))
            content = message.get("content")
            if content is None:
                content = json.dumps(message.get("tool_calls") or "", default=str)
            content = str(content).replace("\r", "")
            if len(content) > _MESSAGE_CHARS:
                half = _MESSAGE_CHARS // 2
                content = content[:half] + " ...[output truncated]... " + content[-half:]
            lines.append(f"[{role}] {content}")
        rendered = "\n".join(lines)

    if len(rendered) > _TRACE_TAIL_CHARS:
        rendered = "...[trace truncated]...\n" + rendered[-_TRACE_TAIL_CHARS:]
    return rendered


def _spec_section(spec: AgentSpec) -> str:
    # No sort_keys: AgentSpec declares system_prompt first, so the prompt is the
    # first thing in the section and the last thing lost to truncation.
    return SPEC_HEADER + "\n" + json.dumps(spec.model_dump(), indent=2)


def _history_section(history: list[IterationRecord]) -> str:
    lines = [HISTORY_HEADER, HISTORY_COLUMNS]
    if not history:
        lines.append("(no prior iterations - this is the first proposal)")
    for record in history:
        lines.append(
            " | ".join(
                [
                    str(record.iteration),
                    str(record.version),
                    _fmt_score(record.score),
                    _fmt_flag(record.improved),
                    ",".join(record.changed_fields) if record.changed_fields else "-",
                    _flat(record.rationale),
                ]
            )
        )
    return "\n".join(lines)


def _movement_section(evaluation: EvaluationSummary) -> str:
    if not evaluation.fixed_tasks and not evaluation.regressed_tasks:
        return "PER-TASK MOVEMENT VS BEST\nNo per-task movement vs the best version."
    lines = ["PER-TASK MOVEMENT VS BEST"]
    if evaluation.fixed_tasks:
        lines.append(f"Improved: {', '.join(evaluation.fixed_tasks)}")
    if evaluation.regressed_tasks:
        lines.append(
            f"REGRESSED (your last change broke these): "
            f"{', '.join(evaluation.regressed_tasks)}"
        )
        lines.append(
            "Keep what fixed the improved tasks, but do not repeat whatever caused "
            "these regressions."
        )
    return "\n".join(lines)


def _tasks_section(evaluation: EvaluationSummary) -> str:
    lines = [f"## LATEST EVALUATION (score={evaluation.score:.4f})", TASKS_COLUMNS]
    if not evaluation.tasks:
        lines.append("(no task results)")
    for task in evaluation.tasks:
        lines.append(
            " | ".join(
                [
                    task.task_id,
                    task.status,
                    _fmt_score(task.reward),
                    _flat(task.remarks, 120),
                ]
            )
        )
    return "\n".join(lines)


def _failure_sort_key(task: TaskOutcome) -> tuple[float, int, str]:
    reward = 0.0 if task.reward is None else float(task.reward)
    return (reward, 0 if task.status == "error" else 1, task.task_id)


def _failure_blocks(evaluation: EvaluationSummary) -> list[str]:
    failing = [t for t in evaluation.tasks if t.status in ("failed", "error")]
    failing.sort(key=_failure_sort_key)
    blocks: list[str] = []
    for task in failing:
        trace = _render_trace(evaluation.traces.get(task.task_id, ""))
        blocks.append(
            f"### {task.task_id} - status={task.status} reward={_fmt_score(task.reward)}\n"
            f"remarks: {_flat(task.remarks, 300)}\n"
            f"trace tail:\n{trace}"
        )
    return blocks


def build_context(
    *,
    spec: AgentSpec,
    evaluation: EvaluationSummary,
    history: list[IterationRecord],
    budget: int,
) -> str:
    """
    Assemble the improver prompt body within a hard character budget.

    Order: current spec (always) -> iteration history table (always) -> per-task
    movement vs the best version (always) -> per-task result table (always) ->
    failure details, worst task first, appended only while the running total
    stays inside ``budget``. The result is finally truncated to ``budget``
    characters, so the returned length is never larger than the budget even
    when the mandatory prefix alone overflows it.
    """
    parts = [
        _spec_section(spec),
        _history_section(list(history)),
        _movement_section(evaluation),
        _tasks_section(evaluation),
    ]
    # +2 per part accounts for the "\n\n" separators (a 2-char overestimate).
    running = sum(len(part) + 2 for part in parts)

    blocks = _failure_blocks(evaluation)
    if blocks:
        running += len(FAILURES_HEADER) + 2
        kept: list[str] = []
        for block in blocks:
            if running + len(block) + 2 > budget:
                break
            kept.append(block)
            running += len(block) + 2
        if kept:
            parts.append(FAILURES_HEADER)
            parts.extend(kept)

    return "\n\n".join(parts)[: max(budget, 0)]
