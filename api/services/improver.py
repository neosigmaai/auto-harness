"""Improver: assembles the optimization prompt and proposes the next AgentSpec."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from api.agent_spec import BASELINE_SYSTEM_PROMPT, AgentSpec
from api.config import BenchmarkConfig, load_config
from api.job_store import IterationRecord

logger = logging.getLogger(__name__)

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
HISTORY_COLUMNS = (
    "iteration | version | score | improved | prompt_len | changed_fields | rationale"
)
TASK_STATEMENTS_HEADER = "## TASK STATEMENTS (failing tasks)"
TASKS_COLUMNS = "task_id | status | reward | remarks"
FAILURES_HEADER = "## FAILURE DETAILS (worst tasks first)"
MOVEMENT_HEADER = "## PER-TASK MOVEMENT VS BEST"

_TRACE_TAIL_CHARS = 4_000
_TRACE_TAIL_MESSAGES = 12
_MESSAGE_CHARS = 600
_TASK_STATEMENT_PREFIX = "Task:\n"
# Soft ceiling on proposed system_prompt length relative to the baseline prompt.
_MAX_PROMPT_LEN_MULT = 1.5


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


def extract_task_statement(trace_text: str) -> str | None:
    """Pull the task instruction from the head of an agent trace.

    HarnessAgent writes the instruction as the first user message with a
    ``Task:\\n`` prefix. We prefer that form; otherwise we fall back to the
    first non-empty user message. Returns ``None`` when nothing usable is found.
    """
    if not trace_text or not trace_text.strip():
        return None
    try:
        data = json.loads(trace_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    for message in data:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith(_TASK_STATEMENT_PREFIX):
            statement = content[len(_TASK_STATEMENT_PREFIX) :].strip()
            return statement or None
        return content.strip()
    return None


def extract_verifier_message(result: dict[str, Any] | None) -> str | None:
    """Best-effort diagnostic string from a Harbor trial ``result.json``."""
    if not isinstance(result, dict):
        return None

    def _as_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            for key in ("message", "error", "stderr", "stdout", "detail", "reason"):
                nested = _as_text(value.get(key))
                if nested:
                    return nested
        return None

    candidates: list[Any] = [
        result.get("exception_info"),
        result.get("failure_message"),
        result.get("error"),
        result.get("message"),
    ]
    vr = result.get("verifier_result")
    if isinstance(vr, dict):
        candidates.extend(
            [
                vr.get("message"),
                vr.get("error"),
                vr.get("stderr"),
                vr.get("stdout"),
                vr.get("exception"),
            ]
        )
    for candidate in candidates:
        text = _as_text(candidate)
        if text:
            return text
    return None


def _spec_section(spec: AgentSpec) -> str:
    # No sort_keys: AgentSpec declares system_prompt first, so the prompt is the
    # first thing in the section and the last thing lost to truncation.
    return SPEC_HEADER + "\n" + json.dumps(spec.model_dump(), indent=2)


def _history_section(history: list[IterationRecord]) -> str:
    lines = [HISTORY_HEADER, HISTORY_COLUMNS]
    if not history:
        lines.append("(no prior iterations - this is the first proposal)")
    for record in history:
        prompt_len = getattr(record, "prompt_length", None)
        lines.append(
            " | ".join(
                [
                    str(record.iteration),
                    str(record.version),
                    _fmt_score(record.score),
                    _fmt_flag(record.improved),
                    "n/a" if prompt_len is None else str(prompt_len),
                    ",".join(record.changed_fields) if record.changed_fields else "-",
                    _flat(record.rationale),
                ]
            )
        )
    return "\n".join(lines)


def _movement_section(evaluation: EvaluationSummary) -> str:
    if not evaluation.fixed_tasks and not evaluation.regressed_tasks:
        return f"{MOVEMENT_HEADER}\nNo per-task movement vs the best version."
    lines = [MOVEMENT_HEADER]
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


def _task_statements_section(evaluation: EvaluationSummary) -> str:
    """Mandatory: task instructions for every failing task (never budget-gated)."""
    failing = [t for t in evaluation.tasks if t.status in ("failed", "error")]
    lines = [TASK_STATEMENTS_HEADER]
    if not failing:
        lines.append("(no failing tasks)")
        return "\n".join(lines)
    for task in failing:
        statement = extract_task_statement(evaluation.traces.get(task.task_id, ""))
        if statement:
            lines.append(f"### {task.task_id}\n{statement}")
        else:
            lines.append(f"### {task.task_id}\n(task statement not available)")
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
    movement vs the best version (always) -> task statements for failing tasks
    (always) -> per-task result table (always) -> failure details (trace tails),
    worst task first, appended only while the running total stays inside
    ``budget``. The result is finally truncated to ``budget`` characters, so the
    returned length is never larger than the budget even when the mandatory
    prefix alone overflows it.
    """
    parts = [
        _spec_section(spec),
        _history_section(list(history)),
        _movement_section(evaluation),
        _task_statements_section(evaluation),
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


# --------------------------------------------------------------------------- #
# Improver implementations
# --------------------------------------------------------------------------- #

FAKE_CONTEXT_BUDGET = 8_000

# Mutable knobs the improver may change. Derived from AgentSpec so bounds in the
# system prompt cannot drift from the Pydantic model (see follow-up M7).
_IMMUTABLE_SPEC_KEYS = frozenset({"system_prompt", "agent_model"})
_ALLOWED_CONFIG_KEYS = frozenset(
    name for name in AgentSpec.model_fields if name not in _IMMUTABLE_SPEC_KEYS
)


def _field_ge_le(name: str) -> tuple[Any, Any]:
    info = AgentSpec.model_fields[name]
    ge = le = None
    for meta in info.metadata:
        meta_ge = getattr(meta, "ge", None)
        meta_le = getattr(meta, "le", None)
        if meta_ge is not None:
            ge = meta_ge
        if meta_le is not None:
            le = meta_le
    return ge, le


def _allowed_config_prompt_text() -> str:
    parts: list[str] = []
    for name in sorted(_ALLOWED_CONFIG_KEYS):
        ge, le = _field_ge_le(name)
        if ge is not None and le is not None:
            parts.append(f"{name} ({ge}-{le})")
        else:
            parts.append(name)
    return ", ".join(parts)


_MAX_SYSTEM_PROMPT_CHARS = max(4000, int(len(BASELINE_SYSTEM_PROMPT) * _MAX_PROMPT_LEN_MULT))

IMPROVER_SYSTEM_PROMPT = (
    "You are an optimization engine for an autonomous terminal-using coding agent.\n"
    "You are given the agent's current specification, the history of previous "
    "attempts with their scores, and the failures from the most recent benchmark "
    "evaluation. Propose ONE focused change most likely to raise the mean reward.\n"
    "You are editing the best-scoring agent so far. Some earlier attempts scored "
    "worse and were discarded - the history shows them so you do not repeat them.\n"
    "\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"system_prompt": "<the FULL replacement system prompt>", '
    '"config_changes": {"max_steps": 100}, '
    '"rationale": "<why this change addresses the observed failures>"}\n'
    "\n"
    "Rules:\n"
    "- system_prompt must be the complete new prompt, never a diff or a patch.\n"
    "- Prefer rewriting or replacing an existing rule over appending new ones. "
    f"Keep system_prompt within {_MAX_SYSTEM_PROMPT_CHARS} characters "
    f"(the current prompt is {len(BASELINE_SYSTEM_PROMPT)}); a longer proposal will be "
    "sent back to you once to shorten.\n"
    f"- config_changes may only contain: {_allowed_config_prompt_text()}. "
    "Omit any key you do not change; use {} to change nothing.\n"
    "- Raising max_steps or exec_timeout_sec has real cost (every iteration "
    "re-runs the full benchmark). Treat those as a last resort after prompt "
    "changes; prefer a tighter prompt over 'try harder'.\n"
    "- Never propose a different model and never invent other keys.\n"
    "- Do not repeat a change the iteration history shows already failed to "
    "improve the score.\n"
    "- The agent has exactly one tool (bash). Do not ask for other tools.\n"
    "- Read TASK STATEMENTS carefully before diagnosing failures; shell noise "
    "in the trace tail is secondary to whether the agent understood the task."
)


class _ProposalRejected(Exception):
    """Internal: the model's reply was unusable; triggers the single retry."""


# Matches the suffix FakeImprover's exhausted branch appends, so it can be
# stripped before appending the next one. Without this, feeding a proposal's
# spec back into propose() repeatedly (as the mock end-to-end loop does) would
# grow system_prompt by one suffix per call forever, eventually overflowing
# AgentSpec's 20_000-char cap and raising - breaking FakeImprover's "never
# raises" invariant. Stripping keeps the prompt length bounded regardless of
# how many times it is fed back in.
_REVISION_SUFFIX_RE = re.compile(r"\n\n\[fake-improver revision \d+\]\Z")


def _strip_revision_suffix(system_prompt: str) -> str:
    return _REVISION_SUFFIX_RE.sub("", system_prompt)


# litellm is imported lazily into this global by _litellm(). Keeping it a module
# attribute (rather than a local import) is what lets tests swap it out with
# monkeypatch.setattr(improver_mod, "litellm", stub) without the real package
# ever being imported. Lazy so that importing api.services (FastAPI app, every
# Postgres test) never requires litellm to be installed.
litellm: Any = None


def _litellm() -> Any:
    global litellm
    if litellm is None:
        import litellm as _litellm_mod

        litellm = _litellm_mod
    return litellm


def _extract_content(response: Any) -> str:
    """Pull the assistant text out of a litellm completion response.

    Raises ``_ProposalRejected``, not ``ImproverError``: an empty or malformed
    *response body* is a rejected proposal like any other and gets the same
    single retry, distinct from a transport-level exception from
    ``litellm.completion`` itself (which is never retried).
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise _ProposalRejected("improver LLM returned no choices")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _ProposalRejected("improver LLM returned no text content")
    return content


class FakeImprover:
    """
    Deterministic improver for tests (mirrors MockBenchmarkRunner's role).

    Per call, in precedence order:
      1. ``mutate`` is applied to the incoming spec when supplied;
      2. otherwise the next scripted proposal is returned;
      3. otherwise (list exhausted) a deterministic derived proposal is returned:
         the incoming spec with ``[fake-improver revision N]`` appended to the
         system prompt (replacing any previous revision suffix, so the prompt
         length stays bounded no matter how many times a proposal is fed back
         in). It never raises and never runs out, for any N.
    """

    def __init__(
        self,
        proposals: list[Proposal] | None = None,
        *,
        mutate: Callable[[AgentSpec], AgentSpec] | None = None,
    ) -> None:
        self._proposals = list(proposals or [])
        self._mutate = mutate
        self.calls = 0
        self.last_prompt = ""
        self.last_response = ""

    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal:
        self.calls += 1
        n = self.calls

        if self._mutate is not None:
            proposal = Proposal(spec=self._mutate(spec), rationale=f"fake improver mutation {n}")
        elif n <= len(self._proposals):
            proposal = self._proposals[n - 1]
        else:
            merged = spec.model_dump()
            base_prompt = _strip_revision_suffix(spec.system_prompt)
            merged["system_prompt"] = f"{base_prompt}\n\n[fake-improver revision {n}]"
            proposal = Proposal(
                spec=AgentSpec.model_validate(merged),
                rationale=f"fake improver deterministic revision {n}",
            )

        self.last_prompt = build_context(
            spec=spec,
            evaluation=evaluation,
            history=history,
            budget=FAKE_CONTEXT_BUDGET,
        )
        self.last_response = json.dumps(
            {
                "system_prompt": proposal.spec.system_prompt,
                "config_changes": {},
                "rationale": proposal.rationale,
            },
            indent=2,
        )
        return proposal


class LLMImprover:
    """Proposes the next AgentSpec with one litellm JSON-mode call (+1 retry)."""

    def __init__(self, *, model: str, budget: int) -> None:
        self.model = model
        self.budget = budget
        self.last_prompt = ""
        self.last_response = ""

    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal:
        client = _litellm()
        context = build_context(
            spec=spec,
            evaluation=evaluation,
            history=history,
            budget=self.budget,
        )
        self.last_prompt = context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": IMPROVER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        last_error = ""
        for attempt in (0, 1):
            if attempt == 1:
                retry_text = (
                    "Your previous response was rejected: "
                    + last_error
                    + "\nReturn a corrected JSON object with the same three keys "
                    "(system_prompt, config_changes, rationale) and nothing else."
                )
                # Never append an empty assistant turn (M13): some providers reject it
                # and turn a recoverable proposal rejection into a hard transport error.
                if self.last_response.strip():
                    messages = messages + [
                        {"role": "assistant", "content": self.last_response},
                        {"role": "user", "content": retry_text},
                    ]
                else:
                    messages = messages + [{"role": "user", "content": retry_text}]
                self.last_prompt = context + "\n\n## RETRY\n" + retry_text

            try:
                response = client.completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:  # noqa: BLE001 - transport failure, no retry
                raise ImproverError(f"improver LLM call failed: {exc}") from exc

            try:
                text = _extract_content(response)
                self.last_response = text
                return self._parse(text, spec, final_attempt=attempt == 1)
            except _ProposalRejected as exc:
                last_error = str(exc)
                logger.warning("improver proposal rejected (attempt %s): %s", attempt, last_error)

        raise ImproverError(f"improver returned an invalid proposal twice: {last_error}")

    def _parse(self, text: str, spec: AgentSpec, *, final_attempt: bool = False) -> Proposal:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise _ProposalRejected(f"response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise _ProposalRejected("response was not valid JSON: expected a JSON object")

        changes = data.get("config_changes") or {}
        if not isinstance(changes, dict):
            raise _ProposalRejected("config_changes must be a JSON object")
        unknown = sorted(set(changes) - _ALLOWED_CONFIG_KEYS)
        if unknown:
            raise _ProposalRejected(
                f"config_changes contains unsupported keys: {unknown}; "
                f"allowed keys are {sorted(_ALLOWED_CONFIG_KEYS)}"
            )

        merged = spec.model_dump()
        prompt = data.get("system_prompt")
        if isinstance(prompt, str) and prompt.strip():
            if len(prompt) > _MAX_SYSTEM_PROMPT_CHARS:
                # The cap is a style preference; losing the iteration is a real cost.
                # Reject once so the retry can nudge the model shorter, but ACCEPT on
                # the final attempt rather than raising — otherwise a prompt a few
                # characters over budget ends the whole job via failed_improve, which
                # is strictly worse than a slightly long prompt.
                if not final_attempt:
                    raise _ProposalRejected(
                        f"system_prompt length {len(prompt)} exceeds the "
                        f"{_MAX_SYSTEM_PROMPT_CHARS}-character budget; rewrite more "
                        "concisely rather than appending rules"
                    )
                logger.warning(
                    "accepting over-budget system_prompt on final attempt: "
                    "%d chars > budget %d",
                    len(prompt),
                    _MAX_SYSTEM_PROMPT_CHARS,
                )
            merged["system_prompt"] = prompt
        merged.update(changes)

        try:
            new_spec = AgentSpec.model_validate(merged)
        except ValidationError as exc:
            raise _ProposalRejected(f"proposal failed AgentSpec validation: {exc}") from exc

        rationale = str(data.get("rationale") or "").strip() or "(no rationale provided)"
        return Proposal(spec=new_spec, rationale=rationale)


class _RaisingImprover:
    """Test double that always fails a propose() call.

    Used wherever a test needs a failing improve step (e.g. the failed_improve
    stop path). FakeImprover deliberately never raises even when its scripted
    list is exhausted, so exercising that failure path requires this separate
    stub instead.
    """

    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal:
        raise ImproverError("synthetic improver failure for tests")


def create_improver(
    config: BenchmarkConfig | None = None,
    *,
    improver_model: str | None = None,
) -> Improver:
    """Factory: FakeImprover for the mock backend, LLMImprover otherwise."""
    cfg = config or load_config()
    if cfg.execution_backend == "mock":
        return FakeImprover()
    return LLMImprover(
        model=improver_model or cfg.improver_model,
        budget=cfg.improver_context_budget,
    )
