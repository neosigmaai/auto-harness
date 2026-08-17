"""The improvement proposer (M4).

Given the current agent, its observed failures (task + reason + trace excerpt), and
the accumulated context of prior attempts, propose a NEW full ``agent.py`` + rationale.

Two implementations behind one Protocol:
  * ``OpenAIProposer`` — real LLM (OpenAI), the graded path.
  * ``MockProposer``   — deterministic, keyless fallback so the loop is always runnable
                         (also used for fast simulated dev). Injects the known-good
                         planning/verification discipline into the system prompt.

Scope (senior guidance): the proposer may rewrite the ENTIRE file (one file), but every
candidate passes a compile-gate before it is allowed to run — a non-importable candidate
is rejected, never executed.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Protocol

from harness_service.config import Settings
from harness_service.constants import ProposerKind
from harness_service.domain import AgentState, BenchmarkResult, Improvement

logger = logging.getLogger("harness.proposer")


# ── compile-gate ──
def validate_candidate(source: str) -> tuple[bool, str]:
    """A candidate must compile and still look like a HarnessAgent."""
    try:
        compile(source, "<candidate_agent>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    if "class HarnessAgent" not in source:
        return False, "missing 'class HarnessAgent'"
    if "async def run" not in source and "def run" not in source:
        return False, "missing agent run() method"
    return True, ""


def _diff_summary(old: str, new: str) -> str:
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return f"+{added}/-{removed} lines ({len(old)}→{len(new)} chars)"


def _failure_digest(result: BenchmarkResult, limit_chars: int = 1500) -> str:
    lines = []
    for o in result.failures:
        excerpt = (o.trace_excerpt or o.failure_reason or "").strip().replace("\n", " ")
        lines.append(f"- {o.task_id}: {excerpt[:limit_chars]}")
    return "\n".join(lines) if lines else "(no failures)"


class Proposer(Protocol):
    kind: ProposerKind

    async def propose(
        self, base: AgentState, result: BenchmarkResult, context: str
    ) -> Improvement:
        ...


_SYSTEM = """\
You optimize an autonomous terminal agent implemented in a single Python file (agent.py).
The agent solves TerminalBench tasks by running bash commands in a Linux container. You may
rewrite the ENTIRE file, but you MUST keep it a valid, importable Python module that defines
`class HarnessAgent(BaseAgent)` with an async `run(...)` method and the module-level `TOOLS`,
`AGENT_INSTRUCTION`, `MODEL`, and `MAX_STEPS` it relies on. Do not change how MODEL is read
from the environment. Improve the agent's ability to actually pass tasks — typical wins:
enforce a plan/TODO, explore the environment first, check command output for errors, verify
the solution before finishing, avoid asking the user questions (act autonomously).

Respond with STRICT JSON: {"rationale": "<why this helps, referencing the failures>",
"new_source": "<the complete new agent.py>"}."""


class OpenAIProposer:
    kind = ProposerKind.OPENAI

    def __init__(self, settings: Settings):
        self._s = settings

    async def propose(
        self, base: AgentState, result: BenchmarkResult, context: str
    ) -> Improvement:
        from openai import AsyncOpenAI

        user = (
            f"CURRENT agent.py:\n```python\n{base.source}\n```\n\n"
            f"OBSERVED FAILURES (task: trace/reason excerpt):\n{_failure_digest(result)}\n\n"
            f"PRIOR ATTEMPTS (accumulated context):\n{context or '(none yet)'}\n\n"
            f"Current train val_score: {result.val_score:.3f}. Propose the next improvement."
        )
        client = AsyncOpenAI(api_key=self._s.openai_api_key)
        resp = await client.chat.completions.create(
            model=self._s.openai_model,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        new_source = data.get("new_source", "")
        rationale = data.get("rationale", "(no rationale)")
        return Improvement(
            proposer=ProposerKind.OPENAI,
            rationale=rationale,
            diff_summary=_diff_summary(base.source, new_source),
            new_agent_source=new_source,
            llm_request={"model": self._s.openai_model, "system": _SYSTEM, "user_excerpt": user[:2000]},
            llm_response={"content": content[:8000]},
        )


class MockProposer:
    """Deterministic, keyless. Injects planning/verification discipline into the prompt.

    Enough to make the loop demonstrably improve (the SimulatedExecutor rewards these
    techniques, mirroring the real levers in program_templates/), then plateau — which
    exercises the accept → reject → patience-stop path without any API cost.
    """

    kind = ProposerKind.MOCK

    def __init__(self, settings: Settings | None = None):
        self._n = 0

    async def propose(
        self, base: AgentState, result: BenchmarkResult, context: str
    ) -> Improvement:
        self._n += 1
        addition = (
            f"\n\n# --- proposed refinement {self._n} ---\n"
            'AGENT_INSTRUCTION += "\\n- Make a TODO plan first, then explore the '
            'environment, check the output of every command, work step by step, and '
            'verify your solution before finishing."\n'
        )
        new_source = base.source + addition
        return Improvement(
            proposer=ProposerKind.MOCK,
            rationale=(
                "Add explicit planning, environment exploration, output-checking, and "
                "verification discipline to the system prompt (addresses the dominant "
                "failure modes: giving up early, not verifying, misreading the task)."
            ),
            diff_summary=_diff_summary(base.source, new_source),
            new_agent_source=new_source,
            llm_request={},
            llm_response={},
        )


def get_proposer(settings: Settings) -> Proposer:
    if settings.openai_api_key:
        return OpenAIProposer(settings)
    logger.info("no OPENAI_API_KEY → using deterministic MockProposer")
    return MockProposer(settings)
