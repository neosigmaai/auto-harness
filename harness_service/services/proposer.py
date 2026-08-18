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
    """Per-task failure summary for the LLM.

    ``failure_reason`` comes FIRST and is never crowded out: it carries the actionable
    detail (harness exception / the agent's own logged error such as a malformed LLM
    request), whereas ``trace_excerpt`` is a raw conversation dump that is often long
    and uninformative when the agent crashed before doing anything.
    """
    lines = []
    for o in result.failures:
        reason = (o.failure_reason or "").strip().replace("\n", " ")
        if not reason:
            reason = (o.trace_excerpt or "").strip().replace("\n", " ")
        lines.append(f"- {o.task_id}: {reason[:limit_chars]}")
    return "\n".join(lines) if lines else "(no failures)"


def _crash_warning(result: BenchmarkResult) -> str:
    """Loud, explicit callout when the previous candidate looks broken rather than weak.

    A total collapse (everything failing, with agent-level errors) almost always means
    the proposed code itself is invalid — not that the strategy was bad. Saying so
    plainly stops the LLM from "improving the strategy" when it should be fixing a bug.
    """
    if not result.outcomes or result.n_passed > 0:
        return ""
    errored = [o for o in result.failures if o.failure_reason and "ERROR:" in o.failure_reason]
    if not errored:
        return ""
    return (
        "\n!!! CRITICAL: EVERY task failed and the agent reported runtime errors. This "
        "almost certainly means the previous agent.py was BROKEN (it crashed before it "
        "could solve anything), not that its strategy was weak. Read the ERROR text "
        "below, find the defect in the code, and FIX IT. Do not simply restate the same "
        "approach.\n"
    )


class Proposer(Protocol):
    kind: ProposerKind

    async def propose(
        self, base: AgentState, result: BenchmarkResult, context: str
    ) -> Improvement:
        ...


_SYSTEM = """\
You optimize an autonomous terminal agent implemented in a SINGLE Python file (agent.py).
The agent solves TerminalBench tasks by running bash commands inside a Linux container.
Your job: given the current agent and its observed failures, return an improved agent.py.

════════════════════════════════════════════════════════════════════════
HOW THE AGENT IS EXECUTED (read carefully — most failed proposals break here)
════════════════════════════════════════════════════════════════════════
Your file is imported by the Harbor benchmark harness, which constructs `HarnessAgent`
and awaits `run(instruction, environment, context)`. Inside `run`, the agent drives a
tool-calling conversation loop via `litellm.acompletion(model=..., messages=messages,
tools=TOOLS, ...)` against an OpenAI-compatible API, and executes bash through
`await environment.exec(command, timeout_sec=...)`.

HARD REQUIREMENTS — violating any of these makes every task score 0.0:

1. The module MUST define `class HarnessAgent(BaseAgent)` with `async def run(self,
   instruction, environment, context)`, plus module-level `TOOLS`, `AGENT_INSTRUCTION`,
   `MODEL`, `MAX_STEPS`. Keep the existing imports and `MODEL = os.environ.get(
   "AGENT_MODEL", ...)` exactly as-is — the harness sets the model via the environment.

2. **THE MESSAGE-PROTOCOL RULE (most common fatal mistake).**
   The `messages` list must always remain a VALID OpenAI chat conversation:
     • A message with `"role": "tool"` is ONLY legal immediately after an assistant
       message that contained `tool_calls`, and it MUST carry a `"tool_call_id"` that
       matches one of those calls' `id`.
     • You may NOT invent `{"role": "tool", "content": ...}` entries to inject data.
       Doing so causes an immediate API rejection:
         BadRequestError: "Invalid Value: 'input.call_id'. Function call output
         requires call_id."
       The agent then dies at step 0 and EVERY task scores 0.0.
     • Do NOT fabricate `{"role": "assistant"}` messages containing fake plans or
       reminders either; assistant turns must come from the model.

   ✗ WRONG — injecting pre-gathered environment data (this is the bug to avoid):
       for cmd in ["ls", "env"]:
           result = await environment.exec(cmd, timeout_sec=10)
           messages.append({"role": "tool", "content": result.stdout})   # FATAL

   ✓ RIGHT — run setup commands first, then fold the output into the USER message:
       recon = []
       for cmd in ["uname -a", "ls -la", "pwd"]:
           r = await environment.exec(cmd, timeout_sec=30)
           recon.append(f"$ {cmd}\\n{_truncate(r.stdout, 1000)}")
       env_summary = "\\n\\n".join(recon)
       messages = [
           {"role": "system", "content": AGENT_INSTRUCTION},
           {"role": "user", "content":
               f"Task:\\n{instruction}\\n\\nEnvironment recon:\\n{env_summary}"},
       ]

   ✓ ALSO RIGHT — steering behaviour via AGENT_INSTRUCTION (safest, high value):
       AGENT_INSTRUCTION = \"\"\"...
       Before acting, write a short TODO plan. Explore before you edit. After every
       command, check the exit code and stderr. Before finishing, re-run the tests or
       otherwise verify your work. Never ask the user a question — you are autonomous.
       \"\"\"

3. Keep the loop's existing error handling and the trace-saving block at the end of
   `run` (it writes trace.json, which is how failures are diagnosed).

4. Never ask the user questions — you are fully autonomous; there is no human to answer.

════════════════════════════════════════════════════════════════════════
TECHNIQUES THAT ACTUALLY RAISE THE SCORE (apply what the failures call for)
════════════════════════════════════════════════════════════════════════
• Enforce a plan/TODO before acting — the single biggest win.
• Explore the environment first (OS, tools, files), but inject that recon as a USER
  message, never as a fake tool result (see the protocol rule above).
• After each command, check the exit code / stderr and react to errors instead of
  plowing ahead.
• Verify the solution (re-run the tests, re-inspect the artifact) BEFORE declaring done.
• Keep per-command timeouts sane; never launch interactive or unbounded/brute-force
  processes that will hang the agent.

════════════════════════════════════════════════════════════════════════
DIAGNOSE FROM THE FAILURE TEXT BEFORE YOU EDIT
════════════════════════════════════════════════════════════════════════
Each failure line leads with its most actionable signal:
• "ERROR: ...BadRequestError / invalid_request_error / call_id..." → the code builds an
  invalid request (usually the message-protocol rule above). This is a CODE BUG — fix it;
  do not merely restate the strategy.
• "NO VERIFIER RESULT ... timed out" → a command hung or ran too long. Add/shorten
  timeouts; avoid interactive or brute-force commands.
• "reward=0.00 ... trace tail: <gave up / asked a question / edited the wrong file>" → a
  STRATEGY weakness. Strengthen planning, exploration, or verification.

Change the SMALLEST thing that addresses the observed failures. Your candidate is only kept
if it BEATS the best score so far — a broken rewrite that regresses every task is reverted
and wastes an iteration.

════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — respond with STRICT JSON and nothing else
════════════════════════════════════════════════════════════════════════
{"rationale": "<2-5 sentences: which failures you target, the specific defect or weakness,
and exactly what you changed>",
 "new_source": "<the COMPLETE new agent.py as a single string>"}
"""


class OpenAIProposer:
    kind = ProposerKind.OPENAI

    def __init__(self, settings: Settings):
        self._s = settings

    async def propose(
        self, base: AgentState, result: BenchmarkResult, context: str
    ) -> Improvement:
        from openai import AsyncOpenAI

        user = (
            f"{_crash_warning(result)}"
            f"CURRENT agent.py:\n```python\n{base.source}\n```\n\n"
            f"OBSERVED FAILURES (task: actionable error/reason first, then trace tail):\n"
            f"{_failure_digest(result)}\n\n"
            f"PRIOR ATTEMPTS (accumulated context — what was already tried and its result):\n"
            f"{context or '(none yet)'}\n\n"
            f"Current train val_score: {result.val_score:.3f} "
            f"({result.n_passed}/{len(result.outcomes)} passed). Propose the next improvement."
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
