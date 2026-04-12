"""Portable prompt and env helpers (no tau2)."""

from __future__ import annotations

import os

if "AGENT_MODEL" not in os.environ:
    raise RuntimeError("AGENT_MODEL env var is not set")

AGENT_MODEL: str = os.environ["AGENT_MODEL"]

AGENT_INSTRUCTION = """
You are a helpful assistant that completes tasks according to the <policy> provided below.
""".strip()

# SWE-Bench: used by swe_agent (v1+ LLM path); v0 stub ignores prompts.
SWE_AGENT_INSTRUCTION = """
You are a software engineering agent. Given a GitHub issue description, you modify the
repository so that the project's tests pass. Produce a valid unified diff as your final
answer when asked for a patch.
""".strip()


def build_system_prompt(instruction: str, domain_policy: str | None) -> str:
    if domain_policy:
        return (
            "<instructions>\n"
            f"{instruction}\n"
            "</instructions>\n"
            "<policy>\n"
            f"{domain_policy}\n"
            "</policy>"
        )
    return instruction


def build_swe_system_prompt(
    instruction: str,
    problem_statement: str,
    extra_context: str | None = None,
) -> str:
    """Build system + issue text for SWE patch generation (no tau2)."""
    parts = [
        "<instructions>\n" + instruction.strip() + "\n</instructions>",
        "<issue>\n" + problem_statement.strip() + "\n</issue>",
    ]
    if extra_context:
        parts.append("<context>\n" + extra_context.strip() + "\n</context>")
    return "\n\n".join(parts)


def reasoning_effort_kwargs() -> dict[str, str]:
    effort = os.environ.get("AGENT_REASONING_EFFORT", "")
    return {"reasoning_effort": effort} if effort else {}
