"""AgentSpec — the mutable surface of the agent under optimization.

The improver may change the system prompt and a handful of bounded knobs; tools
(bash) and the agent loop itself are fixed. Keep this module free of database and
service imports: it is shared by the store, the improver, the API schemas and the
tests, and must stay cheap to import.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Verbatim copy of AGENT_INSTRUCTION in agent/templates/terminal_bench.py.
# tests/test_agent_spec.py asserts the two stay identical.
BASELINE_SYSTEM_PROMPT = """\
You are an autonomous terminal agent. You are given a task and a Linux container.
You solve tasks by executing bash commands. Work step by step.

Rules:
- Read the task carefully before acting.
- Explore the environment first to understand what you have.
- Check command output for errors before proceeding.
- Install missing dependencies as needed.
- Verify your solution before finishing.
- When you are done, send a final text message (no tool call) summarizing what you did.
"""


class AgentSpec(BaseModel):
    """A complete, validated description of a runnable agent.

    ``extra="forbid"`` plus the field bounds are the validation gate on improver
    output: a proposal that does not parse is a failed improve step, never a
    crashed job.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(min_length=1, max_length=20_000)
    agent_model: str = Field(min_length=1, max_length=256)
    max_steps: int = Field(default=80, ge=1, le=200)
    max_output_chars: int = Field(default=8000, ge=500, le=100_000)
    exec_timeout_sec: int = Field(default=120, ge=10, le=1200)


def baseline_spec(agent_model: str) -> AgentSpec:
    """Version 0 of every job: the template's prompt and limits, caller's model."""
    return AgentSpec(system_prompt=BASELINE_SYSTEM_PROMPT, agent_model=agent_model)


def changed_fields(old: AgentSpec, new: AgentSpec) -> list[str]:
    """Sorted names of the fields whose values differ between two specs."""
    old_data = old.model_dump()
    new_data = new.model_dump()
    return sorted(
        key
        for key in set(old_data) | set(new_data)
        if old_data.get(key) != new_data.get(key)
    )
