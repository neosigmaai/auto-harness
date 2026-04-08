"""BaseHarnessAgent — this is the file the coding agent optimizes."""

import os
from dataclasses import dataclass, field

if "AGENT_MODEL" not in os.environ:
    raise RuntimeError("AGENT_MODEL env var is not set")
AGENT_MODEL: str = os.environ["AGENT_MODEL"]
AGENT_REASONING_EFFORT: str = os.environ.get("AGENT_REASONING_EFFORT", "")
MAX_TURNS: int = 30

# ── editable by Claude Code ───────────────────────────────────────

SYSTEM_PROMPT = """
You are a helpful assistant that completes tasks.
""".strip()


@dataclass
class HarnessState:
    messages: list = field(default_factory=list)


class BaseHarnessAgent:
    """
    Generic agent base — Claude Code optimizes this.
    Benchmark-specific wrappers subclass this in agent_tau.py, agent_harbor.py etc.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.state = HarnessState()

    def get_generate_kwargs(self) -> dict:
        if AGENT_REASONING_EFFORT:
            return {"reasoning_effort": AGENT_REASONING_EFFORT}
        return {}

    def get_init_state(self) -> HarnessState:
        return HarnessState()

    def generate_next_message(self, message, state: HarnessState):
        """Generate next message — benchmark wrappers call this."""
        raise NotImplementedError


__all__ = ["BaseHarnessAgent", "HarnessState", "AGENT_MODEL", "MAX_TURNS", "SYSTEM_PROMPT"]