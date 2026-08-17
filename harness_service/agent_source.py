"""Loading the baseline agent under optimization.

The starting point is the read-only template ``agent/templates/terminal_bench.py``
(the same file the CLI loop copies to ``agent/agent.py``). We read its source and
surface a few tunable params so the state model captures them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from harness_service.config import Settings
from harness_service.domain import AgentState

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_TEMPLATE = REPO_ROOT / "agent" / "templates" / "terminal_bench.py"


@lru_cache
def load_baseline_source() -> str:
    if not BASELINE_TEMPLATE.exists():
        raise FileNotFoundError(f"baseline agent template not found: {BASELINE_TEMPLATE}")
    return BASELINE_TEMPLATE.read_text(encoding="utf-8")


def _parse_int(source: str, name: str, default: int) -> int:
    m = re.search(rf"^{name}\s*=\s*(\d+)", source, re.MULTILINE)
    return int(m.group(1)) if m else default


def build_agent_state(source: str, config: dict, settings: Settings) -> AgentState:
    """Assemble an AgentState from a source string + per-job config overrides."""
    return AgentState(
        source=source,
        model=config.get("agent_model", settings.agent_model),
        reasoning_effort=config.get("reasoning_effort", settings.agent_reasoning_effort),
        max_steps=_parse_int(source, "MAX_STEPS", 80),
        max_output_chars=_parse_int(source, "MAX_OUTPUT_CHARS", 8000),
    )


def build_baseline_agent(config: dict, settings: Settings) -> AgentState:
    return build_agent_state(load_baseline_source(), config, settings)
