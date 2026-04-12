"""Stable import path for benchmarks (`from agent.agent import HarnessAgent`, SWE helpers).

Edit `agent/core.py` for instructions; `agent/tau_agent.py` for tau; `agent/swe_agent.py` for SWE.
"""

from agent.core import (
    AGENT_INSTRUCTION,
    AGENT_MODEL,
    SWE_AGENT_INSTRUCTION,
    build_swe_system_prompt,
    build_system_prompt,
)
from agent.swe_agent import SWEInstanceContext, generate_patch, model_name_for_predictions
from agent.tau_agent import HarnessAgent, HarnessState

__all__ = [
    "AGENT_INSTRUCTION",
    "AGENT_MODEL",
    "SWE_AGENT_INSTRUCTION",
    "HarnessAgent",
    "HarnessState",
    "SWEInstanceContext",
    "build_swe_system_prompt",
    "build_system_prompt",
    "generate_patch",
    "model_name_for_predictions",
]
