# Spec loading for the spec-driven HarnessAgent runtime.
#
# STDLIB ONLY, ON PURPOSE. agent/spec_agent.py imports harbor and litellm at module
# scope, so it can never be imported by a unit test; every decision worth testing
# lives here instead. Never import api.* from this file: harbor spawns the agent
# with only the repo root on PYTHONPATH.
#
# DEFAULT_SYSTEM_PROMPT is a duplicate of api.agent_spec.BASELINE_SYSTEM_PROMPT.
# tests/test_spec_agent_runtime.py asserts the two stay byte-identical.
from __future__ import annotations

import json
import os

DEFAULT_SYSTEM_PROMPT = """\
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

DEFAULT_MAX_STEPS = 80
DEFAULT_MAX_OUTPUT_CHARS = 8000
DEFAULT_EXEC_TIMEOUT_SEC = 120
DEFAULT_MODEL = "gpt-5.4"

SPEC_ENV_VAR = "HARNESS_AGENT_SPEC"

_INT_FIELDS = {
    "max_steps": DEFAULT_MAX_STEPS,
    "max_output_chars": DEFAULT_MAX_OUTPUT_CHARS,
    "exec_timeout_sec": DEFAULT_EXEC_TIMEOUT_SEC,
}


def default_spec() -> dict:
    """The terminal-bench template's behaviour, as a spec dict."""
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "agent_model": os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        "max_steps": DEFAULT_MAX_STEPS,
        "max_output_chars": DEFAULT_MAX_OUTPUT_CHARS,
        "exec_timeout_sec": DEFAULT_EXEC_TIMEOUT_SEC,
    }


def load_spec(path: str | None) -> dict:
    """Return a spec dict: template defaults overlaid with JSON read from ``path``.

    Never raises. An unset, missing, unreadable, malformed or partially invalid
    file degrades to the defaults so the agent stays runnable standalone — a
    broken spec must not turn into a crashed benchmark task.

    Only the five known keys are honoured; anything else in the file is ignored,
    so a spec written by a newer AgentSpec still runs on an older runtime.
    """
    spec = default_spec()
    if not path:
        return spec

    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return spec
    if not isinstance(loaded, dict):
        return spec

    for key in spec:
        if key in loaded and loaded[key] is not None:
            spec[key] = loaded[key]

    prompt = spec["system_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        spec["system_prompt"] = DEFAULT_SYSTEM_PROMPT

    model = spec["agent_model"]
    if not isinstance(model, str) or not model.strip():
        spec["agent_model"] = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    for key, fallback in _INT_FIELDS.items():
        try:
            spec[key] = int(spec[key])
        except (TypeError, ValueError):
            spec[key] = fallback

    return spec


def load_spec_from_env() -> dict:
    """Load the spec named by $HARNESS_AGENT_SPEC (defaults when unset)."""
    return load_spec(os.environ.get(SPEC_ENV_VAR))
