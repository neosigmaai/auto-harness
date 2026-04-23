# HarnessAgent for WebArena — starting template.
#
# Thin wrapper around WebArena's `agent.prompts.prompt_constructor.CoTPromptConstructor`
# + `agent.agent.PromptAgent`. The driver (`webarena/driver.py`) has already added
# `webarena_repo/` to `sys.path` and `os.chdir`-ed into it, so the WebArena imports
# below resolve against the clone.
#
# You own this file: change the system prompt (`AGENT_INSTRUCTION`), swap the
# prompt JSON, or override `next_action` entirely. Do not hardcode the model
# or reasoning effort — those come from the harness via env vars.

from __future__ import annotations

import json
import os

import tiktoken

from agent.agent import PromptAgent
from agent.prompts.prompt_constructor import (
    CoTPromptConstructor,
    DirectPromptConstructor,
    PromptConstructor,
)
from llms.lm_config import LMConfig
from llms.tokenizers import Tokenizer

AGENT_MODEL: str = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

AGENT_INSTRUCTION: str = """
You are an autonomous browser agent. Read the accessibility tree carefully, pick ONE precise action per step, and prefer the most direct path to the user's objective. Stop as soon as you have the answer.
""".strip()

# Default to the WebArena-shipped 2-shot CoT prompt. The JSON is produced by
# WebArena's `agent/prompts/to_json.py` — we trigger it on demand below so a
# fresh clone works without a separate build step.
DEFAULT_INSTRUCTION_PATH = "agent/prompts/jsons/p_cot_id_actree_2s.json"


def _ensure_prompt_jsons(default_path: str) -> None:
    """Run WebArena's prompt-to-json converter if the expected JSON is missing."""
    if os.path.exists(default_path):
        return
    try:
        from agent.prompts import to_json
        to_json.run()
    except Exception as e:
        raise RuntimeError(
            f"WebArena prompt JSONs are missing and to_json.run() failed: {e}. "
            f"Expected {default_path}. Ensure webarena_repo is freshly cloned."
        ) from e


def _build_tokenizer(model: str) -> Tokenizer:
    """Tokenizer with a sane fallback for models tiktoken doesn't map yet.

    WebArena's shipped Tokenizer calls `tiktoken.encoding_for_model(model)`
    which raises KeyError on new model names (e.g. post-gpt-4o releases). We
    only need the tokenizer for `max_obs_length` truncation, so any modern
    BPE encoding is fine — fall back to `o200k_base` (gpt-4o family) then
    `cl100k_base` (gpt-3.5/4 family).
    """
    try:
        return Tokenizer("openai", model)
    except KeyError:
        for encoding_name in ("o200k_base", "cl100k_base"):
            try:
                tok = Tokenizer.__new__(Tokenizer)
                tok.tokenizer = tiktoken.get_encoding(encoding_name)
                return tok
            except Exception:
                continue
        raise


def _build_lm_config(model: str) -> LMConfig:
    """LMConfig matching WebArena's default openai chat settings."""
    cfg = LMConfig(provider="openai", model=model, mode="chat")
    cfg.gen_config["temperature"] = 1.0
    cfg.gen_config["top_p"] = 0.9
    cfg.gen_config["context_length"] = 0
    cfg.gen_config["max_tokens"] = 384
    cfg.gen_config["stop_token"] = None
    cfg.gen_config["max_obs_length"] = 1920
    cfg.gen_config["max_retry"] = 1
    return cfg


def _build_prompt_constructor(
    instruction_path: str, lm_config: LMConfig, tokenizer: Tokenizer
) -> PromptConstructor:
    """Pick the constructor class declared in the instruction JSON's meta_data."""
    with open(instruction_path) as f:
        meta = json.load(f)["meta_data"]
    ctor_name = meta.get("prompt_constructor", "CoTPromptConstructor")
    ctor_cls = {
        "CoTPromptConstructor": CoTPromptConstructor,
        "DirectPromptConstructor": DirectPromptConstructor,
    }.get(ctor_name)
    if ctor_cls is None:
        raise ValueError(f"Unknown prompt_constructor: {ctor_name}")
    return ctor_cls(instruction_path, lm_config=lm_config, tokenizer=tokenizer)


class HarnessAgent(PromptAgent):
    """Agent under optimization for WebArena."""

    def __init__(
        self,
        model: str | None = None,
        action_set_tag: str = "id_accessibility_tree",
        instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    ) -> None:
        model = model or AGENT_MODEL
        _ensure_prompt_jsons(instruction_path)

        lm_config = _build_lm_config(model)
        tokenizer = _build_tokenizer(model)
        prompt_constructor = _build_prompt_constructor(
            instruction_path, lm_config, tokenizer
        )

        # Prepend the harness-level instruction to whatever prompt the JSON ships
        # with. Mutating the loaded dict keeps the rest of WebArena's plumbing
        # (examples, template, meta_data) intact.
        if AGENT_INSTRUCTION:
            original_intro = prompt_constructor.instruction.get("intro", "")
            prompt_constructor.instruction["intro"] = (
                AGENT_INSTRUCTION.strip() + "\n\n" + original_intro
            )

        super().__init__(
            action_set_tag=action_set_tag,
            lm_config=lm_config,
            prompt_constructor=prompt_constructor,
        )

    # Override points left deliberately simple — the coding agent can change
    # the system prompt above, swap `instruction_path`, or subclass further
    # (e.g., to tweak early-stop logic). `next_action` / `reset` are inherited.
