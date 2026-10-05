"""Setup helpers for the auto-harness BFCL integration.

Keep this file small and reviewable: import-time checks only, no provisioning.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import tempfile

PINNED_VERSION = "2026.3.23"
PINNED_VERSION_PREFIX = "BFCL_v4"


def _is_anthropic_model(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("anthropic/") or lower.startswith("claude")


def _is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def _check_package_version() -> bool:
    try:
        version = md.version("bfcl-eval")
    except md.PackageNotFoundError:
        print("[prepare] ERROR: bfcl-eval not installed. Add to pyproject.toml and re-sync.")
        return False
    if version != PINNED_VERSION:
        print(
            f"[prepare] ERROR: bfcl-eval=={version} installed, expected {PINNED_VERSION}. "
            f"Re-run scripts/bfcl_spike.py against the new version before changing the pin."
        )
        return False
    return True


def _check_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401
    except ImportError as exc:
        print(
            f"[prepare] ERROR: soundfile not installed: {exc}\n"
            "          bfcl-eval pulls qwen-agent transitively, which requires soundfile.\n"
            "          Add `soundfile` to pyproject.toml and re-sync."
        )
        return False
    return True


def _check_bfcl_imports() -> bool:
    # `bfcl_eval.constants.eval_config` mkdirs at import time. Point it at a
    # throwaway dir so this check doesn't pollute the workspace.
    os.environ.setdefault("BFCL_PROJECT_ROOT", tempfile.mkdtemp(prefix="bfcl-setup-"))
    try:
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX  # noqa: F401
        from bfcl_eval.constants.model_config import (  # noqa: F401
            MODEL_CONFIG_MAPPING,
            ModelConfig,
        )
        from bfcl_eval.model_handler.api_inference.openai_response import (  # noqa: F401
            OpenAIResponsesHandler,
        )
    except ImportError as exc:
        print(f"[prepare] ERROR: bfcl-eval import failed: {exc}")
        return False
    if VERSION_PREFIX != PINNED_VERSION_PREFIX:
        print(
            f"[prepare] ERROR: bfcl-eval VERSION_PREFIX={VERSION_PREFIX!r}, "
            f"expected {PINNED_VERSION_PREFIX!r}. Dataset format changed — "
            f"re-run scripts/bfcl_spike.py."
        )
        return False
    return True


def _check_category(category: str) -> bool:
    """Assert the configured category exists and has at least one packaged task."""
    from pathlib import Path

    import bfcl_eval

    data_dir = Path(bfcl_eval.__file__).parent / "data"
    candidate = data_dir / f"{PINNED_VERSION_PREFIX}_{category}.json"
    if not candidate.exists():
        # List available categories so the user can fix their config.
        available = sorted(
            p.name.removeprefix(f"{PINNED_VERSION_PREFIX}_").removesuffix(".json")
            for p in data_dir.glob(f"{PINNED_VERSION_PREFIX}_*.json")
        )
        print(
            f"[prepare] ERROR: BFCL category {category!r} not found at {candidate}.\n"
            f"          Available: {', '.join(available)}"
        )
        return False
    if candidate.stat().st_size == 0:
        print(f"[prepare] ERROR: BFCL category file {candidate} is empty.")
        return False
    return True


def check_env_bfcl(cfg: dict) -> bool:
    """Validate the BFCL environment for the configured experiment.

    Returns True iff all checks pass; on failure, prints a concrete remediation.
    """
    if not _check_package_version():
        return False
    if not _check_soundfile():
        return False
    if not _check_bfcl_imports():
        return False

    category = cfg.get("category", "multi_turn_base")
    if not _check_category(category):
        return False

    # First-PR scope: OpenAI Responses only. Reject vendor-specific model
    # ids at prepare-time so the failure surfaces here rather than partway
    # through a paid baseline run inside BFCL's subprocess.
    agent_model = cfg.get("agent_model") or "gpt-5.4"
    if _is_anthropic_model(agent_model):
        print(
            f"[prepare] ERROR: BFCL agent_model={agent_model!r} is Anthropic.\n"
            "          This integration ships only the OpenAI Responses path.\n"
            "          Anthropic support needs a separate handler template "
            "(not in this PR)."
        )
        return False
    if _is_gemini_model(agent_model):
        print(
            f"[prepare] ERROR: BFCL agent_model={agent_model!r} is Gemini.\n"
            "          This integration ships only the OpenAI Responses path.\n"
            "          Gemini support needs a separate handler template "
            "(not in this PR)."
        )
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[prepare] ERROR: OPENAI_API_KEY is required for BFCL.\n"
            f"          (agent_model={agent_model!r} resolves to OpenAI Responses.)"
        )
        return False

    print(f"[prepare] BFCL OK: bfcl-eval=={PINNED_VERSION}, category={category}")
    return True
