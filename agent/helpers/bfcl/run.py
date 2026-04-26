"""Subprocess shim for BFCL.

Why a shim and not parent-process monkey-patching: a fresh `python -m bfcl_eval`
subprocess starts a clean interpreter and will not see mutations made in the
parent (e.g., in `BFCLRunner`). Registering the harness handler must therefore
happen inside the subprocess — that is exactly what this shim does.

Usage:
    BFCL_PROJECT_ROOT=<run_root> \
    AGENT_MODEL=<api-model> \
    python -m agent.helpers.bfcl.run generate --model harness-agent ...

Order of operations:
    1. Ensure `BFCL_PROJECT_ROOT` is set BEFORE any `bfcl_eval` import. The
       package's `eval_config` module runs `mkdir(...)` at import time for
       `RESULT_PATH`, `SCORE_PATH`, and `LOCK_DIR`. If we import bfcl_eval
       before setting the env var, those directories land in the package
       default and the run-root override is silently ignored.
    2. Register `harness-agent` in `MODEL_CONFIG_MAPPING`.
    3. Delegate to BFCL's Typer CLI.
"""

from __future__ import annotations

import os
import sys
import tempfile


# Subcommands that actually exercise `harness-agent`. If registration fails
# while one of these is being invoked, the import error must surface — a later
# "model not found" or silently-stock-handler failure would hide the real
# cause. For everything else (read-only listings, version, --help) we degrade
# gracefully so a fresh checkout can run smoke commands before `prepare.py`.
_GENERATIVE_SUBCOMMANDS = frozenset({"generate", "evaluate"})


def _is_generative_invocation(argv: list[str]) -> bool:
    return any(arg in _GENERATIVE_SUBCOMMANDS for arg in argv[1:])


def main() -> None:
    if not os.environ.get("BFCL_PROJECT_ROOT"):
        # Read-only subcommands like `models` and `test-categories` don't write
        # results, but BFCL still calls mkdir at import time. Auto-provide an
        # ephemeral root so manual smoke tests don't need to pre-export it.
        os.environ["BFCL_PROJECT_ROOT"] = tempfile.mkdtemp(prefix="bfcl-shim-")

    needs_harness = _is_generative_invocation(sys.argv)

    try:
        from agent.helpers.bfcl.registry import register_harness_handler

        register_harness_handler()
    except ImportError as exc:
        if needs_harness:
            # generate / evaluate require harness-agent. Re-raise so the user
            # sees the actual root cause (missing template, syntax error in
            # agent/agent.py, broken import) instead of a confusing
            # downstream "model not found" or stock-handler behaviour.
            raise
        # Read-only subcommand against placeholder state — log and continue.
        print(
            f"[bfcl-shim] note: harness-agent not registered ({exc}).\n"
            "[bfcl-shim] Run `python prepare.py` to copy "
            "agent/templates/bfcl.py into agent/agent.py. Read-only "
            "subcommands (test-categories, models, version, scores, results) "
            "work without it.",
            file=sys.stderr,
        )

    from bfcl_eval.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
    sys.exit(0)
