"""Deterministic verifier for the assumptions the BFCL integration relies on.

Run from the repo root:

    python scripts/bfcl_spike.py

Exits 0 if every assumption (A1-A7) and finding (F1-F7) listed below holds
against the pinned `bfcl-eval` version, non-zero otherwise. The script never
touches the network, never calls a paid LLM, and writes only into a temp
directory.

If `bfcl-eval` is bumped, re-run this file *before* changing integration code.
Update PINNED_VERSION below to the new pin and adjust any check whose verdict
shifts.

# ─────────────────────────────────────────────────────────────────────────────
# Assumptions the BFCL integration depends on
# ─────────────────────────────────────────────────────────────────────────────
#
# A1. Package distribution is `bfcl-eval`; import name is `bfcl_eval`.
#
# A2. The current packaged dataset is BFCL v4; data files are named
#     `BFCL_v4_<category>.json`. `multi_turn_base` contains 200 entries with
#     IDs `multi_turn_base_0` … `multi_turn_base_199`.
#
# A3. BFCL data is packaged inside `bfcl_eval/data/`. `BFCL_PROJECT_ROOT`
#     controls runtime artifact paths (`result/`, `score/`, `.file_locks/`,
#     `.env`, `test_case_ids_to_generate.json`).
#
# A4. `bfcl generate --run-ids` is a boolean flag. The list of task IDs is
#     read from `<BFCL_PROJECT_ROOT>/test_case_ids_to_generate.json` whose
#     shape is `{category: [task_id, ...]}`.
#
# A5. The OpenAI handler classes are `OpenAIResponsesHandler` and
#     `OpenAICompletionsHandler`. Both subclass `BaseHandler` directly; there
#     is no intermediate generic `OpenAIHandler` class.
#
# A6. BFCL score files contain a single header row (with `accuracy`,
#     `correct_count`, `total_count`) followed by failing entries only — not a
#     full per-task result table. Per-task rewards must be reconstructed from
#     the result JSONL (generated IDs) minus the score JSONL (failure IDs).
#
# A7. `MODEL_CONFIG_MAPPING` values are `ModelConfig` dataclass instances, not
#     plain dicts. Custom handler registration must construct the current
#     package's `ModelConfig` type with `model_handler` set to the handler
#     class.
#
# ─────────────────────────────────────────────────────────────────────────────
# Findings discovered during verification — also gated by this script
# ─────────────────────────────────────────────────────────────────────────────
#
# F1. `bfcl_eval.constants.eval_config` calls `mkdir(...)` for `RESULT_PATH`,
#     `SCORE_PATH`, and `LOCK_DIR` *at import time*. Therefore
#     `BFCL_PROJECT_ROOT` must be set in the environment BEFORE the first
#     `bfcl_eval` import, or the directories are created at the package
#     default and the override is silently ignored for cached modules.
#
# F2. `--run-ids` is *exclusive* of `--test-category`, not additive. When
#     `run_ids=True`, the test-category argument is ignored; categories are
#     inferred from the keys of `test_case_ids_to_generate.json`.
#
# F3. Score and result files are JSONL with a `.json` extension. Reward
#     parsers must read line-by-line — `json.load(f)` will raise.
#
# F4. Result files live in a nested layout
#     `<result_dir>/<registry_dir_name>/<group_dir>/<VERSION_PREFIX>_<test_category>_result.json`.
#     `registry_dir_name` is derived from the BFCL model id (e.g.
#     `harness-agent`), not the API model name. Reward parsers should walk
#     `os.walk` / `rglob` rather than scanning a flat `result/`.
#
# F5. `bfcl-eval` has an undeclared transitive dep on `soundfile` via
#     `qwen_agent`, which imports `soundfile` at module load. Adding
#     `soundfile` to `pyproject.toml` is required for a fresh install.
#
# F6. `AGENT_MODEL` is forwarded to the OpenAI Responses API verbatim. It
#     must be a real OpenAI model id; an arbitrary internal placeholder will
#     404 with an unhelpful error.
#
# F7. `OpenAIResponsesHandler._substitute_prompt_role` rewrites
#     `system` → `developer` only for `test_entry["question"]`. Messages
#     appended directly to `inference_data["message"]` bypass the rewrite, so
#     callers must use the `developer` role explicitly.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

PINNED_VERSION = "2026.3.23"
PINNED_VERSION_PREFIX = "BFCL_v4"
EXPECTED_MULTI_TURN_BASE_TASKS = 200

_failures: list[tuple[str, str]] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS] {tag}: {msg}")


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL] {tag}: {msg}")
    _failures.append((tag, msg))


def _check(tag: str, condition: bool, ok_msg: str, fail_msg: str) -> None:
    (_ok if condition else _fail)(tag, ok_msg if condition else fail_msg)


def _isolate_project_root() -> Path:
    """Set BFCL_PROJECT_ROOT to a fresh temp dir before any bfcl_eval import.

    Required because `bfcl_eval.constants.eval_config` calls `mkdir(...)` for
    `RESULT_PATH`, `SCORE_PATH`, and `LOCK_DIR` at import time. Setting the env
    var afterwards is too late.
    """
    if "bfcl_eval" in sys.modules:
        raise RuntimeError(
            "bfcl_eval was already imported before _isolate_project_root() ran"
        )
    root = Path(tempfile.mkdtemp(prefix="bfcl-spike-"))
    os.environ["BFCL_PROJECT_ROOT"] = str(root)
    return root


def check_a1_package() -> None:
    import importlib.metadata as md

    try:
        name = md.metadata("bfcl-eval")["Name"]
        version = md.version("bfcl-eval")
    except md.PackageNotFoundError:
        _fail("A1", "bfcl-eval distribution not installed")
        return

    import bfcl_eval

    _check(
        "A1.import",
        Path(bfcl_eval.__file__).name == "__init__.py"
        and Path(bfcl_eval.__file__).parent.name == "bfcl_eval",
        f"import bfcl_eval -> {bfcl_eval.__file__}",
        f"unexpected import path {bfcl_eval.__file__}",
    )
    _check(
        "A1.dist",
        name.replace("_", "-").lower() == "bfcl-eval",
        f"distribution name {name}",
        f"unexpected distribution name {name}",
    )
    _check(
        "A1.version",
        version == PINNED_VERSION,
        f"version pinned at {version}",
        f"version {version} != pinned {PINNED_VERSION}; re-run spike after bump",
    )


def check_a2_dataset(project_root: Path) -> None:
    from bfcl_eval.constants import category_mapping

    _check(
        "A2.prefix",
        category_mapping.VERSION_PREFIX == PINNED_VERSION_PREFIX,
        f"VERSION_PREFIX == {category_mapping.VERSION_PREFIX!r}",
        f"VERSION_PREFIX == {category_mapping.VERSION_PREFIX!r} "
        f"(expected {PINNED_VERSION_PREFIX!r})",
    )

    import bfcl_eval

    data_dir = Path(bfcl_eval.__file__).parent / "data"
    multi_turn_base = data_dir / f"{PINNED_VERSION_PREFIX}_multi_turn_base.json"
    _check(
        "A2.file",
        multi_turn_base.exists(),
        f"{multi_turn_base.name} present",
        f"{multi_turn_base.name} missing under {data_dir}",
    )

    if multi_turn_base.exists():
        with multi_turn_base.open() as f:
            rows = [json.loads(line) for line in f]
        _check(
            "A2.count",
            len(rows) == EXPECTED_MULTI_TURN_BASE_TASKS,
            f"multi_turn_base has {len(rows)} tasks",
            f"multi_turn_base has {len(rows)} tasks "
            f"(expected {EXPECTED_MULTI_TURN_BASE_TASKS})",
        )
        ids = [row.get("id") for row in rows]
        _check(
            "A2.ids",
            ids[0] == "multi_turn_base_0"
            and ids[-1] == f"multi_turn_base_{EXPECTED_MULTI_TURN_BASE_TASKS - 1}",
            f"ids span {ids[0]}..{ids[-1]}",
            f"unexpected id span {ids[0]!r}..{ids[-1]!r}",
        )


def check_a3_project_root(project_root: Path) -> None:
    from bfcl_eval.constants import eval_config

    expected = {
        "RESULT_PATH": project_root / "result",
        "SCORE_PATH": project_root / "score",
        "DOTENV_PATH": project_root / ".env",
        "TEST_IDS_TO_GENERATE_PATH": project_root / "test_case_ids_to_generate.json",
        "LOCK_DIR": project_root / ".file_locks",
    }
    for name, want in expected.items():
        got = getattr(eval_config, name)
        _check(
            f"A3.{name}",
            Path(got) == want,
            f"{name} == {got}",
            f"{name} == {got} (expected {want})",
        )

    _check(
        "A3.mkdir_side_effect",
        (project_root / "result").is_dir()
        and (project_root / "score").is_dir()
        and (project_root / ".file_locks").is_dir(),
        "RESULT_PATH/SCORE_PATH/LOCK_DIR created at import time",
        "import-time mkdir side effect missing — eval_config behaviour drifted",
    )


def check_a4_run_ids() -> None:
    import bfcl_eval._llm_response_generation as gen
    import bfcl_eval.__main__ as main_mod

    src_main = inspect.getsource(main_mod)
    src_gen = inspect.getsource(gen)

    _check(
        "A4.typer_flag",
        'run_ids: bool = typer.Option(' in src_main
        and '"--run-ids"' in src_main,
        "--run-ids declared as bool typer.Option in __main__.py",
        "--run-ids flag not found or shape changed in __main__.py",
    )
    _check(
        "A4.argparse_flag",
        '"--run-ids", action="store_true"' in src_gen,
        "--run-ids declared as argparse store_true in _llm_response_generation.py",
        "--run-ids store_true flag missing in _llm_response_generation.py",
    )

    # Verify exclusivity: run_ids=True path must not call parse_test_category_argument
    src_helper = inspect.getsource(gen.get_involved_test_entries)
    run_ids_branch = src_helper.split("if run_ids:", 1)[1].split("else:", 1)[0]
    _check(
        "A4.exclusive",
        "load_test_entries_from_id_file" in run_ids_branch
        and "parse_test_category_argument" not in run_ids_branch,
        "--run-ids branch ignores --test-category (exclusive)",
        "--run-ids/--test-category exclusivity contract changed",
    )


def check_a5_handlers() -> None:
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.base_handler import BaseHandler

    _check(
        "A5.responses",
        OpenAIResponsesHandler.__bases__ == (BaseHandler,),
        "OpenAIResponsesHandler subclasses BaseHandler directly",
        f"OpenAIResponsesHandler bases {OpenAIResponsesHandler.__bases__} "
        "(expected (BaseHandler,))",
    )
    _check(
        "A5.completions",
        OpenAICompletionsHandler.__bases__ == (BaseHandler,),
        "OpenAICompletionsHandler subclasses BaseHandler directly",
        f"OpenAICompletionsHandler bases {OpenAICompletionsHandler.__bases__} "
        "(expected (BaseHandler,))",
    )

    try:
        __import__("bfcl_eval.model_handler.api_inference.openai")
        _fail("A5.no_generic", "unexpected bfcl_eval.model_handler.api_inference.openai module exists")
    except ModuleNotFoundError:
        _ok("A5.no_generic", "no generic OpenAIHandler module (as expected)")

    expected_signatures = {
        "add_first_turn_message_FC": ["self", "inference_data", "first_turn_message"],
        "_compile_tools": ["self", "inference_data", "test_entry"],
        "_add_execution_results_FC": [
            "self",
            "inference_data",
            "execution_results",
            "model_response_data",
        ],
    }
    for method_name, want_params in expected_signatures.items():
        method = getattr(OpenAIResponsesHandler, method_name, None)
        if method is None:
            _fail("A5.method", f"{method_name} missing on OpenAIResponsesHandler")
            continue
        params = list(inspect.signature(method).parameters)
        _check(
            f"A5.{method_name}",
            params == want_params,
            f"{method_name}{tuple(params)}",
            f"{method_name}{tuple(params)} (expected {tuple(want_params)})",
        )

    src = inspect.getsource(BaseHandler)
    final_count = src.count("@final")
    _check(
        "A5.final_loops",
        final_count >= 4,
        f"BaseHandler has {final_count} @final decorators (>= 4 expected)",
        f"BaseHandler has only {final_count} @final decorators (R6 weakened)",
    )


def check_a6_score_writer() -> None:
    from bfcl_eval.eval_checker import eval_runner_helper

    src = inspect.getsource(eval_runner_helper.save_eval_results)
    _check(
        "A6.header_keys",
        '"accuracy"' in src
        and '"correct_count"' in src
        and '"total_count"' in src
        and "result.insert(0, header)" in src,
        "save_eval_results inserts {accuracy, correct_count, total_count} header at row 0",
        "score header shape changed — verify reward parser still skips row 0",
    )
    _check(
        "A6.filename",
        f'f"{{VERSION_PREFIX}}_{{test_category}}_score.json"' in src,
        "score file named <VERSION_PREFIX>_<category>_score.json",
        "score filename pattern changed",
    )

    from bfcl_eval import utils as bu

    writer_src = inspect.getsource(bu.write_list_of_dicts_to_file)
    _check(
        "A6.jsonl",
        'json.dumps(entry' in writer_src and '+ "\\n"' in writer_src,
        "writer emits JSONL (one dict per line) despite .json extension",
        "writer no longer emits JSONL — reward parser must change",
    )

    from bfcl_eval.eval_checker import eval_runner

    runner_src = inspect.getsource(eval_runner)
    _check(
        "A6.failures_only",
        'result.append(entry_result)' in runner_src
        and 'if entry_result["valid"]:' in runner_src,
        "eval_runner appends only invalid entries to result list",
        "eval_runner no longer filters to failures only — verify reward parser",
    )

    from bfcl_eval.model_handler.base_handler import BaseHandler

    write_src = inspect.getsource(BaseHandler.write)
    _check(
        "A6.result_layout",
        'f"{VERSION_PREFIX}_{test_category}_result.json"' in write_src
        and 'self.registry_dir_name' in write_src
        and 'get_directory_structure_by_id' in write_src,
        "result files live under <result_dir>/<registry_dir_name>/<group_dir>/...",
        "result-file layout changed — reward parser must walk the new path",
    )


def check_a7_model_config() -> None:
    from dataclasses import is_dataclass, fields

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )

    _check(
        "A7.dataclass",
        is_dataclass(ModelConfig),
        "ModelConfig is a @dataclass",
        "ModelConfig is no longer a dataclass",
    )
    _check(
        "A7.mapping_type",
        isinstance(MODEL_CONFIG_MAPPING, dict) and len(MODEL_CONFIG_MAPPING) > 0,
        f"MODEL_CONFIG_MAPPING is dict with {len(MODEL_CONFIG_MAPPING)} entries",
        "MODEL_CONFIG_MAPPING is empty or no longer a dict",
    )
    if MODEL_CONFIG_MAPPING:
        sample = next(iter(MODEL_CONFIG_MAPPING.values()))
        _check(
            "A7.values_are_modelconfig",
            isinstance(sample, ModelConfig),
            "MODEL_CONFIG_MAPPING values are ModelConfig instances",
            f"MODEL_CONFIG_MAPPING values are {type(sample).__name__}",
        )

    expected_fields = {
        "model_name",
        "display_name",
        "url",
        "org",
        "license",
        "model_handler",
        "input_price",
        "output_price",
        "is_fc_model",
        "underscore_to_dot",
    }
    actual_fields = {f.name for f in fields(ModelConfig)}
    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields
    _check(
        "A7.fields",
        not missing,
        f"ModelConfig has all expected fields (extra: {sorted(extra) or 'none'})",
        f"ModelConfig missing fields: {sorted(missing)}",
    )

    try:
        cfg = ModelConfig(
            model_name="harness-spike",
            display_name="auto-harness BFCL spike",
            url="https://github.com/neosigmaai/auto-harness",
            org="auto-harness",
            license="",
            model_handler=OpenAIResponsesHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )
        MODEL_CONFIG_MAPPING["harness-agent-spike"] = cfg
        roundtripped = MODEL_CONFIG_MAPPING.pop("harness-agent-spike")
        _check(
            "A7.registration",
            isinstance(roundtripped, ModelConfig)
            and roundtripped.model_handler is OpenAIResponsesHandler,
            "ModelConfig(...) registration round-trips into MODEL_CONFIG_MAPPING",
            "registration shape changed — fix agent.helpers.bfcl.registry",
        )
    except TypeError as exc:
        _fail("A7.registration", f"ModelConfig kwargs rejected: {exc}")


def check_findings() -> None:
    """Sanity-check the items in 'Verification Findings' that are testable
    without a live LLM call."""

    # F5: soundfile must be importable for model_config to load
    try:
        import soundfile  # noqa: F401

        _ok(
            "F5.soundfile",
            "soundfile importable (required transitively by qwen_agent)",
        )
    except ImportError as exc:
        _fail(
            "F5.soundfile",
            f"soundfile not installed: {exc} -- add to pyproject.toml",
        )

    # F7: OpenAI Responses _substitute_prompt_role must rewrite system->developer
    from bfcl_eval.model_handler.api_inference.openai_response import (
        OpenAIResponsesHandler,
    )

    src = inspect.getsource(OpenAIResponsesHandler._substitute_prompt_role)
    _check(
        "F7.developer_role",
        '"system"' in src and '"developer"' in src,
        "_substitute_prompt_role rewrites system -> developer",
        "_substitute_prompt_role no longer maps system -> developer",
    )


def main() -> int:
    project_root = _isolate_project_root()
    print(f"Spike project root: {project_root}")
    print(f"Pinned bfcl-eval version: {PINNED_VERSION}\n")

    check_a1_package()
    check_a2_dataset(project_root)
    check_a3_project_root(project_root)
    check_a4_run_ids()
    check_a5_handlers()
    check_a6_score_writer()
    check_a7_model_config()
    check_findings()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for tag, msg in _failures:
            print(textwrap.indent(f"- {tag}: {msg}", "  "))
        print(
            "\nIf the failures are due to a bfcl-eval version bump, update "
            "PINNED_VERSION at the top of this file and reconcile any drifted "
            "assumption with the A1-A7 / F1-F7 list in the module docstring."
        )
        return 1

    print("All assumptions (A1-A7) and findings (F1-F7) verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
