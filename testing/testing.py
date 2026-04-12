"""
End-to-end check: SWE-Bench official harness + Docker (one instance).

Validates the same path as ``benchmark.py`` with ``swe_skip_harness: false``:
``swebench`` import, ``docker info``, non-empty predictions, and
``swebench.harness.run_evaluation`` (spawns eval containers).

Requirements:
  - ``pip install -e '.[swe]'`` (or ``uv sync --extra swe``)
  - Docker daemon running (Compose: socket mounted at ``/var/run/docker.sock``)
  - Network on first run (HuggingFace dataset; image pulls)

Uses ``SWE_STUB_PATCH`` so no ``OPENAI_API_KEY`` is required. First runs can take
many minutes (image pull/build + tests).

Usage:
  python testing/testing.py
  docker compose run --rm autoeval python testing/testing.py
  python testing/testing.py --task-id astropy__astropy-12907
  python testing/testing.py --skip-report-check
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Running as ``python testing/testing.py`` sets sys.path[0] to ``testing/``; repo-root
# modules (``benchmark``, ``agent``, …) live one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CONFIG_FILE = "experiment_config.yaml"

# Minimal unified diff so ``all_empty`` is false and the harness actually runs.
_MIN_STUB_PATCH = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-placeholder
+auto-harness-docker-e2e
"""


def _load_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _docker_available() -> bool:
    """
    True if Docker is usable for the SWE harness.

    On the host, ``docker info`` is the check. In the Compose app container there is
    usually no ``docker`` binary—only ``/var/run/docker.sock`` is mounted—so we
    accept a present socket (the harness talks to the API, not the CLI).
    """
    if shutil.which("docker"):
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return r.returncode == 0
    sock = Path("/var/run/docker.sock")
    try:
        return sock.exists() and sock.is_socket()
    except OSError:
        return False


def _require_swebench() -> None:
    try:
        import swebench  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Install the SWE extra: pip install -e '.[swe]' (or uv sync --extra swe)"
        ) from e


def _first_instance_id(dataset_name: str, split: str) -> str:
    from swebench.harness.constants import KEY_INSTANCE_ID
    from swebench.harness.utils import load_swebench_dataset

    instances = load_swebench_dataset(dataset_name, split, None)
    if not instances:
        raise RuntimeError(f"No instances for {dataset_name!r} split={split!r}")
    return str(instances[0][KEY_INSTANCE_ID])


def _verify_latest_harness_report(task_id: str) -> tuple[Path, str]:
    """
    Find a harness report for the latest ``auto-harness-*`` run.

    Returns ``(path, kind)`` where ``kind`` is ``"per_instance"`` or ``"aggregate"``.

    When an instance **errors** (bad patch, harness failure), swebench often writes only a
    **top-level** JSON (``{model}.{run_id}.json`` in ``report_dir``, i.e. cwd) and does not
    create ``logs/run_evaluation/.../<instance_id>/report.json``. Stub patches frequently
    hit that path; Docker + harness still ran end-to-end.
    """
    from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR

    from agent.swe_agent import model_name_for_predictions

    root = Path(RUN_EVALUATION_LOG_DIR)
    candidates = sorted(
        root.glob("auto-harness-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No harness run directory under {root} (expected auto-harness-*). "
            "If the harness failed early, check Docker and swebench logs."
        )
    run_dir = candidates[0]
    model_name = model_name_for_predictions()
    model_safe = model_name.replace("/", "__")
    per_instance = run_dir / model_safe / task_id / LOG_REPORT
    if per_instance.is_file():
        return per_instance, "per_instance"

    # Aggregate report (see harness stdout: "Report written to {model}.{run_id}.json")
    aggregate = Path.cwd() / f"{model_name}.{run_dir.name}.json"
    if aggregate.is_file():
        return aggregate, "aggregate"

    raise FileNotFoundError(
        f"No per-instance report at {per_instance} and no aggregate at {aggregate} "
        f"(run {run_dir.name})."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SWE-Bench Docker harness end-to-end (one instance).",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_FILE,
        help=f"YAML config (default: {CONFIG_FILE})",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="SWE-Bench instance id (default: first id in the configured split)",
    )
    parser.add_argument(
        "--skip-report-check",
        action="store_true",
        help="Do not assert a report file exists under logs/run_evaluation (faster to skip if flaky)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_name = cfg.get("swe_dataset", "SWE-bench/SWE-bench_Lite")
    split = cfg.get("split", "dev")

    print("[testing] Step 1/4: checking Docker ...")
    if not _docker_available():
        print(
            "[testing] ERROR: `docker info` failed. Start Docker and retry "
            "(Compose: mount /var/run/docker.sock).",
            file=sys.stderr,
        )
        return 1

    print("[testing] Step 2/4: checking swebench import ...")
    try:
        _require_swebench()
    except RuntimeError as e:
        print(f"[testing] ERROR: {e}", file=sys.stderr)
        return 1

    task_id = args.task_id
    if not task_id:
        print(f"[testing] resolving first instance id ({dataset_name!r}, split={split!r}) ...")
        try:
            task_id = _first_instance_id(dataset_name, split)
        except Exception as e:
            print(f"[testing] ERROR: {e}", file=sys.stderr)
            return 1

    print(f"[testing] Step 3/4: running harness for instance {task_id!r} ...")
    print(
        "[testing] (This may take a long time on first run: dataset + Docker images + tests.)",
    )

    from benchmark import make_runner_from_config

    # Force real harness; avoid env accidentally skipping Docker eval.
    cfg_run: dict[str, Any] = {**cfg, "benchmark": "swe", "split": split, "swe_skip_harness": False}

    stub_backup = os.environ.get("SWE_STUB_PATCH")
    skip_backup = os.environ.pop("SWE_SKIP_HARNESS", None)
    os.environ["SWE_STUB_PATCH"] = _MIN_STUB_PATCH

    try:
        runner = make_runner_from_config(cfg_run, split=split)
        from agent.swe.runner import SWEBenchRunner

        if not isinstance(runner, SWEBenchRunner):
            print(
                f"[testing] ERROR: expected SWEBenchRunner, got {type(runner)!r}",
                file=sys.stderr,
            )
            return 1

        results = runner.run(task_ids=[task_id])
    except Exception as e:
        print(f"[testing] ERROR: harness run failed: {e}", file=sys.stderr)
        return 1
    finally:
        if skip_backup is not None:
            os.environ["SWE_SKIP_HARNESS"] = skip_backup
        if stub_backup is None:
            os.environ.pop("SWE_STUB_PATCH", None)
        else:
            os.environ["SWE_STUB_PATCH"] = stub_backup

    if len(results) != 1 or task_id not in results:
        print(f"[testing] ERROR: unexpected results: {results!r}", file=sys.stderr)
        return 1

    reward = results[task_id]
    print(f"[testing] benchmark reward for {task_id!r}: {reward:.4f} (expected 0.0 for stub patch)")

    if not args.skip_report_check:
        print("[testing] Step 4/4: verifying harness report on disk ...")
        try:
            report_path, report_kind = _verify_latest_harness_report(task_id)
            data = json.loads(report_path.read_text(encoding="utf-8"))
            if report_kind == "aggregate":
                print(
                    f"[testing] aggregate report OK: {report_path} "
                    "(per-instance report.json missing — usual when the instance errors, "
                    "e.g. stub patch does not apply)",
                )
            else:
                inner = data.get(task_id)
                if isinstance(inner, dict) and "resolved" in inner:
                    print(
                        f"[testing] report resolved={inner.get('resolved')} ({report_path})",
                    )
                else:
                    print(f"[testing] report OK: {report_path}")
        except (OSError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"[testing] ERROR: report verification failed: {e}", file=sys.stderr)
            return 1
    else:
        print("[testing] Step 4/4: skipped (--skip-report-check).")

    print("[testing] OK — Docker + official harness path completed end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
