"""
SWE-Bench run artifacts: manifest, failure labels, one-page summary, log collapse.

Called from ``benchmark.py`` after ``train_results.json`` is written.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from agent.swe.log_collapse import collapse_run_instance_log_text


def _swebench_paths():
    try:
        from swebench.harness.constants import (  # type: ignore[import-untyped]
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
            LOG_INSTANCE,
            LOG_REPORT,
            RUN_EVALUATION_LOG_DIR,
            TESTS_FAILED,
        )

        return {
            "APPLY_PATCH_FAIL": APPLY_PATCH_FAIL,
            "APPLY_PATCH_PASS": APPLY_PATCH_PASS,
            "LOG_INSTANCE": LOG_INSTANCE,
            "LOG_REPORT": LOG_REPORT,
            "RUN_EVALUATION_LOG_DIR": Path(RUN_EVALUATION_LOG_DIR),
            "TESTS_FAILED": TESTS_FAILED,
        }
    except ImportError:
        return {
            "APPLY_PATCH_FAIL": ">>>>> Patch Apply Failed",
            "APPLY_PATCH_PASS": ">>>>> Applied Patch",
            "LOG_INSTANCE": "run_instance.log",
            "LOG_REPORT": "report.json",
            "RUN_EVALUATION_LOG_DIR": Path("logs/run_evaluation"),
            "TESTS_FAILED": ">>>>> Some Tests Failed",
        }


def _repo_prefix(instance_id: str) -> str:
    if "__" in instance_id:
        return instance_id.split("__", 1)[0]
    return instance_id


def _failure_label(
    *,
    score: float,
    patch: str,
    skip_harness: bool,
    harness_run_id: str | None,
    model_safe: str,
    iid: str,
    markers: dict[str, Any],
    run_eval_dir: Path,
) -> tuple[str, dict[str, str | None]]:
    """
    Return (label, paths dict with report_path, log_path, log_summary_relative or None).

    Labels: resolved | empty_patch | harness_skipped | no_per_instance_report |
             apply_failed | tests_failed | unknown
    """
    patch = patch or ""
    paths: dict[str, str | None] = {"report_path": None, "log_path": None, "log_summary_path": None}

    if score >= 0.5:
        return "resolved", paths

    if not patch.strip():
        return "empty_patch", paths

    if skip_harness or not harness_run_id:
        return "harness_skipped", paths

    LOG_REPORT = markers["LOG_REPORT"]
    LOG_INSTANCE = markers["LOG_INSTANCE"]
    base = run_eval_dir / harness_run_id / model_safe / iid
    report_path = base / LOG_REPORT
    log_path = base / LOG_INSTANCE
    paths["report_path"] = str(report_path)
    paths["log_path"] = str(log_path)

    report_data: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            report_data = None

    log_text = ""
    if log_path.is_file():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""

    collapsed = collapse_run_instance_log_text(log_text) if log_text else ""

    APPLY_PATCH_FAIL = markers["APPLY_PATCH_FAIL"]
    TESTS_FAILED = markers["TESTS_FAILED"]

    if report_data is None:
        if collapsed and APPLY_PATCH_FAIL in collapsed:
            return "apply_failed", paths
        return "no_per_instance_report", paths

    inner = report_data.get(iid)
    if isinstance(inner, dict) and inner.get("resolved") is True:
        return "resolved", paths

    if collapsed and APPLY_PATCH_FAIL in collapsed:
        return "apply_failed", paths

    if collapsed and TESTS_FAILED in collapsed:
        return "tests_failed", paths

    if isinstance(inner, dict) and inner.get("resolved") is False:
        return "tests_failed", paths

    return "unknown", paths


def finalize_swe_benchmark_artifacts(
    runner: Any,
    results: dict[str, float],
    train_results_path: str,
) -> None:
    """
    Write ``predictions_latest.jsonl``, ``workspace/swe/last_run.json``,
    ``workspace/swe/last_run_summary.md``, and per-instance collapsed logs under
    ``workspace/swe/last_run_logs/`` for failing instances when logs exist.

    Expects ``runner._swe_artifact_ctx`` set by ``SWEBenchRunner.run``.
    """
    ctx = getattr(runner, "_swe_artifact_ctx", None)
    if not ctx:
        return

    markers = _swebench_paths()
    run_eval_dir: Path = markers["RUN_EVALUATION_LOG_DIR"]
    preds_path = Path(ctx["preds_path"])
    predictions_dir = Path(ctx["predictions_dir"])
    benchmark_run_id = ctx["benchmark_run_id"]
    harness_run_id = ctx.get("harness_run_id")
    skip_harness = bool(ctx["skip_harness"])
    ordered_ids: list[str] = ctx["ordered_ids"]
    predictions_map: dict[str, dict[str, str]] = ctx["predictions_map"]
    KEY_PREDICTION = ctx["KEY_PREDICTION"]
    KEY_MODEL = ctx["KEY_MODEL"]
    if ordered_ids:
        model_safe = str(predictions_map[ordered_ids[0]][KEY_MODEL]).replace("/", "__")
    else:
        model_safe = str(ctx.get("model_name_predictions") or "unknown").replace("/", "__")

    latest = predictions_dir / "predictions_latest.jsonl"
    try:
        if preds_path.is_file():
            latest.write_bytes(preds_path.read_bytes())
    except OSError:
        pass

    summaries_dir = predictions_dir / "last_run_logs"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    instances_out: dict[str, Any] = {}
    for iid in ordered_ids:
        pred = predictions_map.get(iid, {})
        patch = pred.get(KEY_PREDICTION, "")
        score = float(results.get(iid, 0.0))
        label, paths = _failure_label(
            score=score,
            patch=patch,
            skip_harness=skip_harness,
            harness_run_id=harness_run_id,
            model_safe=model_safe,
            iid=iid,
            markers=markers,
            run_eval_dir=run_eval_dir,
        )

        rel_summary: str | None = None
        log_path_str = paths.get("log_path")
        if log_path_str and score < 0.5 and label not in ("harness_skipped", "empty_patch"):
            lp = Path(log_path_str)
            if lp.is_file():
                collapsed = collapse_run_instance_log_text(lp.read_text(encoding="utf-8", errors="replace"))
                cap = 48_000
                if len(collapsed) > cap:
                    collapsed = collapsed[:cap] + "\n[... truncated ...]\n"
                safe_name = iid.replace("/", "__") + ".summary.txt"
                out_f = summaries_dir / safe_name
                out_f.write_text(collapsed, encoding="utf-8")
                try:
                    rel_summary = os.path.relpath(out_f, start=os.getcwd())
                except ValueError:
                    rel_summary = str(out_f)
                paths["log_summary_path"] = rel_summary

        instances_out[iid] = {
            "score": score,
            "failure_label": label,
            "repo_prefix": _repo_prefix(iid),
            "patch_empty": not (patch or "").strip(),
            **paths,
        }

    manifest = {
        "schema_version": 1,
        "benchmark_run_id": benchmark_run_id,
        "train_results_path": train_results_path,
        "split": ctx["split"],
        "dataset_name": ctx["dataset_name"],
        "predictions_file": str(preds_path),
        "predictions_latest": str(latest),
        "swe_harness_run_id": harness_run_id,
        "skip_harness": skip_harness,
        "model_name_predictions": ctx.get("model_name_predictions"),
        "summary_markdown": str(predictions_dir / "last_run_summary.md"),
        "collapsed_logs_dir": str(summaries_dir),
        "instances": instances_out,
    }

    manifest_path = predictions_dir / "last_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # One-page markdown summary
    lines: list[str] = []
    lines.append("# SWE-Bench last run\n")
    lines.append(f"- **benchmark_run_id:** `{benchmark_run_id}`")
    lines.append(f"- **harness_run_id:** `{harness_run_id or 'none'}`")
    lines.append(f"- **split:** `{ctx['split']}`  **dataset:** `{ctx['dataset_name']}`")
    lines.append(f"- **skip_harness:** {skip_harness}")
    lines.append(f"- **predictions:** `{preds_path}` → **`{latest}`** (copy)")
    lines.append(f"- **train_results:** `{train_results_path}`")
    lines.append(f"- **manifest:** `{manifest_path}`")
    lines.append("")

    passed = sum(1 for s in results.values() if s >= 0.5)
    total = len(results)
    lines.append(f"## Scores: {passed}/{total} pass (mean {sum(results.values()) / total if total else 0:.4f})\n")

    labels = [instances_out[i]["failure_label"] for i in ordered_ids if i in instances_out]
    by_label = Counter(labels)
    lines.append("## Failure labels (counts)\n")
    for lab, c in sorted(by_label.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- **{lab}:** {c}")
    lines.append("")

    by_repo = Counter(_repo_prefix(i) for i in ordered_ids if instances_out[i]["score"] < 0.5)
    if by_repo:
        lines.append("## Failing instances by repo prefix\n")
        for repo, c in sorted(by_repo.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- **{repo}:** {c}")
        lines.append("")

    lines.append("## Per-instance\n")
    lines.append("| instance | score | label | log summary |")
    lines.append("|----------|-------|-------|-------------|")
    for iid in ordered_ids:
        row = instances_out[iid]
        sm = row.get("log_summary_path") or "—"
        lines.append(f"| `{iid}` | {row['score']:.1f} | `{row['failure_label']}` | {sm} |")

    summary_path = predictions_dir / "last_run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[swe] manifest: {manifest_path}")
    print(f"[swe] summary:  {summary_path}")
