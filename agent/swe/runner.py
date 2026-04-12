"""
SWE-Bench runner: predictions JSONL + optional official harness (Docker).

V0 default: stub empty patches + skip_harness -> scores 0.0 without Docker.
Set swe_skip_harness: false and install the ``swe`` extra for real evaluation.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from benchmark import BenchmarkRunner

from agent.swe_agent import SWEInstanceContext, generate_patch, model_name_for_predictions


def _require_swebench():
    try:
        from swebench.harness.constants import (  # type: ignore[import-untyped]
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
            LOG_REPORT,
            RUN_EVALUATION_LOG_DIR,
        )
    except ImportError as e:
        raise RuntimeError(
            "SWE-Bench harness requires the 'swe' extra: pip install -e '.[swe]'"
        ) from e
    return KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_REPORT, RUN_EVALUATION_LOG_DIR


def _load_instances(dataset_name: str, split: str, task_ids: list[str] | None) -> list[dict[str, Any]]:
    from swebench.harness.utils import load_swebench_dataset  # type: ignore[import-untyped]

    return load_swebench_dataset(dataset_name, split, task_ids)


def _write_predictions_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prediction_row(instance: dict[str, Any], patch: str) -> dict[str, str]:
    KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, _, _ = _require_swebench()
    return {
        KEY_INSTANCE_ID: str(instance[KEY_INSTANCE_ID]),
        KEY_MODEL: model_name_for_predictions(),
        KEY_PREDICTION: patch,
    }


def _scores_from_stub(instance_ids: list[str]) -> dict[str, float]:
    return {iid: 0.0 for iid in instance_ids}


def _scores_after_harness(
    predictions: dict[str, dict[str, str]],
    instance_ids: list[str],
    run_id: str,
) -> dict[str, float]:
    _, KEY_MODEL, KEY_PREDICTION, LOG_REPORT, RUN_EVALUATION_LOG_DIR = _require_swebench()
    out: dict[str, float] = {}
    for iid in instance_ids:
        pred = predictions.get(iid)
        if not pred or not pred.get(KEY_PREDICTION):
            out[iid] = 0.0
            continue
        report_path = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / pred[KEY_MODEL].replace("/", "__")
            / iid
            / LOG_REPORT
        )
        if not report_path.exists():
            out[iid] = 0.0
            continue
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            resolved = bool(data.get(iid, {}).get("resolved"))
            out[iid] = 1.0 if resolved else 0.0
        except (json.JSONDecodeError, KeyError, TypeError):
            out[iid] = 0.0
    return out


def _run_harness(
    dataset_name: str,
    split: str,
    preds_path: Path,
    instance_ids: list[str] | None,
    max_workers: int,
    timeout: int,
    run_id: str,
    namespace: str | None,
) -> None:
    from swebench.harness.run_evaluation import main as run_eval_main  # type: ignore[import-untyped]

    run_eval_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(preds_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        namespace=namespace,
        rewrite_reports=False,
        modal=False,
        instance_image_tag="latest",
        env_image_tag="latest",
        report_dir=".",
    )


class SWEBenchRunner(BenchmarkRunner):
    """
    SWE-Bench via predictions JSONL + optional ``swebench.harness.run_evaluation``.

    Default (swe_skip_harness / SWE_SKIP_HARNESS): no Docker; stub patches -> 0.0.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        agent_model: str | None = None,
        skip_harness: bool = True,
        use_llm: bool = True,
        max_workers: int = 4,
        timeout: int = 1_800,
        namespace: str | None = "swebench",
        predictions_dir: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "unknown")
        self.skip_harness = skip_harness or os.getenv("SWE_SKIP_HARNESS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.use_llm = use_llm
        self.max_workers = max_workers
        self.timeout = timeout
        self.namespace = namespace
        self.predictions_dir = Path(predictions_dir or os.path.join("workspace", "swe"))

    def run(self, task_ids: list[str] | None = None) -> dict[str, float]:
        instances = _load_instances(self.dataset_name, self.split, task_ids)
        if not instances:
            return {}

        KEY_INSTANCE_ID, _, _, _, _ = _require_swebench()
        rows: list[dict[str, str]] = []
        ordered_ids: list[str] = []
        for inst in instances:
            iid = str(inst[KEY_INSTANCE_ID])
            ordered_ids.append(iid)
            raw_ftp = inst.get("FAIL_TO_PASS") or []
            fail_to_pass = (
                raw_ftp if isinstance(raw_ftp, list)
                else [t.strip() for t in raw_ftp.split(",") if t.strip()]
            )
            hints = (inst.get("hints_text") or "").strip() or None
            ctx = SWEInstanceContext(
                instance_id=iid,
                problem_statement=str(inst.get("problem_statement") or inst.get("text") or ""),
                repo=str(inst.get("repo", "") or None) or None,
                workspace_root=None,
                use_llm=self.use_llm,
                fail_to_pass=fail_to_pass or None,
                hints_text=hints,
            )
            patch = generate_patch(ctx)
            rows.append(_prediction_row(inst, patch))

        preds_path = self.predictions_dir / f"predictions_{uuid.uuid4().hex[:8]}.jsonl"
        _write_predictions_jsonl(preds_path, rows)

        _, KEY_MODEL, KEY_PREDICTION, _, _ = _require_swebench()
        predictions_map = {
            r[KEY_INSTANCE_ID]: {
                KEY_INSTANCE_ID: r[KEY_INSTANCE_ID],
                KEY_MODEL: r[KEY_MODEL],
                KEY_PREDICTION: r[KEY_PREDICTION],
            }
            for r in rows
        }

        benchmark_run_id = uuid.uuid4().hex[:12]
        model_name_predictions = model_name_for_predictions()
        all_empty = all(not (predictions_map[iid].get(KEY_PREDICTION) or "") for iid in ordered_ids)

        harness_run_id: str | None = None
        scores: dict[str, float]
        if self.skip_harness or all_empty:
            print(
                f"[swe] skip_harness={self.skip_harness} empty_patches={all_empty} -> scores 0.0 "
                f"(predictions: {preds_path})"
            )
            scores = _scores_from_stub(ordered_ids)
        else:
            harness_run_id = f"auto-harness-{uuid.uuid4().hex[:12]}"
            print(f"[swe] running official harness run_id={harness_run_id} preds={preds_path}")
            _run_harness(
                self.dataset_name,
                self.split,
                preds_path,
                task_ids,
                self.max_workers,
                self.timeout,
                harness_run_id,
                self.namespace,
            )
            scores = _scores_after_harness(predictions_map, ordered_ids, harness_run_id)

        self._swe_artifact_ctx = {
            "benchmark_run_id": benchmark_run_id,
            "preds_path": str(preds_path),
            "predictions_dir": str(self.predictions_dir),
            "ordered_ids": ordered_ids,
            "predictions_map": predictions_map,
            "skip_harness": self.skip_harness,
            "all_empty": all_empty,
            "harness_run_id": harness_run_id,
            "split": self.split,
            "dataset_name": self.dataset_name,
            "KEY_MODEL": KEY_MODEL,
            "KEY_PREDICTION": KEY_PREDICTION,
            "model_name_predictions": model_name_predictions,
        }
        return scores
