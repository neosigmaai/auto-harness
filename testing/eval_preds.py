"""
Evaluate an existing preds.json (from baseline_mini_swe.py) against the
official SWE-bench harness and print per-instance 0/1 scores.

Usage:
  python testing/eval_preds.py testing/results/<run_id>/preds.json

  # inside Docker (has Docker socket mounted):
  docker compose run --rm autoeval python testing/eval_preds.py testing/results/<run_id>/preds.json

Options:
  --dataset    HuggingFace dataset path (default: princeton-nlp/SWE-Bench_Lite)
  --split      Dataset split (default: dev)
  --workers    Parallel harness workers (default: 4)
  --timeout    Seconds per instance (default: 1800)
  --namespace  Docker image namespace (default: swebench)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require_swebench():
    try:
        from swebench.harness.constants import (  # type: ignore[import-untyped]
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
            LOG_REPORT,
            RUN_EVALUATION_LOG_DIR,
        )
        return KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_REPORT, RUN_EVALUATION_LOG_DIR
    except ImportError as e:
        raise SystemExit(
            "SWE-bench harness not installed. Run: uv sync --extra swe"
        ) from e


def _preds_json_to_jsonl(preds_json: Path, out_jsonl: Path) -> list[str]:
    """Convert mini-swe-agent preds.json → harness-compatible JSONL. Returns ordered instance IDs."""
    KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, _, _ = _require_swebench()

    data = json.loads(preds_json.read_text())
    rows = []
    for iid, entry in data.items():
        rows.append({
            KEY_INSTANCE_ID: iid,
            KEY_MODEL: entry.get("model_name_or_path", "mini-swe-agent"),
            KEY_PREDICTION: entry.get("model_patch", ""),
        })

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return [r[KEY_INSTANCE_ID] for r in rows]


def _run_harness(dataset: str, split: str, jsonl_path: Path, instance_ids: list[str],
                 workers: int, timeout: int, run_id: str, namespace: str) -> None:
    from swebench.harness.run_evaluation import main as run_eval_main  # type: ignore[import-untyped]

    run_eval_main(
        dataset_name=dataset,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(jsonl_path),
        max_workers=workers,
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


def _read_scores(instance_ids: list[str], run_id: str, jsonl_path: Path) -> dict[str, float]:
    KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_REPORT, RUN_EVALUATION_LOG_DIR = _require_swebench()

    # Derive model name from the JSONL (harness uses it as subfolder name)
    model_name = "mini-swe-agent"
    try:
        first_line = jsonl_path.read_text().splitlines()[0]
        model_name = json.loads(first_line).get(KEY_MODEL, model_name)
    except Exception:
        pass
    model_name = model_name.replace("/", "__")

    scores: dict[str, float] = {}
    for iid in instance_ids:
        report_path = RUN_EVALUATION_LOG_DIR / run_id / model_name / iid / LOG_REPORT
        if not report_path.exists():
            scores[iid] = 0.0
            continue
        try:
            data = json.loads(report_path.read_text())
            scores[iid] = 1.0 if data.get(iid, {}).get("resolved") else 0.0
        except Exception:
            scores[iid] = 0.0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate preds.json with SWE-bench harness")
    parser.add_argument("preds_json", type=Path, help="Path to preds.json from baseline_mini_swe.py")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-Bench_Lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--namespace", default="swebench")
    args = parser.parse_args()

    preds_json = args.preds_json.resolve()
    if not preds_json.exists():
        raise SystemExit(f"File not found: {preds_json}")

    run_id = f"eval-baseline-{uuid.uuid4().hex[:10]}"
    jsonl_path = preds_json.parent / f"{run_id}.jsonl"

    print(f"[eval_preds] converting {preds_json.name} → {jsonl_path.name}", flush=True)
    instance_ids = _preds_json_to_jsonl(preds_json, jsonl_path)
    print(f"[eval_preds] {len(instance_ids)} instances to evaluate", flush=True)
    print(f"[eval_preds] run_id: {run_id}", flush=True)

    print("[eval_preds] running harness...", flush=True)
    _run_harness(args.dataset, args.split, jsonl_path, instance_ids,
                 args.workers, args.timeout, run_id, args.namespace)

    scores = _read_scores(instance_ids, run_id, jsonl_path)

    resolved = [iid for iid, s in scores.items() if s == 1.0]
    failed = [iid for iid, s in scores.items() if s == 0.0]
    total = len(scores)

    print()
    print(f"{'='*60}")
    print(f"Results: {len(resolved)}/{total} resolved ({100*len(resolved)/total:.1f}%)")
    print(f"{'='*60}")
    if resolved:
        print("\nRESOLVED:")
        for iid in sorted(resolved):
            print(f"  [1] {iid}")
    if failed:
        print("\nFAILED:")
        for iid in sorted(failed):
            print(f"  [0] {iid}")


if __name__ == "__main__":
    main()
