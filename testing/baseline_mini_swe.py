"""
Run the **local** `mini-swe-agent` repo’s batch SWE-Bench driver (Docker per instance).

This invokes `minisweagent.run.benchmarks.swebench` — the same entry as
`mini-extra swebench` — **not** `auto-harness`’s `agent/swe_agent.py` wrapper.

Expected layout (sibling repos):

    NEOSIGMA/
      auto-harness/          ← this repo (run from here)
      mini-swe-agent/        ← upstream clone

Or set ``MINI_SWE_AGENT_ROOT`` to an absolute path (required inside Docker; see
``docker-compose.yml``).

What it runs (defaults):

- **Subset:** ``lite`` → HuggingFace ``princeton-nlp/SWE-Bench_Lite`` (mini’s mapping)
- **Split:** ``dev`` (~23 instances)
- **Environment:** ``docker`` (SWE-Bench images; needs Docker daemon / socket)
- **Model:** ``gpt-5.4`` or ``AGENT_MODEL``

Outputs go to ``testing/results/<run_id>/`` (``preds.json``, trajectories, logs).

Requirements:

- Editable or PYTHONPATH install of mini-swe-agent **or** run via this script which sets ``PYTHONPATH`` to ``<mini>/src``
- ``OPENAI_API_KEY`` (or provider keys per mini config)
- Docker daemon on the host; the **auto-harness** image includes the ``docker`` **CLI** so mini’s Docker backend works inside Compose (rebuild the image if you see ``No such file or directory: 'docker'``).

Usage::

  # Host (sibling mini-swe-agent)
  python testing/baseline_mini_swe.py

  python testing/baseline_mini_swe.py --workers 4 --model gpt-5.4

  # Docker (compose mounts ../mini-swe-agent → /mini-swe-agent; sets MINI_SWE_AGENT_ROOT)
  docker compose run --rm autoeval python testing/baseline_mini_swe.py

Pass extra flags through to mini’s CLI after ``--``::

  python testing/baseline_mini_swe.py -- --redo-existing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_mini_swe_root() -> Path:
    env = os.environ.get("MINI_SWE_AGENT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"MINI_SWE_AGENT_ROOT is not a directory: {p}")
    # auto-harness/testing/ -> auto-harness -> NEOSIGMA/mini-swe-agent
    sibling = _REPO_ROOT.parent / "mini-swe-agent"
    if sibling.is_dir():
        return sibling.resolve()
    raise FileNotFoundError(
        "Could not find mini-swe-agent. Clone it next to auto-harness "
        "(e.g. NEOSIGMA/mini-swe-agent) or set MINI_SWE_AGENT_ROOT."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run local mini-swe-agent batch SWE-Bench (Lite dev, Docker)",
    )
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("AGENT_MODEL", "gpt-5.4"),
        help="Model name for Litellm (default: AGENT_MODEL or gpt-5.4)",
    )
    p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=2,
        help="Parallel workers (default: 2)",
    )
    p.add_argument(
        "--subset",
        default="lite",
        help="mini-swe-agent subset key or HF path (default: lite)",
    )
    p.add_argument(
        "--split",
        default="dev",
        help="Dataset split (default: dev ≈23 for Lite)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: testing/results/<run_id>)",
    )
    p.add_argument(
        "--environment-class",
        default="docker",
        help="mini environment (default: docker)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args after -- passed to mini swebench (e.g. -- --slice 0:3)",
    )
    args = p.parse_args()
    extra = [x for x in args.extra if x != "--"]

    mini_root = _resolve_mini_swe_root()
    cfg_file = mini_root / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Missing mini-swe-agent config: {cfg_file}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out_dir = args.output
    if out_dir is None:
        out_dir = _REPO_ROOT / "testing" / "results" / run_id
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src = str(mini_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        args.subset,
        "--split",
        args.split,
        "-m",
        args.model,
        "-w",
        str(args.workers),
        "--environment-class",
        args.environment_class,
        "-c",
        str(cfg_file),
        "-o",
        str(out_dir),
    ]
    cmd.extend(extra)

    print(f"[baseline_mini_swe] mini-swe-agent root: {mini_root}", flush=True)
    print(f"[baseline_mini_swe] output: {out_dir}", flush=True)
    print(f"[baseline_mini_swe] command: {' '.join(cmd)}", flush=True)

    if args.dry_run:
        return

    r = subprocess.run(cmd, cwd=str(mini_root), env=env)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
