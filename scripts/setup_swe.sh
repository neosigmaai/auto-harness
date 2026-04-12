#!/usr/bin/env bash
# One-time SWE-Bench setup: .env, data dir, experiment_config.yaml, Docker build.
# Usage (from repo root): bash scripts/setup_swe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[setup_swe] repo: $ROOT"

mkdir -p "$ROOT/data"

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "[setup_swe] Created .env — edit OPENAI_API_KEY before prepare/benchmark."
else
  echo "[setup_swe] .env already exists — not overwriting."
fi

if [[ ! -f "$ROOT/experiment_config.yaml" ]]; then
  cp "$ROOT/experiment_config.swe.yaml.example" "$ROOT/experiment_config.yaml"
  echo "[setup_swe] Created experiment_config.yaml from experiment_config.swe.yaml.example"
else
  echo "[setup_swe] experiment_config.yaml already exists — not overwriting."
  echo "          Compare with experiment_config.swe.yaml.example if you need SWE defaults."
fi

echo "[setup_swe] Building Docker image (uv + swe extra)..."
docker compose build

echo ""
echo "[setup_swe] Done. Next:"
echo "  1. Set OPENAI_API_KEY in .env"
echo "  2. docker compose run --rm autoeval python prepare.py"
echo "  3. docker compose run --rm autoeval python benchmark.py"
echo ""
echo "Docs: SWE-PROGRAM.md  ·  First harness run may pull large images."
