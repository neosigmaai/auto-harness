# auto-harness

> Give a coding agent a benchmark and an agent file. Let it iterate overnight. It reads failures, improves the system prompt and tools, gates every change against a self-maintained eval suite, and repeats.

This repo is a simplified version of our auto-harness agent setup. We demonstrate our system on Tau3 benchmark tasks where the agent’s score improves from 0.56 to 0.78 (~40% jump) while mining failures and auto maintaining live evals. If you are curious to learn more, read the full blog here - https://www.neosigma.ai/blog/self-improving-agentic-systems.

The loop is defined in **`PROGRAM.md`**, with mode-specific detail in **`TAU-PROGRAM.md`** (tau-bench) or **`SWE-PROGRAM.md`** (SWE-Bench), matching `benchmark` in `experiment_config.yaml`. The coding agent edits `agent/` and appends findings to `workspace/learnings.md` after each iteration.

---

## How it works

```
run benchmark → analyze → improve agent/agent.py → gate → record → update learnings → repeat
```

- **`agent/agent.py`** — the agent being optimized (tau2 `HarnessAgent`)
- **`benchmark.py`** — runs your benchmark, returns per-task rewards
- **`gating.py`** — two-step gate: eval suite pass rate + full test val_score
- **`record.py`** — appends iteration results to `workspace/results.tsv`
- **`workspace/suite.json`** — the regression suite the coding agent maintains
- **`workspace/learnings.md`** — persistent log of patterns, what worked, and requests to the human
- **`PROGRAM.md`** — shared loop, gate, rules
- **`TAU-PROGRAM.md`** / **`SWE-PROGRAM.md`** — mode-specific instructions (tau vs SWE)

---

## Quick start — SWE-Bench

For **`agent/swe_agent.py`** and the official Docker harness (`swe_skip_harness: false`), use the bundled script once, then set your API key.

**Requirements:** Docker Desktop (or Docker Engine) running, `OPENAI_API_KEY`, and enough disk space for SWE-Bench images.

```bash
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

bash scripts/setup_swe.sh          # creates .env, data/, experiment_config.yaml, docker compose build
# Edit .env — set OPENAI_API_KEY

docker compose run --rm autoeval python prepare.py
docker compose run --rm autoeval python benchmark.py
```

Details: **`SWE-PROGRAM.md`**. Config reference: **`experiment_config.swe.yaml.example`** (also copied to `experiment_config.yaml` by the script if missing). For a fast dry run without grading patches in Docker, set `swe_skip_harness: true` in `experiment_config.yaml`.

---

## Quick start — tau-bench

**Requirements:** Docker, an `OPENAI_API_KEY`, and a coding agent (Claude Code, Codex CLI, or similar).

```bash
# 1. Clone the repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY; TAU2_DATA_DIR defaults to ./data (created by setup_swe.sh or mkdir -p data)

# 3. Configure the experiment
cp experiment_config.yaml.template experiment_config.yaml
# edit experiment_config.yaml — set benchmark: tau, domain, splits (see below)

# 4. Build the Docker image (installs tau-bench and all deps via uv)
docker compose build

# 5. Initialize the workspace
docker compose run --rm autoeval python prepare.py

# 6. Run the benchmark once to verify it works
docker compose run --rm autoeval python benchmark.py
```

---

## experiment_config.yaml

- **tau:** copy `experiment_config.yaml.template` → `experiment_config.yaml`.
- **SWE:** use `experiment_config.swe.yaml.example` (or run `bash scripts/setup_swe.sh`, which copies it when `experiment_config.yaml` is missing).

| Flag | What to set |
|------|-------------|
| `benchmark` | `tau` or `swe` (SWE needs `uv sync --extra swe` / `.[swe]`). |
| `agent_model` | Model name. |
| `threshold` | Eval-suite gate (Step 1). |
| `domain` | tau only. |
| `split` | `benchmark.py` split (SWE Lite: `dev` ~23, `test` ~300). |
| `gate_split` | Gating Step 2 split. |
| `prepare_baseline_role` | `train` vs full `gate` for `prepare.py` baseline. |
| `max_concurrency` | Parallelism. |
| `mini` | `true` → cap default benchmark, gating Step 2, and prepare to 3 tasks. |
| `mini_task_ids` | Those 3 ids; or SWE: `swe_default_task_ids` (mini uses first 3). |
| `swe_dataset` | HF id when `benchmark: swe`. |
| `swe_skip_harness` | `true` = no Docker harness; `false` = grade patches (needs Docker). |
| `swe_use_llm` | `false` = empty patches. |
| `swe_max_workers`, `swe_timeout`, `swe_namespace` | Harness tuning when `swe_skip_harness: false`. |
| `swe_default_task_ids` | Optional default instance list for `benchmark.py`. |
| `swe_gate_task_ids`, `swe_gate_match_default_task_ids` | Optional gating Step 2 subset. |
| `swe_predictions_dir` | Optional; default `workspace/swe`. |

---

## Running the loop

Point your coding agent at the repo and prompt:

```
Read PROGRAM.md and the program doc for your benchmark mode (TAU-PROGRAM.md or SWE-PROGRAM.md), then start the optimization loop.
```

The agent will read results, diagnose failures, edit `agent/agent.py`, gate the change, record the result, and repeat.

---

## Using tau-bench

tau-bench ([sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)) is included as a dependency in `pyproject.toml` and installed automatically during `docker compose build`.

Set `TAU2_DATA_DIR` in your `.env` to point at the tau2 data directory, then configure the domain and split in `experiment_config.yaml`:

```yaml
domain: "retail"
split: "train"      # benchmark.py iterates on train
gate_split: "test"  # gating.py gates on test
```

```bash
# Run the full tau-bench eval (domain/split from experiment_config.yaml)
docker compose run autoeval python benchmark.py

# Run specific tasks
docker compose run autoeval python benchmark.py --task-ids 0 1 42

# Override domain/split on the command line
docker compose run autoeval python benchmark.py --domain airline --split test
```

---

## Plugging in your own benchmark

Subclass `BenchmarkRunner` in `benchmark.py`:

```python
class MyBenchmarkRunner(BenchmarkRunner):
    def run(self, task_ids=None):
        # call your benchmark CLI or API
        # return {task_id: reward} where reward is 0.0–1.0
        ...
```

Swap it in `gating.py`:

```python
from benchmark import MyBenchmarkRunner
runner = MyBenchmarkRunner()
```

That's it. The loop is benchmark-agnostic.

---

## Eval suite

The coding agent self-maintains `workspace/suite.json` — task IDs it must always pass. When it finds a new failure pattern, it adds the task.

`gating.py` runs two checks before any change is committed:

1. **Eval suite**: tasks in `suite.json` must pass at ≥ threshold (default 80%)
2. **Full test (test split)**: full benchmark on the test split; mean reward must be ≥ the best score seen so far in `results.tsv`

---

## Project structure

```
agent/agent.py          the agent under optimization — only file the coding agent edits
agent/swe/              SWE-Bench runner, diagnostics, log collapse (used when `benchmark: swe`)
benchmark.py            benchmark execution layer (abstract + tau-bench example)
gating.py               two-step gate, calls benchmark.py
prepare.py              workspace initialization (run once)
record.py               appends iteration result to results.tsv
PROGRAM.md              shared loop instructions for the coding agent
TAU-PROGRAM.md          tau-bench mode (when benchmark: tau)
SWE-PROGRAM.md          SWE-Bench mode (when benchmark: swe)
Dockerfile              container definition
docker-compose.yml      mounts agent/ and workspace/, passes env vars
workspace/
  suite.json            regression eval suite (task IDs + threshold)
  learnings.md          persistent log: patterns, what worked, requests to human
  results.tsv           iteration history (val_score, commit, evals, timestamp)
```

---

## Design

- **Program the loop, not the agent directly.** The human steers through `PROGRAM.md` and the mode doc (`TAU-PROGRAM.md` or `SWE-PROGRAM.md`); the coding agent edits under `agent/`.
- **Self-maintained evals.** The coding agent decides which tasks belong in the regression suite — no manual curation needed.
- **Learnings close the feedback loop.** After each iteration the agent writes `workspace/learnings.md`: what it tried, what worked, what it needs from the human (a missing tool, parallelism in the runner, a subagent for a slow step). Read it at session start to restore context instantly.
- **Gate everything.** No change is committed without passing both the eval suite and the full test score gate.

---

## Docker cleanup

Images and containers accumulate across runs:

```bash
# Remove stopped containers
docker container prune -f

# Full cleanup (images, build cache)
docker system prune -a -f
```
