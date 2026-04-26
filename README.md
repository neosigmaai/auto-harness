# auto-harness

> Give a coding agent a benchmark and an agent file. Let it iterate overnight. It reads failures, improves the system prompt and tools, gates every change against a self-maintained eval suite, and repeats.

This repo is a simplified version of our auto-harness agent setup. We demonstrate our system on Tau3 benchmark tasks where the agent's score improves from 0.56 to 0.78 (~40% jump) while mining failures and auto maintaining live evals. If you are curious to learn more, read the full blog here - https://www.neosigma.ai/blog/self-improving-agentic-systems.

The loop is defined in `PROGRAM.md`. The coding agent edits `agent/agent.py` to improve the agent and appends findings to `workspace/learnings.md` after each iteration.

---

## Supported Benchmarks

| Benchmark | Domain | Tasks | Agent Interface |
|-----------|--------|-------|-----------------|
| **tau-bench** | Customer service (retail, airline, telecom) | retail: 114, airline: 50, telecom: 114 | Structured tool calls via tau2 |
| **Terminal-Bench 2.0** | Real-world terminal tasks (coding, sysadmin, security) | 89 | Bash commands via Harbor containers |
| **BIRD-Interact** | Interactive text-to-SQL (multi-turn, CRUD over Postgres) | lite: 300, full: 600 | Google ADK agent against a 3-service environment (user sim, DB env, system agent) |

---

## How it works

```
run benchmark → analyze → improve agent/agent.py → gate → record → update learnings → repeat
```

- **`agent/agent.py`** — the agent being optimized (copied from a benchmark-specific template)
- **`agent/templates/`** — starting-point templates for each benchmark (read-only)
- **`benchmark.py`** — runs your benchmark, returns per-task rewards
- **`gating.py`** — three-step gate: eval suite + full test val_score + suite promotion
- **`record.py`** — appends iteration results to `workspace/results.tsv`
- **`prepare.py`** — sets up workspace, copies templates, runs baseline
- **`program_templates/`** — benchmark-specific PROGRAM.md instructions
- **`PROGRAM.md`** — instructions the coding agent follows (copied from template by prepare.py)

---

## Quick start: Terminal-Bench 2.0

**Requirements:** `harbor` CLI, an `OPENAI_API_KEY`, an `E2B_API_KEY` (or `DAYTONA_API_KEY`), and a coding agent (Claude Code, Codex CLI, or similar).

```bash
# 1. Clone the repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Install harbor
uv tool install harbor

# 3. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY and E2B_API_KEY

# 4. Configure the experiment
cp experiment_config.yaml.template experiment_config.yaml
# edit experiment_config.yaml — uncomment the terminal-bench section

# 5. Initialize workspace + run baseline (runs all 89 tasks, generates train/test split)
python prepare.py

# 6. Start the optimization loop
# Point your coding agent at the repo and prompt:
#   "Read PROGRAM.md and start the optimization loop."
```

## Quick start: BIRD-Interact

**Requirements:** Docker (for Postgres), Python 3.12+, `git-lfs` (for the HF dataset), an `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` depending on model), and a coding agent.

```bash
# 1. Clone this repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 3. Configure the experiment
cp experiment_config.yaml.template experiment_config.yaml
# edit experiment_config.yaml — uncomment the BIRD-INTERACT section

# 4. Initialize — prepare.py auto-provisions everything:
#      - clones BIRD-Interact-ADK into ./bird_interact_adk/ (gitignored)
#      - creates an isolated .venv-adk with the ADK's deps
#      - clones the bird-interact-lite dataset from HuggingFace
#      - starts the Postgres Docker container
#      - runs the baseline (300 tasks) and generates the train/test split
python prepare.py

# 5. Start the optimization loop
# Point your coding agent at the repo and prompt:
#   "Read PROGRAM.md and start the optimization loop."
```

**Ground truth (one-time step):** The public BIRD-Interact dataset ships *without* gold SQL to prevent data leakage. On first run, `prepare.py` will detect this and print the exact email + merge command needed. Briefly:

1. Email `bird.bench25@gmail.com` with subject `[bird-interact-lite GT&Test Cases]`
2. Run the `combine_public_with_gt.py` script shown by prepare.py, using the jsonl you receive
3. Re-run `python prepare.py`

**What the integration adds:**

- `BirdInteractRunner` in `benchmark.py` — spawns the three ADK services (user simulator, DB environment, system agent) per run, drives `orchestrator.runner`, parses results into the harness reward format.
- `agent/helpers/bird_interact/bird_service.py` + `agent/helpers/bird_interact/bird_adk_runtime.py` — the harness-owned wrapper that lets your `agent/agent.py` be served as the BIRD system agent via FastAPI.
- `agent/templates/bird_interact.py` — faithful copy of the stock BIRD-Interact-ADK system agent, copied to `agent/agent.py` by `prepare.py` as the iteration starting point.
- `program_templates/bird_interact.md` — benchmark-specific guidance appended to `PROGRAM.md`.

**Known caveats:**
- GPT-5-family models reject explicit `temperature=0`; the template omits the temperature kwarg for those models (stock behavior preserved for all other models).
- `prepare.py` creates a separate `.venv-adk` inside `bird_interact_adk/` because the ADK's deps (google-adk, psycopg2, etc.) may conflict with other benchmarks' deps.
- Advanced users can point at an existing BIRD-Interact install via `bird_repo` + `bird_python_bin` in `experiment_config.yaml` to skip auto-provisioning.

## Quick start: tau-bench

**Requirements:** Docker, an `OPENAI_API_KEY`, and a coding agent.

```bash
# 1. Clone the repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY

# 3. Configure the experiment
cp experiment_config.yaml.template experiment_config.yaml
# edit experiment_config.yaml — uncomment the tau-bench section

# 4. Build the Docker image (installs tau-bench and all deps via uv)
docker compose build

# 5. Initialize the workspace + run baseline
docker compose run autoeval python prepare.py

# 6. Start the optimization loop
# Point your coding agent at the repo and prompt:
#   "Read PROGRAM.md and start the optimization loop."
```

---

## Running the loop

Point your coding agent at the repo and prompt:

```
Read PROGRAM.md and start the optimization loop.
The baseline is already recorded. Start from step 2 (analyze failures).
```

The agent will read traces, diagnose failures, edit `agent/agent.py`, gate the change, record the result, and repeat.

---

## How benchmarks are structured

### Templates

Each benchmark has two templates:

```
agent/templates/
├── tau_bench.py           # tau-bench agent starting point
├── terminal_bench.py      # terminal-bench agent starting point
└── bird_interact.py       # BIRD-Interact system agent starting point

program_templates/
├── tau_bench.md           # tau-bench PROGRAM.md
├── terminal_bench.md      # terminal-bench PROGRAM.md
└── bird_interact.md       # BIRD-Interact PROGRAM.md
```

`prepare.py` copies the correct templates into `agent/agent.py` and `PROGRAM.md` based on `experiment_config.yaml`. The coding agent then edits `agent/agent.py` freely. To see what it changed:

```bash
diff agent/templates/terminal_bench.py agent/agent.py
```

### Using a different Harbor benchmark

If your benchmark runs via `harbor run`, you only need four steps:

**1. Point to your dataset in `experiment_config.yaml`:**

```yaml
benchmark: "terminal-bench"   # reuses TerminalBenchRunner
dataset: "my-harbor-dataset@1.0"
agent_model: "gpt-4o"
env_provider: "e2b"           # or "daytona" / "docker"
split: "train"
gate_split: "test"
```

**2. Check your verifier's `result.json` schema.**
`TerminalBenchRunner` expects:

```json
{
  "task_name": "<id>",
  "verifier_result": {
    "rewards": { "reward": 0.85 }
  }
}
```

If your verifier writes rewards at a different path, update the parser in `TerminalBenchRunner.run()` in `benchmark.py`.

**3. Update the split directory name (optional).**
The split file is currently saved to `tbench_data/task_split.json`. If you want a separate directory per benchmark, change `SPLIT_FILE` in `TerminalBenchRunner` and update `prepare.py` accordingly.

**4. Add a PROGRAM.md supplement.**
Create `program_templates/<your_benchmark>.md` with benchmark-specific guidance (trace paths, task ID format, known techniques) following the same pattern as `terminal_bench.md`. Then register it in `copy_program_template()` in `prepare.py`.

The train/test split generation, gating, trace copying, and optimization loop all work as-is — no other changes needed.

---

### Plugging in your own benchmark

Subclass `BenchmarkRunner` in `benchmark.py`:

```python
class MyBenchmarkRunner(BenchmarkRunner):
    def run(self, task_ids=None):
        # call your benchmark CLI or API
        # return {task_id: reward} where reward is 0.0–1.0
        ...
```

Add a branch in `gating.py`'s `_create_runners()` and `prepare.py`'s `__main__`. Create templates in `agent/templates/` and `program_templates/`. The loop, gating, recording, and workspace format are all benchmark-agnostic.

---

## Eval suite

The coding agent self-maintains `workspace/suite.json` — task IDs it must always pass.

`gating.py` runs three steps before any change is committed:

1. **Regression suite**: tasks in `suite.json` must pass at ≥ threshold (default 80%)
2. **Full test**: full benchmark on the test split; mean reward must be ≥ the best score seen so far
3. **Suite promotion**: previously-failing tasks that now pass are added to the suite

Steps 1 and 2 run sequentially; Step 2 always runs regardless of Step 1's outcome.

---

## Project structure

```
agent/
  agent.py                  the agent under optimization — only file the coding agent edits
  templates/                read-only starting points for each benchmark
  helpers/
    bird_interact/
      bird_service.py       FastAPI service wrapper for BIRD-Interact system agent
      bird_adk_runtime.py   Google ADK runtime adapter for the BIRD service
      setup.py              prepare.py helpers for BIRD-Interact provisioning
benchmark.py                benchmark execution layer (abstract + tau-bench + terminal-bench + bird-interact)
gating.py                   three-step gate (regression suite → full test → suite promotion)
prepare.py                  workspace setup, template copying, baseline run
record.py                   appends iteration result to results.tsv
PROGRAM.md                  loop instructions for the coding agent (copied from template)
program_templates/          benchmark-specific PROGRAM.md templates
experiment_config.yaml.template   example configs for each benchmark
Dockerfile                  container definition (tau-bench)
docker-compose.yml          mounts agent/ and workspace/ (tau-bench)
workspace/
  suite.json                regression eval suite (task IDs + threshold)
  learnings.md              per-run log: patterns, what worked, requests to human
  results.tsv               iteration history (val_score, commit, evals, timestamp)
  traces/                   agent conversation traces for failure analysis
```

---

## Design

- **Program the loop, not the agent directly.** The human steers through `PROGRAM.md`; the coding agent edits `agent/agent.py`.
- **Benchmark-agnostic loop.** The same gating, recording, and workspace format works for any benchmark that returns per-task rewards.
- **Self-maintained evals.** The coding agent decides which tasks belong in the regression suite — no manual curation needed.
- **Learnings close the feedback loop.** After each iteration the agent writes `workspace/learnings.md`: what it tried, what worked, what it needs from the human.
- **Gate everything.** No change is committed without passing both the eval suite and the full test score gate.
- **Structural anti-cheating.** Test traces are not saved to disk. The coding agent can only read train traces.
