# auto-harness

> Give a coding agent a benchmark and an agent file. Let it iterate overnight. It reads failures, improves the system prompt and tools, gates every change against a self-maintained eval suite, and repeats.

This repo is a simplified version of our auto-harness agent setup. We demonstrate our system on Tau3 benchmark tasks where the agent's score improves from 0.56 to 0.78 (~40% jump) while mining failures and auto maintaining live evals. If you are curious to learn more, read the full blog here - https://www.neosigma.ai/blog/self-improving-agentic-systems.

The loop is defined in `PROGRAM.md`. The coding agent edits `agent/agent.py` to improve the agent and appends findings to `workspace/learnings.md` after each iteration.

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
- **`PROGRAM.md`** — instructions the coding agent follows

### How `benchmark.py` and `agent/` interact

`benchmark.py` is deterministic infrastructure — it never changes. It selects the right agent wrapper based on the `benchmark` field in `experiment_config.yaml` and runs the benchmark, returning `{task_id: reward}`.

`agent/agent.py` is what Claude Code improves. It contains the generic agent loop logic — prompts, state management, tool definitions, turn limits. The benchmark-specific wrappers (`agent_tau.py`, `agent_harbor.py`) import from `agent.py` and plug into their respective frameworks.

```
benchmark.py (deterministic)         agent/ (Claude Code improves this)
─────────────────────────────        ──────────────────────────────────
TauBenchRunner                  →    agent_tau.py (tau2 wrapper)
                                         imports from agent.py
TerminalBenchRunner             →    agent_harbor.py (TerminalBench wrapper)
                                         imports from agent.py
                                     agent.py (generic base — never changes)
```

---

## Quick start

**Requirements:** Docker, an `OPENAI_API_KEY`, and a coding agent (Claude Code, Codex CLI, or similar).

```bash
# 1. Clone the repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY and TAU2_DATA_DIR

# 3. Configure the experiment
cp experiment_config.yaml.template experiment_config.yaml
# edit experiment_config.yaml — set domain, model, etc.

# 4. Build the Docker image (installs tau-bench and all deps via uv)
docker compose build

# 5. Initialize the workspace
docker compose run autoeval python prepare.py

# 6. Run the benchmark once to verify it works
docker compose run autoeval python benchmark.py
```

## Running the loop

Point your coding agent at the repo and prompt:

```
Read PROGRAM.md and start the optimization loop.
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

## Using TerminalBench

[TerminalBench](https://github.com/harbor-framework/terminal-bench) evaluates agents on real terminal tasks — compiling code, setting up servers, debugging systems. It runs each task in an isolated Docker container and scores pass/fail via verification scripts.

Install TerminalBench with required dependencies:

```bash
uv tool install terminal-bench --with openai-agents --with harbor
export OPENAI_API_KEY=your-key
export AGENT_MODEL=gpt-4o
```

Configure `experiment_config.yaml`:

```yaml
benchmark: "terminal_bench"
```

Run a subset of tasks to test:

```bash
tb run \
  --agent-import-path agent.agent_harbor:HarborAgent \
  --model openai/gpt-4o \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  --n-concurrent 1 \
  --output-path jobs \
  --task-id nginx-request-logging
```

Or run via `benchmark.py` (same as tau-bench):

```bash
docker compose run autoeval python benchmark.py --task-ids nginx-request-logging
```

---

## Adding a new benchmark

The system is designed to be benchmark-agnostic. Adding a new benchmark requires three steps:

1. **Create `agent/agent_newbenchmark.py`** — a wrapper that extends `BaseHarnessAgent` and the new benchmark's agent interface. Import `SYSTEM_PROMPT`, `AGENT_MODEL`, and `MAX_TURNS` from `agent/agent.py`.

2. **Add a `NewBenchmarkRunner`** to `benchmark.py` — subclass `BenchmarkRunner`, implement `run()` to invoke the benchmark and return `{task_id: reward}`.

3. **Update `experiment_config.yaml`** — set `benchmark: new_benchmark`.

`agent/agent.py` never changes when adding a new benchmark.

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
agent/agent_tau.py      tau-bench wrapper — extends BaseHarnessAgent + tau2's LLMAgent
agent/agent_harbor.py   TerminalBench wrapper — extends BaseHarnessAgent + TerminalBench's BaseAgent
benchmark.py            benchmark execution layer (abstract + tau-bench + terminal-bench)
gating.py               two-step gate, calls benchmark.py
prepare.py              workspace initialization (run once)
record.py               appends iteration result to results.tsv
PROGRAM.md              loop instructions for the coding agent
Dockerfile              container definition
docker-compose.yml      mounts agent/ and workspace/, passes env vars
workspace/
  suite.json            regression eval suite (task IDs + threshold)
  learnings.md          persistent log: patterns, what worked, requests to human
  results.tsv           iteration history (val_score, commit, evals, timestamp)
```

---

## Design

- **Program the loop, not the agent directly.** The human steers through `PROGRAM.md`; the coding agent edits `agent/agent.py`.
- **Self-maintained evals.** The coding agent decides which tasks belong in the regression suite — no manual curation needed.
- **Learnings close the feedback loop.** After each iteration the agent writes `workspace/learnings.md`: what it tried, what worked, what it needs from the human (a missing tool, parallelism in the runner, a subagent for a slow step). Read it at session start to restore context instantly.
- **Gate everything.** No change is committed without passing both the eval suite and the full test score gate.
- **Benchmark-agnostic agent base.** `agent/agent.py` contains only generic logic — prompts, state, tool definitions. Benchmark-specific wrappers live in `agent_tau.py` and `agent_harbor.py` and import from `agent.py`. Adding a new benchmark never requires touching `agent.py`.
- **Benchmarks own their context.** Each `BenchmarkRunner` in `benchmark.py` knows which agent wrapper to use, how to invoke the benchmark, and how to score results. This keeps `benchmark.py` deterministic and `agent/` focused on what Claude Code should improve.

---

## Docker cleanup

Images and containers accumulate across runs:

```bash
# Remove stopped containers
docker container prune -f

# Full cleanup (images, build cache)
docker system prune -a -f
```