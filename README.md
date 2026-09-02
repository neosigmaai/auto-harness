# auto-harness

> Give a coding agent a benchmark and an agent file. Let it iterate overnight. It reads failures, improves the system prompt and tools, gates every change against a self-maintained eval suite, and repeats.

This repo is a simplified version of our auto-harness agent setup. We demonstrate our system on Tau3 benchmark tasks where the agent's score improves from 0.56 to 0.78 (~40% jump) while mining failures and auto maintaining live evals. If you are curious to learn more, read the full blog here - https://www.neosigma.ai/blog/self-improving-agentic-systems.

The loop is defined in `PROGRAM.md`. The coding agent edits `agent/agent.py` to improve the agent and appends findings to `workspace/learnings.md` after each iteration.

**Service architecture (HTTP API, workers, Harbor, iterative jobs):** see [`docs/architecture/README.md`](docs/architecture/README.md).

---

---

# The optimization service (Milestones 1-4)

Alongside the CLI loop above, this repo exposes an HTTP service that runs the same
idea as a job: submit a task set, and the service evaluates the agent, asks an LLM to
improve its prompt and limits, re-evaluates, and repeats until the score stops
improving or a cap is reached. Full iteration history is persisted and readable over
HTTP.

Architecture reference (diagrams): [`docs/architecture/README.md`](docs/architecture/README.md).
Design spec: [`docs/superpowers/specs/2026-09-02-milestone-4-iterative-loop-design.md`](docs/superpowers/specs/2026-09-02-milestone-4-iterative-loop-design.md).
Known gaps and follow-up work: [`docs/milestone-4-followups.md`](docs/milestone-4-followups.md).

## Setup and run

```bash
# 1. Clone and install
git clone <this-repo> && cd auto-harness
uv sync                      # or: python -m venv .venv && .venv/bin/pip install -e .
uv tool install harbor       # the Terminal-Bench runner (provides the `harbor` CLI)
.venv/bin/pip install litellm  # used by the improver (and by the agent inside Harbor)

# 2. Credentials
cp .env.example .env
# edit .env: set OPENAI_API_KEY (the agent and the improver both use it).
# For a non-Docker sandbox also set E2B_API_KEY / DAYTONA_API_KEY /
# MODAL_TOKEN_ID + MODAL_TOKEN_SECRET and change env_provider in config/benchmark.yaml.

# 3. Postgres (the queue and the history live here)
docker compose up -d postgres

# 4. Choose your execution backend
#    config/benchmark.yaml -> execution_backend: harbor   (real containers)
#                          -> execution_backend: mock     (no Docker, instant, for tests)
#    or override per-process:  export EXECUTION_BACKEND=mock

# 5. Start the API and a worker (separate terminals)
set -a; . ./.env; set +a
.venv/bin/uvicorn api.main:app --port 8000
.venv/bin/python -m worker.main -v

# 6. Submit an optimization job and watch it
.venv/bin/python test_client.py --task-ids fix-git regex-log --max-iterations 3
```

`test_client.py` submits the job, polls it, and prints the structured summary
including the full iteration history and the winning agent spec. Pass
`--mode run` to exercise a single benchmark run without the optimization loop.

Run the test suite with `python -m pytest tests/ -q` (Postgres must be up; the
Postgres-dependent tests skip loudly if it is not).

### API

| Endpoint | Purpose |
|---|---|
| `POST /v1/jobs` | Start an optimization job. Returns `202` + `job_id`. |
| `GET /v1/jobs/{id}` | Status, stop reason, best pointer, and the full iteration history. |
| `GET /v1/jobs/{id}/best` | The winning `AgentSpec` inline. `409` until iteration 0 finishes. |
| `GET /v1/agent-versions/{id}` | Any single agent version: spec, parent, rationale. |
| `POST /v1/runs`, `GET /v1/runs/{id}` | Single benchmark run, no optimization. |
| `GET /tasks`, `GET /health` | Configured task allowlist; liveness. |

### Choosing which tasks to run

`config/benchmark.yaml`'s `default_task_ids` does double duty: it is the set used when a
request omits `task_ids`, **and** the allowlist of IDs the API will accept. A task not in
that list is rejected with `400 unknown_task_ids` — so adding a new Terminal-Bench task
means editing that file (and restarting the API, since config is cached at startup).

```bash
# What will run by default, straight from the server:
curl -s localhost:8000/tasks | python3 -m json.tool

# All 16 configured tasks (omit --task-ids):
python test_client.py

# Specific tasks:
python test_client.py --task-ids fix-git regex-log

# One task, quick smoke test (note: a single task makes the mean reward binary,
# so min_delta cannot filter noise — see docs/milestone-4-followups.md F5):
python test_client.py --task-ids polyglot-c-py --max-iterations 2

# Or straight over HTTP:
curl -s -X POST localhost:8000/v1/jobs -H 'Content-Type: application/json' \
  -d '{"task_ids":["fix-git","regex-log"],"max_iterations":3}'
```

The 16 configured tasks are: `cobol-modernization`, `fix-git`, `prove-plus-comm`,
`overfull-hbox`, `regex-log`, `log-summary-date-ranges`, `openssl-selfsigned-cert`,
`sanitize-git-repo`, `filter-js-from-html`, `sqlite-db-truncate`, `nginx-request-logging`,
`largest-eigenval`, `extract-elf`, `gcode-to-text`, `polyglot-c-py`, `headless-terminal`.

Cost note: every iteration runs **all** the job's tasks in a container. 16 tasks at
`max_concurrency: 2` is roughly 2.7 hours per iteration, so a 3-iteration job on the full
set is an overnight run. Use 2-3 tasks while developing.

## Which Terminal-Bench tasks we selected, and why

The 16 tasks in `config/benchmark.yaml` are both the default subset and the
allowlist accepted by `POST /v1/jobs`:

| Group | Tasks | Why it is in the set |
|---|---|---|
| Version control | `fix-git`, `sanitize-git-repo` | Multi-step stateful work where a wrong command is recoverable — rewards careful verification. `fix-git` is our known-passing control. |
| Text / log processing | `regex-log`, `log-summary-date-ranges`, `filter-js-from-html`, `gcode-to-text` | The most common real terminal work, and the easiest place for a better prompt to pay off (precision about edge cases). |
| Sysadmin and config | `openssl-selfsigned-cert`, `nginx-request-logging`, `headless-terminal` | Requires reading tool docs and getting flags exactly right; punishes guessing. |
| Data | `sqlite-db-truncate` | Schema-aware edits where a plausible-looking command silently does the wrong thing. |
| Numerical | `largest-eigenval` | Needs a library, so it exercises dependency installation. |
| Binary / low-level | `extract-elf` | Forces genuine exploration rather than pattern-matching. |
| Translation / legacy | `cobol-modernization`, `polyglot-c-py` | Hard, conceptual tasks. `polyglot-c-py` is our known-failing case — it is what gives the improver real failure traces to work from. |
| Formal proof | `prove-plus-comm` | Deliberately near the ceiling for a small model, to check the loop degrades gracefully rather than thrashing. |
| Typesetting | `overfull-hbox` | Narrow, verifiable, fiddly — a good regression detector. |

The selection criteria, in order:

1. **A spread of skills**, so an improvement to the prompt has to generalise rather
   than special-case one task.
2. **A known-passing and a known-failing anchor**, verified by real runs: `fix-git`
   scores 1.0 at baseline, `polyglot-c-py` scores 0.0. Without a failure the improver
   has no signal; without a pass you cannot detect a regression.
3. **Deliberate difficulty spread** including one task near the ceiling, so the
   stopping rules get exercised.
4. **Runnable in one sitting** — 16 tasks at `max_concurrency: 2` and a 1200s
   per-task timeout is roughly 2.7 hours per iteration, which sets the
   `evaluate_stale_after_sec` the queue uses.

For a quick smoke test use two or three tasks. Be aware that with a single task the
mean reward is binary, so `min_delta` cannot filter noise — see
[`docs/milestone-4-followups.md`](docs/milestone-4-followups.md) (F5).

## Key design decisions

**Postgres is the whole runtime.** The queue holds typed *steps* (`evaluate` |
`improve`); stateless workers claim one with `SELECT ... FOR UPDATE SKIP LOCKED`, and
the worker that completes a step enqueues its successor **in the same transaction**.
There is no orchestrator process and no extra service. A job is therefore never alive
with nothing queued, and a crash resumes from the last commit. The alternative —
one worker running a whole job — fails badly here, because an evaluate step can take
hours and would be reclaimed mid-flight by the stale sweeper.

**The agent is data, not a file.** Each version is an `AgentSpec` (system prompt,
model, `max_steps`, `max_output_chars`, `exec_timeout_sec`) stored as JSONB and
materialised to JSON per run; a fixed runtime (`agent/spec_agent.py`) reads it. Those
five fields are not invented — they are exactly the tunable constants the existing
hand-written template already had, so version 0 is behaviourally identical to the old
agent and comparisons are meaningful. Keeping the surface *pure data* is what makes
`extra="forbid"` plus numeric bounds a complete validation gate: an unusable proposal
is a failed step, never a crashed job, and there is no code to sandbox. The CLI loop's
`agent/agent.py` is never touched by the service.

**Mean reward decides, but per-task movement is recorded.** A proposal that fixes one
task and breaks another lands on exactly the same mean as one that changed nothing, so
each iteration also reports `fixed_tasks` / `regressed_tasks` against the best prior
iteration — surfaced in the API and fed to the improver. A regression deliberately does
not veto a proposal: on a stochastic agent a single-trial regression is usually noise.

**Proposals always build on the best version, never a regression.** An improve step's
base is always `best_agent_version_id`. Without this, one bad proposal poisons every
later iteration. The rejected attempt stays in the history the improver reads, so it
sees what already failed while editing the best-known spec.

**Failure policy favours keeping work.** A failed *improve* step ends the job
`completed` with `stop_reason="failed_improve"` when a best version exists — the
best agent found so far is a valid answer. A failed *evaluate* step fails the job, so
infrastructure errors never masquerade as "no improvement".

**Traces live in an artifact store, not the repo or the database.** A single passing
task produced ~31k tokens of trace. `ArtifactStore` has a local-disk implementation
today; S3 is a factory change.

## What we would do differently with more time

The full list with reproduction details is in
[`docs/milestone-4-followups.md`](docs/milestone-4-followups.md). The three that matter
most:

1. **Give the improver the task statement.** Traces are truncated from the *tail*, and
   the task instruction is the *head* — so on a 101-message trace the improver diagnosed
   the failure without knowing what the task asked for. It blamed a missing shell utility
   because a shell error was the only concrete thing in its window. This is the single
   biggest limitation on optimization quality today.
2. **Plumb the verifier's actual output through.** The improver currently receives the
   generic string `"Verifier failed"`; Harbor's `result.json` usually explains *why*.
3. **Sample more than once per iteration.** One run per iteration cannot separate a
   lucky pass from a real improvement. `trials_per_iteration` with a mean across trials
   is the honest fix, at linear cost.

Beyond those: cost pressure on the improver (nothing stops it raising `max_steps`
toward 200), a length budget on the prompt (it grew 73% in one iteration), and a test
covering an iteration that *actually improves* — the mock runner's score is a pure
function of `task_id`, so no test currently exercises that path.

## What we chose not to implement, and why

- **Live log streaming.** Polling plus artifact download is sufficient; how humans watch
  a run is presentation, not architecture, and can be added without touching the loop.
- **An S3 artifact backend.** The interface is in place; only the implementation is
  missing. Local disk is correct for a single-host deployment.
- **A best-agent-per-task mapping.** The job optimises one agent across the whole
  submitted task set, which is the intended reading. Per-task bests are derivable from
  the stored history if ever wanted, and on single-trial runs they would mostly be noise.
- **Job cancellation.** The status enum and the idempotency guard both already honour
  `cancelled`; no endpoint exists yet. Note the claim query needs a job-status filter
  before cancellation lands (follow-ups F8).
- **Database migrations.** Tables are created by `init_db()` / `create_all`, consistent
  with the existing codebase. A real deployment wants Alembic.
- **Tool or code mutation by the improver.** Only the prompt and three numeric limits are
  mutable. Letting an LLM author tool schemas or agent code means validating and executing
  model-written code; that is a different project with a different risk profile.

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

**Requirements:** `harbor` CLI, an `OPENAI_API_KEY`, and a coding agent (Claude Code, Codex CLI, or similar). If using a sandboxed `env_provider` (the default), you'll also need its credential: `E2B_API_KEY`, `DAYTONA_API_KEY`, or a Modal token via `modal token new` / `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`. `env_provider: "docker"` needs none of these.

```bash
# 1. Clone the repo
git clone https://github.com/neosigmaai/auto-harness
cd auto-harness

# 2. Install harbor
uv tool install harbor

# 3. Set up environment variables
cp .env.example .env
# edit .env — set OPENAI_API_KEY, plus your sandbox provider's credential
# (E2B_API_KEY, DAYTONA_API_KEY, or MODAL_TOKEN_ID + MODAL_TOKEN_SECRET) —
# not needed if you're using env_provider: "docker"

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
env_provider: "e2b"           # or "daytona" / "modal" / "docker"
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
