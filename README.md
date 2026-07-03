# auto-harness

> Give a coding agent a benchmark and an agent file. Let it iterate overnight. It reads failures, improves the system prompt and tools, gates every change against a self-maintained eval suite, and repeats.

This repo is a simplified version of our auto-harness agent setup. We demonstrate our system on Tau3 benchmark tasks where the agent's score improves from 0.56 to 0.78 (~40% jump) while mining failures and auto maintaining live evals. If you are curious to learn more, read the full blog here - https://www.neosigma.ai/blog/self-improving-agentic-systems.

The loop is defined in `PROGRAM.md`. The coding agent edits `agent/agent.py` to improve the agent and appends findings to `workspace/learnings.md` after each iteration.

---

## Take-home: Agent Optimization Service MVP

This branch adds a small FastAPI backend service for the take-home assignment.
It focuses on the take-home MVP milestones:

- API design with structured request, status, result, and iteration shapes
- asynchronous run lifecycle with polling
- sandboxed Terminal-Bench execution through Harbor and Daytona in real mode
- PostgreSQL persistence for runs, task results, and iteration history
- a single-optimized-version Milestone 4 loop

Milestone 4 runs a narrow closed loop: baseline task execution, compact failure
summary, one structured LLM proposal, restricted `AGENT_INSTRUCTION` patch,
same-task rerun, strict score comparison, and accept/revert. The only optimized
version name is `proposal-1`. Milestone 5 is represented by API-level
`X-Org-Id` / `X-User-Id` / `X-Role` scoping and documented RBAC extensions.

Companion system design notes live in
[docs/takehome_mvp_system_design.md](docs/takehome_mvp_system_design.md).

### Setup and dependencies

Use Python 3.12+ for the service and tests.

```bash
python -m pip install -e .
cp .env.example .env
```

The service does not load `.env` automatically. Source it before starting
Uvicorn, running the real smoke scripts, or running the manual client:

```bash
set -a
source .env
set +a
```

Start a local PostgreSQL database:

```bash
docker run --name autoharness-postgres \
  -e POSTGRES_USER=autoharness \
  -e POSTGRES_PASSWORD=autoharness \
  -e POSTGRES_DB=autoharness \
  -p 5432:5432 \
  -d postgres:16
```

If that container already exists, start it instead:

```bash
docker start autoharness-postgres
```

For real Terminal-Bench mode and Milestone 4 optimization, set:

```text
OPENAI_API_KEY=...
DAYTONA_API_KEY=...
DATABASE_URL=postgresql://autoharness:autoharness@localhost:5432/autoharness
```

The same shell that starts Uvicorn must have these variables available.
Harbor CLI must also be installed for real `mode=real` runs:

```bash
uv tool install harbor
```

For a local demo of the full optimization path, you can disable the conservative
instruction-patch guard so the reviewer can see baseline -> LLM proposal ->
patch -> rerun -> accept/reject in one short run:

```bash
export AUTOHARNESS_DISABLE_PATCH_GUARD=1
```

Only use that flag for the local take-home demo. The default guard rejects
dangerous instruction content and only allows replacing `AGENT_INSTRUCTION` in
`agent/agent.py`.

### Start the service

```bash
set -a
source .env
export AUTOHARNESS_DISABLE_PATCH_GUARD=1
set +a
uvicorn autoharness_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Run one Uvicorn worker for the MVP. The background executor and real-mode
Harbor semaphore are process-local, so multiple workers would split state and
break serialization. A production version should use a durable queue and
separate workers instead.

### Run automated test cases

Run the full service test suite:

```bash
python -m pytest tests/service -q
```

Expected result: all service tests pass. In this branch the normal local result
is currently `103 passed, 7 skipped`; skipped tests are optional environment
cases that depend on local database availability.

Useful focused test commands:

```bash
# API contract, async polling, per-task status, auth boundaries, result reads
python -m pytest tests/service/test_api.py -q

# Run lifecycle, durable resume behavior, optimization accept/reject/revert
python -m pytest tests/service/test_service.py -q

# PostgreSQL persistence and durable queue row transitions
python -m pytest tests/service/test_store.py -q

# AGENT_INSTRUCTION patch safety and proposal cleanup
python -m pytest tests/service/test_agent_patch.py -q

# Result normalization, failure summaries, and request validation
python -m pytest tests/service/test_normalizer.py -q

# Harbor/Terminal-Bench runner adapter behavior with faked subprocesses
python -m pytest tests/service/test_runner.py tests/service/test_terminal_bench_runner.py -q

# Optimizer JSON parsing, prompt compaction, and client/smoke formatting helpers
python -m pytest tests/service/test_optimizer.py tests/service/test_optimize_smoke_check.py tests/service/test_test_client.py -q
```

Run lightweight syntax and formatting checks:

```bash
python -m py_compile autoharness_service/*.py scripts/*.py test_client.py benchmark.py
python -m black --check autoharness_service scripts tests/service test_client.py benchmark.py
python -m isort --check-only autoharness_service scripts tests/service test_client.py benchmark.py
```

### Run the manual real-mode client

`test_client.py` is intentionally real-only for reviewer-facing manual runs. It
submits a run, polls until a terminal status, then prints a structured summary:

```bash
python test_client.py \
  --base-url http://127.0.0.1:8000 \
  --task-id break-filter-js-from-html \
  --mode real \
  --requested-concurrency 1 \
  --max-iterations 0 \
  --timeout-sec 1800
```

Expected result: the client prints `submitted run_id=...`, waits while the
service runs Harbor/Daytona in the background, and finally prints JSON with
`score`, `tasks_passed`, `tasks_failed`, `tasks_infra_failed`,
`failure_summary`, and `iteration_statuses`.

For a small batch, use the built-in real demo task list:

```bash
python test_client.py \
  --base-url http://127.0.0.1:8000 \
  --demo-batch \
  --requested-concurrency 1 \
  --max-iterations 0 \
  --timeout-sec 1800
```

### Run the Milestone 4 optimization smoke test

This is the clearest single command to demonstrate the implemented loop:

```bash
python scripts/optimize_smoke_check.py \
  --base-url http://127.0.0.1:8000 \
  --task-id break-filter-js-from-html \
  --requested-concurrency 1 \
  --poll-interval-sec 5 \
  --timeout-sec 1800
```

Expected result: the script prints a compact guided order and phase updates:

1. `Submit`: `POST /runs` returns a run id immediately.
2. `Baseline`: Harbor/Daytona runs the requested Terminal-Bench task.
3. `Collect`: task result, trace path, result path, and artifact metadata are
   persisted.
4. `Optimize`: the LLM proposes one `AGENT_INSTRUCTION` replacement.
5. `Rerun`: the service applies the patch and reruns the same task ids through
   Harbor/Daytona.
6. `Decide`: the patch is accepted only if rerun score is strictly higher;
   otherwise `agent/agent.py` is restored and the optimized snapshot is
   discarded.
7. `FinalSummary`: the script prints final score, task counts, failed tasks,
   iteration states, and optimization metadata.

Baseline artifacts and rerun artifacts are stored separately under:

```text
workspace/service_runs/<run_id>/tbench_jobs/baseline/
workspace/service_runs/<run_id>/tbench_jobs/proposal-1/
```

### Run the durable queue restart smoke test

This script starts its own Uvicorn process, submits a run while background
execution is disabled, kills that process, restarts the service, and verifies
the queued run is still readable and resumes:

```bash
python scripts/durable_queue_restart_check.py \
  --port 8015 \
  --task-id break-filter-js-from-html \
  --requested-concurrency 1 \
  --poll-interval-sec 5 \
  --timeout-sec 1800
```

Expected result: the script prints the first queued status, restarts the
backend, then polls the same `run_id` until it reaches `succeeded`, `failed`,
or another terminal state. The final section prints artifact locations for each
task.

### What each service test covers

| Test file | Purpose | Expected result |
|-----------|---------|-----------------|
| `tests/service/test_api.py` | FastAPI request and response shapes, immediate `202` submit, polling, per-task lifecycle rows, org/user/role access checks, duplicate task validation, and `409` before results are ready | Passes without real Harbor/Daytona by using fake service wiring |
| `tests/service/test_service.py` | RunService lifecycle, background worker resume, durable task claiming, real-run serialization, runner exception handling, timeout handling, optimization accept/reject/revert, and proposal failure states | Passes with fake runners and fake optimizers |
| `tests/service/test_store.py` | PostgreSQL schema creation, idempotent initialization, org boundaries, task queue claiming, requeue on resume, structured proposal storage, and oversized proposal rejection | Passes when the local test database is reachable; otherwise DB-dependent cases may skip |
| `tests/service/test_agent_patch.py` | `AgentPatchService` only changes top-level `AGENT_INSTRUCTION`, compiles patched `agent.py`, rejects dangerous content by default, supports the local demo guard override, and deletes rejected proposal snapshots | Passes using temporary files |
| `tests/service/test_normalizer.py` | Reward-to-status normalization, infra-vs-agent failure summaries, non-finite reward handling, unsafe task-id rejection, and mode/provider validation | Passes with pure unit tests |
| `tests/service/test_runner.py` | Simulated runner behavior, real adapter per-run jobs directory, per-attempt artifact separation, Terminal-Bench template installation, and lazy import behavior | Passes with monkeypatched runner dependencies |
| `tests/service/test_terminal_bench_runner.py` | Harbor resume behavior, result extraction from pending jobs, and stderr reporting when a resumed Harbor job fails | Passes with fake Harbor subprocess output |
| `tests/service/test_optimizer.py` | Strict optimizer JSON parsing, prompt content, artifact metadata flattening, missing API key handling, and LLM response parsing | Passes with mocked LLM clients |
| `tests/service/test_optimize_smoke_check.py` | Smoke-script guided order, polling phase formatting, rejected-state formatting, and artifact timeline extraction | Passes without starting the service |
| `tests/service/test_test_client.py` | Manual client summary generation, demo auth headers, real-only request validation, demo batch task ids, and polling callbacks | Passes without starting the service |
| `tests/service/test_main.py` | App startup settings and background worker enable/disable flag handling | Passes with settings monkeypatches |
| `tests/service/test_imports.py` | Import-time defaults and avoiding heavyweight benchmark imports during service startup | Passes as a quick import-safety check |

### Selected Terminal-Bench tasks

The reviewer-facing real demo set uses fast Terminal-Bench tasks that exercise
the real Harbor/Daytona path without requiring a large benchmark sweep:

- `break-filter-js-from-html`: recommended one-task smoke; useful for showing
  pass/fail normalization, trace collection, and the Milestone 4 rerun path.
- `multi-source-data-merger`: second demo task used by `--demo-batch`; useful
  for showing batch submission and per-task lifecycle rows.

The internal unit tests still use fake IDs such as `task-pass`, `task-fail`,
and `task-infra`, but those are simulated test fixtures rather than manual
reviewer tasks.

### Key design decisions

- The service treats a run as the MVP batch unit. There is no separate
  `eval_batches` table in this branch.
- Real mode uses Harbor as the Daytona adapter. The service does not call the
  Daytona SDK directly.
- One Uvicorn worker is the safe MVP default because the background executor and
  real-run semaphore live in-process.
- Local MVP uses polling. Daytona webhooks are reserved for production
  lifecycle reconciliation.
- The optimizer creates exactly one structured proposal, applies it only through
  `AgentPatchService`, and only replaces top-level `AGENT_INSTRUCTION` in
  `agent/agent.py`.
- Rejected or non-improving proposals restore both `agent/agent.py` and the
  final visible baseline task rows.
- `X-Org-Id`, `X-User-Id`, and `X-Role` are local demo headers. They demonstrate
  API-level scoping, not production authentication.
- Real Harbor/Daytona runs are serialized in-process and use a per-run
  `workspace/service_runs/<run_id>/tbench_jobs` directory to reduce artifact
  cross-contamination.

### Intentionally not implemented

- Durable queue or worker pool
- Object storage for large traces and logs
- Direct Daytona SDK support
- JWT/OAuth auth and real RBAC
- GateEngine, candidate graphs, beam search, and promotion logic
- Multi-round optimization
- Multiple optimized versions
- Merging optimized versions or optimized task results

### Production follow-ups

- Replace in-process background threads with Redis Streams, Kafka, or another
  durable queue
- Move run artifacts and traces to S3 or MinIO
- Add direct Daytona SDK support with sandbox and command tracking
- Replace demo headers with real identity, audit logging, quotas, and RBAC
- Add GateEngine regression protection and candidate promotion

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
