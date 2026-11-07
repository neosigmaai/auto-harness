# auto-harness

## Agent Optimization Service assignment

The `feat/agent-optimization-service` branch adds a Python/FastAPI + PostgreSQL service with
durable jobs, organization/member access, full agent execution inside Docker, and a measured
optimization loop. Open the editable [final architecture](final_architecture.excalidraw), its
[preview](docs/architecture.png), [design and tradeoffs](docs/DESIGN_PLAN.md), and
[test/experiment evidence](docs/VALIDATION.md). Simulation results are labelled and are not
benchmark improvements. The standalone harness instructions below are separate from this service.
For a five-minute walkthrough and a tested read-only history demo, start with the
[reviewer guide](docs/REVIEWER_GUIDE.md).

The latest corrected live search completed 40 valid observations and three autonomous code
proposals. No candidate earned promotion. In fresh matched confirmation, candidate 2 passed
2/6 trials versus the incumbent's 3/6, despite 3.7% lower model cost and 11.7% lower p95 time.
The service rejected the quality regression and retained the baseline. Separately, prepared
tool images reduced measured repeated readiness to about 0.5–1.4 seconds on four probed tasks;
this is a manual infrastructure improvement with upfront build costs. See the
[final audit](docs/live_prepared_verified_audit.json),
[exact generated sources](docs/prepared_verified_candidates/index.json), and
[measurement limits](docs/VALIDATION.md#corrected-prepared-mode-experiment).

A later [evidence-guided follow-up](docs/live_evidence_guided_audit.json) improved
optimizer context while preserving the evaluation. Its last candidate scored 3/6
versus 2/6 in fresh confirmation, but cost 17.96% more per trial and was rejected
under the same 10% allowance. The run reached its original time limit before
finishing held-back evaluation. This is a small observed quality–cost tradeoff,
not an accepted balanced improvement. The completed run above remains the main demo.

**This service uses only `NEOSIGMA_OPEN_API_KEY`. It never falls back to a personal API key.**
Only the trusted inference gateway sends the company key to the provider. The key is not
passed to candidates or Harbor; the API does not need it for its own work.

### No-cost service demo

```bash
docker compose -f compose.service.yml up --build -d
docker compose -f compose.service.yml run --rm api python -m scripts.bootstrap_demo
# Save one generated service API key as AOS_API_KEY; this is not a provider key.
python test_client.py --max-iterations 2 --json
```

Compose runs the simulator intentionally. Its default ports are 8000 (API) and 5433 (database);
adjust them if occupied. Bootstrap is for a fresh demo database. Organization admins see their
organization's jobs; members see only their own. `GET /docs` exposes the OpenAPI UI.

### Real sandboxed execution

Run the worker on the Docker host, where the Docker CLI/Compose and the repository paths are
available. This avoids pretending a plain API container can provision sibling sandboxes without
Docker access and matching host paths. The tested local provider is Docker, not E2B/EC2.

```bash
python3.12 -m venv .venv-service
.venv-service/bin/pip install -r service/requirements.txt
uv tool install 'harbor==0.20.0'
docker version
harbor --version

# Use your service database; migrate preserves existing service rows.
export DATABASE_URL=postgresql://harness:harness@localhost:5433/harness
.venv-service/bin/python -m service.db migrate
# Fresh database only; otherwise reuse the existing organization and service API key.
.venv-service/bin/python -m scripts.bootstrap_demo

# Terminal 1: API. Company key is not needed here.
EXECUTOR=harbor OPTIMIZER=llm .venv-service/bin/uvicorn service.api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: company key must already be set, or set it securely in the shell.
export EXECUTOR=harbor OPTIMIZER=llm SANDBOX_PROVIDER=docker
.venv-service/bin/python -m service.worker.main

# Terminal 3: use the generated service API key, with an explicit per-job budget.
export AOS_API_KEY='aos_service_key_from_bootstrap'
.venv-service/bin/python test_client.py --max-iterations 2 --patience 2 \
  --max-llm-usd 5 --max-wall-clock-min 90 --timeout 6000 --json
```

Stop the simulated compose worker before starting a real worker on the same database:
`docker compose -f compose.service.yml stop worker`. Mixing worker executor types on one queue
is unsupported. Linux Docker hosts need `host.docker.internal` routing to the gateway; the relay
adds the `host-gateway` entry. The gateway binds an ephemeral port and requires a per-trial token.
Use a dedicated worker machine and network policy for a public deployment.

### Optional prepared tooling images

For repeated experiments, prepare the audited tools once in clean images. First pull the
selected benchmark base images (network/time/disk costs apply):

```bash
.venv-service/bin/python - <<'PY'
import subprocess
from service.tasks_catalog import DEFAULT_TASKS, BY_NAME
for task in DEFAULT_TASKS:
    subprocess.run(['docker', 'pull', '--platform', 'linux/amd64', BY_NAME[task]['docker_image']], check=True)
PY
.venv-service/bin/python -m service.sandbox.prepared_images \
  --tasks fix-git openssl-selfsigned-cert sqlite-db-truncate extract-elf sanitize-git-repo \
          nginx-request-logging largest-eigenval cancel-async-tasks \
          regex-log sparql-university tune-mjcf polyglot-c-py \
  --output runs/prepared-tools-v1/manifest.json
export PREPARED_IMAGES_MANIFEST="$PWD/runs/prepared-tools-v1/manifest.json"
```

Set that variable before starting the API. It freezes the selected immutable image IDs in
each job; the worker uses that saved mapping. Run one preparation command per manifest.
Every attempt gets a fresh writable layer and an offline tool check; candidate code and
credentials never enter preparation images. The original verifier still runs and may use
the network. Changing the recipe/source invalidates stale entries; use a new manifest path
when rebuilding. Missing entries or a different Docker daemon fail closed. This local mode
supports workers on the same daemon; it is not a distributed image registry or cross-host
failover system. Leave the variable unset for the original cold setup mode. Related jobs
must use the same frozen preparation, so old cold histories cannot silently cross modes.

The default `calibrated12` plan freezes 5 feedback, 3 selection and 4 holdout tasks.
Use `--profile stress16` for the earlier 8/4/4 stress plan, or `--tasks` for a custom
list. A profile change starts a distinct experiment; scores across different task
mixes are not evidence of agent improvement. The
[selection rationale](docs/DESIGN_PLAN.md#task-subset-and-model-choice) explains category coverage,
resource limits and sampling limitations. The twelve selected tasks are:

| Role | Tasks | Coverage |
| --- | --- | --- |
| Feedback | `fix-git`, `sanitize-git-repo` | Git recovery and multi-step history changes |
| Feedback | `openssl-selfsigned-cert`, `sqlite-db-truncate`, `extract-elf` | Certificate checks, database recovery and binary inspection |
| Selection | `nginx-request-logging`, `largest-eigenval`, `cancel-async-tasks` | Service configuration, numerical precision and concurrency |
| Holdout | `regex-log`, `sparql-university`, `tune-mjcf`, `polyglot-c-py` | Exact text/query constraints, parameter tuning and constrained code generation |

This small CPU-oriented development set excludes image-heavy, long scientific and coupled
multi-service stress cases. It is not a random sample of the full benchmark; timings include
setup and local architecture emulation. Task roles remain fixed throughout an experiment.

`--max-iterations 0` runs a plain benchmark, including
small smoke subsets. Optimization requires at least three tasks and disjoint evaluation sets.
`SUCCEEDED` means the job lifecycle completed; inspect `stop_reason`, `final_evaluation` and
iteration decisions to determine whether any measured improvement exists.
The initial `baseline_score` uses one selection trial per task; accepted candidates use fresh
matched confirmations. Use `iterations[*].metrics.incumbent` and the matching candidate metrics
for before/after comparisons, rather than subtracting the headline scores with different sample counts.

Every real inference request reserves spending capacity before dispatch. The ceiling is per job;
multiple jobs need a separately allocated overall budget. Ambiguous requests retain a reservation.
Token prices are conservative estimates, not invoices; local Docker compute is not priced as zero.
Deadline/spend exhaustion can leave holdout evidence incomplete. No gain is claimed from that.

Use `--agent-model` and `--optimizer-model` to select priced models explicitly.
The calibrated experiment kept task execution on `gpt-5.4-mini` and tested
`gpt-5.4` for the relatively few code proposals. This is a model-allocation
experiment, not a claim that a larger optimizer is always better. Verified access
and measured outcomes are recorded separately in the validation report.

### Tests and history

```bash
.venv-service/bin/python -m pytest service/tests -q
# Optional isolated PostgreSQL tests; user must have CREATE DATABASE on this test server.
export AOS_TEST_DATABASE_URL=postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test
.venv-service/bin/python -m pytest service/tests -q
.venv-service/bin/python -m service.tests.smoke_simulated
```

Without `AOS_TEST_DATABASE_URL`, database tests skip explicitly. They create/drop dedicated test
databases and do not truncate existing service data. Live model tests require a separate
`--allow-paid` opt-in; ordinary pytest makes no provider calls.

Important endpoints: `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`,
`GET /v1/jobs/{id}/iterations`, `/iterations/{index}`, `/iterations/{index}/attempts`,
`/iterations/{index}/tasks/{task}?include_trace=true`, `/agent`, and `/events`.
Immutable attempts include retries, confirmation and holdout phases. `test_client.py --json`
retrieves the full iteration details and paginated attempts, including traces. Raw transcripts
can be large and may contain task data; protect them as tenant artifacts.

A new experiment can use `--prior-job-id UUID` to snapshot an authorized terminal job's rejected
hypotheses and feedback evidence. The benchmark revision, tasks, split, model and original source
must match. Add `--resume-candidate-index N` to evaluate an interrupted, unconfirmed candidate
before making another LLM proposal. This copies source and diagnostic context; every baseline,
candidate and confirmation result in the new job is freshly measured. It does not reopen the
cancelled job or import its scores. Prior rejection hashes also prevent exact code repetition;
semantic repetition still requires judgment and evidence, not just hashing.

---

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

## First-run checklist

Before running the initialization command in your chosen quick start:

- [ ] **Choose one benchmark and install its prerequisites:** [Terminal-Bench 2.0](#quick-start-terminal-bench-20) needs the `harbor` CLI and a configured sandbox provider (or a running Docker daemon for `env_provider: "docker"`); [BIRD-Interact](#quick-start-bird-interact) needs Python 3.12+, Docker, `git-lfs`, and access to the ground-truth data described below; [tau-bench](#quick-start-tau-bench) uses Docker Compose and the image built by `docker compose build`.
- [ ] **Configure the experiment:** copy [`experiment_config.yaml.template`](experiment_config.yaml.template) to `experiment_config.yaml`, uncomment only the benchmark section you intend to use, and review its models, dataset/domain, and `max_concurrency`. The template is entirely commented out, so copying it alone does not select a benchmark.
- [ ] **Set credentials for the selected models and provider:** use [`.env.example`](.env.example) as a reference. For Terminal-Bench, supply the credential for the selected `env_provider`; Docker needs no sandbox-provider key, but model API credentials are still required. For BIRD-Interact, check both `agent_model` and `user_model`.
- [ ] **Make credentials available to the process:** Docker Compose loads `.env` via `env_file`. When running `python prepare.py` directly, export the required variables in your shell first; `prepare.py` does not load `.env` automatically. For a shell-compatible `.env` you have reviewed, Bash/Zsh users can run `set -a; source .env; set +a` from the repository root.
- [ ] **Plan for a baseline run:** on a fresh workspace, `python prepare.py` (or `docker compose run autoeval python prepare.py`) performs setup and launches the baseline benchmark. This makes model API calls and may incur sandbox-provider charges. Review your provider budget before running it; it is not a setup-only check.

After initialization completes, follow [Running the loop](#running-the-loop). Optimization and [gating](#eval-suite) run additional benchmark tasks and can incur further usage charges.

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
