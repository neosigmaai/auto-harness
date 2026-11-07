# Agent Optimization Service — plan and design (v0, for discussion)

Status: proposal. Nothing here is built yet. Sections marked **[Q]** are questions for you.

---

## 0. One-paragraph summary

We add a `service/` package to the auto-harness repo: a FastAPI control plane, a Postgres-backed
job queue + state store, and one or more worker processes. A worker runs the Terminal-Bench agent
(`agent/agent.py`, the thing being optimized) against a fixed 16-task subset, with every task in its
own sandbox (E2B microVM by default, local Docker for development, Modal as an alternative), all via
the Harbor CLI that the repo already uses. After each benchmark pass the worker builds a compact
failure dossier, asks an LLM for one hypothesis-driven rewrite of `agent.py`, validates it, and
re-runs. Every iteration (agent version, proposal, per-task results, traces, decision) is persisted
and queryable over HTTP. Tenancy (org / role) is enforced in the API layer. The design optimizes
for two kinds of performance: the agent's pass rate on the subset, and the wall-clock / dollar cost
of each optimization iteration.

---

## 1. What the assignment actually rewards

| Milestone | Weight (your call) | What graders will look at |
|---|---|---|
| M1 API design | important | clear schemas, 202 + poll, error handling, test_client works |
| M2 async | important | job lifecycle, crash safety, how API and workers talk |
| **M3 sandbox** | **top** | where the boundary is, lifecycle, what happens on failure, real output captured |
| **M4 loop** | **top** | state across iterations, stop criteria, how context reaches the LLM, full history via API |
| M5 multi-tenancy | important | roles enforced at the API, not just data model |

Constraints: Python/FastAPI, Postgres, sandboxed execution, 10–20 fast representative tasks,
README with task rationale + design decisions + "what I'd do differently", `test_client.py`.

---

## 2. Facts discovered that shape the design

From reading the repo, the installed Harbor (0.20.0) source, and the downloaded TB2 task set:

1. **The agent loop runs host-side; only bash runs in the sandbox.** `HarnessAgent.run()` is Python
   executed by the Harbor process; it calls `environment.exec(cmd)` which runs inside the task
   container/microVM. So "sandboxing the agent" has two layers: the task environment (hard boundary,
   provider-enforced) and the process that runs LLM-rewritten `agent.py` (must not be our API or
   worker process). See §5.
2. **Every TB2 task ships a prebuilt image** (`alexgshaw/<task>:20251031`). Harbor's Docker backend
   pulls it (no local build); the E2B backend builds an E2B *template* from it once per task and
   caches it by alias; Modal uses `Image.from_registry`. So the expensive step is a one-time
   per-task template build, not per-run. We should pre-warm it.
3. **Harbor's output layout** per trial: `result.json` (`results.json` in newer Harbor; we accept both) with reward, timings, exception info,
   `agent/trace.json` (full conversation, written by our agent), `verifier/test-stdout.txt`,
   `verifier/reward.json`. All of it is on disk when the subprocess exits; that is the "real task
   output" M3 asks us to capture.
4. **The repo's `benchmark.py` is single-tenant and side-effectful** (global `workspace/`, prunes
   old dirs, `--jobs-dir` shared). We will not call it from the service; we write a small
   `HarborRunner` that invokes `harbor run` with per-iteration directories and env.
5. **Agent code is selected by import path** (`--agent-import-path agent.agent:HarnessAgent` +
   `PYTHONPATH`). Running one Harbor subprocess per iteration with its own `PYTHONPATH` lets many
   jobs with different agent versions run concurrently without module collisions.
6. **Task metadata**: all 89 tasks declare `agent.timeout_sec`, `verifier.timeout_sec`, cpus,
   memory. 47 tasks have a 900 s agent timeout; several need 3600–12000 s, multi-GB assets, torch,
   QEMU, TeX, or network downloads. That drives the subset (§10).
7. **Local box**: 8 cores / 16 GB, Docker running, Harbor installed, `OPENAI_API_KEY` set. No E2B /
   Modal credentials visible. **[Q1]**

---

## 3. Architecture

```
                 ┌──────────────┐  HTTPS (API key)   ┌───────────────────────────────┐
  test_client.py │  clients     │ ─────────────────▶ │  API  (FastAPI, async, N replicas)│
                 └──────────────┘ ◀───────────────── │  auth · tenancy · validation   │
                                      202 + poll     └───────────────┬───────────────┘
                                                                     │ SQL (asyncpg)
                                                                     ▼
                                                     ┌───────────────────────────────┐
                                                     │ Postgres (single source of truth)│
                                                     │ orgs users keys jobs iterations │
                                                     │ task_runs agent_versions events │
                                                     │ job_queue (SKIP LOCKED leases)  │
                                                     └───────────────┬───────────────┘
                                                                     │ claim / heartbeat / checkpoint
                                                                     ▼
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ Worker (N replicas, asyncio)                                          │
                 │  JobRunner state machine ─ Optimizer (LLM) ─ TraceDigest ─ Reaper     │
                 └───────────────┬───────────────────────────────────────┬──────────────┘
                                 │ spawn, scrubbed env, cgroup/timeout   │ LLM API
                                 ▼                                       ▼
                 ┌──────────────────────────────┐           ┌──────────────────────┐
                 │ Runner subprocess (per iter.) │           │ LLM provider          │
                 │ `harbor run` + agent_vN.py    │           │ proposals, digests    │
                 └───────────────┬──────────────┘           └──────────────────────┘
                                 │ provider API (E2B / Modal / Docker)
                                 ▼
                 ┌──────────────────────────────────────────────────────────────────────┐
                 │ Sandboxes: one per (task, iteration), from the task's prebuilt image  │
                 │  bash exec · tests · artifacts  → result.json, trace.json, test-stdout │
                 └──────────────────────────────────────────────────────────────────────┘
```

Three trust zones, left to right: **control plane** (API + DB, holds tenant data and secrets),
**execution plane** (worker + runner subprocess, holds only the keys a run needs), **untrusted
compute** (sandboxes, where task code and agent-issued commands execute).

### Why these pieces and not others

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Queue | Postgres table + `SELECT … FOR UPDATE SKIP LOCKED` | Redis/Celery, RabbitMQ, SQS | One fewer system; enqueue is in the same transaction as the job row (no dual-write); throughput need is ~tens of jobs/hour, not thousands/sec. Cost: ~1 s poll latency. |
| API ↔ worker | Shared DB only (no RPC) | gRPC/HTTP between them | Workers can crash/scale independently; the DB is the contract. |
| Benchmark driver | Harbor CLI as subprocess | Harbor Python `Job` API in-process | Process boundary around LLM-rewritten code; crash of one run can't take the worker down; trivial to kill on cancel. Cost: progress is read from files, not hooks. |
| Default sandbox | E2B (Firecracker microVM, template per task) | Docker on worker host | Hardware isolation, concurrency not bounded by the worker box, per-sandbox network policy. Docker kept for dev/tests. Modal supported via Harbor if preferred. |
| State per iteration | Checkpointed steps in DB | Re-run job from scratch after crash | A benchmark pass costs minutes and dollars; we must resume, not restart. |
| LLM output | Full new `agent.py` + rationale, in JSON | Unified diff | Diffs from LLMs frequently fail to apply; the file is ~200 lines (~2–3k tokens); we compute the diff server-side for the history. |
| Traces | Compressed JSONB in Postgres (+ summary columns) | S3/GCS | ~50–500 KB per task-run; 16 tasks × 10 iterations ≈ 50 MB/job. Fine in Postgres for this scale; a `blob_ref` column lets us move to object storage later without schema change. |

---

## 4. Data model and job lifecycle

Core tables (all tenant-owned rows carry `org_id`):

- `organizations`, `users`, `api_keys(hash, user_id, org_id, role ∈ {admin, member})`
- `jobs(id, org_id, created_by, status, config jsonb, idempotency_key, best_iteration_id,
  stop_reason, lease_owner, lease_expires_at, lease_epoch, created_at, …)`
- `agent_versions(id, org_id, sha256, content text, parent_id, created_by_iteration_id)` —
  content-addressed, so "revert to parent" and "same code twice" are free.
- `iterations(id, job_id, index, agent_version_id, phase, score, n_pass, n_fail, n_error,
  proposal jsonb, decision ∈ {baseline, accepted, rejected, errored}, started_at, finished_at)`
- `task_runs(id, iteration_id, task_name, status, reward, failure_class, duration_s,
  exit_reason, n_steps, test_stdout text, trace_blob bytea, digest jsonb, sandbox_id)`
- `job_events(id, job_id, ts, kind, payload)` — append-only audit/progress log.

Job state machine (persisted; every transition is a compare-and-set on `status`):

```
QUEUED ─▶ RUNNING ─▶ SUCCEEDED
   │          │  ╲──▶ FAILED      (infra error after retries)
   │          ╰─────▶ CANCELLED   (user) 
   ╰───────────────▶ CANCELLED
```

Inside RUNNING, each iteration is a sub-state machine the worker checkpoints to `iterations.phase`:

```
PROVISION ─▶ BENCHMARK ─▶ COLLECT ─▶ DIGEST ─▶ PROPOSE ─▶ VALIDATE ─▶ (next iteration)
                                                            │
                                        stop? ◀─────────────╯  → FINALIZE (pick best version)
```

Crash semantics: a worker heartbeats `lease_expires_at`; if it dies, another worker claims the job
after expiry and resumes from the last completed phase. BENCHMARK is at-least-once (if the runner
died mid-way we re-run that iteration's benchmark; completed task_runs are kept and skipped via
`-i` task selection). A `lease_epoch` fencing token is written with every update so a zombie worker
that wakes up late cannot overwrite newer state.

---

## 5. The sandbox boundary (M3)

**What runs where**

| Code | Where it executes | Trust |
|---|---|---|
| API, worker, optimizer prompts | control/execution plane | ours |
| `harbor run` + `agent_vN.py` (LLM-rewritten) | runner subprocess, per iteration | semi-trusted |
| bash commands the agent issues, task tests | sandbox per (task, iteration) | untrusted |

**Runner subprocess hardening** (this is the "not in the worker process" boundary):
- Spawned per iteration with `start_new_session=True`, own cwd (`runs/<job>/<iter>/`), scrubbed
  environment: only the LLM key for the agent model and the sandbox provider key; no DB URL, no
  optimizer key, no other tenants' data. Killed as a process group on cancel/timeout.
- Hard wall-clock budget = `max(task agent timeouts) × 1.5 + provisioning slack`.
- The proposed `agent.py` passes a static policy check before it is ever executed: must parse
  (`ast`), must define `HarnessAgent(BaseAgent)` with `run()`, import allowlist (`json, os, re,
  asyncio, litellm, harbor.*`), no `subprocess`/`socket`/`open()` for write outside `self.logs_dir`,
  no reads of `os.environ` except `AGENT_MODEL`/`AGENT_REASONING_EFFORT`. Then a smoke import in a
  throwaway subprocess. Fails → iteration recorded as `rejected: invalid_agent`, LLM gets the error
  as feedback next round.
- In production the runner image is a separate container (`service-runner`) with no Docker socket;
  for local Docker provider mode the runner must talk to Docker, which we document as dev-only.

**Provider facts that matter** (from vendor docs, Nov 2026): E2B Hobby tier allows 20 concurrent
sandboxes, 1 create/s, 1 h max runtime, up to 8 vCPU / 8 GiB each — enough for 16 parallel tasks
at 900 s; Pro lifts this to 100+. Compute is ≈ $0.05/vCPU-h + $0.016/GiB-h, so a 15-minute
1 vCPU / 2 GB trial costs ≈ $0.02 (Daytona is priced identically; Modal sandboxes ≈ $0.03). A full
16-task iteration is therefore well under $0.50 of sandbox time; LLM tokens dominate cost.
E2B builds one template per (image, cpu, mem) and reuses it across jobs; Modal caches
`Image.from_registry` server-side; Docker pulls `alexgshaw/<task>:20251031` once.

**Sandbox lifecycle** (E2B default; Docker/Modal through the same Harbor flag):
1. *Pre-warm* (once per deployment, or lazily on first use): build the E2B template for each of
   the 16 tasks from its prebuilt image. Stored in DB `task_catalog(task_name, provider,
   template_ready_at)`. Removes a multi-minute cold start from the first job.
2. *Create*: Harbor creates one sandbox per task from the template (sub-second on E2B), tags it
   with `session_id` metadata; we extend metadata with `job_id`, `iteration_id`, `lease_epoch`.
3. *Run*: agent phase (network per task policy), then verifier phase in the same sandbox.
4. *Collect*: Harbor downloads `agent/`, `verifier/` logs and writes `result.json`. The worker
   parses these into `task_runs` as files appear (a watcher on the jobs dir gives live per-task
   progress: `7/16 done`).
5. *Destroy*: Harbor deletes the sandbox (`--delete`). A **reaper** task in the worker lists
   provider sandboxes every minute and kills any whose `job_id` is not RUNNING or whose
   `lease_epoch` is stale — this is what handles "the worker died and left 16 microVMs billing".

**Failure taxonomy and what we do**

| Failure | Detection | Handling |
|---|---|---|
| Sandbox fails to start / provider 5xx | Harbor exception in `result.json` | retry trial ×2 (Harbor `-r`), then `task_run.status = infra_error`, excluded from score denominator *and* flagged on the job |
| Agent timeout | `verifier_result` null + timing | `reward = 0`, `failure_class = timeout`; counts as a failure (matches repo semantics) |
| Agent crashed (bad code path) | exception in agent phase | `failure_class = agent_exception`; the traceback is fed to the LLM |
| Verifier timeout | timing | counted as failure, flagged |
| Runner subprocess hangs | wall-clock budget | kill process group, mark iteration `errored`, reaper cleans sandboxes, job retries iteration once |
| Worker dies | lease expiry | another worker resumes at last phase |

---

## 6. The optimization loop (M4)

### 6.1 Algorithm (per job)

```
v0 = template agent (or user-supplied file)          # agent_versions row
run(v0) on T tasks  →  R0                           # baseline iteration
best = (v0, score(R0)); history = []
for i in 1..max_iterations:
    dossier = digest(failures of best.run, history)   # §6.2
    proposal = LLM(dossier)  → {hypothesis, targets, rationale, new_agent_py}
    v_i = validate(proposal.new_agent_py)  (static policy + smoke import)
    R_i = run(v_i)                                   # §6.4 for the cheap variant
    decision = accept if score(R_i) > score(best) + δ   # δ = 0 by default, see 6.3
             | accept if score equal and fewer timeouts/steps (tie-break on cost)
             | reject otherwise
    history.append({v_i, hypothesis, decision, per-task flips})
    if accepted: best = (v_i, R_i)
    stop if any of: no accepted change in `patience` iterations · score == 1.0
                    · budget (LLM $ / sandbox minutes / wall-clock) exhausted · cancelled
finalize: job.best_iteration_id = best; expose diff v0→best and the whole history
```

Hill-climbing from the best-so-far parent, one hypothesis per iteration (the repo's PROGRAM.md
rules exist for good reason: small changes are attributable and revertible).

### 6.2 What the LLM sees (the "accumulated context" question)

The prompt is built from structured pieces, not raw logs, and ordered so the static prefix is
cacheable:

1. **System**: role, the constraints on `agent.py` (what may change: `AGENT_INSTRUCTION`, `TOOLS`
   schema, `MAX_STEPS`, truncation, loop logic; what may not: model, env provider, timeouts), the
   output JSON schema, and the "one hypothesis" rule.
2. **Current best `agent.py`** verbatim.
3. **Scoreboard**: per-task pass/fail over all iterations (a small matrix), so the LLM sees which
   tasks are stable, flaky, or never pass.
4. **Failure dossiers** for failing tasks in the best run (top-k by "fixability", k≈6):
   task instruction (truncated), `failure_class`, agent's last ~8 steps (command + truncated
   output), the final message, verifier `test-stdout` tail, steps used / max, wall time.
   Classes: `timeout`, `gave_up_early`, `wrong_output`, `loop_detected`, `tool_misuse`,
   `env_setup_failure`, `agent_exception`, `infra_error`. Classification is heuristic first
   (cheap, deterministic), LLM-assisted only when ambiguous.
5. **History ledger**: for each previous iteration — hypothesis, 5-line diff summary, decision,
   which tasks flipped. Rejected hypotheses are listed explicitly ("do not retry these verbatim").
6. **Learnings**: a short rolling memo the LLM itself updates each round (the repo's
   `learnings.md` idea, but a DB column).

Token budget target ≈ 25–40k input per proposal with caching on items 1–2.

### 6.3 Noise: the biggest correctness risk in M4

A single run of a stochastic agent on 16 tasks has a standard error around ±1 task (6%). Naive
hill-climbing will "accept" lucky noise and "reject" real gains. Mitigations, cheapest first:

- **Flip confirmation**: when a candidate differs from best only on a few tasks, re-run *only the
  flipped tasks* once more and take the majority. Costs 2–4 task-runs instead of 16.
- **Cost tie-break**: equal scores → prefer fewer timeouts and fewer total steps.
- **δ > 0 option**: `min_improvement` config for users who want conservative acceptance.
- **`eval_repeats` option** (k runs per task, pass@k or mean) for final confirmation of the best
  version only, not every iteration.

### 6.4 Cheaper iterations (system performance)

- **Two-phase evaluation**: run the candidate on *currently failing* tasks first. If none flips to
  pass, reject without touching the passing tasks (saves the pass-rate fraction of every rejected
  iteration, typically 40–60% of sandbox minutes). Only if there is upside do we run the full set
  to check for regressions.
- **Parallel candidates** (`candidates_per_iteration = k`): ask the LLM for k different hypotheses
  (temperature + "distinct from each other"), evaluate all in parallel, accept the best. Trades
  k× sandbox cost for k× fewer serial LLM round-trips. Off by default, on when wall-clock matters.
- **Content-addressed agent versions**: identical code → reuse results (used when the LLM proposes
  a revert or a duplicate).
- **Trace digests computed once** per task_run and stored; the LLM never re-reads raw traces.

### 6.5 Research we borrow from (one line each)

- Reflexion (Shinn et al. 2023): verbal self-reflection on failures stored as memory → our
  learnings memo + failure dossiers.
- GEPA (Agrawal et al. 2025): reflective prompt evolution with execution traces beats RL at ~35×
  fewer rollouts; keep a Pareto set over per-task outcomes rather than one scalar → our scoreboard
  matrix and (optional) parent sampling from the Pareto front instead of the single best.
- Darwin Gödel Machine (Zhang et al. 2025): archive of agent variants, open-ended exploration;
  evaluation noise handled by re-evaluating promising candidates → flip confirmation.
- Agent Workflow Memory (Wang et al. 2024): induce reusable workflows from successful traces →
  passing traces are a resource too; a later iteration can mine "what worked" into the prompt.
- Terminus-2 / Terminal-Bench authors' known wins (already in `program_templates/terminal_bench.md`):
  enforced TODO planning, env bootstrapping, non-interactive mode, verification before finishing.
  These are the hypotheses we expect the LLM to find first; we do not hardcode them.

---

## 7. Async processing (M2)

- `POST /jobs` validates, writes `jobs` + `job_events` in one transaction, returns **202** with
  `{job_id, status_url}`. `Idempotency-Key` header dedupes retries (same key + same org → same job).
- Workers loop: `UPDATE jobs SET status='RUNNING', lease_owner=$w, lease_expires_at=now()+60s,
  lease_epoch=lease_epoch+1 WHERE id = (SELECT id FROM jobs WHERE status='QUEUED' … FOR UPDATE SKIP
  LOCKED LIMIT 1) RETURNING *`. Heartbeat every 15 s. Per-org concurrency cap enforced in the same
  query (`WHERE (SELECT count(*) FROM jobs WHERE org_id=… AND status='RUNNING') < quota`).
- Cancellation is cooperative: API flips `cancel_requested=true`; the worker checks it at phase
  boundaries and on heartbeat, kills the runner, reaps sandboxes, writes `CANCELLED`.
- Progress: `GET /jobs/{id}` is cheap (one row + aggregates); `GET /jobs/{id}/events` for the
  timeline; optional `GET /jobs/{id}/stream` (SSE) later. `test_client.py` polls with backoff.
- Graceful shutdown: SIGTERM → stop claiming, finish current phase or release the lease.

---

## 8. API surface (M1)

```
POST   /v1/jobs                      submit {task_subset?, max_iterations, patience, budget, agent_source?, options}
GET    /v1/jobs/{id}                 status, current iteration, best score, stop_reason
GET    /v1/jobs/{id}/iterations      history: per-iteration score, decision, hypothesis, diff summary
GET    /v1/jobs/{id}/iterations/{n}  full detail: agent file, proposal, per-task results
GET    /v1/jobs/{id}/iterations/{n}/tasks/{task}   trace digest, test output, (raw trace on request)
GET    /v1/jobs/{id}/agent           best agent.py (text) + diff vs baseline
POST   /v1/jobs/{id}/cancel
GET    /v1/jobs?status=&created_by=  list (scoped by tenancy)
GET    /v1/tasks                     catalogue of the 16 tasks with category/difficulty/timeouts
GET    /v1/healthz, /v1/readyz
(admin) POST /v1/orgs, /v1/orgs/{id}/members, /v1/api-keys
```

Result schema (stable from M1, filled by simulation first, real runs later):
`{tasks: [{name, status: passed|failed|error, reward, failure_class, duration_s}], summary: {n, passed,
failed, errored, score}, failure_summary: [..]}`. Errors: RFC 7807 problem+json, 404 vs 403 chosen
so tenants cannot probe other tenants' ids (always 404 across orgs).

Simulation mode (`EXECUTOR=simulated`) returns deterministic fake results so M1/M2/M5 can be tested
without any sandbox or LLM; it is also what CI uses.

---

## 9. Multi-tenancy (M5)

- Auth: `Authorization: Bearer <api-key>`; keys hashed (sha256) at rest; key → user → org + role.
- Roles: `admin` (manage members/keys, see and cancel all org jobs), `member` (submit, see own).
- Enforcement in one FastAPI dependency that yields a `Principal`; every repository method takes it
  and adds `WHERE org_id = :org` (+ `created_by = :user` for members). No endpoint can forget it
  because the repository layer requires the principal argument.
- Defence in depth (optional): Postgres row-level security with `SET LOCAL app.org_id`.
- Quotas per org: max concurrent jobs, max iterations/job, monthly sandbox minutes.

---

## 10. Task subset

### Selection criteria (in priority order)

1. **Fast and bounded**: agent timeout ≤ 900 s, verifier ≤ 900 s, `cpus = 1`, memory ≤ 2–4 GB.
   One iteration then finishes in ≤ 15 min worst case with all tasks in parallel.
2. **Light environment**: no multi-GB assets, no torch/HF model downloads, no QEMU/TeX/Coq-sized
   images, no mandatory internet at run time. Keeps template builds and sandbox starts cheap.
3. **Signal, not noise**: avoid tasks that are near-0% for current frontier models in 15 minutes
   (no gradient to climb) and keep only 2–3 near-100% tasks as regression canaries.
4. **Representative**: cover the TB2 categories (software engineering, sysadmin, security,
   data processing, debugging, file operations, data querying, maths/games) and the difficulty mix.
5. **Fixable by scaffolding**: failure modes that prompt / loop changes can plausibly fix
   (not exploring, not verifying, giving up early, misreading output), not tasks whose failure
   is pure capability.

### Evidence used

Per-task solve rates come from the public TB2 leaderboard dataset (`harborframework/terminal-bench-2-leaderboard`,
10,947 trials, 27 agent/model submissions, 123 attempts per task; analysis by Simon Willison). They are
**mixed-strength averages**, so a strong agent like `gpt-5.4` will sit well above them. That is why the
subset leans toward the 25–70 % band: tasks at 90 %+ average are near-certain passes for our agent and
carry no learning signal; tasks under ~20 % are mostly capability-bound. Timeouts, CPU/memory and image
weight come from each task's `task.toml` and Dockerfile at the registry commit Harbor pins.

### Proposed 16 (all 900 s agent timeout, 1 CPU, ≤ 4 GB, prebuilt image, no run-time downloads)

| # | Task | Category | Difficulty | Avg solve % | Role / why it is in |
|---|---|---|---|---|---|
| 1 | fix-git | software-eng | easy | 96 | canary (regression detector), git state recovery |
| 2 | regex-log | data-processing | medium | 89 | canary; exact-spec reading, self-verification |
| 3 | nginx-request-logging | sysadmin | medium | 96 | canary; service config + verify loop |
| 4 | largest-eigenval | mathematics | medium | 70 | numeric precision, output-format discipline |
| 5 | sparql-university | data-querying | hard | 69 | query construction, iterative checking |
| 6 | extract-elf | file-operations | medium | 57 | binary tooling; needs env bootstrapping |
| 7 | sanitize-git-repo | security | medium | 57 | multi-step history rewriting; easy to stop early |
| 8 | cancel-async-tasks | software-eng | hard | 55 | concurrency semantics; tests are strict |
| 9 | tune-mjcf | scientific-computing | medium | 54 | parameter search against a simulator |
| 10 | configure-git-webserver | sysadmin | hard | 53 | multi-service setup; verification matters |
| 11 | chess-best-move | games | medium | 47 | reasoning + installing/using a tool |
| 12 | adaptive-rejection-sampler | scientific-computing | medium | 41 | algorithm implementation from spec |
| 13 | polyglot-c-py | software-eng | medium | 35 | constrained code generation; headroom |
| 14 | gcode-to-text | file-operations | medium | 25 | format reverse-engineering; headroom |
| 15 | sqlite-db-truncate | debugging | medium | 77 | forensic recovery; careful inspection |
| 16 | openssl-selfsigned-cert | security | medium | 80 | exact-spec command crafting |

Expected baseline for `gpt-5.4` on this set: roughly 55–70 % (the averages above understate a strong
model), which leaves 5–7 tasks for the loop to win and 3 canaries to protect. Category coverage:
software-eng 3, sysadmin 2, security 2, file-ops 2, scientific 2, data-processing 1, data-querying 1,
mathematics 1, games 1, debugging 1.

Alternates (same constraints) if the calibration run shows a task is degenerate for our agent:
`model-extraction-relu-logits` (maths, hard, 24 %), `build-pmars` (software-eng, 71 %),
`count-dataset-tokens` (model-training, 77 %), `build-cython-ext` (debugging, 76 %),
`password-recovery` (security, hard, 76 %, may be CPU-bound), `kv-store-grpc` (software-eng, 85 %).

Deliberately excluded, with the reason the README will state:
- **Infra-flaky**: `db-wal-recovery` (48 of 123 leaderboard trials errored), `caffe-cifar-10`,
  `gpt2-codegolf`, `train-fasttext`, `extract-moves-from-video` (60–90 errored trials each).
- **Heavy images / downloads**: `pytorch-model-recovery`, `sqlite-with-gcov`, `train-fasttext` (10–12 MB
  assets), `hf-model-inference`, `query-optimize`, `mteb-*` (model downloads), `qemu-*`,
  `install-windows-3.11` (QEMU), `overfull-hbox` (TeX Live), `mcmc-sampling-stan`, `rstan-to-pystan`
  (4 CPU / 8 GB), torch tasks.
- **Too slow**: anything with ≥ 1200 s agent timeout (`build-pov-ray` 12000 s, `sam-cell-seg` 7200 s,
  12 tasks at 3600 s, 17 at 1800 s).
- **No signal**: `make-doom-for-mips` (0 of 123), and the ≥ 95 % tasks beyond the three canaries
  (`git-leak-recovery`, `cobol-modernization`, `vulnerable-secret`, `prove-plus-comm`,
  `custom-memory-heap-crash`, `multi-source-data-merger`, `log-summary-date-ranges`).

### Calibration step (part of the plan, not optional)

Before freezing the list: run the baseline agent 2× on the 22 candidates (16 + 6 alternates).
Drop tasks that are 0/2 for both runs unless they are the only representative of a category; keep
at most 3 that are 2/2. Target baseline pass rate 35–55%, which gives the loop the most room to
show improvement without being noise-dominated. The README reports this calibration.

---

## 11. Performance plan, ranked by expected impact

**Agent score (what the loop optimizes)**
1. Failure dossiers with the verifier's actual test output (not just the agent's belief).
2. Scoreboard matrix + rejected-hypothesis ledger so the LLM stops repeating itself.
3. Flip confirmation so accepted changes are real.
4. Optional Pareto-parent sampling (GEPA-style) when hill-climbing stalls.

**Iteration wall-clock and cost**
1. All tasks of an iteration in parallel sandboxes (n = |tasks|). On E2B this makes an iteration
   ≈ the slowest task (5–12 min) rather than the sum. Local Docker on this laptop: n ≈ 5–6.
2. Pre-warmed templates; sandboxes are created from them in well under a second on E2B.
3. Two-phase evaluation (failing tasks first).
4. Prompt caching on the static prefix; digests instead of raw traces.
5. Parallel candidates when the user wants fewer serial rounds.
6. Per-task agent timeout tuned to the subset (900 s) and `MAX_STEPS` enforced, so dead-end runs
   stop costing money.

**Service**
- Async everywhere (asyncpg, httpx), stateless API, SKIP LOCKED queue, indexes on
  `(org_id, created_at)`, `(status, lease_expires_at)`; big blobs compressed and stored off the
  hot row.

---

## 12. Trade-off register

| Axis | Position | Consequence |
|---|---|---|
| Consistency | Strong within Postgres; every state change is a CAS with a fencing epoch | No split-brain between workers; slight write amplification |
| Availability | API and workers are stateless and horizontally scalable; DB is the SPOF (use managed HA Postgres) | Job continues across worker loss; API outage does not lose running jobs |
| Sync vs async | Submit is async (202); reads are sync and cheap; polling, not callbacks | Works for `test_client.py` and behind NAT; SSE is an additive upgrade |
| Isolation | Per-task microVM + per-iteration runner subprocess with scrubbed env | Strong containment of task code; LLM-rewritten agent still runs on our side but in a disposable process with a static policy gate |
| Exactly-once vs at-least-once | At-least-once for BENCHMARK, idempotent elsewhere | Possible duplicate sandbox spend after a crash, never lost or double-counted results |
| Cost vs signal | Single run per task by default + flip confirmation | Cheap iterations; noise controlled where it matters |
| Simplicity vs completeness | Postgres queue, no Redis/Celery; Harbor subprocess, no custom provider SDK code | Fewer moving parts to explain and operate; some latency and less fine-grained control |

---

## 13. Delivery plan (order matters: M3/M4 get the most time)

| Phase | Scope | Done when |
|---|---|---|
| 0 | branch; `service/` skeleton; compose (postgres, api, worker); Alembic; settings; simulated executor | `docker compose up` + `test_client.py` round-trips a simulated job |
| 1 (M1) | schemas, endpoints, problem+json errors, task catalogue | OpenAPI reads well; contract tests pass |
| 2 (M2) | SKIP LOCKED queue, leases, heartbeat, cancel, idempotency, resume-from-phase | kill -9 the worker mid-job → another worker finishes it |
| 3 (M3) | `HarborRunner`, per-iteration dirs, env scrubbing, policy gate, template pre-warm, result/trace ingestion, reaper, failure taxonomy | real 16-task baseline on E2B (or Docker) persisted and visible via API; chaos test: kill runner mid-run, orphan sandboxes get reaped |
| 4 (M4) | digests, prompt builder, proposal schema, validate/apply, accept/stop policy, flip confirmation, two-phase eval, history endpoints | a 5–8 iteration job shows an accepted improvement with full history in `test_client.py` output |
| 5 (M5) | orgs/users/keys/roles, scoping, quotas, tests for cross-tenant 404s | tenancy tests pass |
| 6 | README (tasks + rationale + calibration numbers, design decisions, what I'd do differently), demo run results, PR | PR open |

Rough effort: 0–2 ≈ 25%, 3 ≈ 30%, 4 ≈ 30%, 5–6 ≈ 15%.

---

## 14. Questions for you

- **[Q1] Sandbox provider and budget.** Which do you have credentials for: E2B, Modal, or local
  Docker only? A real 16-task × 6-iteration demo is roughly 100 sandbox-runs; on E2B that is a few
  dollars of compute plus LLM cost. I will design for E2B-default / Docker-dev regardless, but I can
  only *demonstrate* what we have keys for.
- **[Q2] Models.** Agent model stays `gpt-5.4` as in the repo? Which model for the optimizer LLM
  (I would use a stronger reasoning model with a JSON schema; can be Claude or OpenAI)?
- **[Q3] "Improve the performance" — which one?** I plan to optimize both the agent's pass rate
  and the iteration cost/latency. If you had to pick a headline metric for the README, which?
- **[Q4] Repo mechanics.** You do not have push rights to `neosigmaai/auto-harness`, I assume → fork,
  branch, PR against upstream (or PR within the fork)? Which GitHub account?
- **[Q5] Should the optimizer be allowed to change `MAX_STEPS`, the tool schema and the loop code,
  or only the system prompt?** I recommend "whole file minus model/timeouts" as the repo does.
- **[Q6] Any appetite for the optional extras** (parallel candidates, Pareto parents, SSE stream,
  RLS), or keep the core loop tight and spend remaining time on a convincing real run + README?

---

## 15. Cherry-on-top ideas (later, if time remains)

- LiteLLM proxy with per-job virtual keys and budgets → hard cost caps per tenant, and the LLM key
  never reaches the runner at all.
- Run the agent loop *inside* the sandbox (Harbor "installed agent" style) so even the rewritten
  Python never executes on our side; costs a per-sandbox `pip install` unless baked into templates.
- E2B sandbox pause/snapshot after environment setup for tasks with slow setup phases.
- Warm pool of sandboxes per task (create-ahead) to shave the last seconds of provisioning.
- Persist prompt-cache-friendly dossiers; dedupe identical failing traces across iterations.
- Per-task "oracle" diffing: compare the agent's trajectory with the task's reference `solution/`
  (available in the task bundle) when building dossiers — powerful, but must be clearly separated
  from anything the *agent* sees to avoid leaking solutions into the benchmark.
