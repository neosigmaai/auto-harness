# PLAN.md — Auto-Harness Optimization Service (branch: `mvp1`)

> Living document. Updated as we progress. Owner: @vitthal-bhandari.
> Timebox: ~5 hours. Target: **depth on M1–M4**, M5 (multi-tenancy) deferred but schema-ready.

---

## 1. Problem restatement

Build a **backend service** (FastAPI + Postgres) that runs the existing TerminalBench
agent (`agent/agent.py`) against a subset of tasks, observes failures, uses an LLM
(**OpenAI**) to propose improvements to the agent, applies them, and **re-runs to measure
whether performance improved** — looping until it stops improving or hits a max.

The repo already ships the *execution primitive* we wrap:
- `agent/agent.py` — the agent under optimization (system prompt + bash tool + loop).
- `benchmark.py::TerminalBenchRunner.run(task_ids)` — runs the agent in a **sandbox**
  (`env_provider`: e2b/daytona/modal/docker) via `harbor run`, returns `{task_id: reward}`,
  writes traces to `workspace/traces/latest/<task>/{trace.json,result.json}`.

We are NOT re-implementing agent execution. We are building the **service/infrastructure
around it**: HTTP API, async jobs, persisted iteration history, tenancy.

**What's graded (per the brief): the infrastructure, not the LLM call.** Reasoning about job
lifecycle, execution isolation, state across iterations, and access control.

---

## 2. Key decisions & tradeoffs

| Decision | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** | Required/preferred by brief; async-native. |
| Persistence | **Postgres** via SQLAlchemy 2.0 async + asyncpg | Required by brief. |
| Migrations | Bootstrap DDL on startup (create_all) for MVP; note Alembic as next step | Fast to build; MVP-appropriate. |
| Async pipeline built once | **Endpoint always enqueues + client always polls, from M1 onward.** "Simulated vs real" is an **executor** choice, NOT an API-shape choice. | Avoids throwaway inline-M1 code that M2 would rewrite. M1 and M2 collapse into one pipeline; M3 is a clean executor swap. Simulated executor resolves on the first poll. |
| Async processing | DB-backed job queue + in-process asyncio worker claiming rows with `FOR UPDATE SKIP LOCKED` | Survives restart, no extra broker (Redis/Celery). API and worker communicate **through the DB** — clean lifecycle boundary. Note Celery/arq as scale-up path. |
| **Execution isolation** | **Pluggable `Executor` interface**, chosen per-job | The core abstraction. Agent never runs in the API or worker process. |
| — `SimulatedExecutor` | M1 default | Deterministic rewards seeded by `hash(agent_source + task_id)` so improvements can move the score; returns dummy trace/failure text. No external deps → whole system + `test_client.py` run anywhere, instantly. |
| — `HarborExecutor` | real M3 path | Runs a **candidate** agent in an **E2B sandbox** via `harbor`, parses per-task rewards + real trace/stdout for LLM context. **E2B_API_KEY available → this path is validatable, not just documented.** |
| LLM proposer | **OpenAI** (`OPENAI_API_KEY`) | User choice. Structured output → new `agent.py` + rationale. Falls back to a deterministic mock proposer when no key, so M4 is always testable. |
| Sandbox provider | **`env_provider: e2b`** (key on hand) | Dev box lacks `harbor`/`docker` + system py is 3.14; installing `harbor` (needs py 3.12) + E2B key makes M3 runnable for real. |
| **Grading runs the REAL path** (confirmed w/ seniors) | M3 (E2B sandbox) + M4 (OpenAI loop) must genuinely run the agent against 10–20 real TerminalBench tasks and improve iteratively. Simulated stays **dev-only**. | Not "documented but simulated": the graded `python test_client.py` (localhost:8000, confirmed) must be able to drive a real optimize run. At M4, `test_client` default `--mode` → `optimize`; executor from server `DEFAULT_EXECUTOR` (harbor+keys for grading, simulated fallback keyless). |

---

## 3. Architecture

```
                 HTTP (test_client.py)
                        │
                 ┌──────▼───────┐        ┌──────────────────┐
                 │  FastAPI app │◀──────▶│    Postgres       │
                 │  (routers)   │  ORM   │  orgs, users,     │
                 └──────┬───────┘        │  jobs, iterations,│
                        │ enqueue (row)  │  task_results     │
                        │                └────────▲──────────┘
                 ┌──────▼───────────────┐         │ claim/update
                 │ asyncio worker loop  │─────────┘
                 │ (SKIP LOCKED claim)  │
                 └──────┬───────────────┘
                        │ optimize loop (M4)
                 ┌──────▼───────────────┐
                 │  Executor (protocol) │
                 ├───────────┬──────────┤
                 │ Simulated │  Harbor  │  ← agent runs HERE, isolated
                 └───────────┴──────────┘
```

### 3a. State model — "no information is lost" (designed in M1, filled by M4)

This is the part we invest in during M1. Two layers that mirror each other:

**Domain layer** (pure Python, in `harness_service/domain/`) — the vocabulary of the loop,
independent of ORM/HTTP:
- `AgentState` — an immutable snapshot of the agent at one iteration: the full `agent.py`
  **source**, plus surfaced params (`model`, `reasoning_effort`, `MAX_STEPS`,
  `MAX_OUTPUT_CHARS`) and a `content_hash`. This is what makes an iteration reproducible.
- `TaskOutcome` — one task's result: `task_id`, `reward`, `passed`, `duration`,
  `trace_excerpt`, `failure_reason`.
- `BenchmarkResult` — a set of `TaskOutcome`s + derived `val_score`, `n_passed`, `n_failed`.
- `Improvement` — the LLM's proposal: `rationale`, `diff_summary`, `new_agent_source`,
  `proposer` (openai/mock), raw request/response kept for audit.
- `Iteration` — ties it together: `idx`, `AgentState`, `BenchmarkResult`,
  optional `Improvement` (the one that *produced* this state), `accepted`, `decision_reason`.
- `Trajectory` — ordered `Iteration`s + the **accumulated context** (running learnings blob
  fed into each LLM call) + `best_val_score`/`best_iteration_idx`. The single object that
  guarantees nothing is dropped between iterations.

The domain layer is what `test_client.py` and the optimizer both reason over; the ORM is
just its persistence.

**Persistence layer** (SQLAlchemy, 1:1 with the domain objects):
- `organizations(id, name, created_at)`
- `users(id, org_id, email, role[admin|member], api_key, created_at)`
- `jobs(id, org_id, user_id, mode[single_run|optimize], executor, status[queued|running|succeeded|failed|cancelled], config JSONB, subset JSONB, max_iterations, patience, best_val_score, best_iteration_id, accumulated_context TEXT, error, created_at, updated_at, finished_at)`
- `iterations(id, job_id, idx, agent_source TEXT, agent_params JSONB, agent_hash, proposal_rationale TEXT, proposal_diff_summary TEXT, proposer, llm_request JSONB, llm_response JSONB, val_score, n_passed, n_failed, accepted, decision_reason, created_at)`
- `task_results(id, iteration_id, task_id, reward, passed, duration, trace_excerpt TEXT, failure_reason TEXT)`

Nothing is lost: every candidate `agent.py` (accepted **or** rejected), every LLM
request/response, every per-task trace excerpt, and the running context are all persisted
and reachable via the API.

Multi-tenancy columns (`org_id`, `user_id`, `role`, `api_key`) exist from day one;
**enforcement** (M5) is a thin dependency layer added last.

### API (v1)
| Method | Path | Milestone | Purpose |
|---|---|---|---|
| GET | `/health` | — | liveness |
| POST | `/v1/jobs` | M1/M2 | submit run/optimize job → returns `{job_id, status}` **immediately** |
| GET | `/v1/jobs/{id}` | M1/M2 | status + summary (passed/failed/failure summary) |
| GET | `/v1/jobs/{id}/iterations` | M4 | full iteration history |
| GET | `/v1/jobs` | M2/M5 | list jobs (tenant-scoped in M5) |
| POST | `/v1/orgs`, `/v1/orgs/{id}/users` | M5 | provisioning (later) |

Auth: `X-API-Key` header → user → org. Wired as a dependency; M1–M4 accept a default
seeded key, M5 enforces role/ownership.

---

## 4. Milestone plan & status

- [x] **M0 — Scaffold** `harness_service/` package (`constants`, `config`, `domain/`, `db/`, `executors/`, `api/`, `worker.py`), async DB engine/session, `docker-compose.harness.yml` for Postgres, `[service]` extra in `pyproject`. Boots + imports verified; `/health` served; all 5 tables register; `SimulatedExecutor` works.
- [x] **M1 — API + full state model, on dummy data.** ✅ Done + validated e2e against a real (embedded) Postgres: 21/21 checks incl. concurrency (SKIP LOCKED), determinism, auth 401, 404, 422, and lossless history. `test_client.py` verified against a live uvicorn server. Baseline `core` scores 0.25 (3/12) — realistic failure signal for M4.
      - `Executor` protocol + `SimulatedExecutor` returning deterministic dummy rewards + fake trace/failure text.
      - **Async pipeline already in place** (enqueue → worker → poll) — simulated executor just resolves fast.
      - `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/iterations` with clean Pydantic schemas, validation, error handling.
      - Domain layer (`Trajectory`/`Iteration`/`AgentState`/…) + ORM persisting it losslessly (§3a).
      - Clean class-based `test_client.py` (§4a) exercising submit→poll→summary.
- [ ] **M2 — Async processing, formalized.** Confirm submit returns immediately; worker claims via `SKIP LOCKED`; concurrency (multiple jobs), status transitions, restart-safety, timeouts. Mostly hardening what M1 already stood up.
- [ ] **M3 — Real sandbox execution.** `HarborExecutor` runs a **candidate** agent in an **E2B** sandbox, captures real per-task trace/stdout → LLM context. Executor selectable per job. Handle sandbox failure → job `failed` + `None` rewards. Harness integration per §4b.
- [ ] **M4 — Optimization loop.** baseline → observe failures → OpenAI proposes new `agent.py` + rationale → validate (`compile()`) → apply → re-run → accept if `val_score` improved → repeat until no improvement (`patience`) or `max_iterations`. Accumulated context threaded through each LLM call. Every iteration persisted; history via API.
- [ ] **M5 — Multi-tenancy (deferred).** Enforce roles — admins see all org activity, members submit + see only their own. API-level checks on the columns already present.

### 4a. `test_client.py` design (clean, class-based, stateful)
- `HarnessClient` — thin HTTP wrapper (base_url, api_key, `requests.Session`); one method per endpoint.
- `JobRun` — holds state for one submitted job: `job_id`, last polled `status`, cached
  `iterations`, `best_val_score`. Methods: `submit()`, `poll_until_done(interval, timeout)`,
  `summary()` (pretty structured print: per-task pass/fail, val_score per iteration, the
  improvement + rationale at each step, final outcome — the full trajectory when M4 is on).
- `__main__`: parse args (subset, mode, executor, max_iterations) → submit → poll → print.

### 4b. Harness integration for M3 (expected *tiny* changes)
1. **Candidate isolation (likely zero-change):** write each candidate to
   `agent/_candidates/job_<id>.py` and run with
   `agent_import_path="agent._candidates.job_<id>:HarnessAgent"` (already a ctor param;
   importable from repo root). Avoids clobbering tracked `agent/agent.py`; concurrency-safe.
2. **Expose per-task output (small change to `benchmark.py`):** today `run()` copies traces
   to `workspace/traces/latest/` only when `split=="train"` and returns rewards only. Add an
   opt-in to emit per-task trace/stdout into the caller-supplied `jobs_dir` (or return their
   paths) so each job reads its own output in isolation. Exact minimal diff confirmed at M3;
   fall back to reading the run's job_dir if we can avoid touching `benchmark.py` at all.

### Deliverables
- [ ] `test_client.py` — submits a job, polls, prints structured summary incl. full iteration history.
- [ ] `README` section — setup/run, task subset + rationale, design decisions, what's next, what's intentionally skipped.

---

## 5. TerminalBench task subset (10–20, representative + fast)

Selection criteria: spread across categories (coding / sysadmin / data / security),
avoid the longest-running tasks, keep the full subset completing in a reasonable time.
Stored in `harness_service/tasks.py`. **Exact IDs to be confirmed against the live
`terminal-bench@2.0` dataset (`harbor tasks -d terminal-bench@2.0`) when harbor is
available** — simulated mode is dataset-agnostic. Rationale written into README.

---

## 5b. Delivery (confirmed with seniors)
- `origin` is the **upstream** `neosigmaai/auto-harness` (no write access) → deliver via **fork + PR**.
  Flow: `gh auth login` (user-run, credentialed) → `gh repo fork … --remote --remote-name fork`
  → `git push -u fork mvp1` → `gh pr create --repo neosigmaai/auto-harness --head <user>:mvp1`.
- Grader runs the service per README, then `python test_client.py` with **defaults (localhost:8000)** — confirmed.

## 6. Open questions / risks
- **E2B key available** → real Harbor/M3 path is validatable. Still needs `harbor` installed against Python 3.12 (system py here is 3.14, `harbor`/`docker` absent) — one-time setup, documented in README.
- OpenAI proposer must return a *runnable* `agent.py`; guard with `compile()` + fallback to previous good source on failure. Never accept a candidate that doesn't import.
- Candidate `agent.py` files under `agent/_candidates/` are generated artifacts → gitignore them; keep `agent/agent.py` (the tracked template copy) untouched by jobs.
- Keep per-job executor choice explicit so graders can run fast (simulated) or real (harbor+e2b).
- Concurrency: `SimulatedExecutor` is safe; `HarborExecutor` writing candidate modules must use unique per-job paths (covered by §4b.1).

## 7. Progress log
- _2026-08-17_: **Clarifications from seniors.** (a) Delivery = fork + PR (upstream `origin` is read-only); gh not yet authed on this box — user runs `gh auth login`, then fork/push/PR. (b) test_client run = service per README + `python test_client.py` defaults (localhost:8000) — confirmed. (c) **Grading executes the REAL agent against 10–20 TerminalBench tasks + real LLM iterative improvement** → M3/M4 real paths are mandatory and must be validated with actual harbor+E2B+OpenAI (needs keys in `.env` + `uv tool install harbor` + a small budget). Plan updated: simulated is dev-only; test_client default mode → optimize at M4.
- _2026-08-17_: Branch `mvp1` created. Repo reconnaissance done. Plan drafted. Decisions locked: depth M1–M4, OpenAI proposer, simulated-default/harbor-real executor.
- _2026-08-17_: **M1 landed + validated.** Job pipeline live: `tasks.py` (core/smoke subsets), `agent_source.py` (baseline from the terminal_bench template), `services/jobs.py` (create/claim `FOR UPDATE SKIP LOCKED`/persist), `services/runner.py` (baseline Trajectory; M4 extends), real `Worker` (N concurrent claim loops, executor runs outside the DB txn), `api/deps.py` (X-API-Key principal + dev seed), job schemas + routes (`POST /v1/jobs`, `GET /v1/jobs`, `/{id}`, `/{id}/iterations`), and `test_client.py` (`HarnessClient` + stateful `JobRun`). Validated against a real embedded Postgres (`pgserver`) since docker isn't on this box — 21/21 e2e checks, plus `test_client.py` driven against a live uvicorn server. README gained a service quick-start + task-subset rationale.
- _2026-08-17_: **M0 scaffold landed.** Package skeleton under `harness_service/`: constants (enums+defaults), pydantic-settings config, domain dataclasses (`AgentState`/`TaskOutcome`/`BenchmarkResult`/`Improvement`/`Iteration`/`Trajectory`, "produced" semantics), async SQLAlchemy engine + ORM (orgs/users/jobs/iterations/task_results, tenancy columns present), `Executor` protocol + working `SimulatedExecutor` + registry, heartbeat `Worker` (claim stub for M1), FastAPI app w/ lifespan (init_db + worker) and `/health`. Compose file for Postgres; `.env.example` + `pyproject[service]` updated. Verified via 3.12 venv: imports clean, app builds, `/health` served, all tables register.
- _2026-08-17_: Plan refined after design discussion. **E2B key confirmed available** (default `env_provider: e2b`, M3 validatable). M1 reframed as the *investment* milestone: dummy-data executor + front-loaded lossless state model (domain layer §3a + ORM). Locked "simulated vs real = executor choice only; async pipeline built once" so M1≈M2 and M3 is a clean swap. Added `test_client.py` design (§4a) and M3 harness-integration touch-points (§4b).
