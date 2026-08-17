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
| Async processing | DB-backed job queue + in-process asyncio worker claiming rows with `FOR UPDATE SKIP LOCKED` | Survives restart, no extra broker (Redis/Celery). API and worker communicate **through the DB** — clean lifecycle boundary. Note Celery/arq as scale-up path. |
| **Execution isolation** | **Pluggable `Executor` interface**, chosen per-job | The core abstraction. Agent never runs in the API or worker process. |
| — `SimulatedExecutor` | default | Deterministic rewards seeded by `hash(agent_source + task_id)` so improvements can move the score. No external deps → whole system + `test_client.py` run anywhere. |
| — `HarborExecutor` | real M3 path | Writes candidate `agent.py`, runs `harbor` in a sandbox (`env_provider`), parses rewards + trace excerpts. Requires `harbor` + provider key. |
| LLM proposer | **OpenAI** (`OPENAI_API_KEY`) | User choice. Structured output → new `agent.py` + rationale. Falls back to a deterministic mock proposer when no key, so M4 is always testable. |
| Sandbox note | `harbor`/`docker` not installed in this dev box; system py 3.14 | Simulated executor is the dev/test default; Harbor path documented + wired, validated when creds available. |

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

### Data model
- `organizations(id, name, created_at)`
- `users(id, org_id, email, role[admin|member], api_key, created_at)`
- `jobs(id, org_id, user_id, mode[single_run|optimize], executor, status[queued|running|succeeded|failed|cancelled], config JSONB, subset JSONB, max_iterations, error, created_at, updated_at, finished_at)`
- `iterations(id, job_id, idx, agent_source, proposed_improvement, rationale, val_score, accepted, results JSONB, created_at)`
- `task_results(id, iteration_id, task_id, reward, passed, output_excerpt)`

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

- [ ] **M0 — Scaffold** `harness_service/` package, config, DB engine/session, models, app factory, `docker-compose` for Postgres, deps in `pyproject`.
- [ ] **M1 — API design**: `POST /v1/jobs` (single_run) + `GET /v1/jobs/{id}` with clean schema, validation, error handling. Simulated executor. Structured result: passed / failed / failure summary.
- [ ] **M2 — Async**: submit returns immediately (status `queued`); worker claims + runs; caller polls. Job lifecycle in DB.
- [ ] **M3 — Sandbox execution**: `HarborExecutor` runs candidate agent in `harbor` sandbox, captures real task output → used as LLM context. Handle sandbox failure (→ job `failed` with error, `None` rewards). Executor selectable per job.
- [ ] **M4 — Optimization loop**: baseline → observe failures → OpenAI proposes new `agent.py` + rationale → apply → re-run → accept if improved → repeat until no improvement (`patience`) or `max_iterations`. Persist every iteration (agent snapshot, proposal, results, outcome). History via API.
- [ ] **M5 — Multi-tenancy (deferred)**: enforce roles — admins see all org activity, members submit + see only their own. API-level checks.

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

## 6. Open questions / risks
- Real Harbor validation depends on installing `harbor` + a sandbox provider key (E2B). Dev box lacks both → validated in simulated mode, Harbor path documented.
- OpenAI proposer must return a *runnable* `agent.py`; guard with syntax check (`compile()`) + fallback to previous good source on failure. Never accept a candidate that doesn't import.
- Keep per-job executor choice explicit so graders can run fast (simulated) or real (harbor).

## 7. Progress log
- _2026-08-17_: Branch `mvp1` created. Repo reconnaissance done. Plan drafted. Decisions locked: depth M1–M4, OpenAI proposer, simulated-default/harbor-real executor.
