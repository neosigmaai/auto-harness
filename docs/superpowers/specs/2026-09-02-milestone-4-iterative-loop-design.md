# Milestone 4 — Iterative Optimization Loop: Design

**Date:** 2026-09-02
**Status:** Draft for review
**Branch:** `ha/iterative-improvement`

## 1. Goal

Once an agent has run and failures are observed, the service uses an LLM to
propose an improvement, apply it, and re-run the benchmark to check whether
performance improved — looping until performance stops improving or a maximum
number of iterations is reached. The full iteration history (agent state at
each step, proposal, benchmark results, final outcome) is persisted and
accessible via the API.

## 2. Decisions already made (with rationale)

| Decision | Choice |
|---|---|
| Mutable surface | **Prompt + config only.** The improver edits an `AgentSpec` (system prompt, model params, step limits) — pure data, no code generation. Tools stay fixed (bash). |
| Best output | **One best overall agent per job**, chosen by aggregate score. Per-task data remains visible in iteration history. |
| Loop execution | **Typed step queue.** The queue holds `evaluate` and `improve` steps; workers are stateless and claim steps exactly like they claim runs today. No orchestrator process. The worker that completes a step enqueues its successor in the same DB transaction. |
| Agent storage | **Agent versions live in Postgres (JSONB), not in the repo.** `agent/agent.py` is never mutated by the service. |
| Traces | **Artifact store** (local-disk implementation behind an interface), not the repo and not the DB. |
| Log streaming UX | **Deferred.** Not core architecture; `GET /v1/jobs/{id}` polling shows step-level progress, artifacts are retrievable after each evaluate step. |

## 3. Architecture overview

```
POST /v1/jobs
   │  create job, insert AgentSpec v0, enqueue step(evaluate, iter=0)
   ▼
steps table  (Postgres queue, FOR UPDATE SKIP LOCKED — same pattern as runs)
   │
   │  worker claims next step
   ▼
┌───────────────────────────────────────────────────────────────┐
│ evaluate step                                                 │
│   materialize spec JSON → workspace/runs/<run_id>/            │
│   create run row → HarborBenchmarkRunner.execute_sync(run_id) │
│   collect traces → ArtifactStore                              │
│   score = mean reward                                         │
│   [txn] complete step + update job best + stopping check      │
│         → enqueue improve  OR  close job                      │
├───────────────────────────────────────────────────────────────┤
│ improve step                                                  │
│   build context: latest failure traces + iteration history    │
│   LLM call → proposed AgentSpec + rationale                   │
│   validate against schema                                     │
│   [txn] insert agent_version N+1 + complete step              │
│         + enqueue evaluate(iter=N+1)                          │
└───────────────────────────────────────────────────────────────┘
```

The existing `runs` / `run_tasks` tables and `/v1/runs` endpoints are
unchanged. An evaluate step executes its run **inline** (create run row →
`execute_sync`) — the run is a *result record*, not a second queue hop, so
there is no wait state between step and run.

Naming note: queue units are called **steps**, never "tasks" — `task_id`
already means a benchmark task throughout the codebase.

## 4. AgentSpec and the generic runtime

### 4.1 AgentSpec schema (Pydantic, `api/agent_spec.py`)

```python
class AgentSpec(BaseModel):
    system_prompt: str                    # min_length=1, max_length=20_000
    agent_model: str                      # e.g. "gpt-4.1-mini"
    max_steps: int = 80                   # ge=1, le=200
    max_output_chars: int = 8000          # ge=500, le=100_000
    exec_timeout_sec: int = 120           # ge=10, le=1200
    model_config = ConfigDict(extra="forbid")   # reject unknown fields
```

`extra="forbid"` plus bounds are the validation gate on improver output: a
proposal that doesn't parse is a failed improve step, never a crashed job.

Version 0 of every job is built from the current constants in the
terminal-bench template (its `AGENT_INSTRUCTION`, `MAX_STEPS`, etc.), with
`agent_model` from the request or config default.

### 4.2 Generic runtime: `agent/spec_agent.py` (new file)

A copy of the current `HarnessAgent` loop that reads its spec from the JSON
file named by the `HARNESS_AGENT_SPEC` env var instead of module constants.
Fixed bash tool, same trace saving (`HARNESS_SAVE_TRACE`), same token
accounting. If `HARNESS_AGENT_SPEC` is unset it falls back to the template
defaults, so it is runnable standalone.

**Why a new file rather than rewriting `agent/agent.py`:** in the Layer A CLI
loop, `agent/agent.py` is *the file the coding agent edits* (enforced by
`gating.py`), and `prepare.py` overwrites it from templates. Layer B's
spec-driven agent must not share that mutable file. The service never reads
or writes `agent/agent.py` after this change.

### 4.3 Plumbing (small Layer A touch)

`TerminalBenchRunner` (`benchmark.py`) gains two optional constructor params,
defaulting to current behavior:

- `agent_import_path: str = "agent.agent:HarnessAgent"` — Layer B passes
  `"agent.spec_agent:HarnessAgent"`.
- `extra_env: dict[str, str] | None = None` — merged into the `harbor run`
  subprocess environment. Layer B passes
  `{"HARNESS_AGENT_SPEC": <materialized spec path>}`.

The worker materializes the version's spec to
`workspace/runs/<run_id>/agent_spec.json` before invoking the runner. The
harbor CLI spawns the agent on the host (it orchestrates the sandbox via
`environment.exec`), so subprocess env vars reach the agent directly.

`HarborBenchmarkRunner._check_agent_import` checks `agent/spec_agent.py`
for job-driven runs (plain `/v1/runs` keeps the existing check).

## 5. Data model (new tables; existing tables untouched)

### `jobs`

| column | type | notes |
|---|---|---|
| id | UUID pk | |
| status | str | queued \| running \| completed \| failed \| cancelled |
| task_ids | JSONB | validated against config allowlist at POST time |
| agent_model | str | for spec v0 |
| improver_model | str | |
| max_iterations | int | |
| patience | int | consecutive non-improving evaluations before stop |
| min_delta | float | required score improvement |
| current_iteration | int | |
| best_agent_version_id | UUID fk → agent_versions, nullable | |
| best_score | float, nullable | |
| stop_reason | str, nullable | max_iterations \| no_improvement \| budget_exceeded \| failed_improve \| failed |
| created_at / started_at / finished_at | timestamptz | |
| error_code / error_message | str / text | same envelope as runs |

### `agent_versions`

| column | type | notes |
|---|---|---|
| id | UUID pk | |
| job_id | UUID fk → jobs | |
| version | int | unique per job; 0 = baseline |
| parent_version_id | UUID fk, nullable | null for v0 |
| spec | JSONB | the full AgentSpec |
| rationale | text | improver's explanation ("why this change") |
| created_by | str | baseline \| improver |
| created_at | timestamptz | |

Full snapshot per version, not diffs — specs are a few KB and snapshots make
"state of the agent at each step" a single row read.

### `steps`

| column | type | notes |
|---|---|---|
| id | UUID pk | |
| job_id | UUID fk → jobs | |
| type | str | evaluate \| improve |
| status | str | queued \| running \| completed \| failed |
| iteration | int | |
| agent_version_id | UUID fk → agent_versions | version being evaluated / improved upon |
| run_id | UUID fk → runs, nullable | set by evaluate steps |
| score | float, nullable | evaluate: mean reward for this iteration |
| stale_after_sec | int | staleness threshold, computed per step type at enqueue time (see §6) |
| worker_id / claimed_at | str / timestamptz | claim bookkeeping, same as runs |
| created_at / started_at / finished_at | timestamptz | |
| error_code / error_message | str / text | |

### Artifact pointers

Artifacts are addressed by convention, not by table:
`jobs/<job_id>/iterations/<n>/tasks/<task_id>/trace.json` and
`jobs/<job_id>/iterations/<n>/improver/{prompt.txt,response.json}`. The API
can enumerate a prefix; no extra rows to keep consistent.

## 6. Queue mechanics and crash recovery

### Claiming

`PostgresJobStore.claim_next_step(worker_id, stale_after_sec)` mirrors
`PostgresRunStore.claim_next`: requeue stale running steps, then
`SELECT ... FOR UPDATE SKIP LOCKED` ordered by `created_at`.

`worker.process_one` claims **a step first, then falls back to a legacy
standalone run**, so one worker binary serves both `/v1/runs` and `/v1/jobs`.

### Staleness

Evaluate steps are long — worst case
`per_task_timeout × ceil(len(task_ids) / max_concurrency)` (~2.7 h for 16
tasks at concurrency 2). Step staleness is therefore computed per step type:
improve steps use the run default (1800 s); evaluate steps use the formula
above plus 20 % slack, stored on the step row at enqueue time
(`stale_after_sec` column) so the claimer doesn't need config context.

### Transactional chaining

One store method owns every transition:

```python
def complete_step_and_advance(step_id, *, outcome) -> None
```

In a single transaction it: marks the step completed/failed, applies job
updates (score, best pointer, `current_iteration`), and either inserts the
successor step or closes the job with a `stop_reason`. A crash before commit
leaves the step `running` until stale-requeue re-runs it; a crash after
commit leaves the successor queued. There is no window where a job is alive
with nothing queued.

### Idempotent re-execution

A requeued evaluate step creates a **fresh run row** (the orphaned run is
marked failed with `error_code=superseded`); the step's `run_id` always
points at the latest attempt. A requeued improve step simply re-calls the
LLM — no state was committed.

### Failure policy

- Improve step fails (LLM error, invalid spec after 2 retries) → job
  `completed` with `stop_reason=failed_improve` if a best version exists,
  else job `failed`. The best-so-far agent is still a valid answer.
- Evaluate step fails (`ExecutionUnavailableError`, harbor crash) → job
  `failed` with the run's error envelope. Infra failures should not silently
  count as "no improvement".

## 7. The improver (`api/services/improver.py`)

```python
class Improver(Protocol):
    def propose(self, *, spec: AgentSpec, evaluation: EvaluationSummary,
                history: list[IterationRecord]) -> Proposal  # (AgentSpec, rationale)
```

- **`LLMImprover`** — one litellm call with JSON/tool-structured output:
  `{system_prompt, config_changes, rationale}`. Output is validated through
  `AgentSpec`; one retry with the validation error appended, then the step
  fails.
- **`FakeImprover`** — deterministic scripted proposals for tests (mirrors
  `MockBenchmarkRunner`'s role).

### Context assembly (the "accumulated context" requirement)

Each call receives, in order, within a hard character budget (default
~60k chars, config `improver_context_budget`):

1. The current `AgentSpec` (always, full).
2. **Iteration history table** (always, compact): per prior iteration —
   version, one-line change summary, rationale, score, improved-or-not.
   This is what prevents re-proposing failed ideas.
3. **Latest evaluation**: per-task status/reward table, then failure details
   for failed/error tasks — the tail of each `trace.json` (last N messages,
   command outputs truncated), worst tasks first, until the budget is spent.

Only the latest run's traces enter the prompt; older traces stay in the
artifact store. History carries forward as the compact table, not raw text.

## 8. Scoring and stopping

- **Score** = mean reward across the job's tasks, with `None` (error/timeout)
  counted as 0.0. Mean reward beats pass-rate on small task sets because
  partial rewards carry signal (`reward_to_task_status` already preserves
  them).
- **Improved** ⇔ `score > best_score + min_delta`.
- **Stop after an evaluate step when any of:**
  1. `iteration + 1 >= max_iterations` → `stop_reason=max_iterations`
  2. consecutive non-improving evaluations `>= patience` →
     `stop_reason=no_improvement`
  3. wall-clock since job start `> max_job_duration_sec` →
     `stop_reason=budget_exceeded`
- Defaults (config, overridable per job): `max_iterations=5`, `patience=2`,
  `min_delta=0.01`, `max_job_duration_sec=6 h`.
- The baseline evaluation (iteration 0) sets the initial `best_*`; the loop
  therefore always reports a best agent, even if no proposal ever improved.

## 9. Artifact store (`api/services/artifacts.py`)

```python
class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes | Path) -> None
    def get(self, key: str) -> bytes
    def list(self, prefix: str) -> list[str]
    def exists(self, key: str) -> bool
```

`LocalArtifactStore(root=workspace/artifacts)` now; an S3 implementation
later is a drop-in (same factory pattern as `create_runner`). After each
evaluate step the worker copies each trial's `trace.json` (and harbor's
`result.json`) out of the job dir into the store; the improver reads traces
from the store, never from harbor's directory layout. Improver prompt +
response are stored per iteration for auditability.

## 10. API contract

Existing `/v1/runs`, `/tasks`, `/health` are unchanged.

### `POST /v1/jobs` → 202

```json
{
  "task_ids": ["fix-git", "regex-log"],      // optional, same allowlist rules as runs
  "agent_model": "gpt-4.1-mini",             // optional
  "improver_model": "gpt-5.4",               // optional
  "max_iterations": 5,                        // optional
  "patience": 2,                              // optional
  "min_delta": 0.01                           // optional
}
→ { "job_id": "...", "status": "queued", "created_at": "..." }
```

Same validation/error envelope as runs (`unknown_task_ids` 400,
`execution_unavailable` 503, etc.).

### `GET /v1/jobs/{job_id}` → 200

```json
{
  "job_id": "...",
  "status": "running",
  "created_at": "...", "started_at": "...", "finished_at": null,
  "config": { "task_ids": [...], "agent_model": "...", "improver_model": "...",
              "max_iterations": 5, "patience": 2, "min_delta": 0.01 },
  "current_iteration": 2,
  "best": { "agent_version_id": "...", "version": 1, "score": 0.75 },
  "stop_reason": null,
  "iterations": [
    {
      "iteration": 0,
      "agent_version_id": "...", "version": 0,
      "run_id": "...",
      "score": 0.62,
      "improved": true,
      "summary": { "total": 16, "passed": 10, "failed": 4, "error": 2, "...": "..." },
      "proposal": null
    },
    {
      "iteration": 1,
      "agent_version_id": "...", "version": 1,
      "run_id": "...",
      "score": 0.75,
      "improved": true,
      "summary": { "...": "..." },
      "proposal": { "rationale": "Added explicit verification step to prompt...",
                    "changed_fields": ["system_prompt", "max_steps"] }
    }
  ],
  "error": null
}
```

Iterations embed version *references* + proposal metadata; full specs come
from the versions endpoint (keeps the job payload readable).

### `GET /v1/jobs/{job_id}/best` → 200

Full winning `AgentSpec` inline, plus `version`, `score`, and the version's
`rationale`. 404 `job_not_found`; 409 `no_evaluation_yet` if iteration 0
hasn't finished.

### `GET /v1/agent-versions/{version_id}` → 200

Full spec + metadata (job_id, version, parent_version_id, rationale,
created_by, created_at).

## 11. Config additions (`config/benchmark.yaml`)

```yaml
improver_model: gpt-5.4
max_iterations: 5
patience: 2
min_delta: 0.01
max_job_duration_sec: 21600
improver_context_budget: 60000
artifacts_dir: workspace/artifacts
```

All become `BenchmarkConfig` fields with the same validation style as
Milestone 3's additions. Per-job request fields override config defaults.

## 12. Out of scope (deliberately deferred)

- Log streaming UX (SSE / live tail) — polling + artifact download suffices.
- S3/GCS artifact backend — interface is ready, implementation later.
- Per-task best-agent mapping — derivable from history if wanted later.
- Job cancellation endpoint — status enum reserves `cancelled`.
- Alembic migrations — `init_db()` `create_all` covers new tables, consistent
  with current practice.

## 13. Open questions for the assignment designers

1. Improvement metric: pass rate or mean reward? Is single-run comparison
   acceptable given run-to-run variance?
2. Mutable surface: is prompt+config sufficient, or are tool/code changes
   expected?
3. "State of the agent at each step accessible via the API" — inline
   snapshots or referenced versions? (We do references + a fetch endpoint.)
4. Expected iteration counts / cost caps?
5. Should concurrent jobs be supported? (This design supports them; per-run
   spec materialization means no shared mutable files.)

## 14. Testing strategy

All loop logic is testable without Harbor or a real LLM:

- **Unit (no DB):** AgentSpec validation (bounds, extra fields, truncation);
  scoring (mean with `None`→0); stopping rule (table-driven: improves,
  plateaus, regresses, hits max, hits budget); improver context assembly
  (budget trimming, history table shape).
- **Store (Postgres, same skipif guard as `test_api.py`):**
  `claim_next_step` no-double-claim (mirror the existing
  `ThreadPoolExecutor` test); stale evaluate step requeue with per-step
  threshold; `complete_step_and_advance` transitions — evaluate→improve,
  improve→evaluate, each stop reason, orphaned-run supersede.
- **End-to-end (Postgres + `MockBenchmarkRunner` + `FakeImprover`):** a job
  that improves twice then plateaus → stops with `no_improvement`, best
  pointer correct, history complete; a job hitting `max_iterations`; improver
  returning invalid spec → retry then `failed_improve`; worker crash
  simulation (kill between steps → resume).
- **API:** POST validation reuses run tests' patterns; GET shapes; `best`
  404/409 paths.
