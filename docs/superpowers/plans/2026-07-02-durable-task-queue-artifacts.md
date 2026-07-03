# Durable Task Queue And Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Milestones 1-3 explicit: every task has visible lifecycle state, queued/running work survives service restart, and Harbor/Daytona outputs are recorded with durable paths.

**Architecture:** Use Postgres as the durable queue by treating `aos_task_results` rows as task lifecycle records. `RunService` creates queued task rows at submit time, claims them into running state during execution or resume, writes final normalized task results, and computes run completion from terminal task rows. `TerminalBenchRunner` still owns Harbor execution, but now exposes per-task artifact metadata from the retained job directory.

**Tech Stack:** Python 3.13, FastAPI, psycopg, PostgreSQL, pytest, Harbor CLI, Daytona sandbox provider.

## Global Constraints

- Do not add Redis for this milestone.
- Preserve the existing HTTP API shape; improve returned progress from persisted task rows.
- Keep real execution path Harbor-first and Daytona-backed.
- Do not delete existing user changes or reset the worktree.
- Use TDD: each new production behavior starts with a failing test.

---

### Task 1: Persist Task Lifecycle Rows

**Files:**
- Modify: `tests/service/test_service.py`
- Modify: `tests/service/test_store.py`
- Modify: `autoharness_service/store.py`
- Modify: `autoharness_service/models.py`

**Interfaces:**
- Produces: `PostgresStore.create_task_queue(run_id, org_id, task_ids) -> None`
- Produces: `PostgresStore.mark_task_running(run_id, org_id, task_id) -> None`
- Produces: `PostgresStore.upsert_task_result(run_id, org_id, result) -> None`
- Produces: `PostgresStore.requeue_running_tasks(run_id, org_id) -> int`

- [ ] Write failing tests showing submit creates queued task rows and status progress reads queued tasks.
- [ ] Write failing tests showing a task can transition queued -> running -> passed.
- [ ] Implement lifecycle methods using upsert and `started_at` / `completed_at` columns.
- [ ] Run focused service/store tests.

### Task 2: Execute Runs From Durable Task Rows

**Files:**
- Modify: `tests/service/test_service.py`
- Modify: `tests/service/test_api.py`
- Modify: `autoharness_service/service.py`

**Interfaces:**
- Consumes: task lifecycle methods from Task 1.
- Produces: `RunService.resume_incomplete_runs() -> int`
- Produces: per-task execution status visible through `GET /runs/{run_id}`.

- [ ] Write failing tests showing `execute_run` marks current task running while a runner is blocked.
- [ ] Write failing tests showing `resume_incomplete_runs` picks up queued work after a new service instance is created.
- [ ] Change `execute_run` to process task rows, not only a whole batch result blob.
- [ ] Keep existing optimizer iteration behavior after all tasks finish.
- [ ] Run focused API/service tests.

### Task 3: Collect Harbor Artifacts

**Files:**
- Modify: `tests/service/test_runner.py`
- Modify: `tests/service/test_terminal_bench_runner.py`
- Modify: `autoharness_service/runner.py`
- Modify: `benchmark.py`

**Interfaces:**
- Produces: `TerminalBenchRunner.last_artifacts: dict[str, dict[str, str]]`
- Produces: `TerminalBenchRunnerAdapter.last_artifacts: dict[str, dict[str, str]]`

- [ ] Write failing runner tests using a fake Harbor job tree with `result.json`, `job.log`, `trial.log`, `agent/trace.json`, and verifier files.
- [ ] Implement artifact discovery with stable relative/absolute path strings.
- [ ] Merge artifact metadata into normalized task records.
- [ ] Run focused runner tests.

### Task 4: Verification And Review

**Files:**
- Verify: `autoharness_service/*`
- Verify: `tests/service/*`

- [ ] Run `python -m pytest tests/service -q`.
- [ ] Run `python -m black --check agent/agent.py benchmark.py autoharness_service tests/service test_client.py`.
- [ ] Run `python -m isort --check-only agent/agent.py benchmark.py autoharness_service tests/service test_client.py`.
- [ ] Inspect `git diff --check`.
- [ ] Review the final diff for scope, security, restart semantics, and artifact path completeness.
