# Agent Optimization Service MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small FastAPI backend service that accepts benchmark run requests, executes selected Terminal-Bench tasks asynchronously through simulated or Harbor/Daytona-backed runners, persists run/task/iteration state in PostgreSQL, and exposes a polling API plus `test_client.py`.

**Architecture:** Keep the take-home MVP narrow: `FastAPI API -> RunService -> Store -> Runner -> Normalizer -> Store`. Real sandbox execution is Harbor-first (`TerminalBenchRunner -> harbor run --env daytona`), not direct Daytona SDK. Optimization is a minimal LLM proposal/history step, not multi-candidate search, GateEngine, or suite promotion.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic, psycopg 3, PostgreSQL, pytest, FastAPI TestClient/httpx, existing `benchmark.py` TerminalBenchRunner, existing OpenAI dependency.

## Global Constraints

- Work only in `/Users/yinfeiwang/workspace/neosigma/auto-harness/.worktrees/takehome-mvp` on branch `takehome/mvp`.
- `archive` branch is reference only; do not merge/cherry-pick the heavy runtime implementation wholesale.
- `AGENTS.md` must remain ignored and must never be staged or pushed.
- Do not use `git push --force` or `git push --force-with-lease`.
- Do not commit `.env`, `experiment_config.yaml`, `workspace/`, raw benchmark jobs, or API keys.
- Assignment MVP scope is Milestones 1-3 plus minimal Milestone 4 history/proposal; Milestone 5 is header-based mock tenancy and README design.
- Do not implement GateEngine, SuiteStore, CandidateGraphManager, Beam Search, MergeEngine, Kubernetes workers, Redis/Kafka, or direct Daytona SDK in this MVP.
- Real execution path uses Harbor as Daytona adapter: `TerminalBenchRunner -> harbor run --env daytona -> local jobs_dir/artifacts`.
- Local PostgreSQL is written only by the service process; Daytona sandboxes do not connect to local PostgreSQL.
- Simulated mode must work without Daytona, Harbor, or OpenAI keys so reviewers can validate API and lifecycle quickly.
- Real mode may require `OPENAI_API_KEY`, `DAYTONA_API_KEY`, and `harbor` CLI.

---

## File Structure

Create a focused service package and keep the old benchmark loop intact.

```text
autoharness_service/
  __init__.py
  api.py              FastAPI app factory and HTTP routes
  config.py           environment settings and defaults
  models.py           dataclasses/enums used inside the service
  schemas.py          Pydantic request/response models
  store.py            PostgreSQL persistence and schema init
  normalizer.py       task result and failure summary normalization
  runner.py           simulated runner and TerminalBenchRunner adapter
  service.py          RunService orchestration and background execution
  optimizer.py        failure summary -> one LLM proposal
  main.py             uvicorn entrypoint module

tests/service/
  test_normalizer.py
  test_store.py
  test_runner.py
  test_service.py
  test_api.py
  test_optimizer.py

test_client.py        end-to-end submit/poll/summary script
```

Modify:

```text
pyproject.toml        add service/test dependencies
.env.example          add DATABASE_URL and service defaults
.gitignore            ensure AGENTS.md is ignored
README.md             add take-home service setup, API, task selection, design decisions
```

---

## Task 1: Dependencies, Package Skeleton, and Import Baseline

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `.gitignore`
- Create: `autoharness_service/__init__.py`
- Create: `autoharness_service/config.py`
- Create: `tests/service/test_imports.py`

**Interfaces:**
- Produces `autoharness_service.config.ServiceSettings`.
- Produces `autoharness_service.config.load_settings() -> ServiceSettings`.
- Later tasks consume `ServiceSettings.database_url`, `default_model`, `default_sandbox_provider`, `default_mode`, `max_local_concurrency`, and `poll_interval_sec`.

- [ ] **Step 1: Write the failing import/config test**

Create `tests/service/test_imports.py`:

```python
from autoharness_service.config import ServiceSettings, load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUTOHARNESS_SERVICE_MODE", raising=False)

    settings = load_settings()

    assert isinstance(settings, ServiceSettings)
    assert settings.database_url == "postgresql://autoharness:autoharness@localhost:5432/autoharness"
    assert settings.default_mode == "simulated"
    assert settings.default_sandbox_provider == "daytona"
    assert settings.max_local_concurrency == 4
```

- [ ] **Step 2: Run the test and verify it fails before implementation**

Run:

```bash
python -m pytest tests/service/test_imports.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'autoharness_service'
```

- [ ] **Step 3: Add dependencies**

Modify `pyproject.toml` dependencies to include the service runtime and test stack:

```toml
dependencies = [
    "openai",
    "pyyaml",
    "tau2",
    "fastapi",
    "uvicorn[standard]",
    "psycopg[binary]",
    "pytest",
    "httpx",
]
```

- [ ] **Step 4: Add service environment defaults**

Append to `.env.example`:

```text
# Agent Optimization Service
DATABASE_URL=postgresql://autoharness:autoharness@localhost:5432/autoharness
AUTOHARNESS_SERVICE_MODE=simulated
AUTOHARNESS_MAX_LOCAL_CONCURRENCY=4
AUTOHARNESS_POLL_INTERVAL_SEC=2
AUTOHARNESS_HOST=127.0.0.1
AUTOHARNESS_PORT=8000
```

- [ ] **Step 5: Ensure local Codex instructions are ignored**

Confirm `.gitignore` contains:

```text
AGENTS.md
```

If it is missing, add it under the local coding agent config section.

- [ ] **Step 6: Create package skeleton and config implementation**

Create `autoharness_service/__init__.py`:

```python
"""HTTP service for the take-home Agent Optimization Service MVP."""
```

Create `autoharness_service/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATABASE_URL = "postgresql://autoharness:autoharness@localhost:5432/autoharness"


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str
    default_model: str
    default_sandbox_provider: str
    default_mode: str
    max_local_concurrency: int
    poll_interval_sec: float


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def load_settings() -> ServiceSettings:
    return ServiceSettings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        default_model=os.getenv("AGENT_MODEL", "gpt-5.4"),
        default_sandbox_provider=os.getenv("AUTOHARNESS_SANDBOX_PROVIDER", "daytona"),
        default_mode=os.getenv("AUTOHARNESS_SERVICE_MODE", "simulated"),
        max_local_concurrency=_int_from_env("AUTOHARNESS_MAX_LOCAL_CONCURRENCY", 4),
        poll_interval_sec=_float_from_env("AUTOHARNESS_POLL_INTERVAL_SEC", 2.0),
    )
```

- [ ] **Step 7: Run the import test**

Run:

```bash
python -m pytest tests/service/test_imports.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add pyproject.toml .env.example .gitignore autoharness_service/__init__.py autoharness_service/config.py tests/service/test_imports.py
git diff --cached --name-only | rg '(^|/)AGENTS\.md$|(^|/)\.env$' && exit 1 || true
git commit -m "feat: add service package skeleton"
```

---

## Task 2: Domain Models, API Schemas, and Result Normalizer

**Files:**
- Create: `autoharness_service/models.py`
- Create: `autoharness_service/schemas.py`
- Create: `autoharness_service/normalizer.py`
- Create: `tests/service/test_normalizer.py`

**Interfaces:**
- Produces `TaskResultRecord`, `FailureSummary`, `RunStatus`, `TaskStatus`.
- Produces `normalize_reward_result(task_id, reward, trace_path=None, result_path=None, metadata=None) -> TaskResultRecord`.
- Produces `normalize_missing_result(task_id, reason, metadata=None) -> TaskResultRecord`.
- Produces `build_failure_summary(task_results) -> FailureSummary`.
- Produces Pydantic schemas consumed by API and test client.

- [ ] **Step 1: Write normalizer tests first**

Create `tests/service/test_normalizer.py`:

```python
from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_missing_result,
    normalize_reward_result,
)


def test_normalize_reward_passed():
    result = normalize_reward_result("task-pass", 1.0)

    assert result.task_id == "task-pass"
    assert result.status == "passed"
    assert result.reward == 1.0
    assert result.failure_type is None


def test_normalize_reward_failed():
    result = normalize_reward_result("task-fail", 0.0)

    assert result.status == "failed"
    assert result.failure_type == "agent_failed"
    assert result.error_summary == "Verifier reward below pass threshold"


def test_normalize_missing_result_as_infra_failure():
    result = normalize_missing_result("task-missing", "Trial result.json missing")

    assert result.status == "infra_failed"
    assert result.reward is None
    assert result.failure_type == "missing_result"
    assert result.error_summary == "Trial result.json missing"


def test_build_failure_summary_counts_failure_types():
    results = [
        normalize_reward_result("task-pass", 1.0),
        normalize_reward_result("task-fail", 0.0),
        normalize_missing_result("task-missing", "Trial result.json missing"),
    ]

    summary = build_failure_summary(results)

    assert summary.agent_failures == 1
    assert summary.infra_failures == 1
    assert summary.tasks_passed == 1
    assert summary.tasks_total == 3
    assert summary.top_failure_modes == ["agent_failed", "missing_result"]
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/service/test_normalizer.py -q
```

Expected:

```text
ModuleNotFoundError or ImportError for normalizer symbols
```

- [ ] **Step 3: Implement domain models**

Create `autoharness_service/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


RunStatus = str
TaskStatus = str

RUN_STATUSES = {"queued", "running", "succeeded", "failed", "timed_out", "cancelled"}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}
TASK_STATUSES = {"queued", "running", "passed", "failed", "infra_failed", "timed_out"}


@dataclass(frozen=True)
class TaskResultRecord:
    task_id: str
    status: TaskStatus
    reward: float | None
    failure_type: str | None = None
    error_summary: str | None = None
    trace_path: str | None = None
    result_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureSummary:
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    agent_failures: int
    infra_failures: int
    top_failure_modes: list[str]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: RunStatus
    task_ids: list[str]
    mode: str
    model: str
    sandbox_provider: str
    requested_concurrency: int
    max_iterations: int
    org_id: str
    created_by: str
    score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class IterationRecord:
    run_id: str
    iteration_index: int
    status: str
    agent_version: str
    score: float | None = None
    proposal: str | None = None
    accepted: bool | None = None
```

- [ ] **Step 4: Implement Pydantic schemas**

Create `autoharness_service/schemas.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    task_ids: list[str] = Field(min_length=1, max_length=20)
    max_iterations: int = Field(default=0, ge=0, le=1)
    sandbox_provider: Literal["daytona", "e2b", "docker", "simulated"] = "daytona"
    model: str = "gpt-5.4"
    mode: Literal["simulated", "real"] = "simulated"
    requested_concurrency: int = Field(default=1, ge=1, le=8)


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime | None
    status_url: str
    result_url: str


class RunProgress(BaseModel):
    total: int
    queued: int
    running: int
    completed: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: RunProgress
    score: float | None
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    reward: float | None
    failure_type: str | None
    error_summary: str | None
    trace_path: str | None
    result_path: str | None
    metadata: dict[str, Any]


class FailureSummaryResponse(BaseModel):
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    agent_failures: int
    infra_failures: int
    top_failure_modes: list[str]


class RunResultsResponse(BaseModel):
    run_id: str
    status: str
    score: float | None
    tasks_total: int
    tasks_passed: int
    tasks_failed: int
    tasks_infra_failed: int
    task_results: list[TaskResultResponse]
    failure_summary: FailureSummaryResponse


class IterationResponse(BaseModel):
    iteration: int
    agent_version: str
    status: str
    score: float | None
    proposal: str | None
    accepted: bool | None


class IterationsResponse(BaseModel):
    run_id: str
    iterations: list[IterationResponse]


class TaskListResponse(BaseModel):
    tasks: list[str]
```

- [ ] **Step 5: Implement normalizer**

Create `autoharness_service/normalizer.py`:

```python
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from autoharness_service.models import FailureSummary, TaskResultRecord


PASS_THRESHOLD = 0.5


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def normalize_reward_result(
    task_id: str,
    reward: float | None,
    *,
    trace_path: str | None = None,
    result_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskResultRecord:
    if reward is None:
        return normalize_missing_result(
            task_id,
            "Verifier result missing",
            trace_path=trace_path,
            result_path=result_path,
            metadata=metadata,
        )

    if reward >= PASS_THRESHOLD:
        return TaskResultRecord(
            task_id=task_id,
            status="passed",
            reward=float(reward),
            trace_path=trace_path,
            result_path=result_path,
            metadata=_metadata_dict(metadata),
        )

    return TaskResultRecord(
        task_id=task_id,
        status="failed",
        reward=float(reward),
        failure_type="agent_failed",
        error_summary="Verifier reward below pass threshold",
        trace_path=trace_path,
        result_path=result_path,
        metadata=_metadata_dict(metadata),
    )


def normalize_missing_result(
    task_id: str,
    reason: str,
    *,
    trace_path: str | None = None,
    result_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskResultRecord:
    return TaskResultRecord(
        task_id=task_id,
        status="infra_failed",
        reward=None,
        failure_type="missing_result",
        error_summary=reason,
        trace_path=trace_path,
        result_path=result_path,
        metadata=_metadata_dict(metadata),
    )


def build_failure_summary(task_results: Iterable[TaskResultRecord]) -> FailureSummary:
    results = list(task_results)
    failure_modes = Counter(
        result.failure_type
        for result in results
        if result.failure_type is not None
    )
    tasks_passed = sum(1 for result in results if result.status == "passed")
    tasks_failed = sum(1 for result in results if result.status == "failed")
    tasks_infra_failed = sum(
        1 for result in results if result.status in {"infra_failed", "timed_out"}
    )
    agent_failures = sum(
        1 for result in results if result.failure_type == "agent_failed"
    )
    infra_failures = sum(
        1
        for result in results
        if result.failure_type in {"missing_result", "sandbox_timeout", "runner_failed"}
        or result.status in {"infra_failed", "timed_out"}
    )
    return FailureSummary(
        tasks_total=len(results),
        tasks_passed=tasks_passed,
        tasks_failed=tasks_failed,
        tasks_infra_failed=tasks_infra_failed,
        agent_failures=agent_failures,
        infra_failures=infra_failures,
        top_failure_modes=[
            mode for mode, _count in failure_modes.most_common(5)
        ],
    )
```

- [ ] **Step 6: Run normalizer tests**

Run:

```bash
python -m pytest tests/service/test_normalizer.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add autoharness_service/models.py autoharness_service/schemas.py autoharness_service/normalizer.py tests/service/test_normalizer.py
git commit -m "feat: add service schemas and result normalizer"
```

---

## Task 3: PostgreSQL Store and Schema

**Files:**
- Create: `autoharness_service/store.py`
- Create: `tests/service/test_store.py`

**Interfaces:**
- Produces `PostgresStore(database_url: str)`.
- Produces `init_schema() -> None`.
- Produces `create_run(request, org_id, created_by) -> RunRecord`.
- Produces `get_run(run_id, org_id) -> RunRecord | None`.
- Produces `mark_run_running(run_id)`, `mark_run_succeeded(run_id, score)`, `mark_run_failed(run_id, status, error)`.
- Produces `replace_task_results(run_id, task_results) -> None`.
- Produces `list_task_results(run_id) -> list[TaskResultRecord]`.
- Produces `create_iteration(run_id, iteration_index, status, agent_version, score=None, proposal=None, accepted=None) -> IterationRecord`.
- Produces `list_iterations(run_id) -> list[IterationRecord]`.

- [ ] **Step 1: Write row mapping and optional live DB tests**

Create `tests/service/test_store.py`:

```python
import os
import uuid

import pytest

from autoharness_service.normalizer import normalize_reward_result
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.store import PostgresStore


pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set for live Postgres store tests",
)


def test_store_creates_run_and_task_results():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()
    request = RunCreateRequest(
        task_ids=[f"task-{uuid.uuid4()}"],
        mode="simulated",
        requested_concurrency=1,
    )

    run = store.create_run(request, org_id="org-test", created_by="user-test")
    store.mark_run_running(run.run_id)
    store.replace_task_results(
        run.run_id,
        [normalize_reward_result(request.task_ids[0], 1.0)],
    )
    store.mark_run_succeeded(run.run_id, 1.0)

    loaded = store.get_run(run.run_id, org_id="org-test")
    results = store.list_task_results(run.run_id)

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.score == 1.0
    assert results[0].task_id == request.task_ids[0]
    assert results[0].status == "passed"


def test_store_filters_runs_by_org():
    store = PostgresStore(os.environ["DATABASE_URL"])
    store.init_schema()
    request = RunCreateRequest(task_ids=[f"task-{uuid.uuid4()}"], mode="simulated")

    run = store.create_run(request, org_id="org-a", created_by="user-a")

    assert store.get_run(run.run_id, org_id="org-a") is not None
    assert store.get_run(run.run_id, org_id="org-b") is None
```

- [ ] **Step 2: Run tests and verify skip or failure**

Run:

```bash
python -m pytest tests/service/test_store.py -q
```

Expected if `DATABASE_URL` is unset:

```text
2 skipped
```

Expected if `DATABASE_URL` is set before implementation:

```text
ImportError for PostgresStore
```

- [ ] **Step 3: Implement store**

Create `autoharness_service/store.py` with synchronous psycopg access:

```python
from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from autoharness_service.models import IterationRecord, RunRecord, TaskResultRecord
from autoharness_service.schemas import RunCreateRequest


class PostgresStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def _connect(self):
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  id uuid PRIMARY KEY,
                  org_id text NOT NULL,
                  created_by text NOT NULL,
                  status text NOT NULL,
                  mode text NOT NULL,
                  model text NOT NULL,
                  sandbox_provider text NOT NULL,
                  requested_concurrency integer NOT NULL,
                  max_iterations integer NOT NULL,
                  task_ids jsonb NOT NULL,
                  score double precision,
                  error text,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  started_at timestamptz,
                  completed_at timestamptz
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_results (
                  run_id uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  task_id text NOT NULL,
                  status text NOT NULL,
                  reward double precision,
                  failure_type text,
                  error_summary text,
                  trace_path text,
                  result_path text,
                  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY (run_id, task_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS iterations (
                  run_id uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  iteration_index integer NOT NULL,
                  status text NOT NULL,
                  agent_version text NOT NULL,
                  score double precision,
                  proposal text,
                  accepted boolean,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY (run_id, iteration_index)
                )
                """
            )

    def create_run(
        self,
        request: RunCreateRequest,
        *,
        org_id: str,
        created_by: str,
    ) -> RunRecord:
        run_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO runs (
                  id, org_id, created_by, status, mode, model, sandbox_provider,
                  requested_concurrency, max_iterations, task_ids
                )
                VALUES (%s, %s, %s, 'queued', %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    run_id,
                    org_id,
                    created_by,
                    request.mode,
                    request.model,
                    request.sandbox_provider,
                    request.requested_concurrency,
                    request.max_iterations,
                    Jsonb(request.task_ids),
                ),
            ).fetchone()
        return _run_from_row(row)

    def get_run(self, run_id: str, *, org_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = %s AND org_id = %s",
                (run_id, org_id),
            ).fetchone()
        return _run_from_row(row) if row else None

    def mark_run_running(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'running', started_at = COALESCE(started_at, now())
                WHERE id = %s AND status IN ('queued', 'running')
                """,
                (run_id,),
            )

    def mark_run_succeeded(self, run_id: str, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'succeeded', score = %s, completed_at = now()
                WHERE id = %s AND status NOT IN ('succeeded', 'failed', 'timed_out', 'cancelled')
                """,
                (score, run_id),
            )

    def mark_run_failed(self, run_id: str, *, status: str, error: str) -> None:
        if status not in {"failed", "timed_out", "cancelled"}:
            raise ValueError("terminal failure status expected")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = %s, error = %s, completed_at = now()
                WHERE id = %s AND status NOT IN ('succeeded', 'failed', 'timed_out', 'cancelled')
                """,
                (status, error, run_id),
            )

    def replace_task_results(
        self,
        run_id: str,
        task_results: Iterable[TaskResultRecord],
    ) -> None:
        with self._connect() as conn:
            for result in task_results:
                conn.execute(
                    """
                    INSERT INTO task_results (
                      run_id, task_id, status, reward, failure_type, error_summary,
                      trace_path, result_path, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, task_id) DO UPDATE SET
                      status = EXCLUDED.status,
                      reward = EXCLUDED.reward,
                      failure_type = EXCLUDED.failure_type,
                      error_summary = EXCLUDED.error_summary,
                      trace_path = EXCLUDED.trace_path,
                      result_path = EXCLUDED.result_path,
                      metadata = EXCLUDED.metadata
                    """,
                    (
                        run_id,
                        result.task_id,
                        result.status,
                        result.reward,
                        result.failure_type,
                        result.error_summary,
                        result.trace_path,
                        result.result_path,
                        Jsonb(result.metadata),
                    ),
                )

    def list_task_results(self, run_id: str) -> list[TaskResultRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_results WHERE run_id = %s ORDER BY task_id",
                (run_id,),
            ).fetchall()
        return [_task_result_from_row(row) for row in rows]

    def create_iteration(
        self,
        run_id: str,
        *,
        iteration_index: int,
        status: str,
        agent_version: str,
        score: float | None = None,
        proposal: str | None = None,
        accepted: bool | None = None,
    ) -> IterationRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO iterations (
                  run_id, iteration_index, status, agent_version, score, proposal, accepted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, iteration_index) DO UPDATE SET
                  status = EXCLUDED.status,
                  agent_version = EXCLUDED.agent_version,
                  score = EXCLUDED.score,
                  proposal = EXCLUDED.proposal,
                  accepted = EXCLUDED.accepted
                RETURNING *
                """,
                (
                    run_id,
                    iteration_index,
                    status,
                    agent_version,
                    score,
                    proposal,
                    accepted,
                ),
            ).fetchone()
        return _iteration_from_row(row)

    def list_iterations(self, run_id: str) -> list[IterationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM iterations WHERE run_id = %s ORDER BY iteration_index",
                (run_id,),
            ).fetchall()
        return [_iteration_from_row(row) for row in rows]


def _task_ids_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def _run_from_row(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(row["id"]),
        status=str(row["status"]),
        task_ids=_task_ids_from_json(row["task_ids"]),
        mode=str(row["mode"]),
        model=str(row["model"]),
        sandbox_provider=str(row["sandbox_provider"]),
        requested_concurrency=int(row["requested_concurrency"]),
        max_iterations=int(row["max_iterations"]),
        org_id=str(row["org_id"]),
        created_by=str(row["created_by"]),
        score=row["score"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _task_result_from_row(row: dict[str, Any]) -> TaskResultRecord:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return TaskResultRecord(
        task_id=str(row["task_id"]),
        status=str(row["status"]),
        reward=row["reward"],
        failure_type=row["failure_type"],
        error_summary=row["error_summary"],
        trace_path=row["trace_path"],
        result_path=row["result_path"],
        metadata=dict(metadata or {}),
    )


def _iteration_from_row(row: dict[str, Any]) -> IterationRecord:
    return IterationRecord(
        run_id=str(row["run_id"]),
        iteration_index=int(row["iteration_index"]),
        status=str(row["status"]),
        agent_version=str(row["agent_version"]),
        score=row["score"],
        proposal=row["proposal"],
        accepted=row["accepted"],
    )
```

- [ ] **Step 4: Run store tests**

Run without DB:

```bash
python -m pytest tests/service/test_store.py -q
```

Expected:

```text
2 skipped
```

Run with local Postgres if available:

```bash
DATABASE_URL=postgresql://autoharness:autoharness@localhost:5432/autoharness python -m pytest tests/service/test_store.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add autoharness_service/store.py tests/service/test_store.py
git commit -m "feat: add postgres run store"
```

---

## Task 4: Runners and RunService Orchestration

**Files:**
- Create: `autoharness_service/runner.py`
- Create: `autoharness_service/service.py`
- Create: `tests/service/test_runner.py`
- Create: `tests/service/test_service.py`

**Interfaces:**
- Produces `SimulatedBenchmarkRunner.run(task_ids) -> dict[str, float | None]`.
- Produces `TerminalBenchRunnerAdapter.run(task_ids, model, sandbox_provider, requested_concurrency) -> dict[str, float | None]`.
- Produces `RunService.submit_run(request, org_id, created_by) -> RunRecord`.
- Produces `RunService.execute_run(run_id, org_id) -> None`.
- Produces `RunService.get_run_status(run_id, org_id) -> RunStatusResponse | None`.
- Produces `RunService.get_run_results(run_id, org_id) -> RunResultsResponse | None`.

- [ ] **Step 1: Write runner tests**

Create `tests/service/test_runner.py`:

```python
from autoharness_service.runner import SimulatedBenchmarkRunner


def test_simulated_runner_returns_deterministic_rewards():
    runner = SimulatedBenchmarkRunner()

    results = runner.run(["task-pass", "task-fail", "task-infra"])

    assert results["task-pass"] == 1.0
    assert results["task-fail"] == 0.0
    assert results["task-infra"] is None
```

- [ ] **Step 2: Write RunService tests with fake store**

Create `tests/service/test_service.py`:

```python
from datetime import datetime, timezone

from autoharness_service.models import RunRecord
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.schemas import RunCreateRequest
from autoharness_service.service import RunService


class FakeStore:
    def __init__(self):
        self.runs = {}
        self.results = {}
        self.iterations = {}

    def init_schema(self):
        pass

    def create_run(self, request, *, org_id, created_by):
        run = RunRecord(
            run_id="run-1",
            status="queued",
            task_ids=request.task_ids,
            mode=request.mode,
            model=request.model,
            sandbox_provider=request.sandbox_provider,
            requested_concurrency=request.requested_concurrency,
            max_iterations=request.max_iterations,
            org_id=org_id,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
        )
        self.runs[run.run_id] = run
        return run

    def get_run(self, run_id, *, org_id):
        run = self.runs.get(run_id)
        if run is None or run.org_id != org_id:
            return None
        return run

    def mark_run_running(self, run_id):
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(**{**run.__dict__, "status": "running"})

    def mark_run_succeeded(self, run_id, score):
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(**{**run.__dict__, "status": "succeeded", "score": score})

    def mark_run_failed(self, run_id, *, status, error):
        run = self.runs[run_id]
        self.runs[run_id] = RunRecord(**{**run.__dict__, "status": status, "error": error})

    def replace_task_results(self, run_id, task_results):
        self.results[run_id] = list(task_results)

    def list_task_results(self, run_id):
        return self.results.get(run_id, [])

    def create_iteration(self, run_id, **kwargs):
        self.iterations.setdefault(run_id, []).append(kwargs)
        return kwargs

    def list_iterations(self, run_id):
        return []


def test_run_service_executes_simulated_run():
    store = FakeStore()
    service = RunService(store=store, simulated_runner=SimulatedBenchmarkRunner())
    request = RunCreateRequest(
        task_ids=["task-pass", "task-fail"],
        mode="simulated",
        requested_concurrency=1,
    )

    run = service.submit_run(request, org_id="org-1", created_by="user-1", start_background=False)
    service.execute_run(run.run_id, org_id="org-1")

    status = service.get_run_status(run.run_id, org_id="org-1")
    results = service.get_run_results(run.run_id, org_id="org-1")

    assert status is not None
    assert status.status == "succeeded"
    assert status.score == 0.5
    assert results is not None
    assert results.tasks_passed == 1
    assert results.tasks_failed == 1
```

- [ ] **Step 3: Run and verify failures**

Run:

```bash
python -m pytest tests/service/test_runner.py tests/service/test_service.py -q
```

Expected:

```text
ImportError for runner/service symbols
```

- [ ] **Step 4: Implement runners**

Create `autoharness_service/runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


class SimulatedBenchmarkRunner:
    def run(self, task_ids: list[str]) -> dict[str, float | None]:
        results: dict[str, float | None] = {}
        for task_id in task_ids:
            lowered = task_id.lower()
            if "infra" in lowered or "timeout" in lowered:
                results[task_id] = None
            elif "fail" in lowered:
                results[task_id] = 0.0
            else:
                results[task_id] = 1.0
        return results


@dataclass(frozen=True)
class TerminalBenchRunnerAdapter:
    split: str = "train"

    def run(
        self,
        task_ids: list[str],
        *,
        model: str,
        sandbox_provider: str,
        requested_concurrency: int,
    ) -> dict[str, float | None]:
        from benchmark import TerminalBenchRunner

        runner = TerminalBenchRunner(
            agent_model=model,
            split=self.split,
            env_provider=sandbox_provider,
            n_concurrent=max(1, min(requested_concurrency, len(task_ids))),
            jobs_dir="workspace/tbench_jobs",
        )
        return runner.run(task_ids=task_ids)
```

- [ ] **Step 5: Implement RunService**

Create `autoharness_service/service.py`:

```python
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from autoharness_service.models import TaskResultRecord
from autoharness_service.normalizer import (
    build_failure_summary,
    normalize_reward_result,
)
from autoharness_service.runner import SimulatedBenchmarkRunner, TerminalBenchRunnerAdapter
from autoharness_service.schemas import (
    FailureSummaryResponse,
    RunCreateRequest,
    RunResultsResponse,
    RunStatusResponse,
    RunProgress,
    TaskResultResponse,
)


class RunService:
    def __init__(
        self,
        *,
        store: Any,
        simulated_runner: SimulatedBenchmarkRunner | None = None,
        terminal_runner: TerminalBenchRunnerAdapter | None = None,
        max_local_concurrency: int = 4,
    ):
        self.store = store
        self.simulated_runner = simulated_runner or SimulatedBenchmarkRunner()
        self.terminal_runner = terminal_runner or TerminalBenchRunnerAdapter()
        self.max_local_concurrency = max_local_concurrency

    def submit_run(
        self,
        request: RunCreateRequest,
        *,
        org_id: str,
        created_by: str,
        start_background: bool = True,
    ):
        self.store.init_schema()
        run = self.store.create_run(request, org_id=org_id, created_by=created_by)
        self.store.create_iteration(
            run.run_id,
            iteration_index=0,
            status="queued",
            agent_version="initial",
        )
        if start_background:
            thread = threading.Thread(
                target=self.execute_run,
                kwargs={"run_id": run.run_id, "org_id": org_id},
                daemon=True,
            )
            thread.start()
        return run

    def execute_run(self, run_id: str, *, org_id: str) -> None:
        run = self.store.get_run(run_id, org_id=org_id)
        if run is None:
            return
        try:
            self.store.mark_run_running(run_id)
            raw_results = self._run_benchmark(run)
            task_results = self._normalize_results(run.task_ids, raw_results)
            self.store.replace_task_results(run_id, task_results)
            score = _score(task_results)
            self.store.create_iteration(
                run_id,
                iteration_index=0,
                status="completed",
                agent_version="initial",
                score=score,
            )
            self.store.mark_run_succeeded(run_id, score)
        except TimeoutError as exc:
            self.store.mark_run_failed(run_id, status="timed_out", error=str(exc))
        except Exception as exc:
            self.store.mark_run_failed(run_id, status="failed", error=str(exc))

    def _run_benchmark(self, run) -> dict[str, float | None]:
        if run.mode == "simulated":
            return self.simulated_runner.run(run.task_ids)
        return self.terminal_runner.run(
            run.task_ids,
            model=run.model,
            sandbox_provider=run.sandbox_provider,
            requested_concurrency=min(run.requested_concurrency, self.max_local_concurrency),
        )

    def _normalize_results(
        self,
        task_ids: list[str],
        raw_results: dict[str, float | None],
    ) -> list[TaskResultRecord]:
        normalized: list[TaskResultRecord] = []
        for task_id in task_ids:
            trace_path = Path("workspace") / "traces" / "latest" / task_id / "trace.json"
            result_path = Path("workspace") / "traces" / "latest" / task_id / "result.json"
            normalized.append(
                normalize_reward_result(
                    task_id,
                    raw_results.get(task_id),
                    trace_path=str(trace_path) if trace_path.exists() else None,
                    result_path=str(result_path) if result_path.exists() else None,
                    metadata={"source": "harbor" if raw_results else "missing"},
                )
            )
        return normalized

    def get_run_status(self, run_id: str, *, org_id: str) -> RunStatusResponse | None:
        run = self.store.get_run(run_id, org_id=org_id)
        if run is None:
            return None
        task_results = self.store.list_task_results(run_id)
        completed = len(task_results)
        total = len(run.task_ids)
        running = 1 if run.status == "running" and completed < total else 0
        queued = max(total - completed - running, 0)
        return RunStatusResponse(
            run_id=run.run_id,
            status=run.status,
            progress=RunProgress(
                total=total,
                queued=queued,
                running=running,
                completed=completed,
            ),
            score=run.score,
            error=run.error,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )

    def get_run_results(self, run_id: str, *, org_id: str) -> RunResultsResponse | None:
        run = self.store.get_run(run_id, org_id=org_id)
        if run is None:
            return None
        task_results = self.store.list_task_results(run_id)
        summary = build_failure_summary(task_results)
        return RunResultsResponse(
            run_id=run.run_id,
            status=run.status,
            score=run.score,
            tasks_total=summary.tasks_total,
            tasks_passed=summary.tasks_passed,
            tasks_failed=summary.tasks_failed,
            tasks_infra_failed=summary.tasks_infra_failed,
            task_results=[_task_response(result) for result in task_results],
            failure_summary=FailureSummaryResponse(**summary.__dict__),
        )


def _score(task_results: list[TaskResultRecord]) -> float:
    if not task_results:
        return 0.0
    return sum(result.reward or 0.0 for result in task_results) / len(task_results)


def _task_response(result: TaskResultRecord) -> TaskResultResponse:
    return TaskResultResponse(
        task_id=result.task_id,
        status=result.status,
        reward=result.reward,
        failure_type=result.failure_type,
        error_summary=result.error_summary,
        trace_path=result.trace_path,
        result_path=result.result_path,
        metadata=result.metadata,
    )
```

- [ ] **Step 6: Run runner/service tests**

Run:

```bash
python -m pytest tests/service/test_runner.py tests/service/test_service.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add autoharness_service/runner.py autoharness_service/service.py tests/service/test_runner.py tests/service/test_service.py
git commit -m "feat: add benchmark run service"
```

---

## Task 5: FastAPI Routes and App Factory

**Files:**
- Create: `autoharness_service/api.py`
- Create: `autoharness_service/main.py`
- Create: `tests/service/test_api.py`

**Interfaces:**
- Produces `create_app(service: RunService | None = None, start_background: bool = True) -> FastAPI`.
- Produces `POST /runs`, `GET /runs/{run_id}`, `GET /runs/{run_id}/results`, `GET /runs/{run_id}/iterations`, `GET /tasks`.
- Produces header mock tenancy via `X-Org-Id` and `X-User-Id`.

- [ ] **Step 1: Write API tests**

Create `tests/service/test_api.py`:

```python
from fastapi.testclient import TestClient

from autoharness_service.api import create_app
from autoharness_service.runner import SimulatedBenchmarkRunner
from autoharness_service.service import RunService
from tests.service.test_service import FakeStore


def test_api_submit_poll_and_read_results():
    service = RunService(store=FakeStore(), simulated_runner=SimulatedBenchmarkRunner())
    app = create_app(service=service, start_background=False)
    client = TestClient(app)

    create_response = client.post(
        "/runs",
        json={
            "task_ids": ["task-pass", "task-fail"],
            "mode": "simulated",
            "requested_concurrency": 1,
        },
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1"},
    )

    assert create_response.status_code == 202
    run_id = create_response.json()["run_id"]

    service.execute_run(run_id, org_id="org-1")

    status_response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1"},
    )
    results_response = client.get(
        f"/runs/{run_id}/results",
        headers={"X-Org-Id": "org-1", "X-User-Id": "user-1"},
    )

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"
    assert results_response.status_code == 200
    assert results_response.json()["tasks_passed"] == 1
    assert results_response.json()["tasks_failed"] == 1


def test_api_enforces_org_boundary():
    service = RunService(store=FakeStore(), simulated_runner=SimulatedBenchmarkRunner())
    app = create_app(service=service, start_background=False)
    client = TestClient(app)

    create_response = client.post(
        "/runs",
        json={"task_ids": ["task-pass"], "mode": "simulated"},
        headers={"X-Org-Id": "org-a", "X-User-Id": "user-a"},
    )
    run_id = create_response.json()["run_id"]

    response = client.get(
        f"/runs/{run_id}",
        headers={"X-Org-Id": "org-b", "X-User-Id": "user-b"},
    )

    assert response.status_code == 404
```

- [ ] **Step 2: Run API tests and verify failure**

Run:

```bash
python -m pytest tests/service/test_api.py -q
```

Expected:

```text
ImportError for autoharness_service.api
```

- [ ] **Step 3: Implement API app**

Create `autoharness_service/api.py`:

```python
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Response, status

from autoharness_service.config import load_settings
from autoharness_service.schemas import (
    IterationResponse,
    IterationsResponse,
    RunCreateRequest,
    RunCreateResponse,
    TaskListResponse,
)
from autoharness_service.service import RunService
from autoharness_service.store import PostgresStore


DEFAULT_TASKS = [
    "break-filter-js-from-html",
    "task-pass",
    "task-fail",
    "task-infra",
]


def create_app(
    service: RunService | None = None,
    *,
    start_background: bool = True,
) -> FastAPI:
    app = FastAPI(title="Agent Optimization Service", version="0.1.0")
    app.state.service = service or _build_service()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks", response_model=TaskListResponse)
    def list_tasks() -> TaskListResponse:
        return TaskListResponse(tasks=DEFAULT_TASKS)

    @app.post(
        "/runs",
        response_model=RunCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        request: RunCreateRequest,
        response: Response,
        x_org_id: str = Header(default="default-org"),
        x_user_id: str = Header(default="local-user"),
    ) -> RunCreateResponse:
        run = app.state.service.submit_run(
            request,
            org_id=x_org_id,
            created_by=x_user_id,
            start_background=start_background,
        )
        response.headers["Location"] = f"/runs/{run.run_id}"
        return RunCreateResponse(
            run_id=run.run_id,
            status=run.status,
            created_at=run.created_at,
            status_url=f"/runs/{run.run_id}",
            result_url=f"/runs/{run.run_id}/results",
        )

    @app.get("/runs/{run_id}")
    def get_run(
        run_id: str,
        x_org_id: str = Header(default="default-org"),
    ):
        run_status = app.state.service.get_run_status(run_id, org_id=x_org_id)
        if run_status is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run_status

    @app.get("/runs/{run_id}/results")
    def get_results(
        run_id: str,
        x_org_id: str = Header(default="default-org"),
    ):
        run_results = app.state.service.get_run_results(run_id, org_id=x_org_id)
        if run_results is None:
            raise HTTPException(status_code=404, detail="run not found")
        if run_results.status not in {"succeeded", "failed", "timed_out", "cancelled"}:
            raise HTTPException(status_code=409, detail="run is not finished")
        return run_results

    @app.get("/runs/{run_id}/iterations", response_model=IterationsResponse)
    def get_iterations(
        run_id: str,
        x_org_id: str = Header(default="default-org"),
    ) -> IterationsResponse:
        run_status = app.state.service.get_run_status(run_id, org_id=x_org_id)
        if run_status is None:
            raise HTTPException(status_code=404, detail="run not found")
        iterations = app.state.service.store.list_iterations(run_id)
        return IterationsResponse(
            run_id=run_id,
            iterations=[
                IterationResponse(
                    iteration=item.iteration_index,
                    agent_version=item.agent_version,
                    status=item.status,
                    score=item.score,
                    proposal=item.proposal,
                    accepted=item.accepted,
                )
                for item in iterations
            ],
        )

    return app


def _build_service() -> RunService:
    settings = load_settings()
    store = PostgresStore(settings.database_url)
    return RunService(store=store, max_local_concurrency=settings.max_local_concurrency)
```

Create `autoharness_service/main.py`:

```python
from autoharness_service.api import create_app

app = create_app()
```

- [ ] **Step 4: Run API tests**

Run:

```bash
python -m pytest tests/service/test_api.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add autoharness_service/api.py autoharness_service/main.py tests/service/test_api.py
git commit -m "feat: expose optimization service api"
```

---

## Task 6: Minimal LLM Optimizer Proposal

**Files:**
- Create: `autoharness_service/optimizer.py`
- Modify: `autoharness_service/service.py`
- Create: `tests/service/test_optimizer.py`
- Modify: `tests/service/test_service.py`

**Interfaces:**
- Produces `build_optimizer_prompt(task_results, failure_summary) -> str`.
- Produces `Optimizer.propose(task_results, failure_summary, model) -> str`.
- `RunService.execute_run()` stores iteration 1 proposal when `run.max_iterations > 0`.
- The optimizer does not apply patches and does not rerun candidates in this MVP.

- [ ] **Step 1: Write optimizer tests**

Create `tests/service/test_optimizer.py`:

```python
from autoharness_service.normalizer import build_failure_summary, normalize_reward_result
from autoharness_service.optimizer import build_optimizer_prompt


def test_build_optimizer_prompt_contains_failed_tasks():
    results = [
        normalize_reward_result("task-pass", 1.0),
        normalize_reward_result("task-fail", 0.0),
    ]
    summary = build_failure_summary(results)

    prompt = build_optimizer_prompt(results, summary)

    assert "task-fail" in prompt
    assert "agent_failed" in prompt
    assert "one focused improvement" in prompt
```

- [ ] **Step 2: Run optimizer tests and verify failure**

Run:

```bash
python -m pytest tests/service/test_optimizer.py -q
```

Expected:

```text
ImportError for autoharness_service.optimizer
```

- [ ] **Step 3: Implement optimizer**

Create `autoharness_service/optimizer.py`:

```python
from __future__ import annotations

import os

from openai import OpenAI

from autoharness_service.models import FailureSummary, TaskResultRecord


def build_optimizer_prompt(
    task_results: list[TaskResultRecord],
    failure_summary: FailureSummary,
) -> str:
    failed_lines = []
    for result in task_results:
        if result.status == "passed":
            continue
        failed_lines.append(
            f"- {result.task_id}: status={result.status}, "
            f"reward={result.reward}, failure_type={result.failure_type}, "
            f"error={result.error_summary}"
        )
    failures = "\n".join(failed_lines) if failed_lines else "- none"
    return (
        "You are improving a Terminal-Bench bash agent. "
        "Propose one focused improvement to the agent prompt or behavior. "
        "Do not propose multiple candidates or broad rewrites.\n\n"
        f"Summary: passed={failure_summary.tasks_passed}, "
        f"failed={failure_summary.tasks_failed}, "
        f"infra_failed={failure_summary.tasks_infra_failed}, "
        f"failure_modes={failure_summary.top_failure_modes}\n\n"
        f"Failed tasks:\n{failures}\n\n"
        "Return four short sections: hypothesis, proposed_change, expected_effect, risks."
    )


class Optimizer:
    def propose(
        self,
        task_results: list[TaskResultRecord],
        failure_summary: FailureSummary,
        *,
        model: str,
    ) -> str:
        prompt = build_optimizer_prompt(task_results, failure_summary)
        if not os.getenv("OPENAI_API_KEY"):
            return (
                "LLM proposal skipped because OPENAI_API_KEY is not set.\n\n"
                f"Prompt that would be sent:\n{prompt}"
            )
        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You propose concise improvements for coding agents.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text
```

- [ ] **Step 4: Update RunService to persist proposal iteration**

Modify `autoharness_service/service.py`:

```python
from autoharness_service.optimizer import Optimizer
```

Update `RunService.__init__` signature:

```python
def __init__(
    self,
    *,
    store: Any,
    simulated_runner: SimulatedBenchmarkRunner | None = None,
    terminal_runner: TerminalBenchRunnerAdapter | None = None,
    optimizer: Optimizer | None = None,
    max_local_concurrency: int = 4,
):
    self.store = store
    self.simulated_runner = simulated_runner or SimulatedBenchmarkRunner()
    self.terminal_runner = terminal_runner or TerminalBenchRunnerAdapter()
    self.optimizer = optimizer or Optimizer()
    self.max_local_concurrency = max_local_concurrency
```

After `self.store.create_iteration(... iteration_index=0 ...)` in `execute_run`, add:

```python
if run.max_iterations > 0:
    summary = build_failure_summary(task_results)
    proposal = self.optimizer.propose(task_results, summary, model=run.model)
    self.store.create_iteration(
        run_id,
        iteration_index=1,
        status="proposal_created",
        agent_version="proposal-1",
        score=score,
        proposal=proposal,
        accepted=None,
    )
```

- [ ] **Step 5: Extend RunService test for proposal persistence**

Add to `tests/service/test_service.py`:

```python
class FakeOptimizer:
    def propose(self, task_results, failure_summary, *, model):
        return "hypothesis: improve bash verification\nproposed_change: inspect output before final answer"


def test_run_service_records_optimizer_proposal_when_requested():
    store = FakeStore()
    service = RunService(
        store=store,
        simulated_runner=SimulatedBenchmarkRunner(),
        optimizer=FakeOptimizer(),
    )
    request = RunCreateRequest(
        task_ids=["task-fail"],
        mode="simulated",
        requested_concurrency=1,
        max_iterations=1,
    )

    run = service.submit_run(request, org_id="org-1", created_by="user-1", start_background=False)
    service.execute_run(run.run_id, org_id="org-1")

    assert len(store.iterations[run.run_id]) == 3
    assert store.iterations[run.run_id][-1]["status"] == "proposal_created"
    assert "proposed_change" in store.iterations[run.run_id][-1]["proposal"]
```

The expected length is 3 because `submit_run()` creates queued iteration 0, `execute_run()` updates completed iteration 0, and optimizer adds proposal iteration 1.

- [ ] **Step 6: Run optimizer/service tests**

Run:

```bash
python -m pytest tests/service/test_optimizer.py tests/service/test_service.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add autoharness_service/optimizer.py autoharness_service/service.py tests/service/test_optimizer.py tests/service/test_service.py
git commit -m "feat: record llm improvement proposals"
```

---

## Task 7: End-to-End Test Client

**Files:**
- Create: `test_client.py`
- Create: `tests/service/test_test_client.py`

**Interfaces:**
- Produces CLI:
  - `python test_client.py --base-url http://127.0.0.1:8000 --task-id task-pass --task-id task-fail --mode simulated`
- Prints structured JSON summary with run status, results, and iterations.

- [ ] **Step 1: Write a formatting test for the client helper**

Create `tests/service/test_test_client.py`:

```python
from test_client import build_summary


def test_build_summary_includes_results_and_iterations():
    summary = build_summary(
        status={"run_id": "run-1", "status": "succeeded"},
        results={"score": 0.5, "tasks_passed": 1, "tasks_failed": 1},
        iterations={"iterations": [{"iteration": 0, "status": "completed"}]},
    )

    assert summary["run_id"] == "run-1"
    assert summary["status"] == "succeeded"
    assert summary["score"] == 0.5
    assert summary["iterations"][0]["status"] == "completed"
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/service/test_test_client.py -q
```

Expected:

```text
ModuleNotFoundError or ImportError for build_summary
```

- [ ] **Step 3: Implement `test_client.py`**

Create `test_client.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx


TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled"}


def build_summary(
    *,
    status: dict[str, Any],
    results: dict[str, Any],
    iterations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": status["run_id"],
        "status": status["status"],
        "score": results.get("score"),
        "tasks_passed": results.get("tasks_passed"),
        "tasks_failed": results.get("tasks_failed"),
        "tasks_infra_failed": results.get("tasks_infra_failed"),
        "failure_summary": results.get("failure_summary"),
        "failed_task_ids": [
            item["task_id"]
            for item in results.get("task_results", [])
            if item["status"] != "passed"
        ],
        "iterations": iterations.get("iterations", []),
    }


def submit_run(
    client: httpx.Client,
    *,
    task_ids: list[str],
    mode: str,
    max_iterations: int,
    requested_concurrency: int,
) -> str:
    response = client.post(
        "/runs",
        json={
            "task_ids": task_ids,
            "mode": mode,
            "max_iterations": max_iterations,
            "requested_concurrency": requested_concurrency,
            "sandbox_provider": "daytona",
        },
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


def poll_run(
    client: httpx.Client,
    run_id: str,
    *,
    poll_interval_sec: float,
    timeout_sec: int,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        response = client.get(f"/runs/{run_id}")
        response.raise_for_status()
        payload = response.json()
        print(json.dumps({"poll": payload}, indent=2))
        if payload["status"] in TERMINAL_STATUSES:
            return payload
        if time.monotonic() - started > timeout_sec:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_sec}s")
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task-id", action="append", dest="task_ids", required=True)
    parser.add_argument("--mode", choices=["simulated", "real"], default="simulated")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--requested-concurrency", type=int, default=1)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        run_id = submit_run(
            client,
            task_ids=args.task_ids,
            mode=args.mode,
            max_iterations=args.max_iterations,
            requested_concurrency=args.requested_concurrency,
        )
        print(json.dumps({"submitted_run_id": run_id}, indent=2))
        status = poll_run(
            client,
            run_id,
            poll_interval_sec=args.poll_interval_sec,
            timeout_sec=args.timeout_sec,
        )
        results = client.get(f"/runs/{run_id}/results").json()
        iterations = client.get(f"/runs/{run_id}/iterations").json()
        print(json.dumps(build_summary(status=status, results=results, iterations=iterations), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run client helper test**

Run:

```bash
python -m pytest tests/service/test_test_client.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add test_client.py tests/service/test_test_client.py
git commit -m "feat: add service test client"
```

---

## Task 8: README Take-home Instructions and Final Verification

**Files:**
- Modify: `README.md`
- Optional modify: `docs/takehome_mvp_design_zh.md`

**Interfaces:**
- README explains setup, service start, simulated test, real Harbor/Daytona mode, selected task rationale, key design decisions, omissions, and production extensions.

- [ ] **Step 1: Add README section**

Add a top-level section near the Quick Start area:

```markdown
## Take-home: Agent Optimization Service MVP

This branch adds a small FastAPI backend service for the take-home assignment.
It intentionally focuses on Milestones 1-3:

- API design with structured request/response shapes
- asynchronous run lifecycle with polling
- sandboxed Terminal-Bench execution through Harbor + Daytona in real mode
- PostgreSQL persistence for runs, task results, and iteration history

Milestone 4 is represented by a minimal LLM improvement proposal history. The
service does not automatically patch `agent/agent.py` in this MVP. Milestone 5
is represented by API-level `X-Org-Id` / `X-User-Id` scoping and documented
RBAC extensions.

### Setup

```bash
python -m pip install -e .
cp .env.example .env
```

For simulated mode, no sandbox key is required.

For real Terminal-Bench mode, set:

```text
OPENAI_API_KEY=...
DAYTONA_API_KEY=...
DATABASE_URL=postgresql://autoharness:autoharness@localhost:5432/autoharness
```

Install Harbor:

```bash
uv tool install harbor
```

### Start the service

```bash
uvicorn autoharness_service.main:app --host 127.0.0.1 --port 8000
```

### Run the end-to-end client in simulated mode

```bash
python test_client.py \
  --base-url http://127.0.0.1:8000 \
  --task-id task-pass \
  --task-id task-fail \
  --mode simulated
```

### Run a real Harbor/Daytona task

```bash
python test_client.py \
  --base-url http://127.0.0.1:8000 \
  --task-id break-filter-js-from-html \
  --mode real \
  --requested-concurrency 1 \
  --timeout-sec 1800
```

### Selected tasks

The initial real-mode smoke task is `break-filter-js-from-html` because it is a
small Terminal-Bench task that exercises file inspection, shell execution, and
verifier feedback without requiring a large multi-task run. With more runtime,
the representative 10-20 task subset should be chosen by timing tasks and
keeping a mix of file manipulation, coding, parsing, and verifier-failure cases.

### Key design decisions

- The service treats a run as the MVP batch unit. There is no separate
  `eval_batches` table in this branch.
- Real mode uses Harbor as the Daytona adapter. The service does not call the
  Daytona SDK directly.
- Daytona sandboxes do not write to local PostgreSQL. Harbor writes local
  artifacts, and the service normalizes those artifacts into PostgreSQL rows.
- Local MVP uses polling. Daytona webhooks are reserved for production lifecycle
  reconciliation.
- The optimizer stores one LLM proposal and does not do multi-candidate search,
  GateEngine regression suites, suite promotion, or merge evaluation.

### What I would do with more time

- Replace in-process background threads with Redis Streams or Kafka workers.
- Store large traces and logs in S3 or MinIO instead of local paths.
- Add direct Daytona SDK support with sandbox_id/session_id/cmd_id tracking.
- Add GateEngine regression protection and candidate promotion.
- Add JWT/OAuth-backed organization membership and role-based access control.
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest tests/service -q
```

Expected without `DATABASE_URL`:

```text
service tests pass and live Postgres tests skip
```

Expected with local Postgres:

```text
all service tests pass
```

- [ ] **Step 3: Run import check**

Run:

```bash
python -c "from autoharness_service.api import create_app; app = create_app(); print(app.title)"
```

Expected:

```text
Agent Optimization Service
```

- [ ] **Step 4: Manual simulated E2E**

Start service:

```bash
uvicorn autoharness_service.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python test_client.py --task-id task-pass --task-id task-fail --mode simulated
```

Expected summary:

```text
status=succeeded
score=0.5
tasks_passed=1
tasks_failed=1
```

- [ ] **Step 5: Optional real Daytona smoke**

Run only if `OPENAI_API_KEY`, `DAYTONA_API_KEY`, `harbor`, and local Postgres are configured:

```bash
python test_client.py \
  --task-id break-filter-js-from-html \
  --mode real \
  --requested-concurrency 1 \
  --timeout-sec 1800
```

Expected:

```text
The run reaches a terminal status and /results returns one task result.
The reward may be 0.0; that means the task failed, not that the service failed.
```

- [ ] **Step 6: Safety checks before final commit**

Run:

```bash
git status --short
git ls-files AGENTS.md
git diff --check
git diff --cached --name-only | rg '(^|/)AGENTS\.md$|(^|/)\.env$' && exit 1 || true
```

Expected:

```text
AGENTS.md is not tracked.
.env is not staged.
No whitespace errors.
```

- [ ] **Step 7: Commit Task 8**

Run:

```bash
git add README.md docs/takehome_mvp_design_zh.md
git commit -m "docs: document takehome service mvp"
```

---

## Final Verification Checklist

- [ ] `python -m pytest tests/service -q` passes, with live DB tests skipped if `DATABASE_URL` is unset.
- [ ] `python -c "from autoharness_service.api import create_app; print(create_app().title)"` prints `Agent Optimization Service`.
- [ ] `uvicorn autoharness_service.main:app --host 127.0.0.1 --port 8000` starts.
- [ ] `python test_client.py --task-id task-pass --task-id task-fail --mode simulated` completes and prints structured summary.
- [ ] README includes setup/run instructions, selected task rationale, key design decisions, omitted scope, and future work.
- [ ] `git ls-files AGENTS.md` prints nothing.
- [ ] `git status --short` shows only intentional files before each commit and clean after the final commit.

## Scope Review

Covered:

- Milestone 1 API design: `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/results`, `GET /tasks`, structured errors/status/results.
- Milestone 2 async processing: submit returns run id, background thread executes, caller polls status.
- Milestone 3 sandbox execution: real mode delegates to Harbor with `--env daytona`; simulated mode validates lifecycle without external services.
- Milestone 4 minimal history: iteration 0 and optional LLM proposal iteration are persisted and exposed.
- Milestone 5 design hook: mock `X-Org-Id`/`X-User-Id` API-level scoping and README production RBAC notes.

Intentionally not covered:

- automatic patch application
- multi-candidate optimization
- failure clustering
- GateEngine regression suite
- suite promotion
- CandidateGraphManager
- Kubernetes/distributed workers
- direct Daytona SDK implementation
- webhook-based completion
