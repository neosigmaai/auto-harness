# Milestone 4 — Iterative Optimization Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a job-level optimization loop to the benchmark service: an LLM proposes agent-prompt/config improvements from observed failures, the benchmark is re-run, and the loop continues until performance stops improving or a cap is hit — with the full iteration history persisted and exposed over HTTP.

**Architecture:** A new `jobs` resource sits above the existing `runs` primitive. The Postgres queue gains **typed steps** (`evaluate` | `improve`); stateless workers claim one step at a time and, in the same transaction that completes a step, enqueue its successor — so a job never holds a worker across steps and a crash resumes from the last committed step. The agent under optimization is no longer a mutated repo file: each version is an `AgentSpec` (system prompt + model config) stored as JSONB and materialized to disk per run, executed by a fixed, generic runtime (`agent/spec_agent.py`). Agent traces live in an artifact store behind an interface, never in the repo or the database.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy 2.0 (typed ORM, `create_all` — no Alembic), Pydantic v2, psycopg3 / PostgreSQL 16, litellm (improver + agent), Harbor CLI (Terminal-Bench 2.0), pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-milestone-4-iterative-loop-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python ≥3.12.** Every new module starts with `from __future__ import annotations`.
- **Existing behavior is untouched.** `runs`/`run_tasks` tables, `/v1/runs`, `/tasks`, `/health` and the standalone-run worker path keep working exactly as they do today. `agent/agent.py` is never read or written by the service after Task 4.
- **No Alembic.** New tables are created by `init_db()` → `Base.metadata.create_all`, consistent with current practice.
- **Queue pattern is fixed:** `SELECT ... FOR UPDATE SKIP LOCKED`, mirroring `PostgresRunStore.claim_next` (`api/store.py:291-345`). Never poll-and-then-update without the row lock.
- **Naming:** queue units are **steps**. Never call them "tasks" — `task_id` already means a benchmark task throughout this codebase.
- **Mutable agent surface is prompt + config only** (`AgentSpec`). No code generation, no tool invention. Unknown fields from the LLM are rejected (`extra="forbid"`).
- **Score = mean reward**, with a `None` reward (timeout / infra error) counting as `0.0`.
- **Improvement threshold:** `improved ⇔ score > best_score + min_delta`. A score exactly equal to `best_score + min_delta` is *not* an improvement.
- **Stop-reason precedence** (first match wins): `max_iterations` → `no_improvement` → `budget_exceeded`.
- **Failure policy:** a failed *improve* step ends the job `completed` with `stop_reason="failed_improve"` when a best version exists (the best-so-far agent is a valid answer); with no best version the job is `failed`. A failed *evaluate* step always fails the job, so infra errors never masquerade as "no improvement".
- **Error envelope:** all route errors return `ErrorResponse(error=ErrorDetail(code, message, details))` via the `_error(...)` helper pattern from `api/routes/runs.py:23-25`. Error codes used here: `unknown_task_ids` (400), `empty_task_ids` (422), `job_not_found` (404), `agent_version_not_found` (404), `no_evaluation_yet` (409), `execution_unavailable` (503).
- **Config values are defaults, request fields are overrides.** `load_config()` is `lru_cache`d — any test that changes config or `EXECUTION_BACKEND` must call `clear_config_cache()`.
- **Postgres-dependent tests** copy the `_postgres_available()` + `pytestmark = pytest.mark.skipif(...)` guard from `tests/test_api.py:26-42`. Pure-logic tests must not require Postgres, Harbor, or network.
- **No network in tests.** `litellm` is monkeypatched at the module attribute; the mock execution backend (`EXECUTION_BACKEND=mock`) is used for all end-to-end tests.
- **Commit after every task**, message prefix `feat:` / `test:` / `refactor:`.

## Task Map

| # | Task | Deliverable |
|---|------|-------------|
| 1 | Config additions | 7 loop-tuning fields on `BenchmarkConfig` + YAML |
| 2 | `AgentSpec` | Validated, versionable agent definition |
| 3 | Artifact store | `ArtifactStore` interface + local-disk impl + key helpers |
| 4 | Spec-driven runtime | `agent/spec_agent.py` + `extra_env` plumbing through Harbor |
| 5 | ORM models | `jobs`, `agent_versions`, `steps` tables |
| 6 | Scoring & stopping | `mean_reward`, `compute_stop` (pure functions) |
| 7 | `PostgresJobStore` | Typed step queue + transactional step→successor advance |
| 8 | Improver context | Budgeted prompt assembly from history + failure traces |
| 9 | Improvers | `FakeImprover` (tests) + `LLMImprover` (litellm, validated) |
| 10 | `StepExecutor` + worker | Executes both step types; worker claims steps then legacy runs |
| 11 | Job schemas | Pydantic request/response contract |
| 12 | Job routes | `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/best` |
| 13 | Version route + E2E | `GET /v1/agent-versions/{id}` + full mock-loop API test |

**Dependency order:** 1 → 2 → 3 → 4 are foundations (2 must precede 5-7); 5 → 6 → 7 build the queue; 8 → 9 → 10 the execution; 11 → 12 → 13 the API. Tasks 8-9 depend only on Task 2 and Task 7's `IterationRecord`, so they can proceed in parallel with 10 once 7 lands.

---

## Section A — Foundations (Tasks 1-4)

### Task 1: Config additions

**Files:**
- Modify: `api/config.py:19-33` (add 7 fields to `BenchmarkConfig`)
- Modify: `api/config.py:36-73` (parse + validate them in `load_config`)
- Modify: `config/benchmark.yaml:12` (insert the new block after `jobs_dir`)
- Test: `tests/test_config_m4.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BenchmarkConfig.improver_model: str`, `.max_iterations: int`, `.patience: int`,
  `.min_delta: float`, `.max_job_duration_sec: int`, `.improver_context_budget: int`,
  `.artifacts_dir: str`. Every later task reads these off the frozen dataclass;
  `load_config()` / `clear_config_cache()` signatures are unchanged.

**Context an engineer needs before starting:**
- `BenchmarkConfig` is a `@dataclass(frozen=True)`. The first two fields
  (`default_task_ids`, `default_agent_model`) have no default, all others do, so the
  7 new fields must be **appended after `jobs_dir`** or Python raises
  `TypeError: non-default argument follows default argument`.
- `load_config` is decorated `@lru_cache(maxsize=1)`. Tests MUST call
  `clear_config_cache()` before and after touching it, otherwise a config loaded from
  one test's `tmp_path` YAML leaks into the next test. `maxsize=1` also means two
  different `path` arguments evict each other, so no test can rely on a warm cache.
- Existing fields use the `raw.get("k") or default` idiom. The new numeric fields
  deliberately do **not**: `0 or 5` evaluates to `5`, which would silently accept an
  invalid `max_iterations: 0` instead of raising. `min_delta` has the same problem in
  reverse — `0.0` is a legal value and `0.0 or 0.01` would rewrite it to `0.01`. Both
  therefore use an explicit `is None` check. This deviation is intentional; keep it.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_config_m4.py
"""Unit tests for the Milestone 4 BenchmarkConfig additions (no DB, no harbor)."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import BenchmarkConfig, clear_config_cache, load_config

# A minimal valid config; each test appends the field under test.
MINIMAL_YAML = """\
default_agent_model: gpt-4.1-mini
default_task_ids:
  - fix-git
"""


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    """load_config is lru_cached with maxsize=1 — never let one test's config leak."""
    clear_config_cache()
    yield
    clear_config_cache()


def _write_yaml(tmp_path: Path, extra: str) -> str:
    path = tmp_path / "benchmark.yaml"
    path.write_text(MINIMAL_YAML + extra, encoding="utf-8")
    return str(path)


def test_repo_config_provides_m4_defaults() -> None:
    """The checked-in config/benchmark.yaml carries the Milestone 4 defaults."""
    cfg = load_config()
    assert cfg.improver_model == "gpt-5.4"
    assert cfg.max_iterations == 5
    assert cfg.patience == 2
    assert cfg.min_delta == pytest.approx(0.01)
    assert cfg.max_job_duration_sec == 21600
    assert cfg.improver_context_budget == 60000
    assert cfg.artifacts_dir == "workspace/artifacts"


def test_dataclass_defaults_match_contract() -> None:
    """Constructing BenchmarkConfig directly (as tests and workers do) needs no YAML."""
    cfg = BenchmarkConfig(default_task_ids=["fix-git"], default_agent_model="gpt-4.1-mini")
    assert cfg.improver_model == "gpt-5.4"
    assert cfg.max_iterations == 5
    assert cfg.patience == 2
    assert cfg.min_delta == pytest.approx(0.01)
    assert cfg.max_job_duration_sec == 21600
    assert cfg.improver_context_budget == 60000
    assert cfg.artifacts_dir == "workspace/artifacts"


def test_yaml_values_override_defaults(tmp_path: Path) -> None:
    cfg = load_config(
        _write_yaml(
            tmp_path,
            "improver_model: claude-opus-4\n"
            "max_iterations: 9\n"
            "patience: 3\n"
            "min_delta: 0.25\n"
            "max_job_duration_sec: 600\n"
            "improver_context_budget: 1234\n"
            "artifacts_dir: /var/tmp/artifacts\n",
        )
    )
    assert cfg.improver_model == "claude-opus-4"
    assert cfg.max_iterations == 9
    assert cfg.patience == 3
    assert cfg.min_delta == pytest.approx(0.25)
    assert cfg.max_job_duration_sec == 600
    assert cfg.improver_context_budget == 1234
    assert cfg.artifacts_dir == "/var/tmp/artifacts"


def test_min_delta_zero_is_accepted(tmp_path: Path) -> None:
    """0.0 is inside [0, 1) — it must survive parsing, not be defaulted to 0.01."""
    cfg = load_config(_write_yaml(tmp_path, "min_delta: 0.0\n"))
    assert cfg.min_delta == 0.0


@pytest.mark.parametrize(
    "extra,expected_message",
    [
        ("max_iterations: 0\n", "max_iterations"),
        ("max_iterations: -3\n", "max_iterations"),
        ("patience: 0\n", "patience"),
        ("patience: -1\n", "patience"),
        ("max_job_duration_sec: 0\n", "max_job_duration_sec"),
        ("improver_context_budget: 0\n", "improver_context_budget"),
        ("min_delta: 1.0\n", "min_delta"),
        ("min_delta: 1.5\n", "min_delta"),
        ("min_delta: -0.5\n", "min_delta"),
    ],
)
def test_invalid_m4_values_raise(tmp_path: Path, extra: str, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        load_config(_write_yaml(tmp_path, extra))
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_config_m4.py -v`
Expected: FAIL — `test_repo_config_provides_m4_defaults` and
`test_dataclass_defaults_match_contract` fail with
`AttributeError: 'BenchmarkConfig' object has no attribute 'improver_model'`;
the override/validation tests fail with
`TypeError: BenchmarkConfig.__init__() got an unexpected keyword argument 'improver_model'`
or (for the invalid cases) `Failed: DID NOT RAISE <class 'ValueError'>`.
- [ ] **Step 3: Write the implementation**

Full replacement content for `api/config.py`:
```python
"""Load benchmark API configuration from config/benchmark.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "benchmark.yaml"

KNOWN_ENV_PROVIDERS = frozenset({"e2b", "daytona", "modal", "docker"})
KNOWN_EXECUTION_BACKENDS = frozenset({"harbor", "mock"})


@dataclass(frozen=True)
class BenchmarkConfig:
    default_task_ids: list[str]
    default_agent_model: str
    env_provider: str = "docker"
    dataset: str = "terminal-bench@2.0"
    max_concurrency: int = 2
    per_task_timeout: int = 1200
    execution_backend: str = "harbor"
    jobs_dir: str = "workspace/tbench_jobs"
    # --- Milestone 4: iterative optimization loop ---
    improver_model: str = "gpt-5.4"
    max_iterations: int = 5
    patience: int = 2
    min_delta: float = 0.01
    max_job_duration_sec: int = 21600
    improver_context_budget: int = 60000
    artifacts_dir: str = "workspace/artifacts"

    @property
    def known_task_ids(self) -> frozenset[str]:
        """Allowlist of task IDs accepted by POST /v1/runs."""
        return frozenset(self.default_task_ids)


def _positive_int(raw: dict, key: str, default: int) -> int:
    """Read a strictly-positive int.

    Deliberately not `raw.get(key) or default`: that idiom turns an explicit 0 into
    the default instead of rejecting it.
    """
    value = raw.get(key)
    value = default if value is None else int(value)
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value}")
    return value


def _unit_fraction(raw: dict, key: str, default: float) -> float:
    """Read a float in [0, 1). 0.0 is legal, so `or default` cannot be used."""
    value = raw.get(key)
    value = default if value is None else float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{key} must be in [0, 1), got {value}")
    return value


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> BenchmarkConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    task_ids = raw.get("default_task_ids") or []
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"{config_path} must define a non-empty default_task_ids list")

    model = raw.get("default_agent_model") or "gpt-4.1-mini"
    env_provider = str(raw.get("env_provider") or "docker").lower()
    if env_provider not in KNOWN_ENV_PROVIDERS:
        raise ValueError(
            f"Unknown env_provider {env_provider!r}; expected one of {sorted(KNOWN_ENV_PROVIDERS)}"
        )

    backend = (
        os.environ.get("EXECUTION_BACKEND")
        or raw.get("execution_backend")
        or "harbor"
    )
    backend = str(backend).lower()
    if backend not in KNOWN_EXECUTION_BACKENDS:
        raise ValueError(
            f"Unknown execution_backend {backend!r}; expected one of {sorted(KNOWN_EXECUTION_BACKENDS)}"
        )

    return BenchmarkConfig(
        default_task_ids=[str(t) for t in task_ids],
        default_agent_model=str(model),
        env_provider=env_provider,
        dataset=str(raw.get("dataset") or "terminal-bench@2.0"),
        max_concurrency=int(raw.get("max_concurrency") or 2),
        per_task_timeout=int(raw.get("per_task_timeout") or 1200),
        execution_backend=backend,
        jobs_dir=str(raw.get("jobs_dir") or "workspace/tbench_jobs"),
        improver_model=str(raw.get("improver_model") or "gpt-5.4"),
        max_iterations=_positive_int(raw, "max_iterations", 5),
        patience=_positive_int(raw, "patience", 2),
        min_delta=_unit_fraction(raw, "min_delta", 0.01),
        max_job_duration_sec=_positive_int(raw, "max_job_duration_sec", 21600),
        improver_context_budget=_positive_int(raw, "improver_context_budget", 60000),
        artifacts_dir=str(raw.get("artifacts_dir") or "workspace/artifacts"),
    )


def clear_config_cache() -> None:
    load_config.cache_clear()
```

Insert this block into `config/benchmark.yaml` immediately after the
`jobs_dir: workspace/tbench_jobs` line (currently line 12) and before the
`# Default tasks when POST /v1/runs omits task_ids.` comment:
```yaml

# Iterative optimization loop (Milestone 4). Per-job request fields override these.
improver_model: gpt-5.4          # model that proposes AgentSpec changes
max_iterations: 5                # hard cap on evaluate steps per job
patience: 2                      # consecutive non-improving evaluations before stopping
min_delta: 0.01                  # score gain required to count as an improvement
max_job_duration_sec: 21600      # 6 h wall-clock budget per job
improver_context_budget: 60000   # max characters in one improver prompt
artifacts_dir: workspace/artifacts
```
- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_config_m4.py tests/test_reward_mapping.py -v`
Expected: PASS (the existing `test_reward_mapping.py` is included because it also
constructs `BenchmarkConfig` and calls `clear_config_cache()`).
- [ ] **Step 5: Commit**
```bash
git add api/config.py config/benchmark.yaml tests/test_config_m4.py
git commit -m "feat: add iterative-loop settings to BenchmarkConfig"
```

### Task 2: AgentSpec

**Files:**
- Create: `api/agent_spec.py`
- Test: `tests/test_agent_spec.py`

**Interfaces:**
- Consumes: nothing (pure Pydantic; no config, no DB).
- Produces:
  - `BASELINE_SYSTEM_PROMPT: str`
  - `class AgentSpec(BaseModel)` with `model_config = ConfigDict(extra="forbid")` and fields
    `system_prompt: str`, `agent_model: str`, `max_steps: int = 80`,
    `max_output_chars: int = 8000`, `exec_timeout_sec: int = 120`
  - `def baseline_spec(agent_model: str) -> AgentSpec`
  - `def changed_fields(old: AgentSpec, new: AgentSpec) -> list[str]`
  Used by Task 4's drift test, Task 7 (`PostgresJobStore`, spec JSONB round-trip),
  Tasks 8-9 (`Improver` validation gate) and Task 11 (`AgentSpecView`).

**Context an engineer needs before starting:**
- `BASELINE_SYSTEM_PROMPT` must be **character-identical** to `AGENT_INSTRUCTION` in
  `agent/templates/terminal_bench.py:14-25` (that file uses `"""\` so the string does
  *not* start with a newline but *does* end with one, after "...what you did."). A test
  below re-reads the template and asserts equality, so a typo fails loudly rather than
  silently shipping a different baseline.
- `api/agent_spec.py` lives at the top of the `api` package on purpose: it is imported by
  the store, the improver and the schemas, and importing it must not drag in the DB.
  Import nothing from `api.store`, `api.services.*` or `sqlalchemy` here.
- Pydantic v2 is already a dependency (2.13.x). `extra="forbid"` is the validation gate on
  improver output: an unknown key is a failed improve step, never a crashed job.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_agent_spec.py
"""Unit tests for AgentSpec — the mutable surface the improver edits (no DB, no harbor)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.agent_spec import (
    BASELINE_SYSTEM_PROMPT,
    AgentSpec,
    baseline_spec,
    changed_fields,
)
from api.config import REPO_ROOT


def _template_agent_instruction() -> str:
    """Extract AGENT_INSTRUCTION from the template without importing it (needs harbor)."""
    source = (REPO_ROOT / "agent" / "templates" / "terminal_bench.py").read_text(
        encoding="utf-8"
    )
    marker = 'AGENT_INSTRUCTION = """\\\n'
    start = source.index(marker) + len(marker)
    end = source.index('"""', start)
    return source[start:end]


def test_baseline_prompt_is_verbatim_copy_of_template() -> None:
    assert BASELINE_SYSTEM_PROMPT == _template_agent_instruction()


def test_baseline_spec_uses_given_model_and_template_defaults() -> None:
    spec = baseline_spec("gpt-4.1-mini")
    assert spec.agent_model == "gpt-4.1-mini"
    assert spec.system_prompt == BASELINE_SYSTEM_PROMPT
    assert spec.max_steps == 80
    assert spec.max_output_chars == 8000
    assert spec.exec_timeout_sec == 120


def test_valid_spec_round_trips_through_json() -> None:
    spec = AgentSpec(
        system_prompt="do the thing",
        agent_model="claude-sonnet-4",
        max_steps=12,
        max_output_chars=999,
        exec_timeout_sec=45,
    )
    restored = AgentSpec.model_validate(spec.model_dump())
    assert restored == spec
    assert set(spec.model_dump()) == {
        "system_prompt",
        "agent_model",
        "max_steps",
        "max_output_chars",
        "exec_timeout_sec",
    }


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSpec(
            system_prompt="p",
            agent_model="m",
            temperature=0.7,  # not part of the mutable surface
        )


@pytest.mark.parametrize("bad_steps", [0, -1, 201, 10_000])
def test_max_steps_out_of_bounds_is_rejected(bad_steps: int) -> None:
    with pytest.raises(ValidationError):
        AgentSpec(system_prompt="p", agent_model="m", max_steps=bad_steps)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("system_prompt", ""),
        ("system_prompt", "x" * 20_001),
        ("agent_model", ""),
        ("agent_model", "m" * 257),
        ("max_output_chars", 499),
        ("max_output_chars", 100_001),
        ("exec_timeout_sec", 9),
        ("exec_timeout_sec", 1201),
    ],
)
def test_field_bounds_are_enforced(field: str, bad_value: object) -> None:
    kwargs: dict[str, object] = {"system_prompt": "p", "agent_model": "m"}
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        AgentSpec(**kwargs)


@pytest.mark.parametrize(
    "bounds_ok",
    [
        {"max_steps": 1},
        {"max_steps": 200},
        {"max_output_chars": 500},
        {"max_output_chars": 100_000},
        {"exec_timeout_sec": 10},
        {"exec_timeout_sec": 1200},
    ],
)
def test_field_bounds_are_inclusive(bounds_ok: dict) -> None:
    spec = AgentSpec(system_prompt="p", agent_model="m", **bounds_ok)
    for key, value in bounds_ok.items():
        assert getattr(spec, key) == value


def test_changed_fields_detects_exactly_the_differing_fields() -> None:
    old = baseline_spec("gpt-4.1-mini")
    new = old.model_copy(update={"system_prompt": "new prompt", "max_steps": 120})
    assert changed_fields(old, new) == ["max_steps", "system_prompt"]


def test_changed_fields_is_empty_for_identical_specs() -> None:
    old = baseline_spec("gpt-4.1-mini")
    assert changed_fields(old, old.model_copy()) == []


def test_changed_fields_is_sorted_and_covers_every_field() -> None:
    old = baseline_spec("a")
    new = AgentSpec(
        system_prompt="different",
        agent_model="b",
        max_steps=1,
        max_output_chars=500,
        exec_timeout_sec=10,
    )
    assert changed_fields(old, new) == [
        "agent_model",
        "exec_timeout_sec",
        "max_output_chars",
        "max_steps",
        "system_prompt",
    ]
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_agent_spec.py -v`
Expected: FAIL at collection with
`ModuleNotFoundError: No module named 'api.agent_spec'`
- [ ] **Step 3: Write the implementation**
```python
# api/agent_spec.py
"""AgentSpec — the mutable surface of the agent under optimization.

The improver may change the system prompt and a handful of bounded knobs; tools
(bash) and the agent loop itself are fixed. Keep this module free of database and
service imports: it is shared by the store, the improver, the API schemas and the
tests, and must stay cheap to import.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Verbatim copy of AGENT_INSTRUCTION in agent/templates/terminal_bench.py.
# tests/test_agent_spec.py asserts the two stay identical.
BASELINE_SYSTEM_PROMPT = """\
You are an autonomous terminal agent. You are given a task and a Linux container.
You solve tasks by executing bash commands. Work step by step.

Rules:
- Read the task carefully before acting.
- Explore the environment first to understand what you have.
- Check command output for errors before proceeding.
- Install missing dependencies as needed.
- Verify your solution before finishing.
- When you are done, send a final text message (no tool call) summarizing what you did.
"""


class AgentSpec(BaseModel):
    """A complete, validated description of a runnable agent.

    ``extra="forbid"`` plus the field bounds are the validation gate on improver
    output: a proposal that does not parse is a failed improve step, never a
    crashed job.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(min_length=1, max_length=20_000)
    agent_model: str = Field(min_length=1, max_length=256)
    max_steps: int = Field(default=80, ge=1, le=200)
    max_output_chars: int = Field(default=8000, ge=500, le=100_000)
    exec_timeout_sec: int = Field(default=120, ge=10, le=1200)


def baseline_spec(agent_model: str) -> AgentSpec:
    """Version 0 of every job: the template's prompt and limits, caller's model."""
    return AgentSpec(system_prompt=BASELINE_SYSTEM_PROMPT, agent_model=agent_model)


def changed_fields(old: AgentSpec, new: AgentSpec) -> list[str]:
    """Sorted names of the fields whose values differ between two specs."""
    old_data = old.model_dump()
    new_data = new.model_dump()
    return sorted(
        key
        for key in set(old_data) | set(new_data)
        if old_data.get(key) != new_data.get(key)
    )
```
- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_agent_spec.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add api/agent_spec.py tests/test_agent_spec.py
git commit -m "feat: add AgentSpec, the improver's validated mutable surface"
```

### Task 3: Artifact store

**Files:**
- Create: `api/services/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `BenchmarkConfig.artifacts_dir` and `REPO_ROOT` from Task 1.
- Produces:
  - `class ArtifactStore(Protocol)` — `put(key, data) -> None`, `get(key) -> bytes`,
    `list(prefix) -> list[str]`, `exists(key) -> bool`
  - `class LocalArtifactStore` — `__init__(self, root: Path | str) -> None`
  - `def create_artifact_store(config: BenchmarkConfig | None = None) -> ArtifactStore`
  - `def trace_key(job_id: str, iteration: int, task_id: str) -> str`
  - `def result_key(job_id: str, iteration: int, task_id: str) -> str`
  - `def improver_key(job_id: str, iteration: int, name: str) -> str`
  Task 10 (`StepExecutor`) writes traces/results through this; Tasks 8-9 read traces back.

**Context an engineer needs before starting:**
- Keys are **relative POSIX paths**, never OS paths — `"jobs/<id>/iterations/0/..."`.
  The store is the only thing that knows where those land on disk, so an S3 backend can
  drop in later behind the same Protocol (same shape as `create_runner`).
- `put` accepts three input types: `Path` (copy the file), `str` (utf-8 encode) or
  `bytes` (write as-is). `Path` matters because harbor writes `trace.json` on disk and the
  worker should not read a multi-MB file into memory just to hand it over.
- Traversal defence: any key containing `..` is rejected, as are absolute keys and keys
  containing a backslash. The contract names no exception type, so raise plain
  `ValueError`.
- `list(prefix)` filters on the **string** prefix of the relative key, not on directory
  boundaries, so `list("jobs/j1/iterations/0")` and `list("jobs/j1/iter")` both work.
  Directories are never returned — only files.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_artifacts.py
"""Unit tests for the local artifact store (no DB, no harbor, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import REPO_ROOT, BenchmarkConfig
from api.services.artifacts import (
    LocalArtifactStore,
    create_artifact_store,
    improver_key,
    result_key,
    trace_key,
)


def test_put_and_get_bytes_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/tasks/fix-git/trace.json", b'{"a": 1}')
    assert store.get("jobs/j1/iterations/0/tasks/fix-git/trace.json") == b'{"a": 1}'


def test_put_and_get_str_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/improver/prompt.txt", "héllo prompt")
    assert store.get("jobs/j1/iterations/0/improver/prompt.txt") == "héllo prompt".encode()


def test_put_copies_a_file_given_as_path(tmp_path: Path) -> None:
    source = tmp_path / "harbor_out" / "trace.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"[]")
    store = LocalArtifactStore(tmp_path / "store")

    store.put("jobs/j1/iterations/0/tasks/fix-git/trace.json", source)

    assert store.get("jobs/j1/iterations/0/tasks/fix-git/trace.json") == b"[]"
    assert source.exists(), "put must copy, not move, the source file"


def test_put_creates_intermediate_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "nope" / "deeper")
    store.put("a/b/c/d.json", b"1")
    assert (tmp_path / "nope" / "deeper" / "a" / "b" / "c" / "d.json").is_file()


def test_put_overwrites_an_existing_key(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("k.txt", b"first")
    store.put("k.txt", b"second")
    assert store.get("k.txt") == b"second"


def test_exists(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert store.exists("jobs/j1/x.json") is False
    store.put("jobs/j1/x.json", b"{}")
    assert store.exists("jobs/j1/x.json") is True


def test_exists_is_false_for_a_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/x.json", b"{}")
    assert store.exists("jobs/j1") is False


def test_get_missing_key_raises_file_not_found(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("jobs/j1/missing.json")


def test_list_by_prefix_returns_sorted_relative_keys(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/tasks/b/trace.json", b"1")
    store.put("jobs/j1/iterations/0/tasks/a/trace.json", b"1")
    store.put("jobs/j1/iterations/1/tasks/a/trace.json", b"1")
    store.put("jobs/j2/iterations/0/tasks/a/trace.json", b"1")

    assert store.list("jobs/j1/iterations/0") == [
        "jobs/j1/iterations/0/tasks/a/trace.json",
        "jobs/j1/iterations/0/tasks/b/trace.json",
    ]
    assert len(store.list("jobs/j1")) == 3
    assert len(store.list("")) == 4


def test_list_matches_partial_segments(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/a.json", b"1")
    store.put("jobs/j12/a.json", b"1")
    assert store.list("jobs/j1") == ["jobs/j1/a.json", "jobs/j12/a.json"]
    assert store.list("jobs/j1/") == ["jobs/j1/a.json"]


def test_list_of_missing_prefix_is_empty(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert store.list("jobs/nope") == []


@pytest.mark.parametrize(
    "bad_key",
    [
        "jobs/../../etc/passwd",
        "../escape.json",
        "..",
        "/absolute/path.json",
        "jobs\\j1\\trace.json",
        "",
        "jobs/j1/",
    ],
)
def test_unsafe_keys_are_rejected(tmp_path: Path, bad_key: str) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.put(bad_key, b"pwned")
    with pytest.raises(ValueError):
        store.get(bad_key)
    with pytest.raises(ValueError):
        store.exists(bad_key)


def test_traversal_prefix_is_rejected_by_list(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.list("../")


def test_key_helper_formats() -> None:
    assert (
        trace_key("j1", 0, "fix-git")
        == "jobs/j1/iterations/0/tasks/fix-git/trace.json"
    )
    assert (
        result_key("j1", 3, "regex-log")
        == "jobs/j1/iterations/3/tasks/regex-log/result.json"
    )
    assert improver_key("j1", 2, "prompt.txt") == "jobs/j1/iterations/2/improver/prompt.txt"
    assert improver_key("j1", 2, "response.json") == "jobs/j1/iterations/2/improver/response.json"


def test_create_artifact_store_resolves_relative_dir_against_repo_root() -> None:
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir="workspace/artifacts",
    )
    store = create_artifact_store(cfg)
    assert isinstance(store, LocalArtifactStore)
    assert store.root == REPO_ROOT / "workspace" / "artifacts"


def test_create_artifact_store_honours_an_absolute_dir(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir=str(tmp_path / "arts"),
    )
    store = create_artifact_store(cfg)
    assert isinstance(store, LocalArtifactStore)
    assert store.root == tmp_path / "arts"


def test_create_artifact_store_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    """The factory is cheap: directories appear on first put, not on construction."""
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir=str(tmp_path / "lazy"),
    )
    create_artifact_store(cfg)
    assert not (tmp_path / "lazy").exists()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_artifacts.py -v`
Expected: FAIL at collection with
`ModuleNotFoundError: No module named 'api.services.artifacts'`
- [ ] **Step 3: Write the implementation**
```python
# api/services/artifacts.py
"""Artifact storage for job traces and improver prompts/responses.

Artifacts are addressed by convention, not by a database table:

    jobs/<job_id>/iterations/<n>/tasks/<task_id>/trace.json
    jobs/<job_id>/iterations/<n>/tasks/<task_id>/result.json
    jobs/<job_id>/iterations/<n>/improver/{prompt.txt,response.json}

``LocalArtifactStore`` is the only implementation today; an S3/GCS backend is a
drop-in behind ``ArtifactStore`` (same factory pattern as ``create_runner``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from api.config import REPO_ROOT, BenchmarkConfig, load_config


class ArtifactStore(Protocol):
    """Content-addressed-by-convention blob store keyed by relative POSIX paths."""

    def put(self, key: str, data: bytes | str | Path) -> None:
        """Store ``data``; a ``Path`` is copied, a ``str`` is utf-8 encoded."""
        ...

    def get(self, key: str) -> bytes:
        """Return the stored bytes. Raises ``FileNotFoundError`` for unknown keys."""
        ...

    def list(self, prefix: str) -> list[str]:
        """Sorted keys of every stored object whose key starts with ``prefix``."""
        ...

    def exists(self, key: str) -> bool:
        ...


def _validate_prefix(prefix: str) -> str:
    """Reject anything that could escape the store root. Empty means 'everything'."""
    if ".." in prefix or prefix.startswith("/") or "\\" in prefix:
        raise ValueError(f"unsafe artifact key prefix: {prefix!r}")
    return prefix


def _validate_key(key: str) -> str:
    """Validate a key that must name a single object."""
    _validate_prefix(key)
    if not key or key.endswith("/"):
        raise ValueError(f"artifact key must name a file: {key!r}")
    return key


class LocalArtifactStore:
    """Filesystem-backed artifact store rooted at a single directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _validate_key(key)

    def put(self, key: str, data: bytes | str | Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Path):
            shutil.copyfile(data, dest)
        elif isinstance(data, str):
            dest.write_bytes(data.encode("utf-8"))
        else:
            dest.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list(self, prefix: str) -> list[str]:
        _validate_prefix(prefix)
        if not self.root.is_dir():
            return []
        keys = [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        ]
        return sorted(key for key in keys if key.startswith(prefix))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


def create_artifact_store(config: BenchmarkConfig | None = None) -> ArtifactStore:
    """Factory mirroring ``create_runner``: local disk under ``config.artifacts_dir``."""
    cfg = config or load_config()
    root = Path(cfg.artifacts_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return LocalArtifactStore(root)


def trace_key(job_id: str, iteration: int, task_id: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/tasks/{task_id}/trace.json"


def result_key(job_id: str, iteration: int, task_id: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/tasks/{task_id}/result.json"


def improver_key(job_id: str, iteration: int, name: str) -> str:
    return f"jobs/{job_id}/iterations/{iteration}/improver/{name}"
```
- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_artifacts.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add api/services/artifacts.py tests/test_artifacts.py
git commit -m "feat: add local artifact store for traces and improver I/O"
```

### Task 4: Spec-driven agent runtime + harbor plumbing

**Files:**
- Create: `agent/spec_loader.py` (stdlib only — the unit-testable half)
- Create: `agent/spec_agent.py` (imports harbor + litellm — not importable in the test env)
- Modify: `benchmark.py:144-164` (`TerminalBenchRunner.__init__`: add `extra_env` only)
- Modify: `benchmark.py:226-231` (`run()`: apply `self.extra_env` last, after the
  `HARNESS_SAVE_TRACE` block, before `subprocess.run`)
- Modify: `api/services/runner.py:148-183` (`HarborBenchmarkRunner.__init__` +
  `_check_agent_import`)
- Modify: `api/services/runner.py:245-253` (pass both new params to `TerminalBenchRunner`)
- Test: `tests/test_spec_agent_runtime.py`

**Interfaces:**
- Consumes: `BASELINE_SYSTEM_PROMPT` from Task 2 (test-time drift guard only — the agent
  runtime itself must NOT import it).
- Produces:
  - `agent.spec_loader.DEFAULT_SYSTEM_PROMPT: str`, `DEFAULT_MAX_STEPS: int`,
    `DEFAULT_MAX_OUTPUT_CHARS: int`, `DEFAULT_EXEC_TIMEOUT_SEC: int`
  - `agent.spec_loader.default_spec() -> dict`
  - `agent.spec_loader.load_spec(path: str | None) -> dict`
  - `agent.spec_loader.load_spec_from_env() -> dict`
  - `agent.spec_agent.HarnessAgent` — the class harbor loads via
    `--agent-import-path agent.spec_agent:HarnessAgent`
  - `TerminalBenchRunner(..., extra_env: dict[str, str] | None = None)`
  - `HarborBenchmarkRunner(..., agent_import_path: str | None = None,
    extra_env: dict[str, str] | None = None)`
  - `api.services.runner.DEFAULT_AGENT_IMPORT_PATH = "agent.agent:HarnessAgent"`
  Task 10 (`StepExecutor`) constructs `HarborBenchmarkRunner` with
  `agent_import_path="agent.spec_agent:HarnessAgent"` and
  `extra_env={"HARNESS_AGENT_SPEC": <path>, "HARNESS_SAVE_TRACE": "1"}`.

**Context an engineer needs before starting — read all six points:**

1. **Why two agent files.** `agent/spec_agent.py` imports `harbor` and `litellm` at module
   scope, and **neither is installed in the test environment** (verified:
   `python -c "import harbor"` → `ModuleNotFoundError`). So `agent/spec_agent.py` can never
   be imported by a test. All spec-loading logic therefore lives in
   `agent/spec_loader.py`, which is **stdlib only**; `spec_agent.py` imports it and does
   nothing else clever. The unit tests below test `spec_loader` and never touch
   `spec_agent`.
2. **No `api.*` imports in `agent/`.** Harbor spawns the agent with only the repo root on
   `PYTHONPATH`; pulling in `api.config` would drag in yaml/pydantic/sqlalchemy. That is
   why `DEFAULT_SYSTEM_PROMPT` is duplicated in `spec_loader.py` rather than imported from
   `api.agent_spec`. A test below asserts the two copies stay byte-identical, so the
   duplication cannot silently drift.
3. **`agent/` is a namespace package** (no `__init__.py`, verified:
   `agent.__path__` is a `_NamespacePath`). `from agent.spec_loader import ...` works both
   from pytest (`pythonpath = ["."]` in `pyproject.toml`) and under harbor's PYTHONPATH.
   Do **not** add an `__init__.py`.
4. **`agent/agent.py` is off limits.** In the Layer A CLI loop it is the file the coding
   agent edits (enforced by `gating.py`) and `prepare.py` overwrites it from templates.
   Layer B must not share it. Never read or write it from the service after this task.
5. **`agent_import_path` ALREADY EXISTS** in `TerminalBenchRunner.__init__` at
   `benchmark.py:151` with default `"agent.agent:HarnessAgent"`. **Do not re-add it.**
   The only new constructor parameter in this task is `extra_env`.
6. **Why `extra_env` must be applied last.** `run()` currently forces
   `HARNESS_SAVE_TRACE="0"` whenever `self.split != "train"` (`benchmark.py:229-230`), and
   Layer B always passes `split=None`. Applying `extra_env` *after* that block is what lets
   the worker turn tracing back on — without it the improver would receive no traces at
   all. `run()` does a function-local `import subprocess`, so tests monkeypatch
   `subprocess.run` on the real module (attribute lookup happens at call time).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_spec_agent_runtime.py
"""Unit tests for the spec-driven agent runtime and its harbor plumbing.

Nothing here imports agent.spec_agent: it imports harbor and litellm at module
scope and neither is installed in the test environment. The spec-loading logic
lives in agent.spec_loader (stdlib only) precisely so it can be tested here.
"""

from __future__ import annotations

import json
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.spec_loader import (
    DEFAULT_EXEC_TIMEOUT_SEC,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYSTEM_PROMPT,
    default_spec,
    load_spec,
    load_spec_from_env,
)

SPEC_KEYS = {
    "system_prompt",
    "agent_model",
    "max_steps",
    "max_output_chars",
    "exec_timeout_sec",
}


# --------------------------------------------------------------------------
# agent/spec_loader.py
# --------------------------------------------------------------------------


def test_default_prompt_matches_api_agent_spec_baseline() -> None:
    """Drift guard: the agent-side copy and the API-side copy must stay identical."""
    from api.agent_spec import BASELINE_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT == BASELINE_SYSTEM_PROMPT


def test_default_limits_match_api_agent_spec_defaults() -> None:
    from api.agent_spec import baseline_spec

    spec = baseline_spec("gpt-4.1-mini")
    assert DEFAULT_MAX_STEPS == spec.max_steps
    assert DEFAULT_MAX_OUTPUT_CHARS == spec.max_output_chars
    assert DEFAULT_EXEC_TIMEOUT_SEC == spec.exec_timeout_sec


def test_default_spec_has_exactly_the_spec_keys() -> None:
    assert set(default_spec()) == SPEC_KEYS


def test_load_spec_reads_json_from_path(tmp_path: Path) -> None:
    path = tmp_path / "agent_spec.json"
    path.write_text(
        json.dumps(
            {
                "system_prompt": "Be terse. Verify everything.",
                "agent_model": "claude-sonnet-4",
                "max_steps": 30,
                "max_output_chars": 1500,
                "exec_timeout_sec": 45,
            }
        ),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert spec == {
        "system_prompt": "Be terse. Verify everything.",
        "agent_model": "claude-sonnet-4",
        "max_steps": 30,
        "max_output_chars": 1500,
        "exec_timeout_sec": 45,
    }


def test_load_spec_overlays_only_the_provided_fields(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"max_steps": 7}), encoding="utf-8")

    spec = load_spec(str(path))

    assert spec["max_steps"] == 7
    assert spec["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert spec["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert spec["exec_timeout_sec"] == DEFAULT_EXEC_TIMEOUT_SEC


def test_load_spec_ignores_unknown_fields(tmp_path: Path) -> None:
    """A spec written by a newer AgentSpec must not break an older runtime."""
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"max_steps": 11, "temperature": 0.4, "tools": ["python"]}),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert set(spec) == SPEC_KEYS
    assert spec["max_steps"] == 11


def test_load_spec_coerces_numeric_strings(tmp_path: Path) -> None:
    path = tmp_path / "stringy.json"
    path.write_text(
        json.dumps({"max_steps": "25", "max_output_chars": "900", "exec_timeout_sec": "60"}),
        encoding="utf-8",
    )

    spec = load_spec(str(path))

    assert spec["max_steps"] == 25
    assert spec["max_output_chars"] == 900
    assert spec["exec_timeout_sec"] == 60


@pytest.mark.parametrize("path", [None, ""])
def test_load_spec_without_a_path_returns_defaults(path) -> None:
    assert load_spec(path) == default_spec()


def test_load_spec_falls_back_when_file_is_missing(tmp_path: Path) -> None:
    assert load_spec(str(tmp_path / "nope.json")) == default_spec()


def test_load_spec_falls_back_when_file_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_spec(str(path)) == default_spec()


def test_load_spec_falls_back_when_json_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_spec(str(path)) == default_spec()


@pytest.mark.parametrize("bad_prompt", ["", "   ", None, 42])
def test_load_spec_rejects_an_unusable_prompt(tmp_path: Path, bad_prompt) -> None:
    path = tmp_path / "bad_prompt.json"
    path.write_text(json.dumps({"system_prompt": bad_prompt}), encoding="utf-8")
    assert load_spec(str(path))["system_prompt"] == DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("bad_number", ["abc", None, [1]])
def test_load_spec_rejects_an_unusable_number(tmp_path: Path, bad_number) -> None:
    path = tmp_path / "bad_number.json"
    path.write_text(json.dumps({"max_steps": bad_number}), encoding="utf-8")
    assert load_spec(str(path))["max_steps"] == DEFAULT_MAX_STEPS


def test_load_spec_from_env_reads_harness_agent_spec(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "agent_spec.json"
    path.write_text(json.dumps({"max_steps": 3}), encoding="utf-8")
    monkeypatch.setenv("HARNESS_AGENT_SPEC", str(path))

    assert load_spec_from_env()["max_steps"] == 3


def test_load_spec_from_env_without_the_var_returns_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_AGENT_SPEC", raising=False)
    assert load_spec_from_env() == default_spec()


def test_spec_agent_module_exists_and_is_dependency_clean() -> None:
    """Static check: spec_agent.py must not import api.* (it runs under harbor)."""
    from api.config import REPO_ROOT

    source = (REPO_ROOT / "agent" / "spec_agent.py").read_text(encoding="utf-8")
    assert "class HarnessAgent" in source
    assert "from agent.spec_loader import" in source
    assert "HARNESS_SAVE_TRACE" in source
    assert "import api" not in source
    assert "from api" not in source


# --------------------------------------------------------------------------
# benchmark.py: extra_env
# --------------------------------------------------------------------------


def _fake_subprocess_run(captured: dict):
    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    return fake_run


def test_extra_env_overrides_harness_save_trace(tmp_path: Path, monkeypatch) -> None:
    """split=None forces HARNESS_SAVE_TRACE=0; extra_env is applied last and wins."""
    from benchmark import TerminalBenchRunner

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(captured))

    spec_path = str(tmp_path / "agent_spec.json")
    runner = TerminalBenchRunner(
        agent_model="gpt-4.1-mini",
        split=None,
        env_provider="docker",
        n_concurrent=2,
        jobs_dir=str(tmp_path / "jobs"),
        agent_import_path="agent.spec_agent:HarnessAgent",
        extra_env={"HARNESS_AGENT_SPEC": spec_path, "HARNESS_SAVE_TRACE": "1"},
    )
    runner.run(task_ids=["fix-git"])

    assert captured["env"]["HARNESS_SAVE_TRACE"] == "1"
    assert captured["env"]["HARNESS_AGENT_SPEC"] == spec_path
    assert captured["env"]["AGENT_MODEL"] == "gpt-4.1-mini"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent.spec_agent:HarnessAgent"


def test_without_extra_env_behaviour_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    from benchmark import TerminalBenchRunner

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(captured))

    runner = TerminalBenchRunner(
        agent_model="gpt-4.1-mini",
        split=None,
        env_provider="docker",
        n_concurrent=2,
        jobs_dir=str(tmp_path / "jobs"),
    )
    runner.run(task_ids=["fix-git"])

    assert captured["env"]["HARNESS_SAVE_TRACE"] == "0"
    assert "HARNESS_AGENT_SPEC" not in captured["env"]
    cmd = captured["cmd"]
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent.agent:HarnessAgent"


# --------------------------------------------------------------------------
# api/services/runner.py: HarborBenchmarkRunner passthrough
# --------------------------------------------------------------------------


class _FakeRunStore:
    """Minimal duck-typed stand-in for PostgresRunStore (keeps these tests DB-free)."""

    def __init__(self, record) -> None:
        self.record = record
        self.updates: list[dict] = []
        self.task_updates: list[tuple[str, dict]] = []

    def get(self, run_id: str):
        return self.record

    def update(self, run_id: str, **kwargs) -> None:
        self.updates.append(kwargs)

    def set_task(self, run_id: str, task_id: str, **kwargs) -> None:
        self.task_updates.append((task_id, kwargs))


def _run_record(task_ids: list[str]):
    from api.schemas import RunStatus, TaskStatus
    from api.store import RunRecord
    from api.schemas import TaskResult

    return RunRecord(
        run_id="00000000-0000-0000-0000-000000000001",
        status=RunStatus.running,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=None,
        task_ids=task_ids,
        agent_model="gpt-4.1-mini",
        tasks=[TaskResult(task_id=t, status=TaskStatus.queued) for t in task_ids],
    )


def _harbor_runner(store, tmp_path: Path, monkeypatch, **kwargs):
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    hb = runner_mod.HarborBenchmarkRunner(
        store=store,
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        **kwargs,
    )
    monkeypatch.setattr(hb, "check_available", lambda: None)
    return hb


def _capture_terminal_bench_runner(monkeypatch, captured: dict, results: dict):
    import benchmark as benchmark_mod

    class _Recorder:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def run(self, task_ids=None):
            captured["task_ids"] = task_ids
            return results

    monkeypatch.setattr(benchmark_mod, "TerminalBenchRunner", _Recorder)


def test_harbor_runner_passes_agent_import_path_and_extra_env(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}
    _capture_terminal_bench_runner(monkeypatch, captured, {"fix-git": 1.0})
    store = _FakeRunStore(_run_record(["fix-git"]))
    spec_path = str(tmp_path / "workspace" / "runs" / "r1" / "agent_spec.json")

    hb = _harbor_runner(
        store,
        tmp_path,
        monkeypatch,
        agent_import_path="agent.spec_agent:HarnessAgent",
        extra_env={"HARNESS_AGENT_SPEC": spec_path, "HARNESS_SAVE_TRACE": "1"},
    )
    hb.execute_sync("00000000-0000-0000-0000-000000000001")

    assert captured["init"]["agent_import_path"] == "agent.spec_agent:HarnessAgent"
    assert captured["init"]["extra_env"] == {
        "HARNESS_AGENT_SPEC": spec_path,
        "HARNESS_SAVE_TRACE": "1",
    }
    assert captured["task_ids"] == ["fix-git"]


def test_harbor_runner_defaults_keep_the_legacy_agent(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    _capture_terminal_bench_runner(monkeypatch, captured, {"fix-git": 1.0})
    store = _FakeRunStore(_run_record(["fix-git"]))

    hb = _harbor_runner(store, tmp_path, monkeypatch)
    hb.execute_sync("00000000-0000-0000-0000-000000000001")

    assert captured["init"]["agent_import_path"] == "agent.agent:HarnessAgent"
    assert captured["init"]["extra_env"] == {}


def test_check_agent_import_validates_the_spec_agent_file(tmp_path: Path, monkeypatch) -> None:
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    # agent/agent.py is the placeholder — it must NOT be consulted for a job-driven run.
    (agent_dir / "agent.py").write_text(
        "# Placeholder — do not edit this file directly.\n", encoding="utf-8"
    )
    (agent_dir / "spec_agent.py").write_text(
        "class HarnessAgent:\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)

    hb = runner_mod.HarborBenchmarkRunner(
        store=_FakeRunStore(None),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        agent_import_path="agent.spec_agent:HarnessAgent",
    )
    hb._check_agent_import()  # must not raise


def test_check_agent_import_reports_a_missing_spec_agent(tmp_path: Path, monkeypatch) -> None:
    import api.services.runner as runner_mod
    from api.config import BenchmarkConfig

    (tmp_path / "agent").mkdir()
    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)

    hb = runner_mod.HarborBenchmarkRunner(
        store=_FakeRunStore(None),
        config=BenchmarkConfig(
            default_task_ids=["fix-git"],
            default_agent_model="gpt-4.1-mini",
            execution_backend="harbor",
            env_provider="docker",
        ),
        agent_import_path="agent.spec_agent:HarnessAgent",
    )
    with pytest.raises(runner_mod.ExecutionUnavailableError, match="agent/spec_agent.py"):
        hb._check_agent_import()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_spec_agent_runtime.py -v`
Expected: FAIL at collection with
`ModuleNotFoundError: No module named 'agent.spec_loader'`
- [ ] **Step 3: Write the implementation**

**3a. Create `agent/spec_loader.py`** (stdlib only — this is the file the tests import):
```python
# Spec loading for the spec-driven HarnessAgent runtime.
#
# STDLIB ONLY, ON PURPOSE. agent/spec_agent.py imports harbor and litellm at module
# scope, so it can never be imported by a unit test; every decision worth testing
# lives here instead. Never import api.* from this file: harbor spawns the agent
# with only the repo root on PYTHONPATH.
#
# DEFAULT_SYSTEM_PROMPT is a duplicate of api.agent_spec.BASELINE_SYSTEM_PROMPT.
# tests/test_spec_agent_runtime.py asserts the two stay byte-identical.
from __future__ import annotations

import json
import os

DEFAULT_SYSTEM_PROMPT = """\
You are an autonomous terminal agent. You are given a task and a Linux container.
You solve tasks by executing bash commands. Work step by step.

Rules:
- Read the task carefully before acting.
- Explore the environment first to understand what you have.
- Check command output for errors before proceeding.
- Install missing dependencies as needed.
- Verify your solution before finishing.
- When you are done, send a final text message (no tool call) summarizing what you did.
"""

DEFAULT_MAX_STEPS = 80
DEFAULT_MAX_OUTPUT_CHARS = 8000
DEFAULT_EXEC_TIMEOUT_SEC = 120
DEFAULT_MODEL = "gpt-5.4"

SPEC_ENV_VAR = "HARNESS_AGENT_SPEC"

_INT_FIELDS = {
    "max_steps": DEFAULT_MAX_STEPS,
    "max_output_chars": DEFAULT_MAX_OUTPUT_CHARS,
    "exec_timeout_sec": DEFAULT_EXEC_TIMEOUT_SEC,
}


def default_spec() -> dict:
    """The terminal-bench template's behaviour, as a spec dict."""
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "agent_model": os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        "max_steps": DEFAULT_MAX_STEPS,
        "max_output_chars": DEFAULT_MAX_OUTPUT_CHARS,
        "exec_timeout_sec": DEFAULT_EXEC_TIMEOUT_SEC,
    }


def load_spec(path: str | None) -> dict:
    """Return a spec dict: template defaults overlaid with JSON read from ``path``.

    Never raises. An unset, missing, unreadable, malformed or partially invalid
    file degrades to the defaults so the agent stays runnable standalone — a
    broken spec must not turn into a crashed benchmark task.

    Only the five known keys are honoured; anything else in the file is ignored,
    so a spec written by a newer AgentSpec still runs on an older runtime.
    """
    spec = default_spec()
    if not path:
        return spec

    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return spec
    if not isinstance(loaded, dict):
        return spec

    for key in spec:
        if key in loaded and loaded[key] is not None:
            spec[key] = loaded[key]

    prompt = spec["system_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        spec["system_prompt"] = DEFAULT_SYSTEM_PROMPT

    model = spec["agent_model"]
    if not isinstance(model, str) or not model.strip():
        spec["agent_model"] = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    for key, fallback in _INT_FIELDS.items():
        try:
            spec[key] = int(spec[key])
        except (TypeError, ValueError):
            spec[key] = fallback

    return spec


def load_spec_from_env() -> dict:
    """Load the spec named by $HARNESS_AGENT_SPEC (defaults when unset)."""
    return load_spec(os.environ.get(SPEC_ENV_VAR))
```

**3b. Create `agent/spec_agent.py`** (complete file):
```python
# Spec-driven HarnessAgent for Terminal-Bench 2.0 — the service's agent runtime.
#
# Reads its system prompt and limits from the AgentSpec JSON named by
# $HARNESS_AGENT_SPEC, falling back to the template defaults when that is unset or
# unreadable (so it is runnable standalone). Fixed bash tool, same trace saving,
# same token accounting as agent/templates/terminal_bench.py.
#
# This file is NOT agent/agent.py: that file is what the Layer A coding agent edits
# and what prepare.py overwrites from templates. The service never touches it.
# Dependencies are stdlib + litellm + harbor only — no api.* imports, because harbor
# spawns the agent with just the repo root on PYTHONPATH.
import json
import os

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.spec_loader import load_spec_from_env

MODEL = os.environ.get("AGENT_MODEL", "gpt-5.4")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the container. Returns stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _truncate(text: str, limit: int) -> str:
    """Truncate long output, keeping the beginning and end."""
    if not text or len(text) <= limit:
        return text or ""
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - limit} chars truncated] ...\n\n"
        + text[-half:]
    )


class HarnessAgent(BaseAgent):
    """Agent under optimization, configured entirely by its AgentSpec."""

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        spec = load_spec_from_env()
        system_prompt = spec["system_prompt"]
        max_steps = spec["max_steps"]
        max_output_chars = spec["max_output_chars"]
        exec_timeout_sec = spec["exec_timeout_sec"]

        model = self.model_name or spec["agent_model"] or MODEL
        self.logger.info(
            f"spec: model={model} max_steps={max_steps} "
            f"max_output_chars={max_output_chars} exec_timeout_sec={exec_timeout_sec} "
            f"prompt_chars={len(system_prompt)}"
        )

        total_input_tokens = 0
        total_output_tokens = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task:\n{instruction}"},
        ]

        for step in range(max_steps):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                self.logger.error(f"LLM call failed at step {step}: {e}")
                break

            usage = response.usage
            if usage:
                total_input_tokens += usage.prompt_tokens or 0
                total_output_tokens += usage.completion_tokens or 0

            choice = response.choices[0]
            message = choice.message

            # Build the assistant message for history
            assistant_msg = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # If the model returned text without tool calls → task complete
            if not message.tool_calls:
                self.logger.info(f"Agent declared complete at step {step}")
                break

            # Execute each tool call
            for tc in message.tool_calls:
                if tc.function.name != "bash":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.function.name}",
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Error: invalid JSON arguments",
                    })
                    continue

                command = args.get("command", "")
                self.logger.info(f"Step {step} | bash: {command[:200]}")

                result = await environment.exec(command, timeout_sec=exec_timeout_sec)

                output_parts = []
                if result.stdout:
                    output_parts.append(result.stdout)
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                if result.return_code != 0:
                    output_parts.append(f"[exit code: {result.return_code}]")

                output = "\n".join(output_parts) if output_parts else "(no output)"
                output = _truncate(output, max_output_chars)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        # Save full conversation trace for failure analysis (the improver reads these)
        if os.environ.get("HARNESS_SAVE_TRACE", "1") == "1":
            trace_path = self.logs_dir / "trace.json"
            try:
                with open(trace_path, "w") as f:
                    json.dump(messages, f, indent=2, default=str)
                self.logger.info(f"Trace saved to {trace_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save trace: {e}")

        # Populate context
        context.n_input_tokens = total_input_tokens
        context.n_output_tokens = total_output_tokens
```

**3c. Modify `benchmark.py` — two edits, nothing else.**

Edit 1: add exactly one parameter to `TerminalBenchRunner.__init__` (`benchmark.py:144-164`).
`agent_import_path` is already there at line 151 — leave it alone. Replace the
signature's last line and the assignment block so it reads:
```python
    def __init__(
        self,
        agent_model: str | None = None,
        split: str | None = "train",
        env_provider: str = "e2b",
        n_concurrent: int = 50,
        dataset: str = "terminal-bench@2.0",
        agent_import_path: str = "agent.agent:HarnessAgent",
        per_task_timeout: int = 1200,
        jobs_dir: str = "workspace/tbench_jobs",
        reasoning_effort: str | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        self.agent_model = agent_model or os.getenv("AGENT_MODEL", "gpt-5.4")
        self.split = split
        self.env_provider = env_provider
        self.n_concurrent = n_concurrent
        self.dataset = dataset
        self.agent_import_path = agent_import_path
        self.per_task_timeout = per_task_timeout
        self.jobs_dir = jobs_dir
        self.reasoning_effort = reasoning_effort
        # Extra env vars for the `harbor run` subprocess, applied LAST in run() so a
        # caller can override anything we set (notably HARNESS_SAVE_TRACE).
        self.extra_env = dict(extra_env or {})
```

Edit 2: in `run()`, immediately after the existing `HARNESS_SAVE_TRACE` block
(`benchmark.py:226-230`) and before the `# Subprocess timeout` comment, insert:
```python
        # Applied last, deliberately: split=None (the API/job path) forces
        # HARNESS_SAVE_TRACE="0" above, and the iterative loop needs traces back on.
        env.update(self.extra_env)
```
So that region becomes:
```python
        # Disable trace saving for test/baseline runs (prevent coding agent from reading test traces).
        # split=None means the baseline all-tasks run; the train/test split doesn't exist yet so
        # we can't know which tasks are test tasks — safest to save nothing.
        if self.split != "train":
            env["HARNESS_SAVE_TRACE"] = "0"

        # Applied last, deliberately: split=None (the API/job path) forces
        # HARNESS_SAVE_TRACE="0" above, and the iterative loop needs traces back on.
        env.update(self.extra_env)

        # Subprocess timeout: generous for full dataset, computed for splits
```

**3d. Modify `api/services/runner.py` — three edits.**

Edit 1: add a module constant next to `logger` (after `api/services/runner.py:17`):
```python
DEFAULT_AGENT_IMPORT_PATH = "agent.agent:HarnessAgent"
```

Edit 2: replace `HarborBenchmarkRunner.__init__` and `_check_agent_import`
(`api/services/runner.py:148-183`) with:
```python
    def __init__(
        self,
        store: PostgresRunStore,
        *,
        config: BenchmarkConfig | None = None,
        agent_import_path: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.config = config or load_config()
        # Layer B (jobs) passes "agent.spec_agent:HarnessAgent"; /v1/runs keeps agent/agent.py.
        self.agent_import_path = agent_import_path or DEFAULT_AGENT_IMPORT_PATH
        self.extra_env = dict(extra_env or {})

    def check_available(self) -> None:
        if shutil.which("harbor") is None:
            raise ExecutionUnavailableError(
                "harbor CLI not found on PATH (install with: uv tool install harbor)"
            )

        self._check_agent_import()
        self._check_env_provider()

    def _agent_module_relpath(self) -> str:
        """"agent.spec_agent:HarnessAgent" -> "agent/spec_agent.py"."""
        module = self.agent_import_path.split(":", 1)[0]
        return "/".join(module.split(".")) + ".py"

    def _check_agent_import(self) -> None:
        rel = self._agent_module_relpath()
        agent_path = REPO_ROOT / rel
        if rel == "agent/agent.py":
            hint = "Copy agent/templates/terminal_bench.py to agent/agent.py."
        else:
            hint = f"{rel} ships with the repo — restore it from git."

        if not agent_path.is_file():
            raise ExecutionUnavailableError(f"{rel} is missing. {hint}")
        source = agent_path.read_text(encoding="utf-8")
        if "Placeholder — do not edit" in source:
            raise ExecutionUnavailableError(f"{rel} is still the placeholder. {hint}")
        if "class HarnessAgent" not in source:
            raise ExecutionUnavailableError(f"{rel} has no HarnessAgent class. {hint}")
```
(The existing `tests/test_reward_mapping.py::test_harbor_runner_rejects_placeholder_agent`
matches on `"placeholder"`, which these messages preserve.)

Edit 3: pass both through when building the runner
(`api/services/runner.py:245-253`):
```python
            tbr = TerminalBenchRunner(
                agent_model=record.request.agent_model,
                split=None,  # do not require tbench_data/task_split.json
                env_provider=self.config.env_provider,
                n_concurrent=self.config.max_concurrency,
                dataset=self.config.dataset,
                per_task_timeout=self.config.per_task_timeout,
                jobs_dir=jobs_dir,
                agent_import_path=self.agent_import_path,
                extra_env=self.extra_env,
            )
```

Leave `create_runner` unchanged: it serves `/v1/runs`, whose behaviour must not move.
Task 10's `StepExecutor` constructs `HarborBenchmarkRunner` directly with the two new
keyword arguments.
- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_spec_agent_runtime.py tests/test_reward_mapping.py tests/test_agent_spec.py -v`
Expected: PASS (the existing reward-mapping suite is included because Edit 2 rewrites
`_check_agent_import`, which it covers).
- [ ] **Step 5: Commit**
```bash
git add agent/spec_loader.py agent/spec_agent.py benchmark.py api/services/runner.py tests/test_spec_agent_runtime.py
git commit -m "feat: add spec-driven agent runtime and harbor extra_env plumbing"
```

---

**Additions beyond CONTRACT.md made in this section** (flagged as the contract requires):

- `api/config.py`: private helpers `_positive_int(raw, key, default)` and
  `_unit_fraction(raw, key, default)`; the new numeric fields deliberately do not use the
  `raw.get(...) or default` idiom (it cannot reject `0` and it rewrites a legal
  `min_delta: 0.0`).
- `api/services/artifacts.py`: private `_validate_key` / `_validate_prefix`; unsafe keys
  raise plain `ValueError` (the contract names no exception type).
- `agent/spec_loader.py` is a **new module not named in the contract**. It exists because
  `agent/spec_agent.py` imports harbor and litellm at module scope and neither is
  installed in the test environment, so `load_spec` would otherwise be untestable. It
  exports `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_MAX_STEPS`, `DEFAULT_MAX_OUTPUT_CHARS`,
  `DEFAULT_EXEC_TIMEOUT_SEC`, `DEFAULT_MODEL`, `SPEC_ENV_VAR`, `default_spec()`,
  `load_spec(path)` and `load_spec_from_env()`.
- `api/services/runner.py`: module constant `DEFAULT_AGENT_IMPORT_PATH` and private
  `HarborBenchmarkRunner._agent_module_relpath()`.

---

## Section B — Data Model, Scoring & Step Queue (Tasks 5-7)

These three tasks add the persistence layer for the iterative loop: the three new
tables, the pure scoring/stopping functions, and the transactional step queue that
workers claim from. Nothing here imports from `worker/*` or `api/routes/*`.

**Prerequisites from earlier sections:** Task 1 (`BenchmarkConfig` new fields) and
Task 2 (`api/agent_spec.py`) must be merged before Task 6/7 tests can run. Task 5 has
no prerequisites.

**Environment:** all Postgres tests need `docker compose up -d postgres`. Run them with
`DATABASE_URL=postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness` (that is also
the in-test default). Without Postgres the tests skip rather than fail.

---

### Task 5: ORM models for jobs, agent versions and steps

**Files:**
- Modify: `api/models.py:8-10` (imports: add `UniqueConstraint`, add `JSONB`)
- Modify: `api/models.py:57` (append three new model classes after `RunTaskRow`)
- Test: `tests/test_job_models.py`

**Interfaces:**
- Consumes: `api.models.Base` (already defined at `api/models.py:13-14`),
  `api.db.init_db(*, url: str | None = None) -> None`,
  `api.db.get_engine(*, url: str | None = None, force_new: bool = False) -> Engine`,
  `api.db.get_session_factory(*, url: str | None = None) -> sessionmaker[Session]`,
  `api.db.reset_engine() -> None`.
- Produces:
  - `class JobRow(Base)` — `__tablename__ = "jobs"`; columns `id, status, task_ids,
    agent_model, improver_model, max_iterations, patience, min_delta,
    max_job_duration_sec, evaluate_stale_after_sec, current_iteration,
    non_improving_streak, best_agent_version_id, best_score, stop_reason, created_at,
    started_at, finished_at, error_code, error_message`.
  - `class AgentVersionRow(Base)` — `__tablename__ = "agent_versions"`; columns
    `id, job_id, version, parent_version_id, spec, rationale, created_by, created_at`;
    `UniqueConstraint("job_id", "version", name="uq_agent_versions_job_version")`.
  - `class StepRow(Base)` — `__tablename__ = "steps"`; columns `id, job_id, type,
    status, iteration, agent_version_id, run_id, score, stale_after_sec, worker_id,
    claimed_at, created_at, started_at, finished_at, error_code, error_message`.
  - Task 7 (`api/job_store.py`) imports all three.

**Design notes (read before writing code):**

1. `best_agent_version_id` on `JobRow` and `parent_version_id` on `AgentVersionRow` are
   **plain `UUID` columns with no `ForeignKey`**. A real FK `jobs.best_agent_version_id
   → agent_versions.id` would form a cycle with `agent_versions.job_id → jobs.id`, and
   `create_all` cannot order a cyclic pair without `use_alter`; more importantly the
   cycle breaks `ON DELETE CASCADE` deletion of a job. The contract lists both as bare
   "UUID nullable" for exactly this reason.
2. `steps.agent_version_id` and `steps.run_id` are also plain `UUID` columns (per the
   contract). Keeping `run_id` FK-free means deleting/clearing `runs` never blocks on
   `steps`, which matters because `PostgresRunStore.clear()` is called by every existing
   test fixture.
3. `job_id` on both child tables **does** get `ForeignKey("jobs.id",
   ondelete="CASCADE")` plus `index=True` — that is what makes the cascade test pass.
4. Do **not** add `relationship()` attributes. `RunRow.tasks` has one, but a
   relationship on `JobRow` would make SQLAlchemy cascade deletes in Python (loading
   every child) and would change the flush ordering semantics that Task 7 relies on.
   The DB-level `ON DELETE CASCADE` is the cascade.
5. `task_ids` and `spec` are `JSONB` (`from sqlalchemy.dialects.postgresql import
   JSONB`). Type them as `Mapped[list[str]]` and `Mapped[dict]` respectively — verified
   working with SQLAlchemy 2.0.52 + psycopg 3.
6. `created_at` uses `server_default=func.now()` like `RunRow.created_at`, but Task 7
   always passes an explicit Python `_utcnow()` value so ordering is stable inside one
   transaction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_models.py`:

```python
"""ORM-level tests for the Milestone 4 jobs / agent_versions / steps tables."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, func, inspect, select
from sqlalchemy.exc import IntegrityError, OperationalError

from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.models import AgentVersionRow, JobRow, StepRow

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)

SPEC = {
    "system_prompt": "You are a terminal agent.",
    "agent_model": "gpt-4.1-mini",
    "max_steps": 80,
    "max_output_chars": 8000,
    "exec_timeout_sec": 120,
}


def _truncate(factory) -> None:
    session = factory()
    try:
        session.execute(delete(StepRow))
        session.execute(delete(AgentVersionRow))
        session.execute(delete(JobRow))
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def factory():
    # DATABASE_URL must be in the environment BEFORE get_session_factory(): db.py only
    # caches the global engine when called with url=None, which reads the env var.
    os.environ["DATABASE_URL"] = DATABASE_URL
    reset_engine()
    init_db(url=DATABASE_URL)
    session_factory = get_session_factory()
    _truncate(session_factory)
    yield session_factory
    _truncate(session_factory)
    reset_engine()


def _new_job_row(job_id: uuid.UUID, now: datetime) -> JobRow:
    return JobRow(
        id=job_id,
        status="queued",
        task_ids=["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=5,
        patience=2,
        min_delta=0.01,
        max_job_duration_sec=21600,
        evaluate_stale_after_sec=3600,
        current_iteration=0,
        non_improving_streak=0,
        created_at=now,
    )


def test_init_db_creates_the_three_job_tables(factory) -> None:
    inspector = inspect(get_engine())
    names = set(inspector.get_table_names())
    assert {"jobs", "agent_versions", "steps"} <= names

    job_columns = {c["name"] for c in inspector.get_columns("jobs")}
    assert "evaluate_stale_after_sec" in job_columns
    assert "max_job_duration_sec" in job_columns
    assert "non_improving_streak" in job_columns
    assert "best_agent_version_id" in job_columns

    step_columns = {c["name"] for c in inspector.get_columns("steps")}
    assert {"type", "iteration", "agent_version_id", "stale_after_sec"} <= step_columns


def test_deleting_a_job_cascades_versions_and_steps(factory) -> None:
    job_id = uuid.uuid4()
    version_id = uuid.uuid4()
    step_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    session = factory()
    try:
        session.add(_new_job_row(job_id, now))
        session.flush()
        session.add(
            AgentVersionRow(
                id=version_id,
                job_id=job_id,
                version=0,
                parent_version_id=None,
                spec=SPEC,
                rationale="baseline",
                created_by="baseline",
                created_at=now,
            )
        )
        session.flush()
        session.add(
            StepRow(
                id=step_id,
                job_id=job_id,
                type="evaluate",
                status="queued",
                iteration=0,
                agent_version_id=version_id,
                stale_after_sec=3600,
                created_at=now,
            )
        )
        session.commit()

        assert session.scalar(select(func.count()).select_from(AgentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(StepRow)) == 1

        # Bulk DELETE so the database (not the ORM) performs the cascade.
        session.execute(delete(JobRow).where(JobRow.id == job_id))
        session.commit()

        assert session.scalar(select(func.count()).select_from(JobRow)) == 0
        assert session.scalar(select(func.count()).select_from(AgentVersionRow)) == 0
        assert session.scalar(select(func.count()).select_from(StepRow)) == 0
    finally:
        session.close()


def test_agent_version_number_is_unique_per_job(factory) -> None:
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    session = factory()
    try:
        session.add(_new_job_row(job_id, now))
        session.flush()
        session.add(
            AgentVersionRow(
                id=uuid.uuid4(),
                job_id=job_id,
                version=0,
                spec=SPEC,
                rationale="baseline",
                created_by="baseline",
                created_at=now,
            )
        )
        session.commit()

        session.add(
            AgentVersionRow(
                id=uuid.uuid4(),
                job_id=job_id,
                version=0,
                spec=SPEC,
                rationale="duplicate",
                created_by="improver",
                created_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    finally:
        session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_models.py -v`

Expected: FAIL at collection with
`ImportError: cannot import name 'AgentVersionRow' from 'api.models'`.

- [ ] **Step 3: Write the implementation**

Replace the import block at `api/models.py:8-10`:

```python
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
```

Append to the end of `api/models.py` (after `RunTaskRow`, line 57):

```python
class JobRow(Base):
    """An iterative-improvement job (Milestone 4)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    task_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    agent_model: Mapped[str] = mapped_column(String(256), nullable=False)
    improver_model: Mapped[str] = mapped_column(String(256), nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    patience: Mapped[int] = mapped_column(Integer, nullable=False)
    min_delta: Mapped[float] = mapped_column(Float, nullable=False)
    max_job_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluate_stale_after_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    current_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    non_improving_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Deliberately NOT a ForeignKey: jobs -> agent_versions -> jobs would be a cycle
    # and would break ON DELETE CASCADE for the job.
    best_agent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    best_score: Mapped[float | None] = mapped_column(Float)
    stop_reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class AgentVersionRow(Base):
    """Immutable snapshot of an AgentSpec for one job iteration."""

    __tablename__ = "agent_versions"
    __table_args__ = (
        UniqueConstraint("job_id", "version", name="uq_agent_versions_job_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    spec: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_by: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StepRow(Base):
    """A unit of queued work for a job: an evaluate step or an improve step."""

    __tablename__ = "steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    score: Mapped[float | None] = mapped_column(Float)
    stale_after_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_models.py tests/test_api.py -v`

Expected: PASS (3 new tests pass; the 10 existing `test_api.py` tests still pass —
`create_all` is additive and `runs`/`run_tasks` are untouched).

- [ ] **Step 5: Commit**

```bash
git add api/models.py tests/test_job_models.py
git commit -m "feat: add jobs, agent_versions and steps ORM models"
```

---

### Task 6: Scoring and the stopping rule

**Files:**
- Create: `api/services/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: nothing from this repo (stdlib only — deliberately import-free so it stays
  unit-testable without a DB or config).
- Produces:
  - `def mean_reward(rewards: Iterable[float | None]) -> float` — `None` counts as
    `0.0`; empty iterable returns `0.0`.
  - `@dataclass(frozen=True) class StopDecision` with fields `improved: bool`,
    `should_stop: bool`, `stop_reason: str | None`, `non_improving_streak: int`.
  - ```python
    def compute_stop(
        *,
        iteration: int,
        score: float,
        best_score: float | None,
        prior_non_improving_streak: int,
        max_iterations: int,
        patience: int,
        min_delta: float,
        elapsed_sec: float,
        max_job_duration_sec: int,
    ) -> StopDecision
    ```
  - Task 7 (`api/job_store.PostgresJobStore._advance_evaluate`) calls `compute_stop`;
    Task 10 (`worker/steps.StepExecutor`) calls `mean_reward`.

**Semantics that the tests pin down:**
- `improved = best_score is None or score > best_score + min_delta` — strict `>`, so a
  score exactly equal to `best_score + min_delta` is **not** an improvement.
- `non_improving_streak = 0` if improved else `prior_non_improving_streak + 1`.
- Stop precedence, first match wins: `max_iterations` (`iteration + 1 >=
  max_iterations`) → `no_improvement` (`streak >= patience`) → `budget_exceeded`
  (`elapsed_sec > max_job_duration_sec`) → no stop. The precedence is observable: a
  final iteration that also exhausted patience reports `max_iterations`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
"""Pure unit tests for scoring and the stopping rule (no DB, no config)."""

from __future__ import annotations

import pytest

from api.services.scoring import StopDecision, compute_stop, mean_reward


def test_mean_reward_empty_is_zero() -> None:
    assert mean_reward([]) == 0.0


def test_mean_reward_all_none_is_zero() -> None:
    assert mean_reward([None, None]) == 0.0


def test_mean_reward_counts_none_as_zero() -> None:
    # 1.0 + 0.0 + 0.5 + 0.0 over 4 tasks
    assert mean_reward([1.0, None, 0.5, 0.0]) == pytest.approx(0.375)


def test_mean_reward_all_passing() -> None:
    assert mean_reward([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_mean_reward_accepts_ints_and_generators() -> None:
    assert mean_reward(r for r in [1, 0, None]) == pytest.approx(1 / 3)


BASE = {
    "iteration": 0,
    "score": 0.5,
    "best_score": None,
    "prior_non_improving_streak": 0,
    "max_iterations": 5,
    "patience": 2,
    "min_delta": 0.01,
    "elapsed_sec": 10.0,
    "max_job_duration_sec": 21600,
}


@pytest.mark.parametrize(
    "name,overrides,expected",
    [
        (
            "first_iteration_always_improves",
            {"iteration": 0, "score": 0.0, "best_score": None},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "improvement_resets_streak",
            {"iteration": 2, "score": 0.60, "best_score": 0.50,
             "prior_non_improving_streak": 1},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "non_improvement_increments_streak",
            {"iteration": 1, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 0},
            StopDecision(
                improved=False, should_stop=False, stop_reason=None, non_improving_streak=1
            ),
        ),
        (
            "patience_reached_stops_with_no_improvement",
            {"iteration": 2, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 1, "patience": 2},
            StopDecision(
                improved=False,
                should_stop=True,
                stop_reason="no_improvement",
                non_improving_streak=2,
            ),
        ),
        (
            "max_iterations_wins_over_no_improvement",
            {"iteration": 4, "max_iterations": 5, "score": 0.40, "best_score": 0.50,
             "prior_non_improving_streak": 1, "patience": 2},
            StopDecision(
                improved=False,
                should_stop=True,
                stop_reason="max_iterations",
                non_improving_streak=2,
            ),
        ),
        (
            "budget_exceeded_when_no_other_rule_fires",
            {"iteration": 1, "score": 0.70, "best_score": 0.50,
             "elapsed_sec": 100.0, "max_job_duration_sec": 60},
            StopDecision(
                improved=True,
                should_stop=True,
                stop_reason="budget_exceeded",
                non_improving_streak=0,
            ),
        ),
        (
            "budget_not_exceeded_at_exact_limit",
            {"iteration": 1, "score": 0.70, "best_score": 0.50,
             "elapsed_sec": 60.0, "max_job_duration_sec": 60},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "min_delta_boundary_is_not_an_improvement",
            # 0.5 + 0.01 == 0.51 exactly in IEEE754, and 0.51 > 0.51 is False.
            {"iteration": 1, "score": 0.51, "best_score": 0.50, "min_delta": 0.01},
            StopDecision(
                improved=False, should_stop=False, stop_reason=None, non_improving_streak=1
            ),
        ),
        (
            "just_past_min_delta_is_an_improvement",
            {"iteration": 1, "score": 0.52, "best_score": 0.50, "min_delta": 0.01},
            StopDecision(
                improved=True, should_stop=False, stop_reason=None, non_improving_streak=0
            ),
        ),
        (
            "single_iteration_job_stops_immediately",
            {"iteration": 0, "max_iterations": 1, "score": 0.3, "best_score": None},
            StopDecision(
                improved=True,
                should_stop=True,
                stop_reason="max_iterations",
                non_improving_streak=0,
            ),
        ),
    ],
)
def test_compute_stop_table(name: str, overrides: dict, expected: StopDecision) -> None:
    kwargs = {**BASE, **overrides}
    assert compute_stop(**kwargs) == expected, name


def test_stop_decision_is_frozen() -> None:
    decision = compute_stop(**BASE)
    with pytest.raises(Exception):
        decision.improved = False  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`

Expected: FAIL at collection with
`ModuleNotFoundError: No module named 'api.services.scoring'`.

- [ ] **Step 3: Write the implementation**

Create `api/services/scoring.py`:

```python
"""Scoring and stopping rules for the iterative improvement loop.

Deliberately dependency-free (stdlib only) so it can be unit-tested without a
database, config file or LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

STOP_MAX_ITERATIONS = "max_iterations"
STOP_NO_IMPROVEMENT = "no_improvement"
STOP_BUDGET_EXCEEDED = "budget_exceeded"


def mean_reward(rewards: Iterable[float | None]) -> float:
    """Mean reward across a run's tasks; ``None`` (error/timeout) counts as 0.0.

    An empty iterable scores 0.0 rather than raising, so a run that produced no
    task rows is simply the worst possible score.
    """
    values = [0.0 if reward is None else float(reward) for reward in rewards]
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass(frozen=True)
class StopDecision:
    """Outcome of the stopping check performed after an evaluate step."""

    improved: bool
    should_stop: bool
    stop_reason: str | None
    non_improving_streak: int


def compute_stop(
    *,
    iteration: int,
    score: float,
    best_score: float | None,
    prior_non_improving_streak: int,
    max_iterations: int,
    patience: int,
    min_delta: float,
    elapsed_sec: float,
    max_job_duration_sec: int,
) -> StopDecision:
    """Decide whether the job improved and whether the loop should stop.

    ``improved`` requires a strict gain of more than ``min_delta`` over the best
    score so far; the first evaluation (``best_score is None``) always improves.

    Stop precedence, first match wins:
      1. ``max_iterations``  — this was the last allowed iteration.
      2. ``no_improvement``  — the non-improving streak reached ``patience``.
      3. ``budget_exceeded`` — wall-clock since job start passed the budget.
    """
    improved = best_score is None or score > best_score + min_delta
    streak = 0 if improved else prior_non_improving_streak + 1

    stop_reason: str | None = None
    if iteration + 1 >= max_iterations:
        stop_reason = STOP_MAX_ITERATIONS
    elif streak >= patience:
        stop_reason = STOP_NO_IMPROVEMENT
    elif elapsed_sec > max_job_duration_sec:
        stop_reason = STOP_BUDGET_EXCEEDED

    return StopDecision(
        improved=improved,
        should_stop=stop_reason is not None,
        stop_reason=stop_reason,
        non_improving_streak=streak,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`

Expected: PASS (15 tests: 5 `mean_reward`, 10 parametrized `compute_stop`, plus the
frozen check).

- [ ] **Step 5: Commit**

```bash
git add api/services/scoring.py tests/test_scoring.py
git commit -m "feat: add mean_reward and compute_stop scoring rules"
```

---

### Task 7: PostgresJobStore — the transactional step queue

**Files:**
- Create: `api/job_store.py`
- Test: `tests/test_job_store.py`

**Interfaces:**
- Consumes:
  - `api.agent_spec.AgentSpec` (Pydantic v2 model with fields `system_prompt: str`,
    `agent_model: str`, `max_steps: int`, `max_output_chars: int`,
    `exec_timeout_sec: int`; `model_config = ConfigDict(extra="forbid")`)
  - `api.agent_spec.baseline_spec(agent_model: str) -> AgentSpec`
  - `api.agent_spec.changed_fields(old: AgentSpec, new: AgentSpec) -> list[str]`
    (sorted field names that differ)
  - `api.services.scoring.compute_stop(*, iteration: int, score: float,
    best_score: float | None, prior_non_improving_streak: int, max_iterations: int,
    patience: int, min_delta: float, elapsed_sec: float,
    max_job_duration_sec: int) -> StopDecision` (Task 6)
  - `api.models.JobRow`, `api.models.AgentVersionRow`, `api.models.StepRow` (Task 5)
  - `api.db.get_session_factory(*, url: str | None = None) -> sessionmaker[Session]`
  - `api.store._utcnow() -> datetime` (the repo's timestamp helper)
  - `api.schemas.RunStatus` (str enum: `queued|running|completed|failed|cancelled`) —
    reused verbatim for both job status and step status
- Produces (all imported by later tasks):
  - `StepRecord`, `AgentVersionRecord`, `IterationRecord`, `JobRecord`,
    `EvaluateOutcome`, `ImproveOutcome` — frozen dataclasses exactly as in the contract
  - `STEP_EVALUATE = "evaluate"`, `STEP_IMPROVE = "improve"`,
    `IMPROVE_STALE_AFTER_SEC = 1800`
  - ```python
    class PostgresJobStore:
        def __init__(self, session_factory=None) -> None
        def create_job(self, *, task_ids: list[str], agent_model: str,
                       improver_model: str, max_iterations: int, patience: int,
                       min_delta: float, max_job_duration_sec: int,
                       evaluate_stale_after_sec: int) -> JobRecord
        def get_job(self, job_id: str) -> JobRecord | None
        def get_agent_version(self, version_id: str) -> AgentVersionRecord | None
        def claim_next_step(self, worker_id: str) -> StepRecord | None
        def complete_step_and_advance(self, step_id: str,
                                     outcome: EvaluateOutcome | ImproveOutcome) -> None
        def fail_step(self, step_id: str, *, error_code: str,
                      error_message: str) -> None
        def clear(self) -> None
    ```
  - `job_store = PostgresJobStore()` — module-level default instance, mirroring
    `api/store.py:358` (`store = PostgresRunStore()`), for `api/main.create_app` to
    fall back on (Task 12).
  - Task 8/9 (`api/services/improver.py`) imports `IterationRecord`; Task 10
    (`worker/steps.py`) imports `PostgresJobStore`, `StepRecord`, `EvaluateOutcome`,
    `ImproveOutcome`; Tasks 12-13 (routes) import `PostgresJobStore`, `JobRecord`,
    `AgentVersionRecord`.

**Three gotchas verified against Postgres 16 / SQLAlchemy 2.0.52 / psycopg 3 before
writing this plan — do not skip them:**

1. **Insert order inside `create_job` must be forced with `flush()`.** With no
   `relationship()` between `JobRow` and its children, SQLAlchemy's unit of work
   orders mappers by sort key (roughly `module.ClassName`), so
   `api.models.AgentVersionRow` flushes **before** `api.models.JobRow` and you get
   `ForeignKeyViolation: Key (job_id)=(...) is not present in table "jobs"`. Fix:
   `session.add(job); session.flush(); session.add(version); session.flush();
   session.add(step); session.commit()`. `flush()` does not commit — this is still one
   transaction, as the contract requires.
2. **Per-row staleness needs a SQL-side interval.** `timedelta` cannot be used because
   the threshold is a column. The predicate is
   `steps.claimed_at < now() - make_interval(secs => steps.stale_after_sec)`, expressed
   as a `text()` clause dropped into `update(StepRow).where(...)`. Verified rendering:
   `UPDATE steps SET status=%(status)s::VARCHAR, ... WHERE steps.status = %(status_1)s
   ::VARCHAR AND steps.claimed_at IS NOT NULL AND steps.claimed_at < now() -
   make_interval(secs => steps.stale_after_sec)` — the `text()` fragment must spell the
   table name `steps.` because the bulk UPDATE does not alias it. (The typed
   alternative `StepRow.claimed_at < func.now() - func.make_interval(0, 0, 0, 0, 0, 0,
   StepRow.stale_after_sec)` also executes correctly, but the `secs =>` form is what
   the contract specifies and reads better.)
3. **Lock ordering is step-then-job everywhere.** `claim_next_step`,
   `complete_step_and_advance` and `fail_step` all lock the `steps` row first and the
   `jobs` row second. Keep that order in any new method or two workers racing on the
   same job can deadlock.

Additional decisions this task locks in:
- `EvaluateOutcome.score is None` with no `error_code` is stored as `0.0` (consistent
  with `mean_reward` treating `None` as zero); a `None` score therefore never reaches
  `compute_stop`.
- The next version number is `max(agent_versions.version) + 1` for the job, read inside
  the same transaction, rather than `step.iteration + 1` — the unique constraint then
  can never be violated even if a step were retried.
- `elapsed_sec` is measured from `job.started_at` (falling back to `job.created_at`).
- `fail_step` is the worker's "unexpected exception" path: it fails the step and fails
  the job with the same envelope, but leaves an already-terminal job untouched.

#### Cycle 1 — `create_job`, `get_job`, `get_agent_version`, `clear`

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_store.py`:

```python
"""PostgresJobStore tests: queue claiming, transitions and history."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from api.agent_spec import baseline_spec
from api.config import clear_config_cache
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import (
    EvaluateOutcome,
    ImproveOutcome,
    PostgresJobStore,
)
from api.models import AgentVersionRow, JobRow, StepRow

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)


@pytest.fixture()
def job_store() -> PostgresJobStore:
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    store = PostgresJobStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


def _create_job(
    store: PostgresJobStore,
    *,
    task_ids: list[str] | None = None,
    max_iterations: int = 5,
    patience: int = 2,
    min_delta: float = 0.01,
    max_job_duration_sec: int = 21600,
    evaluate_stale_after_sec: int = 3600,
):
    return store.create_job(
        task_ids=task_ids or ["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=max_iterations,
        patience=patience,
        min_delta=min_delta,
        max_job_duration_sec=max_job_duration_sec,
        evaluate_stale_after_sec=evaluate_stale_after_sec,
    )


def test_create_job_inserts_v0_and_one_queued_evaluate_step(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store)

    assert job.status == "queued"
    assert job.task_ids == ["fix-git", "regex-log"]
    assert job.agent_model == "gpt-4.1-mini"
    assert job.improver_model == "gpt-5.4"
    assert job.current_iteration == 0
    assert job.best_agent_version_id is None
    assert job.best_score is None
    assert job.stop_reason is None
    assert job.started_at is None and job.finished_at is None

    # Exactly one iteration record, from the queued evaluate step.
    assert len(job.iterations) == 1
    it0 = job.iterations[0]
    assert it0.iteration == 0
    assert it0.version == 0
    assert it0.status == "queued"
    assert it0.score is None
    assert it0.improved is None
    assert it0.rationale is None
    assert it0.changed_fields == []

    # v0 spec is the baseline spec for the requested model.
    version = job_store.get_agent_version(it0.agent_version_id)
    assert version is not None
    assert version.version == 0
    assert version.parent_version_id is None
    assert version.created_by == "baseline"
    assert version.rationale == "baseline"
    assert version.spec == baseline_spec("gpt-4.1-mini")

    # Row-level shape: one job, one version, one queued evaluate step.
    session = get_session_factory()()
    try:
        assert len(list(session.scalars(select(JobRow)))) == 1
        assert len(list(session.scalars(select(AgentVersionRow)))) == 1
        steps = list(session.scalars(select(StepRow)))
        assert len(steps) == 1
        assert steps[0].type == "evaluate"
        assert steps[0].status == "queued"
        assert steps[0].iteration == 0
        assert steps[0].stale_after_sec == 3600
        assert steps[0].run_id is None
    finally:
        session.close()


def test_get_job_returns_none_for_unknown_or_malformed_id(
    job_store: PostgresJobStore,
) -> None:
    assert job_store.get_job(str(uuid.uuid4())) is None
    assert job_store.get_job("not-a-uuid") is None
    assert job_store.get_agent_version("not-a-uuid") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_store.py -v`

Expected: FAIL at collection with `ModuleNotFoundError: No module named 'api.job_store'`.

- [ ] **Step 3: Write the implementation**

Create `api/job_store.py`:

```python
"""Postgres-backed store for iterative improvement jobs and their step queue.

Mirrors ``api/store.py``: sessions come from ``self._factory()()`` and are always
closed in a ``finally`` block. The steps table is a queue claimed with
``SELECT ... FOR UPDATE SKIP LOCKED``, exactly like ``PostgresRunStore.claim_next``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select

from api.agent_spec import AgentSpec, baseline_spec, changed_fields
from api.db import get_session_factory
from api.models import AgentVersionRow, JobRow, StepRow
from api.schemas import RunStatus
from api.store import _utcnow

STEP_EVALUATE = "evaluate"
STEP_IMPROVE = "improve"

#: Improve steps are a single LLM call; the run default is plenty.
IMPROVE_STALE_AFTER_SEC = 1800

CREATED_BY_BASELINE = "baseline"
CREATED_BY_IMPROVER = "improver"


@dataclass(frozen=True)
class StepRecord:
    """Everything a worker needs to execute one claimed step."""

    step_id: str
    job_id: str
    type: str
    iteration: int
    agent_version_id: str
    version: int
    spec: AgentSpec
    task_ids: list[str]
    agent_model: str
    improver_model: str
    run_id: str | None
    stale_after_sec: int


@dataclass(frozen=True)
class AgentVersionRecord:
    version_id: str
    job_id: str
    version: int
    parent_version_id: str | None
    spec: AgentSpec
    rationale: str
    created_by: str
    created_at: datetime


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    agent_version_id: str
    version: int
    run_id: str | None
    score: float | None
    improved: bool | None
    rationale: str | None
    changed_fields: list[str]
    status: str


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: str
    task_ids: list[str]
    agent_model: str
    improver_model: str
    max_iterations: int
    patience: int
    min_delta: float
    current_iteration: int
    best_agent_version_id: str | None
    best_version: int | None
    best_score: float | None
    stop_reason: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_code: str | None
    error_message: str | None
    iterations: list[IterationRecord] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluateOutcome:
    """Result of an evaluate step. ``score`` is the mean reward for the iteration."""

    run_id: str
    score: float | None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ImproveOutcome:
    """Result of an improve step. ``spec is None`` means the proposal was unusable."""

    spec: AgentSpec | None
    rationale: str = ""
    error_code: str | None = None
    error_message: str | None = None


def _uuid_or_none(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _version_to_record(row: AgentVersionRow) -> AgentVersionRecord:
    return AgentVersionRecord(
        version_id=str(row.id),
        job_id=str(row.job_id),
        version=row.version,
        parent_version_id=(
            str(row.parent_version_id) if row.parent_version_id is not None else None
        ),
        spec=AgentSpec.model_validate(row.spec),
        rationale=row.rationale,
        created_by=row.created_by,
        created_at=row.created_at,
    )


def _build_iterations(
    steps: list[StepRow],
    versions: dict[UUID, AgentVersionRow],
    min_delta: float,
) -> list[IterationRecord]:
    """One IterationRecord per evaluate step, ordered by iteration.

    ``improved`` is derived here rather than stored: an iteration improved when its
    score beats the best score of all *strictly earlier* evaluate steps by more than
    ``min_delta``. The first scored iteration always improved. Steps with no score yet
    (queued/running/failed) report ``improved=None``.
    """
    evaluates = sorted(
        (s for s in steps if s.type == STEP_EVALUATE),
        key=lambda s: (s.iteration, s.created_at),
    )
    records: list[IterationRecord] = []
    earlier_best: float | None = None

    for step in evaluates:
        version = versions.get(step.agent_version_id)
        rationale: str | None = None
        changed: list[str] = []
        if version is not None and version.parent_version_id is not None:
            rationale = version.rationale or None
            parent = versions.get(version.parent_version_id)
            if parent is not None:
                changed = changed_fields(
                    AgentSpec.model_validate(parent.spec),
                    AgentSpec.model_validate(version.spec),
                )

        improved: bool | None = None
        if step.score is not None:
            improved = earlier_best is None or step.score > earlier_best + min_delta
            earlier_best = (
                step.score if earlier_best is None else max(earlier_best, step.score)
            )

        records.append(
            IterationRecord(
                iteration=step.iteration,
                agent_version_id=str(step.agent_version_id),
                version=version.version if version is not None else -1,
                run_id=str(step.run_id) if step.run_id is not None else None,
                score=step.score,
                improved=improved,
                rationale=rationale,
                changed_fields=changed,
                status=step.status,
            )
        )
    return records


class PostgresJobStore:
    """Job/agent-version/step store; also the step queue."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _factory(self):
        return self._session_factory or get_session_factory()

    def create_job(
        self,
        *,
        task_ids: list[str],
        agent_model: str,
        improver_model: str,
        max_iterations: int,
        patience: int,
        min_delta: float,
        max_job_duration_sec: int,
        evaluate_stale_after_sec: int,
    ) -> JobRecord:
        """Insert the job, agent version 0 and the iteration-0 evaluate step.

        All three inserts happen in ONE transaction. The explicit ``flush()`` calls
        are required: with no ORM relationships between these tables, SQLAlchemy would
        otherwise order the INSERTs by mapper sort key and try ``agent_versions``
        before ``jobs``, violating the foreign key.
        """
        now = _utcnow()
        job_id = uuid.uuid4()
        version_id = uuid.uuid4()
        spec = baseline_spec(agent_model)

        session = self._factory()()
        try:
            session.add(
                JobRow(
                    id=job_id,
                    status=RunStatus.queued.value,
                    task_ids=list(task_ids),
                    agent_model=agent_model,
                    improver_model=improver_model,
                    max_iterations=max_iterations,
                    patience=patience,
                    min_delta=min_delta,
                    max_job_duration_sec=max_job_duration_sec,
                    evaluate_stale_after_sec=evaluate_stale_after_sec,
                    current_iteration=0,
                    non_improving_streak=0,
                    created_at=now,
                )
            )
            session.flush()
            session.add(
                AgentVersionRow(
                    id=version_id,
                    job_id=job_id,
                    version=0,
                    parent_version_id=None,
                    spec=spec.model_dump(),
                    rationale=CREATED_BY_BASELINE,
                    created_by=CREATED_BY_BASELINE,
                    created_at=now,
                )
            )
            session.flush()
            session.add(
                StepRow(
                    id=uuid.uuid4(),
                    job_id=job_id,
                    type=STEP_EVALUATE,
                    status=RunStatus.queued.value,
                    iteration=0,
                    agent_version_id=version_id,
                    stale_after_sec=evaluate_stale_after_sec,
                    created_at=now,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        record = self.get_job(str(job_id))
        assert record is not None
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        uid = _uuid_or_none(job_id)
        if uid is None:
            return None
        session = self._factory()()
        try:
            job = session.get(JobRow, uid)
            if job is None:
                return None

            versions = {
                row.id: row
                for row in session.scalars(
                    select(AgentVersionRow)
                    .where(AgentVersionRow.job_id == uid)
                    .order_by(AgentVersionRow.version)
                )
            }
            steps = list(
                session.scalars(
                    select(StepRow)
                    .where(StepRow.job_id == uid)
                    .order_by(StepRow.iteration, StepRow.created_at)
                )
            )

            best_version: int | None = None
            if job.best_agent_version_id is not None:
                best_row = versions.get(job.best_agent_version_id)
                best_version = best_row.version if best_row is not None else None

            return JobRecord(
                job_id=str(job.id),
                status=job.status,
                task_ids=list(job.task_ids),
                agent_model=job.agent_model,
                improver_model=job.improver_model,
                max_iterations=job.max_iterations,
                patience=job.patience,
                min_delta=job.min_delta,
                current_iteration=job.current_iteration,
                best_agent_version_id=(
                    str(job.best_agent_version_id)
                    if job.best_agent_version_id is not None
                    else None
                ),
                best_version=best_version,
                best_score=job.best_score,
                stop_reason=job.stop_reason,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                error_code=job.error_code,
                error_message=job.error_message,
                iterations=_build_iterations(steps, versions, job.min_delta),
            )
        finally:
            session.close()

    def get_agent_version(self, version_id: str) -> AgentVersionRecord | None:
        uid = _uuid_or_none(version_id)
        if uid is None:
            return None
        session = self._factory()()
        try:
            row = session.get(AgentVersionRow, uid)
            if row is None:
                return None
            return _version_to_record(row)
        finally:
            session.close()

    def clear(self) -> None:
        """Delete all job data. Order matters: children before parents."""
        session = self._factory()()
        try:
            session.execute(delete(StepRow))
            session.execute(delete(AgentVersionRow))
            session.execute(delete(JobRow))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Default process-wide store (API and worker construct their own as needed).
job_store = PostgresJobStore()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_store.py -v`

Expected: PASS (2 tests).

#### Cycle 2 — `claim_next_step`

- [ ] **Step 5: Write the failing tests**

Append to `tests/test_job_store.py`:

```python
def test_claim_next_step_marks_step_running_and_job_running(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store)

    step = job_store.claim_next_step("w1")
    assert step is not None
    assert step.job_id == job.job_id
    assert step.type == "evaluate"
    assert step.iteration == 0
    assert step.version == 0
    assert step.task_ids == ["fix-git", "regex-log"]
    assert step.agent_model == "gpt-4.1-mini"
    assert step.improver_model == "gpt-5.4"
    assert step.run_id is None
    assert step.stale_after_sec == 3600
    assert step.spec == baseline_spec("gpt-4.1-mini")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.started_at is not None
    assert refreshed.iterations[0].status == "running"

    # Nothing else is queued.
    assert job_store.claim_next_step("w2") is None

    session = get_session_factory()()
    try:
        row = session.scalars(select(StepRow)).one()
        assert row.status == "running"
        assert row.worker_id == "w1"
        assert row.claimed_at is not None
        assert row.started_at is not None
    finally:
        session.close()


def test_two_workers_do_not_double_claim_steps(job_store: PostgresJobStore) -> None:
    j1 = _create_job(job_store, task_ids=["fix-git"])
    j2 = _create_job(job_store, task_ids=["regex-log"])

    def claim(worker_id: str):
        return job_store.claim_next_step(worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(claim, "w1")
        f2 = pool.submit(claim, "w2")
        claimed = [f1.result(), f2.result()]

    assert None not in claimed
    assert {c.job_id for c in claimed} == {j1.job_id, j2.job_id}
    assert len({c.step_id for c in claimed}) == 2

    # Third claim finds nothing queued.
    assert job_store.claim_next_step("w3") is None

    for record in claimed:
        job = job_store.get_job(record.job_id)
        assert job is not None
        assert job.status == "running"


def test_stale_running_step_is_requeued_and_reclaimable(
    job_store: PostgresJobStore,
) -> None:
    _create_job(job_store, evaluate_stale_after_sec=0)

    first = job_store.claim_next_step("w1")
    assert first is not None
    assert first.stale_after_sec == 0

    # With stale_after_sec=0 the row is stale as soon as now() moves past claimed_at.
    time.sleep(0.05)

    second = job_store.claim_next_step("w2")
    assert second is not None
    assert second.step_id == first.step_id

    session = get_session_factory()()
    try:
        row = session.scalars(select(StepRow)).one()
        assert row.status == "running"
        assert row.worker_id == "w2"
    finally:
        session.close()


def test_claim_next_step_returns_none_when_no_jobs(job_store: PostgresJobStore) -> None:
    assert job_store.claim_next_step("w1") is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_job_store.py -k claim -v`

Expected: FAIL with
`AttributeError: 'PostgresJobStore' object has no attribute 'claim_next_step'`.

- [ ] **Step 7: Write the implementation**

In `api/job_store.py`, extend the SQLAlchemy import line to:

```python
from sqlalchemy import delete, select, text, update
```

and add this module-level constant just below `IMPROVE_STALE_AFTER_SEC`:

```python
#: Per-row staleness predicate. The threshold lives in the row itself
#: (``steps.stale_after_sec``) because evaluate steps can legitimately run for hours,
#: so no Python-side timedelta can express it. The table name must be spelled out:
#: a bulk UPDATE does not alias the target table.
_STALE_STEP_PREDICATE = text(
    "steps.claimed_at < now() - make_interval(secs => steps.stale_after_sec)"
)
```

Add this method to `PostgresJobStore` (after `create_job`):

```python
    def claim_next_step(self, worker_id: str) -> StepRecord | None:
        """Atomically claim the next queued step (or requeue-and-claim a stale one).

        Uses SELECT ... FOR UPDATE SKIP LOCKED so concurrent workers never claim the
        same step. Locks the step row first and the job row second — every method in
        this class uses that order to avoid deadlocks.
        """
        session = self._factory()()
        try:
            now = _utcnow()

            # Requeue steps whose own stale_after_sec has elapsed (best-effort).
            session.execute(
                update(StepRow)
                .where(
                    StepRow.status == RunStatus.running.value,
                    StepRow.claimed_at.is_not(None),
                    _STALE_STEP_PREDICATE,
                )
                .values(
                    status=RunStatus.queued.value,
                    worker_id=None,
                    claimed_at=None,
                    started_at=None,
                )
            )

            step = session.scalar(
                select(StepRow)
                .where(StepRow.status == RunStatus.queued.value)
                .order_by(StepRow.created_at, StepRow.iteration)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if step is None:
                session.commit()
                return None

            step.status = RunStatus.running.value
            step.worker_id = worker_id
            step.claimed_at = now
            step.started_at = now

            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is None:
                # Job vanished under us; drop the claim rather than run an orphan.
                session.rollback()
                return None
            if job.status == RunStatus.queued.value:
                job.status = RunStatus.running.value
                if job.started_at is None:
                    job.started_at = now

            version = session.get(AgentVersionRow, step.agent_version_id)
            if version is None:
                session.rollback()
                return None

            record = StepRecord(
                step_id=str(step.id),
                job_id=str(step.job_id),
                type=step.type,
                iteration=step.iteration,
                agent_version_id=str(step.agent_version_id),
                version=version.version,
                spec=AgentSpec.model_validate(version.spec),
                task_ids=list(job.task_ids),
                agent_model=job.agent_model,
                improver_model=job.improver_model,
                run_id=str(step.run_id) if step.run_id is not None else None,
                stale_after_sec=step.stale_after_sec,
            )
            session.commit()
            return record
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_job_store.py -v`

Expected: PASS (6 tests).

#### Cycle 3 — `complete_step_and_advance`, `fail_step`, derived history

- [ ] **Step 9: Write the failing tests**

Append to `tests/test_job_store.py`:

```python
def _complete_evaluate(
    job_store: PostgresJobStore,
    worker_id: str = "w1",
    *,
    score: float | None = 0.5,
    error_code: str | None = None,
    error_message: str | None = None,
) -> str:
    """Claim the next (evaluate) step and complete it. Returns the step id."""
    step = job_store.claim_next_step(worker_id)
    assert step is not None and step.type == "evaluate"
    job_store.complete_step_and_advance(
        step.step_id,
        EvaluateOutcome(
            run_id=str(uuid.uuid4()),
            score=score,
            error_code=error_code,
            error_message=error_message,
        ),
    )
    return step.step_id


def test_evaluate_improvement_enqueues_improve_step(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    step = job_store.claim_next_step("w1")
    assert step is not None
    run_id = str(uuid.uuid4())

    job_store.complete_step_and_advance(
        step.step_id, EvaluateOutcome(run_id=run_id, score=0.5)
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.finished_at is None
    assert refreshed.stop_reason is None
    assert refreshed.best_score == pytest.approx(0.5)
    assert refreshed.best_agent_version_id == step.agent_version_id
    assert refreshed.best_version == 0
    assert refreshed.iterations[0].status == "completed"
    assert refreshed.iterations[0].score == pytest.approx(0.5)
    assert refreshed.iterations[0].improved is True
    assert refreshed.iterations[0].run_id == run_id

    # An improve step for the same iteration is now claimable.
    improve = job_store.claim_next_step("w1")
    assert improve is not None
    assert improve.type == "improve"
    assert improve.iteration == 0
    assert improve.agent_version_id == step.agent_version_id
    assert improve.stale_after_sec == 1800


def test_evaluate_hitting_max_iterations_completes_job(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=1)
    _complete_evaluate(job_store, score=0.25)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "max_iterations"
    assert refreshed.finished_at is not None
    assert refreshed.best_score == pytest.approx(0.25)

    # No successor step was enqueued.
    assert job_store.claim_next_step("w1") is None
    session = get_session_factory()()
    try:
        assert len(list(session.scalars(select(StepRow)))) == 1
    finally:
        session.close()


def test_improve_ok_inserts_version_one_and_next_evaluate(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)

    improve = job_store.claim_next_step("w1")
    assert improve is not None and improve.type == "improve"
    new_spec = improve.spec.model_copy(
        update={"system_prompt": "Verify every command before finishing.", "max_steps": 90}
    )
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(spec=new_spec, rationale="Add an explicit verification step."),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.current_iteration == 1
    assert refreshed.stop_reason is None

    next_step = job_store.claim_next_step("w1")
    assert next_step is not None
    assert next_step.type == "evaluate"
    assert next_step.iteration == 1
    assert next_step.version == 1
    assert next_step.spec == new_spec
    assert next_step.stale_after_sec == 3600

    version = job_store.get_agent_version(next_step.agent_version_id)
    assert version is not None
    assert version.version == 1
    assert version.parent_version_id == improve.agent_version_id
    assert version.created_by == "improver"
    assert version.rationale == "Add an explicit verification step."


def test_improve_error_with_existing_best_completes_failed_improve(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=None,
            error_code="improver_error",
            error_message="LLM returned invalid JSON twice",
        ),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "failed_improve"
    assert refreshed.finished_at is not None
    # The best-so-far agent is still a valid answer, so no job-level error.
    assert refreshed.error_code is None
    assert refreshed.best_agent_version_id is not None
    assert job_store.claim_next_step("w1") is None

    session = get_session_factory()()
    try:
        improve_row = session.scalars(
            select(StepRow).where(StepRow.type == "improve")
        ).one()
        assert improve_row.status == "failed"
        assert improve_row.error_code == "improver_error"
    finally:
        session.close()


def test_improve_error_without_best_fails_job(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(job_store, score=0.5)
    improve = job_store.claim_next_step("w1")
    assert improve is not None

    # Force the "no best yet" branch: clear the best pointer the baseline set.
    session = get_session_factory()()
    try:
        row = session.get(JobRow, uuid.UUID(job.job_id))
        assert row is not None
        row.best_agent_version_id = None
        row.best_score = None
        session.commit()
    finally:
        session.close()

    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=None, error_code="improver_error", error_message="no usable proposal"
        ),
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.stop_reason == "failed"
    assert refreshed.error_code == "improver_error"
    assert refreshed.error_message == "no usable proposal"
    assert refreshed.finished_at is not None


def test_evaluate_error_fails_job_with_copied_error(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=5)
    _complete_evaluate(
        job_store,
        score=None,
        error_code="execution_unavailable",
        error_message="harbor CLI not found",
    )

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_code == "execution_unavailable"
    assert refreshed.error_message == "harbor CLI not found"
    assert refreshed.finished_at is not None
    assert refreshed.best_agent_version_id is None
    assert refreshed.best_score is None
    assert refreshed.stop_reason is None
    assert refreshed.iterations[0].status == "failed"
    assert refreshed.iterations[0].score is None
    assert refreshed.iterations[0].improved is None
    assert job_store.claim_next_step("w1") is None


def test_fail_step_fails_step_and_job(job_store: PostgresJobStore) -> None:
    job = _create_job(job_store)
    step = job_store.claim_next_step("w1")
    assert step is not None

    job_store.fail_step(step.step_id, error_code="internal_error", error_message="boom")

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.error_code == "internal_error"
    assert refreshed.error_message == "boom"
    assert refreshed.iterations[0].status == "failed"
    assert job_store.claim_next_step("w1") is None


def test_get_job_iterations_improved_flags_and_changed_fields(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=4, min_delta=0.01)

    # Iteration 0: baseline scores 0.50.
    _complete_evaluate(job_store, score=0.50)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    proposal = improve.spec.model_copy(
        update={"system_prompt": "Check your work.", "max_steps": 100}
    )
    job_store.complete_step_and_advance(
        improve.step_id, ImproveOutcome(spec=proposal, rationale="Verify before exit.")
    )

    # Iteration 1: improved to 0.70.
    _complete_evaluate(job_store, score=0.70)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert len(refreshed.iterations) == 2

    it0, it1 = refreshed.iterations
    assert it0.iteration == 0
    assert it0.version == 0
    assert it0.improved is True
    assert it0.rationale is None
    assert it0.changed_fields == []
    assert it0.score == pytest.approx(0.50)

    assert it1.iteration == 1
    assert it1.version == 1
    assert it1.improved is True
    assert it1.rationale == "Verify before exit."
    assert it1.changed_fields == ["max_steps", "system_prompt"]
    assert it1.score == pytest.approx(0.70)

    assert refreshed.best_score == pytest.approx(0.70)
    assert refreshed.best_version == 1
    assert refreshed.best_agent_version_id == it1.agent_version_id
    assert refreshed.current_iteration == 1


def test_non_improving_iteration_reports_improved_false_and_keeps_best(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=4, patience=2, min_delta=0.01)

    _complete_evaluate(job_store, score=0.60)

    improve = job_store.claim_next_step("w1")
    assert improve is not None
    job_store.complete_step_and_advance(
        improve.step_id,
        ImproveOutcome(
            spec=improve.spec.model_copy(update={"max_steps": 120}),
            rationale="More steps.",
        ),
    )

    # Iteration 1 lands exactly on the min_delta boundary -> NOT an improvement.
    _complete_evaluate(job_store, score=0.61)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.iterations[1].improved is False
    assert refreshed.iterations[1].changed_fields == ["max_steps"]
    assert refreshed.best_score == pytest.approx(0.60)
    assert refreshed.best_version == 0

    # Streak is 1 of patience 2, so the loop continues with another improve step.
    nxt = job_store.claim_next_step("w1")
    assert nxt is not None and nxt.type == "improve" and nxt.iteration == 1


def test_patience_exhausted_stops_with_no_improvement(
    job_store: PostgresJobStore,
) -> None:
    job = _create_job(job_store, max_iterations=10, patience=2, min_delta=0.01)

    _complete_evaluate(job_store, score=0.60)
    for max_steps in (110, 120):
        improve = job_store.claim_next_step("w1")
        assert improve is not None and improve.type == "improve"
        job_store.complete_step_and_advance(
            improve.step_id,
            ImproveOutcome(
                spec=improve.spec.model_copy(update={"max_steps": max_steps}),
                rationale=f"Try {max_steps} steps.",
            ),
        )
        _complete_evaluate(job_store, score=0.40)

    refreshed = job_store.get_job(job.job_id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.stop_reason == "no_improvement"
    assert refreshed.best_score == pytest.approx(0.60)
    assert refreshed.best_version == 0
    assert len(refreshed.iterations) == 3
    assert [i.improved for i in refreshed.iterations] == [True, False, False]
    assert job_store.claim_next_step("w1") is None
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_job_store.py -k "evaluate or improve or fail_step or iterations or patience" -v`

Expected: FAIL with
`AttributeError: 'PostgresJobStore' object has no attribute 'complete_step_and_advance'`.

- [ ] **Step 11: Write the implementation**

In `api/job_store.py`, extend the SQLAlchemy import line once more and add the scoring
import:

```python
from sqlalchemy import delete, func, select, text, update

from api.services.scoring import compute_stop
```

Add these methods to `PostgresJobStore` (after `claim_next_step`):

```python
    def complete_step_and_advance(
        self, step_id: str, outcome: EvaluateOutcome | ImproveOutcome
    ) -> None:
        """Close a step and, in the SAME transaction, advance the job.

        Either the successor step is enqueued or the job reaches a terminal state, so
        there is never a live job with nothing queued. A crash before commit leaves the
        step ``running`` until stale-requeue picks it up again.
        """
        uid = _uuid_or_none(step_id)
        if uid is None:
            return

        session = self._factory()()
        try:
            step = session.get(StepRow, uid, with_for_update=True)
            if step is None:
                session.commit()
                return
            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is None:
                session.commit()
                return

            now = _utcnow()
            if isinstance(outcome, EvaluateOutcome):
                _apply_evaluate_outcome(session, step, job, outcome, now)
            else:
                _apply_improve_outcome(session, step, job, outcome, now)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def fail_step(self, step_id: str, *, error_code: str, error_message: str) -> None:
        """Fail a step and its job (worker-level unexpected failures)."""
        uid = _uuid_or_none(step_id)
        if uid is None:
            return

        session = self._factory()()
        try:
            step = session.get(StepRow, uid, with_for_update=True)
            if step is None:
                session.commit()
                return
            now = _utcnow()
            step.status = RunStatus.failed.value
            step.error_code = error_code
            step.error_message = error_message
            step.finished_at = now

            job = session.get(JobRow, step.job_id, with_for_update=True)
            if job is not None and job.status in _ACTIVE_JOB_STATUSES:
                job.status = RunStatus.failed.value
                job.error_code = error_code
                job.error_message = error_message
                job.finished_at = now

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

Add these module-level helpers below `_build_iterations` (they take an open session and
never commit — the caller owns the transaction):

```python
_ACTIVE_JOB_STATUSES = frozenset({RunStatus.queued.value, RunStatus.running.value})

STOP_FAILED_IMPROVE = "failed_improve"
STOP_FAILED = "failed"


def _elapsed_sec(job: JobRow, now: datetime) -> float:
    started = job.started_at or job.created_at
    return (now - started).total_seconds()


def _apply_evaluate_outcome(
    session,
    step: StepRow,
    job: JobRow,
    outcome: EvaluateOutcome,
    now: datetime,
) -> None:
    step.run_id = _uuid_or_none(outcome.run_id)
    step.finished_at = now

    if outcome.error_code:
        # Infra failure: never counted as "no improvement".
        step.status = RunStatus.failed.value
        step.error_code = outcome.error_code
        step.error_message = outcome.error_message
        job.status = RunStatus.failed.value
        job.error_code = outcome.error_code
        job.error_message = outcome.error_message
        job.finished_at = now
        return

    score = 0.0 if outcome.score is None else float(outcome.score)
    step.status = RunStatus.completed.value
    step.score = score

    decision = compute_stop(
        iteration=step.iteration,
        score=score,
        best_score=job.best_score,
        prior_non_improving_streak=job.non_improving_streak,
        max_iterations=job.max_iterations,
        patience=job.patience,
        min_delta=job.min_delta,
        elapsed_sec=_elapsed_sec(job, now),
        max_job_duration_sec=job.max_job_duration_sec,
    )

    job.non_improving_streak = decision.non_improving_streak
    if decision.improved:
        job.best_score = score
        job.best_agent_version_id = step.agent_version_id

    if decision.should_stop:
        job.status = RunStatus.completed.value
        job.stop_reason = decision.stop_reason
        job.finished_at = now
        return

    session.add(
        StepRow(
            id=uuid.uuid4(),
            job_id=job.id,
            type=STEP_IMPROVE,
            status=RunStatus.queued.value,
            iteration=step.iteration,
            agent_version_id=step.agent_version_id,
            stale_after_sec=IMPROVE_STALE_AFTER_SEC,
            created_at=now,
        )
    )


def _apply_improve_outcome(
    session,
    step: StepRow,
    job: JobRow,
    outcome: ImproveOutcome,
    now: datetime,
) -> None:
    step.finished_at = now

    if outcome.error_code or outcome.spec is None:
        error_code = outcome.error_code or "invalid_proposal"
        error_message = outcome.error_message or "Improver returned no valid AgentSpec"
        step.status = RunStatus.failed.value
        step.error_code = error_code
        step.error_message = error_message
        job.finished_at = now
        if job.best_agent_version_id is not None:
            # A best-so-far agent is still a valid answer for the job.
            job.status = RunStatus.completed.value
            job.stop_reason = STOP_FAILED_IMPROVE
        else:
            job.status = RunStatus.failed.value
            job.stop_reason = STOP_FAILED
            job.error_code = error_code
            job.error_message = error_message
        return

    step.status = RunStatus.completed.value

    next_version_number = (
        session.scalar(
            select(func.max(AgentVersionRow.version)).where(
                AgentVersionRow.job_id == job.id
            )
        )
        or 0
    ) + 1
    next_iteration = step.iteration + 1
    new_version_id = uuid.uuid4()

    session.add(
        AgentVersionRow(
            id=new_version_id,
            job_id=job.id,
            version=next_version_number,
            parent_version_id=step.agent_version_id,
            spec=outcome.spec.model_dump(),
            rationale=outcome.rationale,
            created_by=CREATED_BY_IMPROVER,
            created_at=now,
        )
    )
    session.flush()

    job.current_iteration = next_iteration
    session.add(
        StepRow(
            id=uuid.uuid4(),
            job_id=job.id,
            type=STEP_EVALUATE,
            status=RunStatus.queued.value,
            iteration=next_iteration,
            agent_version_id=new_version_id,
            stale_after_sec=job.evaluate_stale_after_sec,
            created_at=now,
        )
    )
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_job_store.py tests/test_job_models.py tests/test_scoring.py tests/test_api.py -v`

Expected: PASS — 16 `test_job_store.py` tests, 3 `test_job_models.py`, 16
`test_scoring.py`, and the existing `test_api.py` suite unaffected.

- [ ] **Step 13: Commit**

```bash
git add api/job_store.py tests/test_job_store.py
git commit -m "feat: add PostgresJobStore step queue with transactional advance"
```

---

### Additions to the contract made by this section

Everything above uses the contract's names verbatim. Three items are *additions* that
later sections may rely on, all in `api/job_store.py`:

1. Module-level constants `STEP_EVALUATE = "evaluate"`, `STEP_IMPROVE = "improve"`,
   `IMPROVE_STALE_AFTER_SEC = 1800`, `CREATED_BY_BASELINE = "baseline"`,
   `CREATED_BY_IMPROVER = "improver"`, `STOP_FAILED_IMPROVE = "failed_improve"`,
   `STOP_FAILED = "failed"` — so Tasks 10-13 compare against names, not literals.
   `api/services/scoring.py` likewise exports `STOP_MAX_ITERATIONS`,
   `STOP_NO_IMPROVEMENT`, `STOP_BUDGET_EXCEEDED`.
2. `job_store = PostgresJobStore()` module-level default instance, mirroring
   `api/store.py:358`, for `create_app`'s `job_store or default_job_store` fallback.
3. `ImproveOutcome` with `spec=None` and no `error_code` is normalized to
   `error_code="invalid_proposal"` on the step and job.

`JobRecord.iterations` is declared as `field(default_factory=list)` so the dataclass can
be constructed in tests without a history; every store method still passes it
explicitly.

---

## Section C — Improver & Worker Execution (Tasks 8-10)

### Task 8: Improver context assembly

Builds the *pure* half of `api/services/improver.py`: the data types the improver
speaks in, the `Improver` protocol, and `build_context()` — the function that turns
(current spec + iteration history + latest evaluation + traces) into one budgeted
prompt string. No DB, no LLM, no filesystem: every test in this task is a pure
function call.

**Files:**
- Create: `api/services/improver.py`
- Test: `tests/test_improver_context.py`

**Interfaces:**
- Consumes (from earlier tasks, exact):
  - `api.agent_spec.AgentSpec(BaseModel)` — fields in declaration order:
    `system_prompt: str` (min 1, max 20_000), `agent_model: str` (min 1, max 256),
    `max_steps: int = 80` (1-200), `max_output_chars: int = 8000` (500-100_000),
    `exec_timeout_sec: int = 120` (10-1200); `model_config = ConfigDict(extra="forbid")`.
  - `api.job_store.IterationRecord` — frozen dataclass with
    `iteration: int`, `agent_version_id: str`, `version: int`, `run_id: str | None`,
    `score: float | None`, `improved: bool | None`, `rationale: str | None`,
    `changed_fields: list[str]`, `status: str`.
- Produces (Tasks 9-10 and Section D rely on these):
  - `TaskOutcome(task_id: str, status: str, reward: float | None, remarks: str | None)` — frozen dataclass; `status` is the string value `"passed" | "failed" | "error"`.
  - `EvaluationSummary(score: float, tasks: list[TaskOutcome], traces: dict[str, str])` — frozen dataclass; `traces` maps `task_id` → trace text already read out of the artifact store.
  - `Proposal(spec: AgentSpec, rationale: str)` — frozen dataclass.
  - `class ImproverError(Exception)`.
  - `class Improver(Protocol)` with `propose(self, *, spec: AgentSpec, evaluation: EvaluationSummary, history: list[IterationRecord]) -> Proposal`.
  - `build_context(*, spec: AgentSpec, evaluation: EvaluationSummary, history: list[IterationRecord], budget: int) -> str`.

**Contract decisions fixed by this task (Task 9 and 10 depend on them):**

1. Section order is always: `## CURRENT AGENT SPEC (JSON)`,
   `## ITERATION HISTORY (oldest first)`, `## LATEST EVALUATION (score=…)`,
   `## FAILURE DETAILS (worst tasks first)`. Sections are joined by a blank line;
   **no section ever contains a blank line internally**, so a consumer (and the
   tests) can split sections on `"\n\n"`.
2. Sections 1-3 are mandatory and are always emitted in full. Only section 4 is
   budget-gated: failure blocks are appended one at a time while the running total
   stays within `budget`, worst task first.
3. The final return value is hard-truncated with `[:budget]`. This makes
   `len(build_context(...)) <= budget` an unconditional invariant even when the
   mandatory prefix alone is larger than the budget. Because the spec section is
   first and `system_prompt` is `AgentSpec`'s first declared field (so it is the
   first key of `json.dumps(spec.model_dump(), indent=2)` — **do not pass
   `sort_keys=True`**), the current prompt is the last thing to be lost.
4. Failure ordering key: `(reward or 0.0, 0 if status == "error" else 1, task_id)` —
   lowest reward first, and an `error` (whose reward is always `None` → `0.0`) sorts
   ahead of a `failed` task with reward `0.0`.
5. Every table cell is flattened to one line and length-capped by `_flat()`, so a
   multi-line rationale or remark can never break the "one row per iteration"
   shape.

- [ ] **Step 1: Write the failing test**
```python
"""Pure tests for improver context assembly (no DB, no LLM, no network)."""

from __future__ import annotations

import json

from api.agent_spec import AgentSpec
from api.job_store import IterationRecord
from api.services.improver import (
    EvaluationSummary,
    TaskOutcome,
    build_context,
)


def _spec(system_prompt: str = "BASE PROMPT") -> AgentSpec:
    return AgentSpec(system_prompt=system_prompt, agent_model="gpt-4.1-mini")


def _iteration(
    *,
    iteration: int,
    version: int,
    score: float | None,
    improved: bool | None,
    changed_fields: list[str],
    rationale: str | None,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        agent_version_id=f"00000000-0000-0000-0000-00000000000{version}",
        version=version,
        run_id=f"11111111-1111-1111-1111-11111111111{iteration}",
        score=score,
        improved=improved,
        rationale=rationale,
        changed_fields=changed_fields,
        status="completed",
    )


def _history() -> list[IterationRecord]:
    return [
        _iteration(
            iteration=0,
            version=0,
            score=0.5,
            improved=True,
            changed_fields=[],
            rationale="baseline",
        ),
        _iteration(
            iteration=1,
            version=1,
            score=0.5,
            improved=False,
            changed_fields=["max_steps", "system_prompt"],
            rationale="Told the agent to verify\nits work before finishing",
        ),
    ]


def _trace(text: str) -> str:
    return json.dumps(
        [
            {"role": "system", "content": "you are an agent"},
            {"role": "assistant", "content": "running the build"},
            {"role": "tool", "content": text},
        ]
    )


def test_history_table_has_one_row_per_iteration_and_rationales() -> None:
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=0.5, tasks=[], traces={}),
        history=_history(),
        budget=60_000,
    )

    block = ctx.split("## ITERATION HISTORY (oldest first)\n", 1)[1].split("\n\n", 1)[0]
    lines = block.strip().splitlines()

    assert lines[0] == "iteration | version | score | improved | changed_fields | rationale"
    assert len(lines) == 3, lines
    assert lines[1].startswith("0 | 0 | 0.5000 | yes | - | baseline")
    assert lines[2].startswith("1 | 1 | 0.5000 | no | max_steps,system_prompt | ")
    # Multi-line rationales are flattened onto their single row.
    assert "Told the agent to verify its work before finishing" in lines[2]


def test_failure_details_are_worst_first() -> None:
    tasks = [
        TaskOutcome(task_id="t-pass", status="passed", reward=1.0, remarks=None),
        TaskOutcome(task_id="t-partial", status="failed", reward=0.4, remarks="Partial reward 0.4"),
        TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed"),
        TaskOutcome(task_id="t-err", status="error", reward=None, remarks="sandbox timeout"),
    ]
    traces = {
        "t-partial": _trace("partial trace body"),
        "t-zero": _trace("zero trace body"),
        "t-err": _trace("error trace body"),
    }
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=0.35, tasks=tasks, traces=traces),
        history=_history(),
        budget=60_000,
    )

    assert "## FAILURE DETAILS (worst tasks first)" in ctx
    i_err = ctx.index("### t-err")
    i_zero = ctx.index("### t-zero")
    i_partial = ctx.index("### t-partial")
    assert i_err < i_zero < i_partial
    # Passing tasks appear in the result table but get no failure block.
    assert "### t-pass" not in ctx
    # JSON message traces are rendered as role-tagged lines.
    assert "[tool] error trace body" in ctx


def test_output_never_exceeds_budget_with_huge_trace() -> None:
    tasks = [TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed")]
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.0,
            tasks=tasks,
            traces={"t-zero": "X" * 500_000},
        ),
        history=_history(),
        budget=5_000,
    )

    assert len(ctx) <= 5_000
    assert "BASE PROMPT" in ctx


def test_no_failure_section_when_all_tasks_pass() -> None:
    tasks = [
        TaskOutcome(task_id="t-a", status="passed", reward=1.0, remarks=None),
        TaskOutcome(task_id="t-b", status="passed", reward=1.0, remarks=None),
    ]
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(score=1.0, tasks=tasks, traces={}),
        history=_history(),
        budget=60_000,
    )

    assert "FAILURE DETAILS" not in ctx
    assert "## LATEST EVALUATION (score=1.0000)" in ctx


def test_current_spec_survives_tiny_budget() -> None:
    ctx = build_context(
        spec=_spec(),
        evaluation=EvaluationSummary(
            score=0.0,
            tasks=[TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed")],
            traces={"t-zero": _trace("body")},
        ),
        history=_history(),
        budget=300,
    )

    assert len(ctx) <= 300
    assert ctx.startswith("## CURRENT AGENT SPEC (JSON)")
    assert "BASE PROMPT" in ctx
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_improver_context.py -v`
Expected: FAIL — collection error `ModuleNotFoundError: No module named 'api.services.improver'`

- [ ] **Step 3: Write the implementation**

Create `api/services/improver.py` with the pure half only (Task 9 appends
`FakeImprover`, `LLMImprover`, `create_improver` below `build_context`):

```python
"""Improver: assembles the optimization prompt and proposes the next AgentSpec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from api.agent_spec import AgentSpec
from api.job_store import IterationRecord

# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskOutcome:
    """One benchmark task's result, flattened for prompt rendering."""

    task_id: str
    status: str  # "passed" | "failed" | "error"
    reward: float | None
    remarks: str | None


@dataclass(frozen=True)
class EvaluationSummary:
    """The latest evaluation: aggregate score, per-task results, and trace text."""

    score: float
    tasks: list[TaskOutcome]
    traces: dict[str, str]  # task_id -> trace text (already read from artifacts)


@dataclass(frozen=True)
class Proposal:
    spec: AgentSpec
    rationale: str


class ImproverError(Exception):
    """Raised when the improver cannot produce a valid proposal."""


class Improver(Protocol):
    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal: ...


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #

SPEC_HEADER = "## CURRENT AGENT SPEC (JSON)"
HISTORY_HEADER = "## ITERATION HISTORY (oldest first)"
HISTORY_COLUMNS = "iteration | version | score | improved | changed_fields | rationale"
TASKS_COLUMNS = "task_id | status | reward | remarks"
FAILURES_HEADER = "## FAILURE DETAILS (worst tasks first)"

_TRACE_TAIL_CHARS = 4_000
_TRACE_TAIL_MESSAGES = 12
_MESSAGE_CHARS = 600


def _flat(text: str | None, limit: int = 200) -> str:
    """Collapse a value onto one length-capped line (tables must stay row-shaped)."""
    if text is None:
        return "-"
    one_line = " ".join(str(text).split())
    if not one_line:
        return "-"
    if len(one_line) > limit:
        one_line = one_line[: limit - 1] + "…"
    return one_line


def _fmt_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.4f}"


def _fmt_flag(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _render_trace(text: str) -> str:
    """Render the tail of a trace: last N messages, each output truncated."""
    if not text or not text.strip():
        return "(no trace captured)"

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None

    rendered = text
    if isinstance(data, list):
        tail = data[-_TRACE_TAIL_MESSAGES:]
        lines: list[str] = []
        if len(data) > len(tail):
            lines.append(f"...[{len(data) - len(tail)} earlier messages omitted]...")
        for message in tail:
            if not isinstance(message, dict):
                lines.append(str(message)[:_MESSAGE_CHARS])
                continue
            role = str(message.get("role", "?"))
            content = message.get("content")
            if content is None:
                content = json.dumps(message.get("tool_calls") or "", default=str)
            content = str(content).replace("\r", "")
            if len(content) > _MESSAGE_CHARS:
                half = _MESSAGE_CHARS // 2
                content = content[:half] + " ...[output truncated]... " + content[-half:]
            lines.append(f"[{role}] {content}")
        rendered = "\n".join(lines)

    if len(rendered) > _TRACE_TAIL_CHARS:
        rendered = "...[trace truncated]...\n" + rendered[-_TRACE_TAIL_CHARS:]
    return rendered


def _spec_section(spec: AgentSpec) -> str:
    # No sort_keys: AgentSpec declares system_prompt first, so the prompt is the
    # first thing in the section and the last thing lost to truncation.
    return SPEC_HEADER + "\n" + json.dumps(spec.model_dump(), indent=2)


def _history_section(history: list[IterationRecord]) -> str:
    lines = [HISTORY_HEADER, HISTORY_COLUMNS]
    if not history:
        lines.append("(no prior iterations - this is the first proposal)")
    for record in history:
        lines.append(
            " | ".join(
                [
                    str(record.iteration),
                    str(record.version),
                    _fmt_score(record.score),
                    _fmt_flag(record.improved),
                    ",".join(record.changed_fields) if record.changed_fields else "-",
                    _flat(record.rationale),
                ]
            )
        )
    return "\n".join(lines)


def _tasks_section(evaluation: EvaluationSummary) -> str:
    lines = [f"## LATEST EVALUATION (score={evaluation.score:.4f})", TASKS_COLUMNS]
    if not evaluation.tasks:
        lines.append("(no task results)")
    for task in evaluation.tasks:
        lines.append(
            " | ".join(
                [
                    task.task_id,
                    task.status,
                    _fmt_score(task.reward),
                    _flat(task.remarks, 120),
                ]
            )
        )
    return "\n".join(lines)


def _failure_sort_key(task: TaskOutcome) -> tuple[float, int, str]:
    reward = 0.0 if task.reward is None else float(task.reward)
    return (reward, 0 if task.status == "error" else 1, task.task_id)


def _failure_blocks(evaluation: EvaluationSummary) -> list[str]:
    failing = [t for t in evaluation.tasks if t.status in ("failed", "error")]
    failing.sort(key=_failure_sort_key)
    blocks: list[str] = []
    for task in failing:
        trace = _render_trace(evaluation.traces.get(task.task_id, ""))
        blocks.append(
            f"### {task.task_id} - status={task.status} reward={_fmt_score(task.reward)}\n"
            f"remarks: {_flat(task.remarks, 300)}\n"
            f"trace tail:\n{trace}"
        )
    return blocks


def build_context(
    *,
    spec: AgentSpec,
    evaluation: EvaluationSummary,
    history: list[IterationRecord],
    budget: int,
) -> str:
    """
    Assemble the improver prompt body within a hard character budget.

    Order: current spec (always) -> iteration history table (always) -> per-task
    result table (always) -> failure details, worst task first, appended only
    while the running total stays inside ``budget``. The result is finally
    truncated to ``budget`` characters, so the returned length is never larger
    than the budget even when the mandatory prefix alone overflows it.
    """
    parts = [
        _spec_section(spec),
        _history_section(list(history)),
        _tasks_section(evaluation),
    ]
    # +2 per part accounts for the "\n\n" separators (a 2-char overestimate).
    running = sum(len(part) + 2 for part in parts)

    blocks = _failure_blocks(evaluation)
    if blocks:
        running += len(FAILURES_HEADER) + 2
        kept: list[str] = []
        for block in blocks:
            if running + len(block) + 2 > budget:
                break
            kept.append(block)
            running += len(block) + 2
        if kept:
            parts.append(FAILURES_HEADER)
            parts.extend(kept)

    return "\n\n".join(parts)[: max(budget, 0)]
```

- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_improver_context.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add api/services/improver.py tests/test_improver_context.py
git commit -m "feat: budgeted improver context assembly"
```

---

### Task 9: FakeImprover and LLMImprover

Appends the two `Improver` implementations plus the factory to
`api/services/improver.py`.

**Files:**
- Modify: `api/services/improver.py` (append below `build_context`; also extend the
  import block at the top)
- Test: `tests/test_improver.py`

**Interfaces:**
- Consumes: everything from Task 8 (`TaskOutcome`, `EvaluationSummary`, `Proposal`,
  `ImproverError`, `Improver`, `build_context`); `api.agent_spec.AgentSpec`;
  `api.config.BenchmarkConfig` / `load_config()` — the frozen config dataclass now
  carries `improver_model: str = "gpt-5.4"`, `improver_context_budget: int = 60000`
  and the pre-existing `execution_backend: str`.
- Produces:
  - `FakeImprover(proposals: list[Proposal] | None = None, *, mutate: Callable[[AgentSpec], AgentSpec] | None = None)` with `propose(*, spec, evaluation, history) -> Proposal`, plus observable attributes `calls: int`, `last_prompt: str`, `last_response: str`.
  - `LLMImprover(*, model: str, budget: int)` with the same `propose(...)` signature, plus `last_prompt: str`, `last_response: str`.
  - `create_improver(config: BenchmarkConfig | None = None, *, improver_model: str | None = None) -> Improver`.

**Behavioral decisions fixed here (Task 10's tests depend on them):**

1. **`FakeImprover` never raises.** Precedence per call: if `mutate` was given it is
   applied to the incoming spec; else if the scripted `proposals` list still has an
   entry for this call index, that entry is returned; else — *the exhausted case* —
   it returns a **deterministic derived proposal**: the incoming spec with
   `"\n\n[fake-improver revision N]"` appended to `system_prompt` (N = 1-based call
   count) and rationale `"fake improver deterministic revision N"`. It cycles
   forever rather than raising, so a mock end-to-end job is always driven by the
   stopping rule and never by improver exhaustion. Tests that need a failing
   improve step use their own two-line stub that raises `ImproverError` (see
   Task 10, `_RaisingImprover`).
2. **`litellm` is imported lazily**, inside `_litellm()`, into the module-level
   global `litellm` (initialised to `None`). Rationale: `api.services` is imported
   by the FastAPI app and by every Postgres test; only the *improve step* needs
   litellm, so a lazy import keeps `import api.services.improver` working in
   environments without litellm installed (the same pattern `benchmark.py` uses for
   its own heavy imports). Because the global exists, tests set
   `monkeypatch.setattr(improver_mod, "litellm", stub)` and `_litellm()` returns the
   stub without ever importing the real package — no network is reachable.
3. **`config_changes` key allowlist**: only `max_steps`, `max_output_chars`,
   `exec_timeout_sec`. `agent_model` and `system_prompt` may not be smuggled through
   `config_changes` (the prompt is set by the top-level `system_prompt` key; the
   model is fixed by the job). An unknown or disallowed key is a rejected proposal →
   retry, not a crash.
4. **Exactly one retry.** Attempt 0 sends system+context. If parsing or `AgentSpec`
   validation fails, attempt 1 re-sends the same messages plus the model's rejected
   reply and a user turn containing the normalized error text. A second failure
   raises `ImproverError`. A transport-level exception from `litellm.completion`
   raises `ImproverError` immediately (no retry — retrying transport belongs to
   litellm itself).

- [ ] **Step 1: Write the failing test**
```python
"""Tests for FakeImprover, LLMImprover and create_improver (no network)."""

from __future__ import annotations

import json

import pytest

from api.agent_spec import AgentSpec
from api.config import BenchmarkConfig
from api.job_store import IterationRecord
from api.services import improver as improver_mod
from api.services.improver import (
    EvaluationSummary,
    FakeImprover,
    ImproverError,
    LLMImprover,
    Proposal,
    TaskOutcome,
    create_improver,
)


def _spec(system_prompt: str = "BASE PROMPT") -> AgentSpec:
    return AgentSpec(system_prompt=system_prompt, agent_model="gpt-4.1-mini", max_steps=80)


def _evaluation() -> EvaluationSummary:
    return EvaluationSummary(
        score=0.5,
        tasks=[
            TaskOutcome(task_id="t-pass", status="passed", reward=1.0, remarks=None),
            TaskOutcome(task_id="t-zero", status="failed", reward=0.0, remarks="Verifier failed"),
        ],
        traces={"t-zero": json.dumps([{"role": "tool", "content": "boom"}])},
    )


def _history() -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=0,
            agent_version_id="00000000-0000-0000-0000-000000000000",
            version=0,
            run_id="11111111-1111-1111-1111-111111111111",
            score=0.5,
            improved=True,
            rationale="baseline",
            changed_fields=[],
            status="completed",
        )
    ]


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _StubLitellm:
    """Stands in for the litellm module attribute; records every call."""

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def completion(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _Response(self.payloads[index])


def _payload(**overrides) -> str:  # noqa: ANN003
    body = {
        "system_prompt": "IMPROVED PROMPT",
        "config_changes": {"max_steps": 120},
        "rationale": "Added a verification step",
    }
    body.update(overrides)
    return json.dumps(body)


def test_fake_improver_returns_scripted_proposals_in_order() -> None:
    first = Proposal(spec=_spec("FIRST"), rationale="first")
    second = Proposal(spec=_spec("SECOND"), rationale="second")
    fake = FakeImprover([first, second])

    got_first = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())
    got_second = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert got_first is first
    assert got_second is second
    assert fake.calls == 2


def test_fake_improver_cycles_deterministic_revision_when_exhausted() -> None:
    fake = FakeImprover()

    first = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())
    second = fake.propose(spec=first.spec, evaluation=_evaluation(), history=_history())

    assert first.spec.system_prompt == "BASE PROMPT\n\n[fake-improver revision 1]"
    assert first.rationale == "fake improver deterministic revision 1"
    assert second.spec.system_prompt.endswith("[fake-improver revision 2]")
    # Exhaustion is not an error: FakeImprover keeps producing valid proposals.
    assert first.spec.max_steps == 80


def test_fake_improver_applies_mutate_callable() -> None:
    fake = FakeImprover(mutate=lambda spec: spec.model_copy(update={"max_steps": 150}))

    proposal = fake.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.max_steps == 150
    assert proposal.spec.system_prompt == "BASE PROMPT"


def test_llm_improver_merges_config_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm([_payload()])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "IMPROVED PROMPT"
    assert proposal.spec.max_steps == 120
    # Untouched fields are carried over from the base spec.
    assert proposal.spec.agent_model == "gpt-4.1-mini"
    assert proposal.spec.exec_timeout_sec == 120
    assert proposal.rationale == "Added a verification step"
    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == "gpt-5.4"
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    assert "BASE PROMPT" in stub.calls[0]["messages"][-1]["content"]
    assert llm.last_response == _payload()


def test_llm_improver_retries_once_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(["this is not json at all", _payload()])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.system_prompt == "IMPROVED PROMPT"
    assert len(stub.calls) == 2
    retry_text = stub.calls[1]["messages"][-1]["content"]
    assert "not valid JSON" in retry_text


def test_llm_improver_raises_after_two_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(["nope", "still nope"])
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    with pytest.raises(ImproverError) as excinfo:
        llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert "invalid proposal twice" in str(excinfo.value)
    assert len(stub.calls) == 2


def test_llm_improver_rejects_out_of_bounds_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(
        [
            _payload(config_changes={"max_steps": 9999}),
            _payload(config_changes={"max_steps": 120}),
        ]
    )
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.max_steps == 120
    assert len(stub.calls) == 2
    retry_text = stub.calls[1]["messages"][-1]["content"]
    assert "AgentSpec validation" in retry_text
    assert "max_steps" in retry_text


def test_llm_improver_rejects_unknown_config_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubLitellm(
        [
            _payload(config_changes={"agent_model": "gpt-4o", "tools": ["python"]}),
            _payload(),
        ]
    )
    monkeypatch.setattr(improver_mod, "litellm", stub)

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    proposal = llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert proposal.spec.agent_model == "gpt-4.1-mini"
    assert "unsupported keys" in stub.calls[1]["messages"][-1]["content"]


def test_llm_improver_wraps_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def completion(self, **kwargs):  # noqa: ANN003, ANN201
            raise RuntimeError("connection reset")

    monkeypatch.setattr(improver_mod, "litellm", _Boom())

    llm = LLMImprover(model="gpt-5.4", budget=20_000)
    with pytest.raises(ImproverError) as excinfo:
        llm.propose(spec=_spec(), evaluation=_evaluation(), history=_history())

    assert "connection reset" in str(excinfo.value)


def _config(backend: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        execution_backend=backend,
    )


def test_create_improver_returns_fake_for_mock_backend() -> None:
    assert isinstance(create_improver(_config("mock")), FakeImprover)


def test_create_improver_returns_llm_for_harbor_backend() -> None:
    improver = create_improver(_config("harbor"), improver_model="gpt-5.4-mini")

    assert isinstance(improver, LLMImprover)
    assert improver.model == "gpt-5.4-mini"
    assert improver.budget == 60_000
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_improver.py -v`
Expected: FAIL — collection error `ImportError: cannot import name 'FakeImprover' from 'api.services.improver'`

- [ ] **Step 3: Write the implementation**

First extend the import block at the top of `api/services/improver.py`:

```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from api.agent_spec import AgentSpec
from api.config import BenchmarkConfig, load_config
from api.job_store import IterationRecord

logger = logging.getLogger(__name__)
```

Then append everything below to the end of the module (after `build_context`):

```python
# --------------------------------------------------------------------------- #
# Improver implementations
# --------------------------------------------------------------------------- #

FAKE_CONTEXT_BUDGET = 8_000

_ALLOWED_CONFIG_KEYS = frozenset({"max_steps", "max_output_chars", "exec_timeout_sec"})

IMPROVER_SYSTEM_PROMPT = (
    "You are an optimization engine for an autonomous terminal-using coding agent.\n"
    "You are given the agent's current specification, the history of previous "
    "attempts with their scores, and the failures from the most recent benchmark "
    "evaluation. Propose ONE focused change most likely to raise the mean reward.\n"
    "\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"system_prompt": "<the FULL replacement system prompt>", '
    '"config_changes": {"max_steps": 100}, '
    '"rationale": "<why this change addresses the observed failures>"}\n'
    "\n"
    "Rules:\n"
    "- system_prompt must be the complete new prompt, never a diff or a patch.\n"
    "- config_changes may only contain: max_steps (1-200), max_output_chars "
    "(500-100000), exec_timeout_sec (10-1200). Omit any key you do not change; "
    "use {} to change nothing.\n"
    "- Never propose a different model and never invent other keys.\n"
    "- Do not repeat a change the iteration history shows already failed to "
    "improve the score.\n"
    "- The agent has exactly one tool (bash). Do not ask for other tools."
)


class _ProposalRejected(Exception):
    """Internal: the model's reply was unusable; triggers the single retry."""


# litellm is imported lazily into this global by _litellm(). Keeping it a module
# attribute (rather than a local import) is what lets tests swap it out with
# monkeypatch.setattr(improver_mod, "litellm", stub) without the real package
# ever being imported. Lazy so that importing api.services (FastAPI app, every
# Postgres test) never requires litellm to be installed.
litellm: Any = None


def _litellm() -> Any:
    global litellm
    if litellm is None:
        import litellm as _litellm_mod

        litellm = _litellm_mod
    return litellm


def _extract_content(response: Any) -> str:
    """Pull the assistant text out of a litellm completion response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise ImproverError("improver LLM returned no choices")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ImproverError("improver LLM returned no text content")
    return content


class FakeImprover:
    """
    Deterministic improver for tests (mirrors MockBenchmarkRunner's role).

    Per call, in precedence order:
      1. ``mutate`` is applied to the incoming spec when supplied;
      2. otherwise the next scripted proposal is returned;
      3. otherwise (list exhausted) a deterministic derived proposal is returned:
         the incoming spec with ``[fake-improver revision N]`` appended to the
         system prompt. It never raises and never runs out.
    """

    def __init__(
        self,
        proposals: list[Proposal] | None = None,
        *,
        mutate: Callable[[AgentSpec], AgentSpec] | None = None,
    ) -> None:
        self._proposals = list(proposals or [])
        self._mutate = mutate
        self.calls = 0
        self.last_prompt = ""
        self.last_response = ""

    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal:
        self.calls += 1
        n = self.calls

        if self._mutate is not None:
            proposal = Proposal(spec=self._mutate(spec), rationale=f"fake improver mutation {n}")
        elif n <= len(self._proposals):
            proposal = self._proposals[n - 1]
        else:
            merged = spec.model_dump()
            merged["system_prompt"] = f"{spec.system_prompt}\n\n[fake-improver revision {n}]"
            proposal = Proposal(
                spec=AgentSpec.model_validate(merged),
                rationale=f"fake improver deterministic revision {n}",
            )

        self.last_prompt = build_context(
            spec=spec,
            evaluation=evaluation,
            history=history,
            budget=FAKE_CONTEXT_BUDGET,
        )
        self.last_response = json.dumps(
            {
                "system_prompt": proposal.spec.system_prompt,
                "config_changes": {},
                "rationale": proposal.rationale,
            },
            indent=2,
        )
        return proposal


class LLMImprover:
    """Proposes the next AgentSpec with one litellm JSON-mode call (+1 retry)."""

    def __init__(self, *, model: str, budget: int) -> None:
        self.model = model
        self.budget = budget
        self.last_prompt = ""
        self.last_response = ""

    def propose(
        self,
        *,
        spec: AgentSpec,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
    ) -> Proposal:
        client = _litellm()
        context = build_context(
            spec=spec,
            evaluation=evaluation,
            history=history,
            budget=self.budget,
        )
        self.last_prompt = context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": IMPROVER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        last_error = ""
        for attempt in (0, 1):
            if attempt == 1:
                retry_text = (
                    "Your previous response was rejected: "
                    + last_error
                    + "\nReturn a corrected JSON object with the same three keys "
                    "(system_prompt, config_changes, rationale) and nothing else."
                )
                messages = messages + [
                    {"role": "assistant", "content": self.last_response},
                    {"role": "user", "content": retry_text},
                ]
                self.last_prompt = context + "\n\n## RETRY\n" + retry_text

            try:
                response = client.completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:  # noqa: BLE001 - transport failure, no retry
                raise ImproverError(f"improver LLM call failed: {exc}") from exc

            text = _extract_content(response)
            self.last_response = text
            try:
                return self._parse(text, spec)
            except _ProposalRejected as exc:
                last_error = str(exc)
                logger.warning("improver proposal rejected (attempt %s): %s", attempt, last_error)

        raise ImproverError(f"improver returned an invalid proposal twice: {last_error}")

    def _parse(self, text: str, spec: AgentSpec) -> Proposal:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise _ProposalRejected(f"response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise _ProposalRejected("response was not valid JSON: expected a JSON object")

        changes = data.get("config_changes") or {}
        if not isinstance(changes, dict):
            raise _ProposalRejected("config_changes must be a JSON object")
        unknown = sorted(set(changes) - _ALLOWED_CONFIG_KEYS)
        if unknown:
            raise _ProposalRejected(
                f"config_changes contains unsupported keys: {unknown}; "
                f"allowed keys are {sorted(_ALLOWED_CONFIG_KEYS)}"
            )

        merged = spec.model_dump()
        prompt = data.get("system_prompt")
        if isinstance(prompt, str) and prompt.strip():
            merged["system_prompt"] = prompt
        merged.update(changes)

        try:
            new_spec = AgentSpec.model_validate(merged)
        except ValidationError as exc:
            raise _ProposalRejected(f"proposal failed AgentSpec validation: {exc}") from exc

        rationale = str(data.get("rationale") or "").strip() or "(no rationale provided)"
        return Proposal(spec=new_spec, rationale=rationale)


def create_improver(
    config: BenchmarkConfig | None = None,
    *,
    improver_model: str | None = None,
) -> Improver:
    """Factory: FakeImprover for the mock backend, LLMImprover otherwise."""
    cfg = config or load_config()
    if cfg.execution_backend == "mock":
        return FakeImprover()
    return LLMImprover(
        model=improver_model or cfg.improver_model,
        budget=cfg.improver_context_budget,
    )
```

- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_improver.py tests/test_improver_context.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**
```bash
git add api/services/improver.py tests/test_improver.py
git commit -m "feat: FakeImprover and litellm-backed LLMImprover"
```

---

### Task 10: StepExecutor + worker wiring

Executes both step types and teaches the worker to serve the step queue before the
legacy run queue. This is the task that closes the loop: after it, a job created
through `PostgresJobStore.create_job` runs to completion with nothing but repeated
`process_one` calls.

**Files:**
- Create: `worker/steps.py`
- Modify: `worker/main.py:13-22` (imports), `worker/main.py:39-76` (`process_one`),
  `worker/main.py:79-113` (`run_loop`)
- Test: `tests/test_job_worker.py`

**Interfaces:**
- Consumes:
  - `api.config.REPO_ROOT: Path`, `api.config.BenchmarkConfig` (fields used here:
    `execution_backend`, `improver_context_budget`; plus whatever
    `HarborBenchmarkRunner` reads itself).
  - `api.agent_spec.AgentSpec` (via `spec.model_dump()`).
  - `api.job_store.PostgresJobStore` — `claim_next_step(worker_id) -> StepRecord | None`,
    `get_job(job_id) -> JobRecord | None`,
    `complete_step_and_advance(step_id, outcome) -> None`,
    `fail_step(step_id, *, error_code, error_message) -> None`.
  - `api.job_store.StepRecord` — `step_id`, `job_id`, `type`, `iteration`,
    `agent_version_id`, `version`, `spec: AgentSpec`, `task_ids: list[str]`,
    `agent_model`, `improver_model`, `run_id`, `stale_after_sec`.
  - `api.job_store.JobRecord.iterations: list[IterationRecord]` (ordered by
    `iteration`; each has `run_id`, `score`, `status`).
  - `api.job_store.EvaluateOutcome(run_id: str, score: float | None, error_code: str | None = None, error_message: str | None = None)`.
  - `api.job_store.ImproveOutcome(spec: AgentSpec | None, rationale: str = "", error_code: str | None = None, error_message: str | None = None)`.
  - `api.services.artifacts.ArtifactStore` (`put(key, bytes|str|Path)`, `get(key) -> bytes`, `exists(key) -> bool`), `create_artifact_store(config=None)`, `trace_key(job_id, iteration, task_id)`, `improver_key(job_id, iteration, name)`.
  - `api.services.scoring.mean_reward(rewards: Iterable[float | None]) -> float`.
  - `api.services.improver` — `Improver`, `ImproverError`, `EvaluationSummary`,
    `TaskOutcome`, `build_context`, `create_improver`.
  - `api.store.PostgresRunStore` — `create(*, task_ids, agent_model) -> RunRecord`,
    `get(run_id) -> RunRecord | None`. `RunRecord` exposes `.status: RunStatus`,
    `.error: RunError | None`, `.tasks: list[TaskResult]` where `TaskResult` has
    `.task_id: str`, `.status: TaskStatus` (an enum — use `.status.value`),
    `.reward: float | None`, `.remarks: str | None`.
  - `api.services.runner` — `MockBenchmarkRunner(store, *, step_delay_sec=0.05)`,
    `HarborBenchmarkRunner(store, *, config, agent_import_path=None, extra_env=None)`
    (both expose `execute_sync(run_id) -> None`), `ExecutionUnavailableError`.
- Produces:
  - `worker.steps.StepExecutor(job_store, run_store, *, config, improver, artifacts, step_delay_sec=0.05)` with `execute(step: StepRecord) -> None`.
  - `worker.steps.SPEC_AGENT_IMPORT_PATH = "agent.spec_agent:HarnessAgent"`.
  - `worker.main.process_one(store, runner, *, worker_id, stale_after_sec, job_store=None, step_executor=None) -> bool` — **positional/keyword shape of the existing four parameters is unchanged**, so `tests/test_api.py` keeps passing untouched.

**Mechanics fixed here:**

- The evaluate step always creates a **fresh** run row, so a stale-requeued step
  re-runs cleanly; the step's `run_id` is reported through `EvaluateOutcome.run_id`,
  which `complete_step_and_advance` writes onto the step row.
- The spec is materialized to `workspace/runs/<run_id>/agent_spec.json`, which is
  also the `jobs_dir` `HarborBenchmarkRunner` uses for that run — the harbor job
  tree lands next to the spec file, so trace collection only has to walk that one
  directory.
- Traces are located by walking `workspace/runs/<run_id>` for any `trace.json`
  whose parent directory is named `agent` — harbor's layout is
  `<jobs_dir>/<harbor_job>/<task_id>__<trial>/agent/trace.json`, and the task id is
  `trial_dir.name.rsplit("__", 1)[0]`. Walking instead of hard-coding the depth
  keeps this robust to harbor changing its nesting. **The mock backend produces no
  traces at all**, which is fine: `EvaluationSummary.traces` is then `{}` and
  `build_context` renders `(no trace captured)`.
- Score is `mean_reward([t.reward for t in record.tasks])` — `None` rewards count
  as `0.0`, per the global constraint.
- If the run row itself came back failed (both runners write their own
  `RunError` before returning), its `code`/`message` are propagated into
  `EvaluateOutcome`, which fails the step *and* the job.
- The improve step reads the latest **completed** evaluate iteration from
  `job_store.get_job(...)` rather than trusting `step.run_id` (an improve step's own
  `run_id` is `None`).
- `result_key()` from Task 3 is deliberately unused by this section — only
  `trace_key()` and `improver_key()` are written here.

#### Cycle 1 — StepExecutor drives a mock job to completion

- [ ] **Step 1: Write the failing test**
```python
"""End-to-end job loop tests: Postgres + mock backend + FakeImprover."""

from __future__ import annotations

import json
import os
import shutil

import pytest
from sqlalchemy.exc import OperationalError

from api.config import REPO_ROOT, clear_config_cache, load_config
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import PostgresJobStore
from api.schemas import RunStatus, TaskStatus
from api.services.artifacts import LocalArtifactStore
from api.services.improver import FakeImprover, ImproverError
from api.services.runner import MockBenchmarkRunner
from api.store import PostgresRunStore
from worker.main import process_one
from worker.steps import StepExecutor

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)

# MockBenchmarkRunner buckets task_id by sha256 % 5: "fix-git" -> passed (1.0),
# "regex-log" -> failed (0.0). The mean reward is therefore exactly 0.5 on every
# iteration, so a mock job can never improve and always plateaus.
TASK_IDS = ["fix-git", "regex-log"]
PLATEAU_SCORE = 0.5


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)


@pytest.fixture()
def stores(tmp_path):
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    factory = get_session_factory()
    run_store = PostgresRunStore(session_factory=factory)
    job_store = PostgresJobStore(session_factory=factory)
    job_store.clear()
    run_store.clear()

    yield run_store, job_store, LocalArtifactStore(tmp_path)

    job_store.clear()
    run_store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


class _RaisingImprover:
    """Improver that always fails, to exercise the failed_improve path."""

    def propose(self, *, spec, evaluation, history):  # noqa: ANN001, ANN201
        raise ImproverError("no proposal today")


def _executor(run_store, job_store, artifacts, improver):  # noqa: ANN001, ANN201
    return StepExecutor(
        job_store,
        run_store,
        config=load_config(),
        improver=improver,
        artifacts=artifacts,
        step_delay_sec=0.0,
    )


def _make_job(job_store, *, max_iterations: int, patience: int):  # noqa: ANN001, ANN201
    return job_store.create_job(
        task_ids=list(TASK_IDS),
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=max_iterations,
        patience=patience,
        min_delta=0.01,
        max_job_duration_sec=3600,
        evaluate_stale_after_sec=1800,
    )


def _drain(run_store, job_store, executor, runner, *, limit: int = 20) -> int:
    """Call process_one until it reports no work; returns the number of units done."""
    done = 0
    for _ in range(limit):
        did_work = process_one(
            run_store,
            runner,
            worker_id="worker-test",
            stale_after_sec=1800,
            job_store=job_store,
            step_executor=executor,
        )
        if not did_work:
            break
        done += 1
    return done


def _cleanup_run_dirs(job) -> None:  # noqa: ANN001
    for iteration in job.iterations:
        if iteration.run_id:
            shutil.rmtree(REPO_ROOT / "workspace" / "runs" / iteration.run_id, ignore_errors=True)


def test_mock_job_plateaus_and_stops_with_no_improvement(stores) -> None:
    run_store, job_store, artifacts = stores
    improver = FakeImprover()
    executor = _executor(run_store, job_store, artifacts, improver)
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    job = _make_job(job_store, max_iterations=3, patience=1)

    # evaluate(0) -> improve(0) -> evaluate(1) -> stop. Exactly three units.
    assert _drain(run_store, job_store, executor, runner) == 3

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "no_improvement"
    assert final.best_agent_version_id is not None
    assert final.best_version == 0
    assert final.best_score == pytest.approx(PLATEAU_SCORE)

    assert [it.iteration for it in final.iterations] == [0, 1]
    assert final.iterations[0].score == pytest.approx(PLATEAU_SCORE)
    assert final.iterations[0].improved is True
    assert final.iterations[0].changed_fields == []
    assert final.iterations[1].score == pytest.approx(PLATEAU_SCORE)
    assert final.iterations[1].improved is False
    assert final.iterations[1].changed_fields == ["system_prompt"]
    assert final.iterations[1].rationale == "fake improver deterministic revision 1"

    assert improver.calls == 1
    _cleanup_run_dirs(final)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_job_worker.py::test_mock_job_plateaus_and_stops_with_no_improvement -v`
Expected: FAIL — collection error `ModuleNotFoundError: No module named 'worker.steps'`

- [ ] **Step 3: Write the implementation**

Create `worker/steps.py`:

```python
"""Worker-side execution of job steps (evaluate / improve)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from api.agent_spec import AgentSpec
from api.config import REPO_ROOT, BenchmarkConfig
from api.job_store import (
    EvaluateOutcome,
    ImproveOutcome,
    IterationRecord,
    PostgresJobStore,
    StepRecord,
)
from api.services.artifacts import ArtifactStore, improver_key, trace_key
from api.services.improver import (
    EvaluationSummary,
    Improver,
    ImproverError,
    Proposal,
    TaskOutcome,
    build_context,
)
from api.services.runner import (
    ExecutionUnavailableError,
    HarborBenchmarkRunner,
    MockBenchmarkRunner,
)
from api.services.scoring import mean_reward
from api.store import PostgresRunStore

logger = logging.getLogger("worker.steps")

SPEC_AGENT_IMPORT_PATH = "agent.spec_agent:HarnessAgent"


class StepExecutor:
    """Executes one claimed step and advances the job in the same call."""

    def __init__(
        self,
        job_store: PostgresJobStore,
        run_store: PostgresRunStore,
        *,
        config: BenchmarkConfig,
        improver: Improver,
        artifacts: ArtifactStore,
        step_delay_sec: float = 0.05,
    ) -> None:
        self.job_store = job_store
        self.run_store = run_store
        self.config = config
        self.improver = improver
        self.artifacts = artifacts
        self.step_delay_sec = step_delay_sec

    # ----------------------------------------------------------------- #
    # Dispatch
    # ----------------------------------------------------------------- #

    def execute(self, step: StepRecord) -> None:
        if step.type == "evaluate":
            self._evaluate(step)
        elif step.type == "improve":
            self._improve(step)
        else:
            self.job_store.fail_step(
                step.step_id,
                error_code="internal_error",
                error_message=f"unknown step type {step.type!r}",
            )

    # ----------------------------------------------------------------- #
    # Evaluate
    # ----------------------------------------------------------------- #

    def _evaluate(self, step: StepRecord) -> None:
        record = self.run_store.create(
            task_ids=list(step.task_ids),
            agent_model=step.spec.agent_model,
        )
        run_id = record.run_id
        logger.info(
            "evaluate step_id=%s job_id=%s iteration=%s version=%s run_id=%s tasks=%s",
            step.step_id,
            step.job_id,
            step.iteration,
            step.version,
            run_id,
            step.task_ids,
        )

        try:
            spec_path = self._materialize_spec(run_id, step.spec)
            runner = self._build_runner(spec_path)
            runner.execute_sync(run_id)

            finished = self.run_store.get(run_id)
            if finished is None:
                raise RuntimeError(f"run {run_id} disappeared during evaluation")

            if finished.error is not None:
                outcome = EvaluateOutcome(
                    run_id=run_id,
                    score=None,
                    error_code=finished.error.code,
                    error_message=finished.error.message,
                )
            else:
                copied = self._store_traces(step, run_id, [t.task_id for t in finished.tasks])
                score = mean_reward([t.reward for t in finished.tasks])
                logger.info(
                    "evaluate done run_id=%s score=%.4f traces=%s",
                    run_id,
                    score,
                    copied,
                )
                outcome = EvaluateOutcome(run_id=run_id, score=score)
        except ExecutionUnavailableError as exc:
            logger.error("evaluate unavailable step_id=%s: %s", step.step_id, exc)
            outcome = EvaluateOutcome(
                run_id=run_id,
                score=None,
                error_code="execution_unavailable",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evaluate failed step_id=%s", step.step_id)
            outcome = EvaluateOutcome(
                run_id=run_id,
                score=None,
                error_code="internal_error",
                error_message=str(exc),
            )

        self.job_store.complete_step_and_advance(step.step_id, outcome)

    def _run_dir(self, run_id: str) -> Path:
        return REPO_ROOT / "workspace" / "runs" / run_id

    def _materialize_spec(self, run_id: str, spec: AgentSpec) -> Path:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "agent_spec.json"
        path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")
        return path

    def _build_runner(self, spec_path: Path) -> MockBenchmarkRunner | HarborBenchmarkRunner:
        if self.config.execution_backend == "mock":
            return MockBenchmarkRunner(store=self.run_store, step_delay_sec=self.step_delay_sec)
        return HarborBenchmarkRunner(
            self.run_store,
            config=self.config,
            agent_import_path=SPEC_AGENT_IMPORT_PATH,
            extra_env={
                "HARNESS_AGENT_SPEC": str(spec_path),
                "HARNESS_SAVE_TRACE": "1",
            },
        )

    def _store_traces(self, step: StepRecord, run_id: str, task_ids: list[str]) -> int:
        """
        Copy harbor trial traces into the artifact store.

        Harbor writes <jobs_dir>/<job>/<task_id>__<trial>/agent/trace.json; the
        mock backend writes nothing, in which case this is a no-op.
        """
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir():
            return 0
        known = set(task_ids)
        copied = 0
        for trace_path in sorted(run_dir.rglob("trace.json")):
            if trace_path.parent.name != "agent":
                continue
            task_id = trace_path.parent.parent.name.rsplit("__", 1)[0]
            if task_id not in known:
                continue
            try:
                self.artifacts.put(trace_key(step.job_id, step.iteration, task_id), trace_path)
                copied += 1
            except Exception:  # noqa: BLE001
                logger.warning("failed to store trace for task_id=%s run_id=%s", task_id, run_id)
        return copied

    # ----------------------------------------------------------------- #
    # Improve
    # ----------------------------------------------------------------- #

    def _improve(self, step: StepRecord) -> None:
        job = self.job_store.get_job(step.job_id)
        if job is None:
            self.job_store.fail_step(
                step.step_id,
                error_code="internal_error",
                error_message=f"job {step.job_id} disappeared",
            )
            return

        latest = self._latest_evaluation(job.iterations)
        if latest is None:
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="improver_failed",
                    error_message="no completed evaluation to improve on",
                ),
            )
            return

        record = self.run_store.get(latest.run_id or "")
        if record is None:
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="improver_failed",
                    error_message=f"evaluation run {latest.run_id} not found",
                ),
            )
            return

        evaluation = EvaluationSummary(
            score=float(latest.score or 0.0),
            tasks=[
                TaskOutcome(
                    task_id=t.task_id,
                    status=t.status.value,
                    reward=t.reward,
                    remarks=t.remarks,
                )
                for t in record.tasks
            ],
            traces=self._read_traces(step.job_id, latest.iteration, [t.task_id for t in record.tasks]),
        )
        history = list(job.iterations)

        logger.info(
            "improve step_id=%s job_id=%s iteration=%s from_score=%.4f traces=%s",
            step.step_id,
            step.job_id,
            step.iteration,
            evaluation.score,
            len(evaluation.traces),
        )

        try:
            proposal = self.improver.propose(
                spec=step.spec,
                evaluation=evaluation,
                history=history,
            )
        except ImproverError as exc:
            logger.error("improver failed step_id=%s: %s", step.step_id, exc)
            self._persist_improver_io(step, evaluation, history, error=str(exc))
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="improver_failed",
                    error_message=str(exc),
                ),
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("improve step crashed step_id=%s", step.step_id)
            self._persist_improver_io(step, evaluation, history, error=str(exc))
            self.job_store.complete_step_and_advance(
                step.step_id,
                ImproveOutcome(
                    spec=None,
                    error_code="internal_error",
                    error_message=str(exc),
                ),
            )
            return

        self._persist_improver_io(step, evaluation, history, proposal=proposal)
        self.job_store.complete_step_and_advance(
            step.step_id,
            ImproveOutcome(spec=proposal.spec, rationale=proposal.rationale),
        )

    @staticmethod
    def _latest_evaluation(iterations: list[IterationRecord]) -> IterationRecord | None:
        completed = [
            it
            for it in iterations
            if it.status == "completed" and it.run_id is not None and it.score is not None
        ]
        return completed[-1] if completed else None

    def _read_traces(self, job_id: str, iteration: int, task_ids: list[str]) -> dict[str, str]:
        traces: dict[str, str] = {}
        for task_id in task_ids:
            key = trace_key(job_id, iteration, task_id)
            try:
                if self.artifacts.exists(key):
                    traces[task_id] = self.artifacts.get(key).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                logger.warning("could not read trace artifact %s", key)
        return traces

    def _persist_improver_io(
        self,
        step: StepRecord,
        evaluation: EvaluationSummary,
        history: list[IterationRecord],
        *,
        proposal: Proposal | None = None,
        error: str | None = None,
    ) -> None:
        prompt = getattr(self.improver, "last_prompt", "") or build_context(
            spec=step.spec,
            evaluation=evaluation,
            history=history,
            budget=self.config.improver_context_budget,
        )
        if proposal is not None:
            body = json.dumps(
                {"rationale": proposal.rationale, "spec": proposal.spec.model_dump()},
                indent=2,
            )
        else:
            body = json.dumps(
                {
                    "error": error or "unknown improver error",
                    "raw_response": getattr(self.improver, "last_response", ""),
                },
                indent=2,
            )
        try:
            self.artifacts.put(improver_key(step.job_id, step.iteration, "prompt.txt"), prompt)
            self.artifacts.put(improver_key(step.job_id, step.iteration, "response.json"), body)
        except Exception:  # noqa: BLE001
            logger.warning(
                "failed to persist improver artifacts job_id=%s iteration=%s",
                step.job_id,
                step.iteration,
            )
```

Then rewrite `worker/main.py`'s import block (lines 13-22) to add the four new
imports:

```python
from api.config import clear_config_cache, load_config
from api.db import init_db
from api.job_store import PostgresJobStore
from api.schemas import RunError, RunStatus
from api.services.artifacts import create_artifact_store
from api.services.improver import create_improver
from api.services.runner import (
    ExecutionUnavailableError,
    HarborBenchmarkRunner,
    MockBenchmarkRunner,
    create_runner,
)
from api.store import PostgresRunStore, _utcnow
from worker.steps import StepExecutor
```

Replace `process_one` (lines 39-76) with the step-first version. The first four
parameters keep their exact names and positions, so existing callers and
`tests/test_api.py` are unaffected:

```python
def process_one(
    store: PostgresRunStore,
    runner: MockBenchmarkRunner | HarborBenchmarkRunner,
    *,
    worker_id: str,
    stale_after_sec: int,
    job_store: PostgresJobStore | None = None,
    step_executor: StepExecutor | None = None,
) -> bool:
    """
    Claim and execute one unit of work. Returns True if work was done.

    Job steps take priority; when no step is queued (or the worker was built
    without a job store) this falls back to the legacy standalone-run queue.
    """
    if job_store is not None and step_executor is not None:
        step = job_store.claim_next_step(worker_id)
        if step is not None:
            logger.info(
                "claimed step_id=%s type=%s job_id=%s iteration=%s",
                step.step_id,
                step.type,
                step.job_id,
                step.iteration,
            )
            try:
                step_executor.execute(step)
            except Exception as exc:  # noqa: BLE001
                logger.exception("step executor crashed step_id=%s", step.step_id)
                job_store.fail_step(
                    step.step_id,
                    error_code="internal_error",
                    error_message=str(exc),
                )
            return True

    run_id = store.claim_next(worker_id, stale_after_sec=stale_after_sec)
    if run_id is None:
        return False
    logger.info("claimed run_id=%s", run_id)
    try:
        runner.execute_sync(run_id)
    except ExecutionUnavailableError as exc:
        store.update(
            run_id,
            status=RunStatus.failed,
            finished_at=_utcnow(),
            error=RunError(code="execution_unavailable", message=str(exc)),
        )
        logger.error("execution unavailable run_id=%s: %s", run_id, exc)
    except Exception as exc:  # noqa: BLE001
        store.update(
            run_id,
            status=RunStatus.failed,
            finished_at=_utcnow(),
            error=RunError(code="internal_error", message=str(exc)),
        )
        logger.exception("worker failed run_id=%s", run_id)

    record = store.get(run_id)
    logger.info(
        "finished run_id=%s status=%s",
        run_id,
        record.status.value if record else "missing",
    )
    return True
```

Replace `run_loop` (lines 79-113) so it builds the job-side collaborators and
passes them through:

```python
def run_loop(
    *,
    poll_interval: float = 1.0,
    stale_after_sec: int = 1800,
    step_delay_sec: float = 0.05,
    max_jobs: int | None = None,
) -> None:
    clear_config_cache()
    cfg = load_config()
    init_db()
    store = PostgresRunStore()
    runner = create_runner(store, config=cfg, step_delay_sec=step_delay_sec)
    job_store = PostgresJobStore()
    artifacts = create_artifact_store(cfg)
    improver = create_improver(cfg)
    step_executor = StepExecutor(
        job_store,
        store,
        config=cfg,
        improver=improver,
        artifacts=artifacts,
        step_delay_sec=step_delay_sec,
    )
    worker_id = default_worker_id()
    logger.info(
        "worker starting id=%s backend=%s env_provider=%s improver=%s",
        worker_id,
        cfg.execution_backend,
        cfg.env_provider,
        type(improver).__name__,
    )

    jobs_done = 0
    while not _shutdown:
        did_work = process_one(
            store,
            runner,
            worker_id=worker_id,
            stale_after_sec=stale_after_sec,
            job_store=job_store,
            step_executor=step_executor,
        )
        if did_work:
            jobs_done += 1
            if max_jobs is not None and jobs_done >= max_jobs:
                logger.info("reached max_jobs=%s; exiting", max_jobs)
                break
            continue
        time.sleep(poll_interval)
```

- [ ] **Step 4: Run the cycle-1 test**
Run: `pytest tests/test_job_worker.py::test_mock_job_plateaus_and_stops_with_no_improvement -v`
Expected: PASS

#### Cycle 2 — the evaluate step materializes the spec

- [ ] **Step 5: Add the failing test**

Append to `tests/test_job_worker.py`:

```python
def test_evaluate_step_materializes_agent_spec(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, FakeImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # max_iterations=1 stops right after the baseline evaluation.
    job = _make_job(job_store, max_iterations=1, patience=2)
    assert _drain(run_store, job_store, executor, runner) == 1

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "max_iterations"

    iteration = final.iterations[0]
    assert iteration.run_id is not None
    spec_path = REPO_ROOT / "workspace" / "runs" / iteration.run_id / "agent_spec.json"
    assert spec_path.is_file()

    written = json.loads(spec_path.read_text(encoding="utf-8"))
    version = job_store.get_agent_version(iteration.agent_version_id)
    assert written["system_prompt"] == version.spec.system_prompt
    assert written["agent_model"] == "gpt-4.1-mini"
    assert written["max_steps"] == version.spec.max_steps

    _cleanup_run_dirs(final)
```

- [ ] **Step 6: Run it**
Run: `pytest tests/test_job_worker.py::test_evaluate_step_materializes_agent_spec -v`
Expected: PASS — no new implementation needed; this pins `_materialize_spec`'s
path and contents, which Task 4's `agent/spec_agent.py` reads via
`HARNESS_AGENT_SPEC`.

#### Cycle 3 — the legacy standalone-run path still works

- [ ] **Step 7: Add the regression test**

Append to `tests/test_job_worker.py`:

```python
def test_process_one_falls_back_to_standalone_run(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, FakeImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # No job exists, so no step is claimable: the plain /v1/runs path must run.
    record = run_store.create(task_ids=["fix-git"], agent_model="gpt-4.1-mini")

    assert process_one(
        run_store,
        runner,
        worker_id="worker-test",
        stale_after_sec=1800,
        job_store=job_store,
        step_executor=executor,
    ) is True

    final = run_store.get(record.run_id)
    assert final.status == RunStatus.completed
    assert final.tasks[0].task_id == "fix-git"
    assert final.tasks[0].status == TaskStatus.passed
    assert final.tasks[0].reward == pytest.approx(1.0)
```

- [ ] **Step 8: Run it**
Run: `pytest tests/test_job_worker.py::test_process_one_falls_back_to_standalone_run -v`
Expected: PASS

#### Cycle 4 — a failing improver ends the job as failed_improve

- [ ] **Step 9: Add the failing test**

Append to `tests/test_job_worker.py`:

```python
def test_improver_error_completes_job_with_failed_improve(stores) -> None:
    run_store, job_store, artifacts = stores
    executor = _executor(run_store, job_store, artifacts, _RaisingImprover())
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    # patience=2 keeps the loop alive past the baseline evaluation, so the
    # improve step actually gets to run and fail.
    job = _make_job(job_store, max_iterations=3, patience=2)

    # evaluate(0) -> improve(0) fails -> job closes. Exactly two units.
    assert _drain(run_store, job_store, executor, runner) == 2

    final = job_store.get_job(job.job_id)
    assert final.status == "completed"
    assert final.stop_reason == "failed_improve"
    # The best-so-far agent is still a valid answer.
    assert final.best_agent_version_id is not None
    assert final.best_score == pytest.approx(PLATEAU_SCORE)
    assert [it.iteration for it in final.iterations] == [0]

    # The improver failure is recorded as an artifact for auditability.
    response = json.loads(artifacts.get("jobs/%s/iterations/0/improver/response.json" % job.job_id))
    assert response["error"] == "no proposal today"

    _cleanup_run_dirs(final)
```

- [ ] **Step 10: Run it**
Run: `pytest tests/test_job_worker.py::test_improver_error_completes_job_with_failed_improve -v`
Expected: PASS

- [ ] **Step 11: Run the full suite to verify nothing regressed**
Run: `pytest tests/ -v`
Expected: PASS — including the pre-existing `tests/test_api.py` worker tests, which
call `process_one` with the original four arguments.

- [ ] **Step 12: Commit**
```bash
git add worker/steps.py worker/main.py tests/test_job_worker.py
git commit -m "feat: StepExecutor executes evaluate/improve steps and worker claims steps first"
```

---

**Additions to the contract made by this section** (nothing renamed, nothing
dropped):

- `worker.steps.SPEC_AGENT_IMPORT_PATH = "agent.spec_agent:HarnessAgent"` — the
  constant form of the string the contract spells inline.
- `api/services/improver.py` module-level names beyond the contract, all internal:
  `SPEC_HEADER`, `HISTORY_HEADER`, `HISTORY_COLUMNS`, `TASKS_COLUMNS`,
  `FAILURES_HEADER`, `IMPROVER_SYSTEM_PROMPT`, `FAKE_CONTEXT_BUDGET`,
  `_ALLOWED_CONFIG_KEYS`, `_ProposalRejected`, `_litellm()`, `litellm` (the
  monkeypatch seam).
- `FakeImprover` and `LLMImprover` both expose `last_prompt` / `last_response`;
  `FakeImprover` also exposes `calls`. `StepExecutor._persist_improver_io` reads
  `last_prompt` / `last_response` via `getattr(..., "")`, so any object satisfying
  the bare `Improver` protocol still works.
- `process_one` gains two keyword-only optional parameters (`job_store`,
  `step_executor`); the existing four are untouched.
- `api/services/__init__.py` is deliberately **not** extended with improver
  re-exports — doing so would make every `import api.services` pull in
  `api.job_store`. Import `api.services.improver` directly.

---

## Section D — API Surface (Tasks 11-13)

This section adds the public HTTP surface for iterative-improvement jobs: the
Pydantic models, `POST /v1/jobs`, `GET /v1/jobs/{job_id}`,
`GET /v1/jobs/{job_id}/best`, and `GET /v1/agent-versions/{version_id}`.

Everything here consumes the interfaces from Tasks 1-10 exactly as spelled in
`/tmp/m4-plan-parts/CONTRACT.md`. Nothing in this section changes existing
`/v1/runs`, `/tasks` or `/health` behaviour, and nothing here rewrites
`tests/test_api.py`.

**Repo facts you need (verified, do not re-derive):**

- `api/schemas.py` currently ends at line 134 (`class ErrorResponse`). It already
  defines `RunStatus`, `TaskStatus`, `RunSummary`, `TaskResult`, `RunError`,
  `ErrorDetail`, `ErrorResponse` and `CreateRunRequest` (whose `task_ids`
  validator you mirror at `api/schemas.py:51-56`).
- `api/routes/runs.py:23-25` is the `_error` helper; `api/routes/runs.py:28-29` is
  `_get_store`. Copy both shapes verbatim.
- `api/main.py:18-23` is the `create_app` signature; `:44-45` sets
  `app.state.store`; `:50-51` includes the routers; `:57-76` is the
  `RequestValidationError` handler that turns an empty `task_ids` into a **422**
  with `error.code == "empty_task_ids"` (this applies to jobs for free, because
  the handler keys off the `task_ids` location, not the model).
- `api/store.py:32` `compute_summary(tasks: list[TaskResult]) -> RunSummary`.
- `api/store.py:197-200` — `PostgresRunStore.get` returns `None` when
  `UUID(run_id)` raises `ValueError`. `PostgresJobStore.get_job` /
  `get_agent_version` do the same (Task 7), which is why a malformed id is a 404
  and not a 500.
- `api/db.py:75` `ping_db() -> bool`.
- `api/config.py` — `BenchmarkConfig` is a frozen dataclass with
  `known_task_ids` (property, `frozenset(default_task_ids)`), `default_task_ids`
  (16 ids), `default_agent_model`, `max_concurrency=2`, `per_task_timeout=1200`,
  `execution_backend`; Task 1 adds `improver_model`, `max_iterations`,
  `patience`, `min_delta`, `max_job_duration_sec`, `improver_context_budget`,
  `artifacts_dir`.
- Pydantic 2.13 accepts numeric constraints on optional fields
  (`int | None = Field(default=None, ge=1, le=50)`) — verified in this repo's
  venv, no `Annotated` gymnastics needed.

---

### Task 11: Pydantic schemas for jobs

**Files:**
- Modify: `api/schemas.py:134` (append a new `# ── Jobs ──` block after
  `class ErrorResponse`; nothing above line 134 changes)
- Test: `tests/test_job_schemas.py`

**Why a separate test file:** these tests are pure (no Postgres). Putting them
in `tests/test_jobs_api.py` would put them under that file's
`pytestmark = pytest.mark.skipif(not _postgres_available(), ...)` guard and they
would silently skip on a machine without Postgres. `tests/test_reward_mapping.py`
is the existing precedent for an unguarded pure-unit test file.

**Interfaces:**

- Consumes (from Task 2, `api/agent_spec.py`):
  ```python
  class AgentSpec(BaseModel):
      model_config = ConfigDict(extra="forbid")
      system_prompt: str = Field(min_length=1, max_length=20_000)
      agent_model: str = Field(min_length=1, max_length=256)
      max_steps: int = Field(default=80, ge=1, le=200)
      max_output_chars: int = Field(default=8000, ge=500, le=100_000)
      exec_timeout_sec: int = Field(default=120, ge=10, le=1200)

  def baseline_spec(agent_model: str) -> AgentSpec
  ```
- Consumes (existing, `api/schemas.py`): `RunStatus`, `RunSummary`, `RunError`.
- Produces (used by Tasks 12-13 and by any client):
  ```python
  class CreateJobRequest(BaseModel):
      task_ids: list[str] | None
      agent_model: str | None
      improver_model: str | None
      max_iterations: int | None      # ge=1, le=50
      patience: int | None            # ge=1, le=10
      min_delta: float | None         # ge=0.0, lt=1.0

  class CreateJobResponse(BaseModel):
      job_id: str; status: RunStatus; created_at: datetime

  class JobConfigEcho(BaseModel):
      task_ids: list[str]; agent_model: str; improver_model: str
      max_iterations: int; patience: int; min_delta: float

  class JobBest(BaseModel):
      agent_version_id: str; version: int; score: float | None

  class ProposalView(BaseModel):
      rationale: str; changed_fields: list[str]

  class IterationView(BaseModel):
      iteration: int; agent_version_id: str; version: int
      run_id: str | None; score: float | None; improved: bool | None
      summary: RunSummary | None; proposal: ProposalView | None

  class JobResponse(BaseModel):
      job_id: str; status: RunStatus; created_at: datetime
      started_at: datetime | None; finished_at: datetime | None
      config: JobConfigEcho; current_iteration: int
      best: JobBest | None; stop_reason: str | None
      iterations: list[IterationView]; error: RunError | None

  class AgentSpecView(BaseModel):
      system_prompt: str; agent_model: str; max_steps: int
      max_output_chars: int; exec_timeout_sec: int

  class BestAgentResponse(BaseModel):
      job_id: str; agent_version_id: str; version: int
      score: float | None; rationale: str; spec: AgentSpecView

  class AgentVersionResponse(BaseModel):
      agent_version_id: str; job_id: str; version: int
      parent_version_id: str | None; rationale: str; created_by: str
      created_at: datetime; spec: AgentSpecView
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_schemas.py`:

```python
"""Pure schema tests for the job API models (no database required)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.agent_spec import baseline_spec
from api.schemas import (
    AgentSpecView,
    CreateJobRequest,
    JobConfigEcho,
    JobResponse,
    RunStatus,
)


def test_create_job_request_accepts_empty_body_and_applies_no_defaults() -> None:
    body = CreateJobRequest()
    assert body.task_ids is None
    assert body.agent_model is None
    assert body.improver_model is None
    assert body.max_iterations is None
    assert body.patience is None
    assert body.min_delta is None


def test_create_job_request_accepts_full_body() -> None:
    body = CreateJobRequest(
        task_ids=["fix-git", "regex-log"],
        agent_model="gpt-4.1-mini",
        improver_model="gpt-5.4",
        max_iterations=3,
        patience=1,
        min_delta=0.0,
    )
    assert body.task_ids == ["fix-git", "regex-log"]
    assert body.max_iterations == 3
    assert body.patience == 1
    assert body.min_delta == 0.0


def test_create_job_request_rejects_empty_task_ids() -> None:
    with pytest.raises(ValidationError) as exc:
        CreateJobRequest(task_ids=[])
    assert "non-empty" in str(exc.value)


def test_create_job_request_allows_null_task_ids() -> None:
    assert CreateJobRequest(task_ids=None).task_ids is None


@pytest.mark.parametrize("value", [0, -1, 51])
def test_create_job_request_rejects_out_of_range_max_iterations(value: int) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(max_iterations=value)


@pytest.mark.parametrize("value", [0, 11])
def test_create_job_request_rejects_out_of_range_patience(value: int) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(patience=value)


@pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
def test_create_job_request_rejects_out_of_range_min_delta(value: float) -> None:
    with pytest.raises(ValidationError):
        CreateJobRequest(min_delta=value)


def test_job_response_serializes_with_empty_iterations() -> None:
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    response = JobResponse(
        job_id="11111111-1111-1111-1111-111111111111",
        status=RunStatus.queued,
        created_at=now,
        config=JobConfigEcho(
            task_ids=["fix-git"],
            agent_model="gpt-4.1-mini",
            improver_model="gpt-5.4",
            max_iterations=5,
            patience=2,
            min_delta=0.01,
        ),
        current_iteration=0,
    )
    dumped = response.model_dump()
    assert dumped["iterations"] == []
    assert dumped["best"] is None
    assert dumped["stop_reason"] is None
    assert dumped["started_at"] is None
    assert dumped["finished_at"] is None
    assert dumped["error"] is None
    assert dumped["config"]["task_ids"] == ["fix-git"]


def test_agent_spec_view_round_trips_every_agent_spec_field() -> None:
    spec = baseline_spec("gpt-4.1-mini")
    view = AgentSpecView(**spec.model_dump())
    assert view.model_dump() == spec.model_dump()
    assert set(view.model_dump()) == {
        "system_prompt",
        "agent_model",
        "max_steps",
        "max_output_chars",
        "exec_timeout_sec",
    }
    assert view.agent_model == "gpt-4.1-mini"
    assert view.system_prompt == spec.system_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_schemas.py -v`

Expected: FAIL at collection with
`ImportError: cannot import name 'AgentSpecView' from 'api.schemas'`
(after Task 2 exists; if `api/agent_spec.py` is also missing you get
`ModuleNotFoundError: No module named 'api.agent_spec'` — finish Task 2 first).

- [ ] **Step 3: Write the implementation**

Append to `api/schemas.py` (after line 134, i.e. after `class ErrorResponse`):

```python
# ── Jobs: iterative optimization loop (Milestone 4) ────────────────────────


class CreateJobRequest(BaseModel):
    """Body for POST /v1/jobs. Every field is optional; omitted fields fall
    back to config defaults at the route layer, not here."""

    task_ids: list[str] | None = Field(
        default=None,
        description="Tasks to evaluate each iteration. Omit or null to use the configured default subset.",
    )
    agent_model: str | None = Field(
        default=None,
        description="Optional LLM model override for the harness agent (spec v0).",
    )
    improver_model: str | None = Field(
        default=None,
        description="Optional LLM model override for the improver.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum evaluate iterations before stopping.",
    )
    patience: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Consecutive non-improving evaluations tolerated before stopping.",
    )
    min_delta: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Score increase required to count an iteration as an improvement.",
    )

    @field_validator("task_ids")
    @classmethod
    def task_ids_not_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("task_ids must be non-empty when provided")
        return value


class CreateJobResponse(BaseModel):
    job_id: str
    status: RunStatus
    created_at: datetime


class JobConfigEcho(BaseModel):
    task_ids: list[str]
    agent_model: str
    improver_model: str
    max_iterations: int
    patience: int
    min_delta: float


class JobBest(BaseModel):
    agent_version_id: str
    version: int
    score: float | None = None


class AgentSpecView(BaseModel):
    """Read-only mirror of api.agent_spec.AgentSpec for API responses."""

    system_prompt: str
    agent_model: str
    max_steps: int
    max_output_chars: int
    exec_timeout_sec: int


class ProposalView(BaseModel):
    rationale: str
    changed_fields: list[str] = Field(default_factory=list)


class IterationView(BaseModel):
    iteration: int
    agent_version_id: str
    version: int
    run_id: str | None = None
    score: float | None = None
    improved: bool | None = None
    summary: RunSummary | None = None
    proposal: ProposalView | None = None


class JobResponse(BaseModel):
    job_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    config: JobConfigEcho
    current_iteration: int = 0
    best: JobBest | None = None
    stop_reason: str | None = None
    iterations: list[IterationView] = Field(default_factory=list)
    error: RunError | None = None


class BestAgentResponse(BaseModel):
    job_id: str
    agent_version_id: str
    version: int
    score: float | None = None
    rationale: str = ""
    spec: AgentSpecView


class AgentVersionResponse(BaseModel):
    agent_version_id: str
    job_id: str
    version: int
    parent_version_id: str | None = None
    rationale: str = ""
    created_by: str
    created_at: datetime
    spec: AgentSpecView
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_schemas.py -v`

Expected: PASS (12 tests, no skips — this file has no Postgres guard).

Also confirm nothing existing regressed:
Run: `pytest tests/test_reward_mapping.py tests/test_job_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/schemas.py tests/test_job_schemas.py
git commit -m "feat: add job API request/response schemas"
```

---

### Task 12: Job routes (`/v1/jobs`)

**Files:**
- Create: `api/routes/jobs.py`
- Modify: `api/main.py:13` (import `agent_versions`/`jobs` routers — the
  `agent_versions` import lands in Task 13), `api/main.py:15` (import
  `PostgresJobStore`), `api/main.py:18-23` (`create_app` signature gains
  `job_store`), `api/main.py:44-45` (set `app.state.job_store`),
  `api/main.py:50-51` (include the jobs router)
- Test: `tests/test_jobs_api.py`

**Interfaces:**

- Consumes (Task 7, `api/job_store.py`):
  ```python
  @dataclass(frozen=True)
  class StepRecord:
      step_id: str; job_id: str; type: str; iteration: int
      agent_version_id: str; version: int; spec: AgentSpec
      task_ids: list[str]; agent_model: str; improver_model: str
      run_id: str | None; stale_after_sec: int

  @dataclass(frozen=True)
  class AgentVersionRecord:
      version_id: str; job_id: str; version: int
      parent_version_id: str | None; spec: AgentSpec; rationale: str
      created_by: str; created_at: datetime

  @dataclass(frozen=True)
  class IterationRecord:
      iteration: int; agent_version_id: str; version: int
      run_id: str | None; score: float | None; improved: bool | None
      rationale: str | None; changed_fields: list[str]; status: str

  @dataclass(frozen=True)
  class JobRecord:
      job_id: str; status: str; task_ids: list[str]; agent_model: str
      improver_model: str; max_iterations: int; patience: int; min_delta: float
      current_iteration: int; best_agent_version_id: str | None
      best_version: int | None; best_score: float | None
      stop_reason: str | None; created_at: datetime
      started_at: datetime | None; finished_at: datetime | None
      error_code: str | None; error_message: str | None
      iterations: list[IterationRecord]

  @dataclass(frozen=True)
  class EvaluateOutcome:
      run_id: str; score: float | None
      error_code: str | None = None; error_message: str | None = None

  class PostgresJobStore:
      def __init__(self, session_factory=None) -> None
      def create_job(self, *, task_ids: list[str], agent_model: str,
                     improver_model: str, max_iterations: int, patience: int,
                     min_delta: float, max_job_duration_sec: int,
                     evaluate_stale_after_sec: int) -> JobRecord
      def get_job(self, job_id: str) -> JobRecord | None
      def get_agent_version(self, version_id: str) -> AgentVersionRecord | None
      def claim_next_step(self, worker_id: str) -> StepRecord | None
      def complete_step_and_advance(self, step_id: str, outcome) -> None
      def fail_step(self, step_id: str, *, error_code: str, error_message: str) -> None
      def clear(self) -> None
  ```
- Consumes (existing): `PostgresRunStore.get(run_id) -> RunRecord | None`
  (`RunRecord.tasks: list[TaskResult]`), `compute_summary(tasks) -> RunSummary`
  (both from `api.store`), `load_config()`, `ping_db()`.
- Produces:
  ```python
  # api/routes/jobs.py
  router: APIRouter                                    # prefix="/v1/jobs"
  def _get_job_store(request: Request) -> PostgresJobStore
  def evaluate_stale_after_sec(task_count: int, cfg: BenchmarkConfig) -> int
  def spec_view(spec: AgentSpec) -> AgentSpecView
  def _job_to_response(record: JobRecord,
                       run_store: PostgresRunStore | None = None) -> JobResponse
  # api/main.py
  def create_app(*, store: PostgresRunStore | None = None,
                 job_store: PostgresJobStore | None = None,
                 database_url: str | None = None,
                 init_database: bool = True) -> FastAPI
  ```
  `spec_view` and `_job_to_response` are imported by Task 13
  (`api/routes/agent_versions.py` imports `spec_view`).

**Design decision — per-iteration `summary` (read this before reviewing):**
`JobRecord`/`IterationRecord` carry no `RunSummary`; the design spec's
`GET /v1/jobs/{id}` example does show a per-iteration `summary`. Resolution:
`_job_to_response` takes an **optional** `run_store` and, when it is provided,
fills each `IterationView.summary` with
`compute_summary(run_store.get(it.run_id).tasks)` for every iteration that has a
`run_id`. The route always passes `request.app.state.store`, so the live API
matches the spec example. `run_store=None` (used by unit callers) leaves
`summary` as `None`. This costs one extra `runs` read per iteration; job payloads
have at most `max_iterations` (≤50) iterations, so the N+1 is bounded and
acceptable — do not "optimize" it into a join without a measurement.

**Design decision — `evaluate_stale_after_sec`:** computed at POST time from
config so the claimer needs no config context (spec §6). The exact expression is
```python
int(math.ceil(task_count / cfg.max_concurrency) * cfg.per_task_timeout * 1.2)
```
For the 16 default task ids at `max_concurrency=2`, `per_task_timeout=1200`:
`ceil(16/2) * 1200 * 1.2 = 8 * 1440 = 11520` seconds.

**Design decision — falsy overrides:** `min_delta=0.0`, and in principle any
numeric override, must be resolved with `is None` checks, **not** `or`. `0.0 or
cfg.min_delta` silently drops a caller's explicit `0.0`. `max_iterations` and
`patience` cannot be `0` (schema `ge=1`) but use the same `is None` form for
consistency.

#### Cycle A — POST /v1/jobs and GET /v1/jobs/{job_id}

- [ ] **Step 1: Write the failing test**

Create `tests/test_jobs_api.py`:

```python
"""API tests for the Milestone 4 iterative-improvement job endpoints."""

from __future__ import annotations

import math
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api.config import clear_config_cache, load_config
from api.db import get_engine, get_session_factory, init_db, reset_engine
from api.job_store import EvaluateOutcome, PostgresJobStore
from api.main import create_app
from api.store import PostgresRunStore

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://auto:auto@127.0.0.1:5432/auto_harness",
)


def _postgres_available() -> bool:
    reset_engine()
    try:
        engine = get_engine(url=DATABASE_URL, force_new=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False
    finally:
        reset_engine()


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not available (docker compose up -d postgres)",
)


@pytest.fixture()
def db_store() -> PostgresRunStore:
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["EXECUTION_BACKEND"] = "mock"
    clear_config_cache()
    reset_engine()
    init_db(url=DATABASE_URL)
    store = PostgresRunStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()
    reset_engine()
    clear_config_cache()
    os.environ.pop("EXECUTION_BACKEND", None)


@pytest.fixture()
def job_store(db_store: PostgresRunStore) -> PostgresJobStore:
    store = PostgresJobStore(session_factory=get_session_factory())
    store.clear()
    yield store
    store.clear()


@pytest.fixture()
def client(db_store: PostgresRunStore, job_store: PostgresJobStore) -> TestClient:
    clear_config_cache()
    app = create_app(
        store=db_store,
        job_store=job_store,
        database_url=DATABASE_URL,
        init_database=True,
    )
    with TestClient(app) as test_client:
        yield test_client
    clear_config_cache()


def test_post_job_with_empty_body_uses_config_defaults(client: TestClient) -> None:
    cfg = load_config()
    resp = client.post("/v1/jobs", json={})
    assert resp.status_code == 202
    created = resp.json()
    assert created["job_id"]
    assert created["status"] == "queued"
    assert created["created_at"]

    got = client.get(f"/v1/jobs/{created['job_id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["job_id"] == created["job_id"]
    assert body["status"] == "queued"
    assert body["config"]["task_ids"] == cfg.default_task_ids
    assert body["config"]["agent_model"] == cfg.default_agent_model
    assert body["config"]["improver_model"] == cfg.improver_model
    assert body["config"]["max_iterations"] == cfg.max_iterations
    assert body["config"]["patience"] == cfg.patience
    assert body["config"]["min_delta"] == cfg.min_delta


def test_post_job_honours_explicit_overrides(client: TestClient) -> None:
    resp = client.post(
        "/v1/jobs",
        json={
            "task_ids": ["fix-git", "regex-log"],
            "agent_model": "override-agent",
            "improver_model": "override-improver",
            "max_iterations": 2,
            "patience": 1,
            "min_delta": 0.0,
        },
    )
    assert resp.status_code == 202
    body = client.get(f"/v1/jobs/{resp.json()['job_id']}").json()
    assert body["config"] == {
        "task_ids": ["fix-git", "regex-log"],
        "agent_model": "override-agent",
        "improver_model": "override-improver",
        "max_iterations": 2,
        "patience": 1,
        "min_delta": 0.0,
    }


def test_post_job_unknown_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/jobs", json={"task_ids": ["not-a-real-task"]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unknown_task_ids"
    assert "not-a-real-task" in body["error"]["details"]["unknown"]


def test_post_job_empty_task_ids(client: TestClient) -> None:
    resp = client.post("/v1/jobs", json={"task_ids": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_task_ids"


def test_get_job_not_found(client: TestClient) -> None:
    resp = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"


def test_fresh_job_shows_queued_iteration_zero_and_no_best(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"task_ids": ["fix-git"]}).json()["job_id"]
    body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["current_iteration"] == 0
    assert body["best"] is None
    assert body["stop_reason"] is None
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["error"] is None
    # create_job enqueues an evaluate step at iteration 0, and get_job returns one
    # IterationRecord per evaluate step regardless of status — so a fresh job reports
    # exactly one iteration, still queued, with no score and no proposal yet.
    assert len(body["iterations"]) == 1
    it = body["iterations"][0]
    assert it["iteration"] == 0
    assert it["version"] == 0
    assert it["score"] is None
    assert it["run_id"] is None
    assert it["proposal"] is None
    assert it["summary"] is None


def test_evaluate_stale_after_sec_uses_config_formula(client: TestClient) -> None:
    cfg = load_config()
    from api.routes.jobs import evaluate_stale_after_sec

    expected = int(math.ceil(4 / cfg.max_concurrency) * cfg.per_task_timeout * 1.2)
    assert evaluate_stale_after_sec(4, cfg) == expected
```

> **Convention this pins down (RESOLVED — Task 7 must match):**
> `PostgresJobStore.get_job` emits one `IterationRecord` per evaluate step
> **regardless of the step's status**, so a freshly created job reports exactly one
> iteration: `iteration=0`, `version=0`, `status="queued"`, `run_id=None`,
> `score=None`, no proposal. This is what
> `test_fresh_job_shows_queued_iteration_zero_and_no_best` asserts. Rationale: an
> in-flight iteration should be visible to a polling client, which is the whole
> point of exposing history over the API — hiding a running iteration would make a
> job look idle for the hours an evaluate step takes. Task 7 must NOT filter queued
> or running steps out of `get_job`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jobs_api.py -v`

Expected: FAIL at collection with
`TypeError: create_app() got an unexpected keyword argument 'job_store'`
(or, if Task 7 is not merged yet,
`ModuleNotFoundError: No module named 'api.job_store'`).

- [ ] **Step 3: Write the implementation**

Create `api/routes/jobs.py`:

```python
"""Iterative-improvement job submission and status endpoints."""

from __future__ import annotations

import math

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.agent_spec import AgentSpec
from api.config import BenchmarkConfig, load_config
from api.db import ping_db
from api.job_store import JobRecord, PostgresJobStore
from api.schemas import (
    AgentSpecView,
    BestAgentResponse,
    CreateJobRequest,
    CreateJobResponse,
    ErrorDetail,
    ErrorResponse,
    IterationView,
    JobBest,
    JobConfigEcho,
    JobResponse,
    ProposalView,
    RunError,
    RunStatus,
)
from api.store import PostgresRunStore, compute_summary

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_job_store(request: Request) -> PostgresJobStore:
    return request.app.state.job_store


def _get_run_store(request: Request) -> PostgresRunStore:
    return request.app.state.store


def evaluate_stale_after_sec(task_count: int, cfg: BenchmarkConfig) -> int:
    """Worst-case wall clock for one evaluate step, plus 20% slack.

    Tasks run `max_concurrency` at a time, each bounded by `per_task_timeout`.
    """
    waves = math.ceil(task_count / cfg.max_concurrency)
    return int(waves * cfg.per_task_timeout * 1.2)


def spec_view(spec: AgentSpec) -> AgentSpecView:
    return AgentSpecView(**spec.model_dump())


def _job_to_response(
    record: JobRecord,
    run_store: PostgresRunStore | None = None,
) -> JobResponse:
    """Map a JobRecord onto the wire shape.

    When `run_store` is supplied, each iteration's `summary` is enriched from
    the run row that iteration produced; without it `summary` stays None.
    """
    iterations: list[IterationView] = []
    for it in record.iterations:
        summary = None
        if run_store is not None and it.run_id:
            run = run_store.get(it.run_id)
            if run is not None:
                summary = compute_summary(run.tasks)

        proposal = None
        if it.rationale is not None:
            proposal = ProposalView(
                rationale=it.rationale,
                changed_fields=list(it.changed_fields),
            )

        iterations.append(
            IterationView(
                iteration=it.iteration,
                agent_version_id=it.agent_version_id,
                version=it.version,
                run_id=it.run_id,
                score=it.score,
                improved=it.improved,
                summary=summary,
                proposal=proposal,
            )
        )

    best = None
    if record.best_agent_version_id is not None:
        best = JobBest(
            agent_version_id=record.best_agent_version_id,
            version=record.best_version if record.best_version is not None else 0,
            score=record.best_score,
        )

    error = None
    if record.error_code:
        error = RunError(code=record.error_code, message=record.error_message or "")

    return JobResponse(
        job_id=record.job_id,
        status=RunStatus(record.status),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        config=JobConfigEcho(
            task_ids=list(record.task_ids),
            agent_model=record.agent_model,
            improver_model=record.improver_model,
            max_iterations=record.max_iterations,
            patience=record.patience,
            min_delta=record.min_delta,
        ),
        current_iteration=record.current_iteration,
        best=best,
        stop_reason=record.stop_reason,
        iterations=iterations,
        error=error,
    )


@router.post(
    "",
    response_model=CreateJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_job(
    body: CreateJobRequest,
    request: Request,
) -> CreateJobResponse | JSONResponse:
    cfg = load_config()
    job_store = _get_job_store(request)

    if body.task_ids is None:
        task_ids = list(cfg.default_task_ids)
    else:
        unknown = [tid for tid in body.task_ids if tid not in cfg.known_task_ids]
        if unknown:
            return _error(
                400,
                "unknown_task_ids",
                f"Unknown task_ids: {unknown}",
                details={"unknown": unknown},
            )
        task_ids = list(body.task_ids)

    agent_model = body.agent_model or cfg.default_agent_model
    improver_model = body.improver_model or cfg.improver_model
    # `is None` (not `or`): an explicit min_delta=0.0 must survive.
    max_iterations = cfg.max_iterations if body.max_iterations is None else body.max_iterations
    patience = cfg.patience if body.patience is None else body.patience
    min_delta = cfg.min_delta if body.min_delta is None else body.min_delta

    if not ping_db():
        return _error(
            503,
            "execution_unavailable",
            "Database is unavailable; cannot enqueue job",
        )

    try:
        record = job_store.create_job(
            task_ids=task_ids,
            agent_model=agent_model,
            improver_model=improver_model,
            max_iterations=max_iterations,
            patience=patience,
            min_delta=min_delta,
            max_job_duration_sec=cfg.max_job_duration_sec,
            evaluate_stale_after_sec=evaluate_stale_after_sec(len(task_ids), cfg),
        )
    except Exception as exc:  # noqa: BLE001
        return _error(503, "execution_unavailable", f"Failed to enqueue job: {exc}")

    return CreateJobResponse(
        job_id=record.job_id,
        status=RunStatus.queued,
        created_at=record.created_at,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job(job_id: str, request: Request) -> JobResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_job(job_id)
    if record is None:
        return _error(404, "job_not_found", f"No job found with id {job_id}")
    return _job_to_response(record, run_store=_get_run_store(request))


@router.get(
    "/{job_id}/best",
    response_model=BestAgentResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def get_best_agent(job_id: str, request: Request) -> BestAgentResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_job(job_id)
    if record is None:
        return _error(404, "job_not_found", f"No job found with id {job_id}")
    if record.best_agent_version_id is None:
        return _error(
            409,
            "no_evaluation_yet",
            f"Job {job_id} has no completed evaluation yet",
        )

    version = job_store.get_agent_version(record.best_agent_version_id)
    if version is None:
        return _error(
            404,
            "agent_version_not_found",
            f"No agent version found with id {record.best_agent_version_id}",
        )

    return BestAgentResponse(
        job_id=record.job_id,
        agent_version_id=version.version_id,
        version=version.version,
        score=record.best_score,
        rationale=version.rationale,
        spec=spec_view(version.spec),
    )
```

Now edit `api/main.py`. Four edits, shown as before/after.

Line 13 — add the jobs router import (`agent_versions` is added in Task 13):

```python
# before
from api.routes import runs, tasks
# after
from api.routes import jobs, runs, tasks
```

Line 15 — import the job store:

```python
# before
from api.store import PostgresRunStore, store as default_store
# after
from api.job_store import PostgresJobStore
from api.store import PostgresRunStore, store as default_store
```

Lines 18-23 — signature:

```python
# before
def create_app(
    *,
    store: PostgresRunStore | None = None,
    database_url: str | None = None,
    init_database: bool = True,
) -> FastAPI:
# after
def create_app(
    *,
    store: PostgresRunStore | None = None,
    job_store: PostgresJobStore | None = None,
    database_url: str | None = None,
    init_database: bool = True,
) -> FastAPI:
```

Lines 44-45 and 50-51 — wire state and routers:

```python
# before
    run_store = store or default_store
    app.state.store = run_store

    # Eager-load config so misconfiguration fails at startup.
    load_config()

    app.include_router(tasks.router)
    app.include_router(runs.router)
# after
    run_store = store or default_store
    app.state.store = run_store
    # PostgresJobStore() resolves its session factory lazily, so constructing a
    # default here does not touch the database at import time.
    app.state.job_store = job_store or PostgresJobStore()

    # Eager-load config so misconfiguration fails at startup.
    load_config()

    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jobs_api.py -v`

Expected: PASS (7 tests).

Run: `pytest tests/test_api.py -v`

Expected: PASS (the existing suite calls `create_app(store=..., ...)`; the new
`job_store` kwarg is optional, so `app.state.job_store` falls back to a default
`PostgresJobStore()` bound to the same test engine).

#### Cycle B — GET /v1/jobs/{job_id}/best

- [ ] **Step 5: Write the failing test**

Append to `tests/test_jobs_api.py`:

```python
def test_best_is_409_before_any_evaluation(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"task_ids": ["fix-git"]}).json()["job_id"]
    resp = client.get(f"/v1/jobs/{job_id}/best")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "no_evaluation_yet"


def test_best_not_found_for_unknown_job(client: TestClient) -> None:
    resp = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000/best")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"


def test_best_returns_winning_spec_inline(
    client: TestClient,
    db_store: PostgresRunStore,
    job_store: PostgresJobStore,
) -> None:
    created = client.post(
        "/v1/jobs",
        json={"task_ids": ["fix-git", "regex-log"], "agent_model": "winning-model"},
    ).json()
    job_id = created["job_id"]

    # Complete iteration 0's evaluate step by hand (no worker involved).
    step = job_store.claim_next_step("manual-worker")
    assert step is not None
    assert step.type == "evaluate"
    assert step.iteration == 0
    assert step.version == 0
    run = db_store.create(task_ids=step.task_ids, agent_model=step.spec.agent_model)
    job_store.complete_step_and_advance(
        step.step_id,
        EvaluateOutcome(run_id=run.run_id, score=0.5),
    )

    resp = client.get(f"/v1/jobs/{job_id}/best")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["agent_version_id"] == step.agent_version_id
    assert body["version"] == 0
    assert body["score"] == pytest.approx(0.5)
    assert body["rationale"] == "baseline"
    assert body["spec"]["agent_model"] == "winning-model"
    assert body["spec"]["system_prompt"]
    assert body["spec"]["max_steps"] >= 1

    # The job view now shows the scored iteration and its run summary.
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["best"] == {
        "agent_version_id": step.agent_version_id,
        "version": 0,
        "score": pytest.approx(0.5),
    }
    assert len(job["iterations"]) == 1
    iteration = job["iterations"][0]
    assert iteration["iteration"] == 0
    assert iteration["run_id"] == run.run_id
    assert iteration["score"] == pytest.approx(0.5)
    assert iteration["improved"] is True
    assert iteration["proposal"] is None
    assert iteration["summary"]["total"] == 2
    assert iteration["summary"]["pending"] == 2
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_jobs_api.py::test_best_returns_winning_spec_inline -v`

Expected: FAIL — before the `/best` route exists the request falls through to
`GET /{job_id}` with `job_id="<uuid>/best"`, so the assertion breaks with
`assert 404 == 200`. (`test_best_is_409_before_any_evaluation` fails the same
way.)

- [ ] **Step 7: Confirm the implementation**

The `/best` route and the summary enrichment are already in the
`api/routes/jobs.py` written in Step 3 — no new production code is needed for
this cycle. If Step 6 still fails after Step 3 is in place, check route
registration order: `GET "/{job_id}/best"` and `GET "/{job_id}"` are distinct
literal paths, so declaration order does not matter, but both must be on the
same `router` instance.

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_jobs_api.py tests/test_api.py tests/test_job_schemas.py -v`

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add api/routes/jobs.py api/main.py tests/test_jobs_api.py
git commit -m "feat: add /v1/jobs create, status and best-agent endpoints"
```

---

### Task 13: Agent-version route and end-to-end API test

**Files:**
- Create: `api/routes/agent_versions.py`
- Modify: `api/main.py:13` (import `agent_versions`), `api/main.py:50-52`
  (include the new router)
- Test: `tests/test_jobs_api.py` (append)

**Interfaces:**

- Consumes:
  - `PostgresJobStore.get_agent_version(version_id) -> AgentVersionRecord | None`
    — returns `None` for a malformed id, mirroring `PostgresRunStore.get`'s
    `try: UUID(x) except ValueError: return None` at `api/store.py:197-200`.
  - `PostgresJobStore.claim_next_step(worker_id) -> StepRecord | None`.
  - `api.routes.jobs.spec_view(spec: AgentSpec) -> AgentSpecView` (Task 12).
  - Task 10, `worker/steps.py`:
    ```python
    class StepExecutor:
        def __init__(self, job_store: PostgresJobStore, run_store: PostgresRunStore, *,
                     config: BenchmarkConfig, improver: Improver,
                     artifacts: ArtifactStore, step_delay_sec: float = 0.05) -> None
        def execute(self, step: StepRecord) -> None
    ```
  - Task 10, `worker/main.py`:
    ```python
    def process_one(store: PostgresRunStore,
                    runner: MockBenchmarkRunner | HarborBenchmarkRunner, *,
                    worker_id: str, stale_after_sec: int,
                    job_store: PostgresJobStore | None = None,
                    step_executor: StepExecutor | None = None) -> bool
    ```
    Claims a step first (when `job_store` and `step_executor` are given), else
    falls back to the untouched standalone-run path. **This exact signature is an
    addition to the contract — see "Contract additions" at the end of this file.**
  - Task 3, `api/services/artifacts.py`: `LocalArtifactStore(root: Path | str)`.
  - Tasks 8-9, `api/services/improver.py`:
    `create_improver(config=None, *, improver_model=None) -> Improver` returning
    `FakeImprover` when `config.execution_backend == "mock"`;
    `FakeImprover(proposals: list[Proposal] | None = None, *, mutate=None)`.
- Produces:
  ```python
  # api/routes/agent_versions.py
  router: APIRouter    # prefix="/v1/agent-versions"
  ```

#### Cycle A — GET /v1/agent-versions/{version_id}

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs_api.py`:

```python
def test_agent_version_not_found_for_random_uuid(client: TestClient) -> None:
    resp = client.get("/v1/agent-versions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_version_not_found"


def test_agent_version_not_found_for_malformed_id(client: TestClient) -> None:
    resp = client.get("/v1/agent-versions/not-a-uuid")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_version_not_found"


def test_agent_version_returns_baseline_v0(
    client: TestClient, job_store: PostgresJobStore
) -> None:
    created = client.post(
        "/v1/jobs",
        json={"task_ids": ["fix-git"], "agent_model": "baseline-model"},
    ).json()

    step = job_store.claim_next_step("manual-worker")
    assert step is not None
    version_id = step.agent_version_id

    resp = client.get(f"/v1/agent-versions/{version_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_version_id"] == version_id
    assert body["job_id"] == created["job_id"]
    assert body["version"] == 0
    assert body["parent_version_id"] is None
    assert body["created_by"] == "baseline"
    assert body["rationale"] == "baseline"
    assert body["created_at"]
    assert body["spec"]["agent_model"] == "baseline-model"
    assert body["spec"]["system_prompt"]
    assert body["spec"]["max_steps"] >= 1
    assert body["spec"]["max_output_chars"] >= 500
    assert body["spec"]["exec_timeout_sec"] >= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jobs_api.py::test_agent_version_not_found_for_random_uuid -v`

Expected: FAIL with `assert 404 == 404` passing on status but
`KeyError: 'error'` — actually FastAPI returns
`{"detail":"Not Found"}` for an unregistered path, so the failure is
`KeyError: 'error'` on `resp.json()["error"]`.

- [ ] **Step 3: Write the implementation**

Create `api/routes/agent_versions.py`:

```python
"""Agent version lookup endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.job_store import PostgresJobStore
from api.routes.jobs import spec_view
from api.schemas import AgentVersionResponse, ErrorDetail, ErrorResponse

router = APIRouter(prefix="/v1/agent-versions", tags=["agent-versions"])


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _get_job_store(request: Request) -> PostgresJobStore:
    return request.app.state.job_store


@router.get(
    "/{version_id}",
    response_model=AgentVersionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_agent_version(
    version_id: str, request: Request
) -> AgentVersionResponse | JSONResponse:
    job_store = _get_job_store(request)
    record = job_store.get_agent_version(version_id)
    if record is None:
        # get_agent_version also returns None for a malformed UUID, mirroring
        # PostgresRunStore.get — a bad id is "not found", never a 500.
        return _error(
            404,
            "agent_version_not_found",
            f"No agent version found with id {version_id}",
        )

    return AgentVersionResponse(
        agent_version_id=record.version_id,
        job_id=record.job_id,
        version=record.version,
        parent_version_id=record.parent_version_id,
        rationale=record.rationale,
        created_by=record.created_by,
        created_at=record.created_at,
        spec=spec_view(record.spec),
    )
```

Edit `api/main.py` again — two edits:

```python
# before (line 13)
from api.routes import jobs, runs, tasks
# after
from api.routes import agent_versions, jobs, runs, tasks
```

```python
# before (router registration)
    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
# after
    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(agent_versions.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jobs_api.py -v`

Expected: PASS (13 tests).

#### Cycle B — one end-to-end API test through the worker

- [ ] **Step 5: Write the failing test**

Append to `tests/test_jobs_api.py` (and extend the import block at the top of
the file with the four new imports shown first):

```python
# ── add to the imports at the top of tests/test_jobs_api.py ────────────────
# from pathlib import Path
#
# from api.services.improver import FakeImprover, create_improver
# from api.services.runner import MockBenchmarkRunner
# from worker.main import process_one
# from worker.steps import StepExecutor


def _drive_job_to_completion(
    job_store: PostgresJobStore,
    run_store: PostgresRunStore,
    artifacts_root: Path,
    *,
    max_steps: int = 24,
) -> None:
    """Run the worker step loop synchronously until nothing is claimable."""
    cfg = load_config()
    assert cfg.execution_backend == "mock"

    improver = create_improver(cfg)
    assert isinstance(improver, FakeImprover)

    from api.services.artifacts import LocalArtifactStore

    executor = StepExecutor(
        job_store,
        run_store,
        config=cfg,
        improver=improver,
        artifacts=LocalArtifactStore(artifacts_root),
        step_delay_sec=0.0,
    )
    runner = MockBenchmarkRunner(store=run_store, step_delay_sec=0.0)

    for _ in range(max_steps):
        did_work = process_one(
            run_store,
            runner,
            worker_id="e2e-worker",
            stale_after_sec=1800,
            job_store=job_store,
            step_executor=executor,
        )
        if not did_work:
            return
    raise AssertionError(f"job did not settle within {max_steps} worker steps")


def test_job_end_to_end_through_worker(
    client: TestClient,
    db_store: PostgresRunStore,
    job_store: PostgresJobStore,
    tmp_path,
) -> None:
    created = client.post(
        "/v1/jobs",
        json={
            "task_ids": ["fix-git", "regex-log"],
            "agent_model": "e2e-model",
            "max_iterations": 3,
            "patience": 2,
            "min_delta": 0.0,
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    _drive_job_to_completion(job_store, db_store, tmp_path / "artifacts")

    body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["status"] == "completed"
    # Fully determined: MockBenchmarkRunner scores fix-git 1.0 and regex-log 0.0 every
    # time, so score is 0.5 at every iteration. With min_delta=0.0 iteration 0 improves
    # (best is None) and 1-2 do not; iteration 2 satisfies BOTH max_iterations and
    # patience, and stop precedence puts max_iterations first.
    assert body["stop_reason"] == "max_iterations"
    assert body["current_iteration"] == 2
    assert len(body["iterations"]) == 3
    assert [it["improved"] for it in body["iterations"]] == [True, False, False]
    assert body["best"]["version"] == 0
    assert body["best"]["score"] == pytest.approx(0.5)
    assert body["finished_at"] is not None
    assert body["error"] is None
    assert body["iterations"], "expected at least the baseline iteration"

    for index, iteration in enumerate(body["iterations"]):
        assert iteration["iteration"] == index
        assert iteration["run_id"]
        assert iteration["score"] is not None
        assert iteration["summary"] is not None
        assert iteration["summary"]["total"] == 2
        assert iteration["summary"]["pending"] == 0
    assert body["iterations"][0]["proposal"] is None
    assert body["iterations"][0]["improved"] is True

    best = client.get(f"/v1/jobs/{job_id}/best")
    assert best.status_code == 200
    best_body = best.json()
    assert best_body["spec"]["agent_model"] == "e2e-model"
    assert best_body["score"] == pytest.approx(body["best"]["score"])
    assert best_body["version"] == body["best"]["version"]

    version = client.get(f"/v1/agent-versions/{best_body['agent_version_id']}")
    assert version.status_code == 200
    assert version.json()["spec"] == best_body["spec"]
```

Notes for the reviewer, so none of this reads as accidental:

- `EXECUTION_BACKEND=mock` is set by the `db_store` fixture (copied from
  `tests/test_api.py:48`), which is why `create_improver(cfg)` returns
  `FakeImprover` and `StepExecutor` builds a `MockBenchmarkRunner` internally.
- `MockBenchmarkRunner._outcome_for` is a pure function of `task_id`
  (`api/services/runner.py:119-140`), so every iteration scores identically
  (0.5 for these two tasks). With `min_delta=0.0`, iteration 0 improves (best is
  `None`) and later iterations do not. The job terminates on `max_iterations` at
  iteration 2 — deterministically, so the assertion is a single literal.
  `failed_improve` is NOT reachable here: `FakeImprover` never raises and never
  exhausts (Task 9 gives it a deterministic `[fake-improver revision N]` fallback
  forever), so only an improver that raises can produce that stop reason — which is
  what Task 10's `_RaisingImprover` test covers instead.
- `best["version"]` is 0 here (no iteration beats the baseline under a
  deterministic mock), so the `/best` spec is the baseline spec and its
  `agent_model` is exactly the requested `"e2e-model"` — the property the test
  actually cares about (the request's model survives into the winning spec).
- `pending == 0` in every summary proves the enrichment really read the run rows
  the worker completed, not a placeholder.

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_jobs_api.py::test_job_end_to_end_through_worker -v`

Expected: FAIL with
`TypeError: process_one() got an unexpected keyword argument 'job_store'`
until Task 10 lands; after Task 10 it passes with no further production changes
in this section.

- [ ] **Step 7: Run the whole suite**

Run: `pytest -v`

Expected: PASS. If Postgres is down, `tests/test_api.py` and
`tests/test_jobs_api.py` skip and `tests/test_job_schemas.py` plus
`tests/test_reward_mapping.py` still pass.

- [ ] **Step 8: Commit**

```bash
git add api/routes/agent_versions.py api/main.py tests/test_jobs_api.py
git commit -m "feat: add /v1/agent-versions endpoint and job API end-to-end test"
```

---

## Final state of `api/main.py` (for reference)

After Tasks 12-13, `create_app` reads:

```python
def create_app(
    *,
    store: PostgresRunStore | None = None,
    job_store: PostgresJobStore | None = None,
    database_url: str | None = None,
    init_database: bool = True,
) -> FastAPI:
    ...
    run_store = store or default_store
    app.state.store = run_store
    app.state.job_store = job_store or PostgresJobStore()

    load_config()

    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(agent_versions.router)
```

The `RequestValidationError` handler at `api/main.py:57-76` is **not** modified:
it already maps any `task_ids` "non-empty" validation error to 422
`empty_task_ids`, which covers `CreateJobRequest` because it matches on the
error location, not the model class. Out-of-range `max_iterations` / `patience` /
`min_delta` therefore return 422 with the generic `validation_error` code and the
raw pydantic errors under `details.errors` — intentional, and consistent with how
runs report unexpected validation failures.

## Contract additions (flagged per CONTRACT.md's instruction)

1. **`worker.main.process_one` signature.** The contract states that
   `process_one` claims a step first and otherwise falls back to the existing
   run path, but does not spell the signature. This section's end-to-end test
   assumes:
   ```python
   def process_one(store: PostgresRunStore,
                   runner: MockBenchmarkRunner | HarborBenchmarkRunner, *,
                   worker_id: str, stale_after_sec: int,
                   job_store: PostgresJobStore | None = None,
                   step_executor: StepExecutor | None = None) -> bool
   ```
   Backwards compatible: `tests/test_api.py` calls it with the old four
   arguments and keeps working. If Task 10 chooses different keyword names,
   `_drive_job_to_completion` in `tests/test_jobs_api.py` is the single place to
   adjust.
2. **Default job store in `create_app`.** `api/store.py` exposes a module-level
   `store = PostgresRunStore()`; the contract does not require the same for jobs,
   so `create_app` constructs `PostgresJobStore()` inline when no store is
   injected. If Task 7 adds a module-level `job_store = PostgresJobStore()`,
   import it as `default_job_store` and use `job_store or default_job_store`
   instead — behaviourally identical.
3. **`get_job` and queued evaluate steps — RESOLVED.** `get_job` emits one
   `IterationRecord` per evaluate step regardless of status, so a fresh job reports
   one queued iteration (not an empty list).
   `test_fresh_job_shows_queued_iteration_zero_and_no_best` asserts this, and Task 7
   must implement it that way — an in-flight iteration has to be visible to a
   polling client. See the inline note in Task 12 Cycle A.
4. **`api/config.py` must export `BenchmarkConfig`** (it already does) —
   `api/routes/jobs.py` imports it for the `evaluate_stale_after_sec` type hint.

---

## Appendix — Cross-Section Reconciliations

The four sections were authored against a shared interface contract. Where two
sections documented alternative choices, **this appendix is the tie-breaker** —
it wins over any inline note.

### A1. `create_app`'s job-store fallback

Task 7 defines a module-level `job_store = PostgresJobStore()` in `api/job_store.py`,
mirroring `store = PostgresRunStore()` at `api/store.py:358`. Task 12's
`create_app` must therefore use the existing codebase pattern:

```python
from api.job_store import PostgresJobStore, job_store as default_job_store
...
app.state.job_store = job_store or default_job_store
```

Do **not** construct `PostgresJobStore()` inline in `create_app` (the alternative
Task 12 documents). Rationale: consistency with `run_store = store or default_store`
already in `api/main.py:44`.

### A2. Named constants, not string literals

Task 6 exports `STOP_MAX_ITERATIONS`, `STOP_NO_IMPROVEMENT`, `STOP_BUDGET_EXCEEDED`
from `api/services/scoring.py`. Task 7 exports `STEP_EVALUATE`, `STEP_IMPROVE`,
`IMPROVE_STALE_AFTER_SEC`, `CREATED_BY_BASELINE`, `CREATED_BY_IMPROVER`,
`STOP_FAILED_IMPROVE`, `STOP_FAILED` from `api/job_store.py`. Tasks 10-13 compare
against these names rather than bare strings. Response bodies still serialize the
underlying string values, so the API contract in the spec is unchanged.

### A3. `get_job` includes queued and running evaluate steps

**Resolved:** `get_job` emits one `IterationRecord` per evaluate step regardless of
status. A freshly created job reports exactly one iteration (`iteration=0`,
`version=0`, `status="queued"`, `run_id=None`, `score=None`, `rationale=None`,
`changed_fields=[]`).

Task 7 must not filter queued or running steps out, and Task 12's test is
`test_fresh_job_shows_queued_iteration_zero_and_no_best` asserting one queued
iteration — **not** an empty list. Rationale: an evaluate step can run for hours;
hiding an in-flight iteration would make a working job look idle to a polling
client, defeating the purpose of exposing history over the API.

### A4. Invalid proposals

`ImproveOutcome(spec=None)` with no `error_code` is normalized by Task 7 to
`error_code="invalid_proposal"`. Task 10 should still set an explicit
`error_code="improver_failed"` when `ImproverError` is raised, so the two failure
modes stay distinguishable in `steps.error_code`.

### A5. Insert ordering inside `create_job` (do not "simplify" this)

`JobRow`, `AgentVersionRow` and `StepRow` deliberately have **no** SQLAlchemy
`relationship()` between them. Without one, the unit of work orders mappers by sort
key and flushes `AgentVersionRow` before `JobRow`, producing
`ForeignKeyViolation: Key (job_id)=(...) is not present in table "jobs"` — this was
reproduced against live Postgres while writing Task 7. `create_job` therefore calls
`session.flush()` explicitly between the three adds. It remains ONE transaction;
the flushes only fix statement order. Do not remove them, and do not "fix" this by
adding relationships (`best_agent_version_id` and `parent_version_id` stay bare
UUIDs to avoid an FK cycle that would break `ON DELETE CASCADE` on jobs).

### A6. Config validation deliberately departs from the `or` house style

`api/config.py` currently reads values as `raw.get("x") or default`. Task 1 does **not**
use that pattern for the new numeric fields, and this is correct, not an oversight:
`0 or 5` evaluates to `5`, so `max_iterations: 0` would be silently accepted as `5`,
and `0.0 or 0.01` would silently rewrite a legal `min_delta: 0.0` to `0.01`. Task 1
introduces `_positive_int` / `_unit_fraction` helpers that test for `None` explicitly
and raise `ValueError` on out-of-range values. Keep them.

The same `is None` discipline applies to request-level overrides in Task 12 — see the
`min_delta` note there.

### A7. `agent/spec_loader.py` is a new module, and `agent/` stays a namespace package

The contract named only `agent/spec_agent.py`. Task 4 splits the spec loading into a
stdlib-only `agent/spec_loader.py` because `harbor` and `litellm` are **not installed**
in this environment, so `spec_agent.py` cannot be imported by a unit test at all. The
loader is tested directly; `spec_agent.py` gets a static source assertion only.

Do **not** add `agent/__init__.py`. `agent/` is currently a namespace package, and
`from agent.spec_loader import load_spec` already resolves under both pytest
(`pythonpath = ["."]`) and the harbor subprocess (repo root on `PYTHONPATH`).

### A8. `process_one`'s signature — Task 10 is authoritative

Task 10 keeps `process_one`'s four existing parameters positionally identical and adds
two optional keyword arguments:

```python
def process_one(store, runner, *, worker_id, stale_after_sec,
                job_store=None, step_executor=None) -> bool
```

This is why `tests/test_api.py` needs no edit. Task 13's end-to-end test must call it
with the `job_store=` / `step_executor=` keywords. A worker claims a **step first** and
falls back to the legacy standalone-run path only when no step is claimable.

### A9. Improver test doubles

`FakeImprover` **never raises** — `mutate` wins if given, else the scripted list in
order, else a deterministic `[fake-improver revision N]` prompt suffix forever. Tests
that need an improver failure use the separate `_RaisingImprover` stub defined in
Task 9, not a "exhausted FakeImprover". Task 13's `failed_improve` test must use
`_RaisingImprover`.

`litellm` is reached through a module-level `litellm` global populated lazily by
`_litellm()`, which keeps `api.services` importable without litellm installed and gives
tests the seam `monkeypatch.setattr(improver_mod, "litellm", stub)`. No test touches the
network.

### A10. `result_key` is defined but unused

Task 3 defines `result_key()` alongside `trace_key()`; Task 10 only writes traces and
improver artifacts. `result_key` is retained because per-task `result.json` is the
natural companion artifact and harbor already produces it — wire it up only if a later
milestone needs verifier detail. It is not dead code to be deleted, nor a gap to fill.

### A11. `api/services/__init__.py` is intentionally left alone

Task 10 does not re-export the new services there. Doing so would drag `api.job_store`
(and therefore the whole ORM) into every `import api.services`, which the Milestone 3
comment in that file was written to avoid.
