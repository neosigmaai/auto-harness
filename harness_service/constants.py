"""Closed vocabularies (enums) and tunable defaults.

Every enum subclasses ``str`` so it serializes to JSON transparently and maps to
a plain VARCHAR column (no native Postgres enum types → no migration friction).
"""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobMode(str, Enum):
    SINGLE_RUN = "single_run"  # one benchmark run, no optimization (M1–M3)
    OPTIMIZE = "optimize"      # the iterative optimization loop (M4)


class ExecutorKind(str, Enum):
    SIMULATED = "simulated"    # deterministic dummy execution (M1 default)
    HARBOR = "harbor"          # real agent run in an E2B sandbox (M3)


class ProposerKind(str, Enum):
    OPENAI = "openai"          # OpenAI proposes agent.py improvements (M4)
    MOCK = "mock"              # deterministic fallback when no API key


class IterationDecision(str, Enum):
    BASELINE = "baseline"      # idx 0 — no proposal produced it
    ACCEPTED = "accepted"      # improvement kept (val_score rose)
    REJECTED = "rejected"      # worse/equal → discarded, but still persisted
    ERROR = "error"            # candidate didn't compile / run crashed


class Role(str, Enum):
    ADMIN = "admin"            # manages the org, sees all activity (M5)
    MEMBER = "member"          # submits jobs, sees only their own (M5)


# ── Tunable defaults (overridable per-job via the request body, or via env) ──
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_PATIENCE = 2               # stop after N consecutive non-improving iters
DEFAULT_SUBSET = "core"           # named task subset (see tasks.py)
TRACE_EXCERPT_CHARS = 4000        # per-task output retained as LLM context
MIN_IMPROVEMENT = 1e-9            # val_score delta that counts as "better"

# Client polling defaults (test_client.py)
CLIENT_POLL_INTERVAL_S = 3.0
CLIENT_POLL_TIMEOUT_S = 1800.0
