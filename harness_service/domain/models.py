"""Domain dataclasses — the lossless in-memory state model.

Design rule that guarantees reproducibility (PLAN.md §3a): an ``Iteration``
carries the ``Improvement`` that *produced* its ``AgentState``, plus the
``context_snapshot`` used to generate that improvement. Any iteration therefore
replays exactly — same source, same context, same LLM I/O — with no reference to
mutable outside state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from harness_service.constants import (
    MIN_IMPROVEMENT,
    IterationDecision,
    ProposerKind,
)


def source_hash(source: str) -> str:
    """Stable content hash of an agent source — used for dedup + reproducibility."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentState:
    """A reproducible snapshot of the agent at one iteration."""

    source: str  # the FULL agent.py text
    model: str
    reasoning_effort: str | None = None
    max_steps: int = 80
    max_output_chars: int = 8000
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            # frozen dataclass → bypass the setattr guard
            object.__setattr__(self, "content_hash", source_hash(self.source))

    @property
    def params(self) -> dict:
        """The non-source fields, for JSONB storage / display."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_steps": self.max_steps,
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True)
class TaskOutcome:
    """One task's result within a benchmark run."""

    task_id: str
    reward: float | None  # None = infra error/timeout → counts as 0.0 in val_score
    passed: bool
    duration_s: float | None = None
    trace_excerpt: str | None = None  # real per-task output → LLM context (M3)
    failure_reason: str | None = None  # one-line diagnosis for the summary

    @property
    def effective_reward(self) -> float:
        return 0.0 if self.reward is None else self.reward


@dataclass(frozen=True)
class BenchmarkResult:
    """A full set of task outcomes + derived aggregates."""

    outcomes: tuple[TaskOutcome, ...]

    @property
    def val_score(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.effective_reward for o in self.outcomes) / len(self.outcomes)

    @property
    def n_passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def n_failed(self) -> int:
        return len(self.outcomes) - self.n_passed

    @property
    def failures(self) -> tuple[TaskOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.passed)


@dataclass(frozen=True)
class Improvement:
    """The LLM proposal that produced an AgentState (None on the baseline)."""

    proposer: ProposerKind
    rationale: str
    diff_summary: str
    new_agent_source: str
    llm_request: dict = field(default_factory=dict)   # kept verbatim for audit
    llm_response: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Iteration:
    """One step of the loop — fully self-describing (see module docstring)."""

    idx: int
    agent_state: AgentState
    result: BenchmarkResult
    decision: IterationDecision
    decision_reason: str = ""
    improvement: Improvement | None = None  # the proposal that PRODUCED this state
    context_snapshot: str = ""              # accumulated context AS OF this iteration

    @property
    def val_score(self) -> float:
        return self.result.val_score


@dataclass(frozen=True)
class Trajectory:
    """The whole run. Ensures nothing is dropped between iterations."""

    iterations: tuple[Iteration, ...] = ()

    def with_iteration(self, it: Iteration) -> "Trajectory":
        return Trajectory(iterations=self.iterations + (it,))

    @property
    def best(self) -> Iteration | None:
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda it: it.val_score)

    @property
    def best_val_score(self) -> float:
        b = self.best
        return b.val_score if b else 0.0

    def is_improvement(self, candidate_score: float) -> bool:
        return candidate_score > self.best_val_score + MIN_IMPROVEMENT

    @property
    def consecutive_non_improving(self) -> int:
        """Trailing run of iterations that did not beat the best-so-far."""
        count = 0
        best = -1.0
        seen_best = False
        for it in self.iterations:
            if it.val_score > best + MIN_IMPROVEMENT:
                best = it.val_score
                count = 0
                seen_best = True
            elif seen_best:
                count += 1
        return count

    def build_context(self) -> str:
        """Fold prior attempts into the blob fed to the next LLM call."""
        lines: list[str] = []
        for it in self.iterations:
            head = f"[iter {it.idx}] val_score={it.val_score:.3f} ({it.decision.value})"
            if it.improvement:
                head += f" — {it.improvement.rationale.strip()[:200]}"
            lines.append(head)
            for f in it.result.failures:
                reason = (f.failure_reason or "").strip()[:160]
                lines.append(f"    FAIL {f.task_id}: {reason}")
        return "\n".join(lines)
