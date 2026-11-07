"""Optimizer contract + simulated implementation. The LLM-backed optimizer (M4) implements the same Protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class HistoryEntry:
    index: int
    hypothesis: str | None
    decision: str | None
    score: float | None
    flipped_to_pass: list[str] = field(default_factory=list)
    flipped_to_fail: list[str] = field(default_factory=list)
    decision_reason: str | None = None
    change_summary: str | None = None
    feedback_summary: dict[str, Any] | None = None
    source_job_id: str | None = None
    agent_sha256: str | None = None


@dataclass
class OptimizerInput:
    iteration_index: int
    agent_source: str                      # current best (parent)
    scoreboard: dict[str, list[bool | None]]  # task -> outcome per iteration (None = not run)
    failure_dossiers: list[dict]           # digests of failing tasks in the parent run
    history: list[HistoryEntry]
    learnings: str | None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    hypothesis: str
    targets: list[str]
    rationale: str
    new_source: str
    change_summary: str
    learnings: str | None = None
    raw: dict | None = None
    objective: str = "quality"

    def to_json(self) -> dict:
        return {"hypothesis": self.hypothesis, "targets": self.targets, "rationale": self.rationale,
                "change_summary": self.change_summary, "objective": self.objective}


class Optimizer(Protocol):
    name: str

    async def propose(self, inp: OptimizerInput) -> Proposal: ...


class SimulatedOptimizer:
    """Appends a marker line per iteration; the simulated executor reads those markers as 'improvement'."""
    name = "simulated"

    async def propose(self, inp: OptimizerInput) -> Proposal:
        classes: dict[str, int] = {}
        for d in inp.failure_dossiers:
            fc = (d.get("stats") or {}).get("failure_class") or "unknown"
            classes[fc] = classes.get(fc, 0) + 1
        top = max(classes, key=classes.get) if classes else "none"
        targets = [d["task_name"] for d in inp.failure_dossiers][:4]
        tried = {h.hypothesis for h in inp.history if h.decision == "rejected"}
        n = inp.iteration_index
        hyp = f"Address '{top}' failures (variant {n})"
        if hyp in tried:
            hyp += " [retry with different wording]"
        new_src = inp.agent_source.rstrip("\n") + f"\n# improvement {n}: {hyp}\n"
        return Proposal(hypothesis=hyp, targets=targets, rationale=f"{classes} observed in parent run",
                        new_source=new_src, change_summary=f"+1 line (simulated change for {top})",
                        learnings=f"iter {n}: most common failure class was {top}")
