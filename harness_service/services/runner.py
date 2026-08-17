"""Run orchestrator.

SINGLE_RUN: one baseline benchmark run over the whole subset.
OPTIMIZE (M4): the iterative loop — baseline on TRAIN, then propose → compile-gate →
apply → re-run on TRAIN → accept-if-improved / reject-and-revert, until improvement
stalls (patience) or max_iterations. The best agent is finally scored on the held-out
TEST split to measure generalization.

Structural inspiration: the blog's optimization loop (analyze failures → propose harness
change → gate on val_score vs best-seen → accept/revert → stop on budget), deliberately
simplified for a take-home (see PLAN.md §M4-scale): no production-traffic batches, no
failure clustering, no separate regression suite — TRAIN performance is the gate.

Failure handling — a single bad iteration never kills the job:
  * proposer error  → record an ERROR iteration, keep the best, continue (patience still applies)
  * candidate won't compile → ERROR iteration, never executed
  * executor error on re-run → ERROR iteration, keep the best, continue
A hard failure of the *baseline* run propagates → the worker marks the job FAILED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from harness_service.agent_source import build_agent_state, build_baseline_agent
from harness_service.config import Settings
from harness_service.constants import (
    DEFAULT_TRAIN_RATIO,
    SPLIT_SEED,
    IterationDecision,
    JobMode,
)
from harness_service.domain import (
    AgentState,
    BenchmarkResult,
    Improvement,
    Iteration,
    Trajectory,
)
from harness_service.executors.base import Executor
from harness_service.services.jobs import JobContext
from harness_service.services.proposer import get_proposer, validate_candidate

logger = logging.getLogger("harness.runner")


@dataclass
class RunOutcome:
    trajectory: Trajectory
    train_subset: list[str] = field(default_factory=list)
    test_subset: list[str] = field(default_factory=list)
    test_result: BenchmarkResult | None = None
    baseline_val_score: float | None = None


async def execute_job(ctx: JobContext, executor: Executor, settings: Settings) -> RunOutcome:
    if ctx.mode == JobMode.SINGLE_RUN:
        return await _single_run(ctx, executor, settings)
    return await _optimize(ctx, executor, settings)


async def _single_run(ctx: JobContext, executor: Executor, settings: Settings) -> RunOutcome:
    agent = build_baseline_agent(ctx.config, settings)
    logger.info("job %s: single_run (%d tasks, executor=%s)",
                ctx.id, len(ctx.subset), executor.kind.value)
    result = await executor.run(agent, ctx.subset)
    baseline = Iteration(
        idx=0, agent_state=agent, result=result,
        decision=IterationDecision.BASELINE,
        decision_reason="baseline agent, no proposal applied",
    )
    return RunOutcome(
        trajectory=Trajectory().with_iteration(baseline),
        train_subset=ctx.subset,
        baseline_val_score=result.val_score,
    )


async def _optimize(ctx: JobContext, executor: Executor, settings: Settings) -> RunOutcome:
    from harness_service.tasks import split_train_test

    ratio = float(ctx.config.get("train_ratio", DEFAULT_TRAIN_RATIO))
    train = ctx.config.get("train_tasks") or None
    test = ctx.config.get("test_tasks") or None
    if train is None or test is None:
        train, test = split_train_test(ctx.subset, ratio, SPLIT_SEED)
    logger.info("job %s: optimize — train=%d test=%d max_iter=%d patience=%d",
                ctx.id, len(train), len(test), ctx.max_iterations, ctx.patience)

    proposer = get_proposer(settings)

    # ── iteration 0: baseline on TRAIN ──
    best_agent = build_baseline_agent(ctx.config, settings)
    base_result = await executor.run(best_agent, train)
    baseline = Iteration(
        idx=0, agent_state=best_agent, result=base_result,
        decision=IterationDecision.BASELINE, decision_reason="baseline on train split",
    )
    trajectory = Trajectory().with_iteration(baseline)
    best_result = base_result
    logger.info("job %s: baseline train val=%.3f", ctx.id, base_result.val_score)

    # ── improvement iterations ──
    for idx in range(1, ctx.max_iterations + 1):
        context = trajectory.build_context()
        it = await self_step(ctx, idx, executor, settings, proposer,
                             best_agent, best_result, context, train)
        trajectory = trajectory.with_iteration(it)
        if it.decision == IterationDecision.ACCEPTED:
            best_agent, best_result = it.agent_state, it.result
            logger.info("job %s: iter %d ACCEPTED train val=%.3f", ctx.id, idx, it.val_score)
        else:
            logger.info("job %s: iter %d %s (%s)", ctx.id, idx, it.decision.value,
                        it.decision_reason[:80])
        if trajectory.consecutive_non_improving >= ctx.patience:
            logger.info("job %s: patience (%d) reached — stopping", ctx.id, ctx.patience)
            break

    # ── held-out TEST evaluation of the best agent ──
    test_result = None
    if test:
        try:
            test_result = await executor.run(best_agent, test)
            logger.info("job %s: held-out test val=%.3f (train best=%.3f)",
                        ctx.id, test_result.val_score, best_result.val_score)
        except Exception:
            logger.exception("job %s: test evaluation failed (non-fatal)", ctx.id)

    return RunOutcome(
        trajectory=trajectory, train_subset=train, test_subset=test,
        test_result=test_result, baseline_val_score=base_result.val_score,
    )


async def self_step(
    ctx: JobContext, idx: int, executor: Executor, settings: Settings, proposer,
    best_agent: AgentState, best_result: BenchmarkResult, context: str, train: list[str],
) -> Iteration:
    """One improvement attempt, fully wrapped so it can never crash the loop."""
    # 1. propose
    try:
        improvement: Improvement = await proposer.propose(best_agent, best_result, context)
    except Exception as exc:
        logger.exception("job %s: proposer failed at iter %d", ctx.id, idx)
        return _error_iteration(idx, best_agent, context, f"proposer error: {exc!r}")

    # 2. compile-gate
    ok, why = validate_candidate(improvement.new_agent_source)
    if not ok:
        return _error_iteration(
            idx, best_agent, context, f"candidate rejected (compile-gate): {why}", improvement,
        )

    candidate = build_agent_state(improvement.new_agent_source, ctx.config, settings)

    # 3. apply + re-run on TRAIN
    try:
        result = await executor.run(candidate, train)
    except Exception as exc:
        logger.exception("job %s: executor failed on candidate at iter %d", ctx.id, idx)
        return _error_iteration(idx, candidate, context, f"executor error: {exc!r}", improvement)

    # 4. gate: accept only if it beats the best-seen train score
    improved = result.val_score > best_result.val_score + 1e-9
    decision = IterationDecision.ACCEPTED if improved else IterationDecision.REJECTED
    reason = (f"train val {result.val_score:.3f} "
              f"{'>' if improved else '≤'} best {best_result.val_score:.3f}"
              + ("" if improved else " → reverted"))
    return Iteration(
        idx=idx, agent_state=candidate, result=result, decision=decision,
        decision_reason=reason, improvement=improvement, context_snapshot=context,
    )


def _error_iteration(
    idx: int, agent: AgentState, context: str, reason: str,
    improvement: Improvement | None = None,
) -> Iteration:
    return Iteration(
        idx=idx, agent_state=agent, result=BenchmarkResult(outcomes=()),
        decision=IterationDecision.ERROR, decision_reason=reason,
        improvement=improvement, context_snapshot=context,
    )
