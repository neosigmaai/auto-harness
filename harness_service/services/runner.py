"""Run orchestrator — turns a claimed job into a Trajectory.

M1 produces the baseline iteration only (idx 0). M4 extends ``execute_job`` with
the propose → apply → re-run → accept/reject loop; the return type (a Trajectory)
already accommodates many iterations, so M4 is an in-place extension.
"""

from __future__ import annotations

import logging

from harness_service.agent_source import build_baseline_agent
from harness_service.config import Settings
from harness_service.constants import IterationDecision, JobMode
from harness_service.domain import Iteration, Trajectory
from harness_service.executors.base import Executor
from harness_service.services.jobs import JobContext

logger = logging.getLogger("harness.runner")


async def execute_job(ctx: JobContext, executor: Executor, settings: Settings) -> Trajectory:
    agent = build_baseline_agent(ctx.config, settings)
    logger.info(
        "job %s: baseline run (executor=%s, %d tasks)",
        ctx.id, executor.kind.value, len(ctx.subset),
    )
    result = await executor.run(agent, ctx.subset)
    baseline = Iteration(
        idx=0,
        agent_state=agent,
        result=result,
        decision=IterationDecision.BASELINE,
        decision_reason="baseline agent, no proposal applied",
        improvement=None,
        context_snapshot="",
    )
    trajectory = Trajectory().with_iteration(baseline)

    if ctx.mode == JobMode.OPTIMIZE:
        # M4 lands the propose→apply→re-run→accept loop here, appending iterations
        # to `trajectory` until improvement stalls (patience) or max_iterations.
        logger.info("job %s: optimize mode — loop lands in M4, baseline only for now", ctx.id)

    return trajectory
