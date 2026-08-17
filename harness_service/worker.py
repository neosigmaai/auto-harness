"""Background job worker.

Runs ``worker_concurrency`` claim loops. Each loop:
  1. claims one QUEUED job (``FOR UPDATE SKIP LOCKED``) and flips it to RUNNING,
  2. runs the executor OUTSIDE the DB transaction (agent execution is isolated),
  3. persists the resulting Trajectory and marks the job SUCCEEDED, or records
     the error and marks it FAILED.

Because claim + processing are in separate transactions, a crash mid-processing
leaves the job RUNNING (recoverable) rather than holding a lock forever.
"""

from __future__ import annotations

import asyncio
import logging

from harness_service.config import Settings
from harness_service.db import session_scope
from harness_service.executors import get_executor
from harness_service.services import jobs as jobs_svc
from harness_service.services.jobs import JobContext
from harness_service.services.runner import execute_job

logger = logging.getLogger("harness.worker")


class Worker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._tasks:
            return
        self._stop.clear()
        n = max(1, self._settings.worker_concurrency)
        for i in range(n):
            self._tasks.append(asyncio.create_task(self._run_forever(i), name=f"harness-worker-{i}"))
        logger.info("worker started (%d loops, poll=%.1fs)", n, self._settings.worker_poll_interval_s)

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks = []
        logger.info("worker stopped")

    async def _run_forever(self, loop_id: int) -> None:
        while not self._stop.is_set():
            try:
                handled = await self._claim_one()
                if not handled:
                    await asyncio.sleep(self._settings.worker_poll_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker loop %d iteration failed", loop_id)
                await asyncio.sleep(self._settings.worker_poll_interval_s)

    async def _claim_one(self) -> bool:
        # 1. claim (short transaction)
        async with session_scope() as s:
            job = await jobs_svc.claim_next_job(s)
            if job is None:
                return False
            ctx = JobContext.from_orm(job)
        logger.info("claimed job %s (mode=%s, executor=%s)", ctx.id, ctx.mode.value, ctx.executor.value)

        # 2. process (no DB transaction held while the agent runs)
        try:
            executor = get_executor(ctx.executor, self._settings)
            outcome = await execute_job(ctx, executor, self._settings)
        except Exception as exc:
            logger.exception("job %s processing failed", ctx.id)
            async with session_scope() as s:
                await jobs_svc.mark_failed(s, ctx.id, repr(exc))
            return True

        # 3. persist result (separate transaction)
        async with session_scope() as s:
            await jobs_svc.persist_success(s, ctx.id, outcome)
        logger.info("job %s SUCCEEDED (train best=%.3f, test=%s)", ctx.id,
                    outcome.trajectory.best_val_score,
                    None if outcome.test_result is None else round(outcome.test_result.val_score, 3))
        return True
