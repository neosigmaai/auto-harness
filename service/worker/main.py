"""Worker process: claim jobs from Postgres, run them, heartbeat leases, stop gracefully."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid

from .. import repo
from ..config import settings
from ..db import close_pool, get_pool
from ..repo import Lease, LeaseLost
from .job_runner import JobRunner
from .optimizer import SimulatedOptimizer

log = logging.getLogger("worker")


def make_executor():
    if settings.executor == "simulated":
        from .executors.simulated import SimulatedExecutor
        return SimulatedExecutor()
    if settings.executor == "harbor":
        from .executors.harbor_runner import HarborExecutor
        return HarborExecutor()
    raise ValueError(f"unknown executor {settings.executor}")


def make_optimizer(pool, lease, job):
    if settings.optimizer == "simulated":
        return SimulatedOptimizer()
    if settings.optimizer == "llm":
        from .llm_optimizer import LLMOptimizer
        return LLMOptimizer(pool, lease, job)
    raise ValueError(f"unknown optimizer {settings.optimizer}")


async def _heartbeat_loop(pool, lease: Lease, runner: JobRunner, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.heartbeat_sec)
            return
        except asyncio.TimeoutError:
            pass
        try:
            async with pool.acquire() as c:
                cancel = await repo.heartbeat(c, lease, settings.lease_ttl_sec)
            if cancel:
                runner.request_stop("cancelled")
                await runner.executor.cancel()
                return
        except LeaseLost:
            log.warning("heartbeat: lease lost for %s; cancelling local work", lease.job_id)
            runner.request_stop("lease_lost")
            await runner.executor.cancel()
            return
        except Exception:
            # Continuing external work while ownership is unverifiable can spend
            # another worker's budget. Stop; the expired lease permits recovery.
            log.exception("heartbeat failed; stopping local work until ownership can be recovered")
            runner.request_stop("database_unavailable")
            await runner.executor.cancel()
            return


async def _run_one(pool, job, worker_id: str, shutting_down: asyncio.Event | None = None):
    lease = Lease(job["id"], worker_id, job["lease_epoch"])
    try:
        runner = JobRunner(pool, job, lease, make_executor(), make_optimizer(pool, lease, job))
    except Exception as exc:
        async with pool.acquire() as conn:
            await repo.finish_job(conn, lease, "FAILED", "configuration_error", error=str(exc)[:1000])
        return
    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat_loop(pool, lease, runner, stop))
    async def on_shutdown():
        if shutting_down is not None:
            await shutting_down.wait()
            runner.request_stop("shutdown")
            await runner.executor.cancel()
    shutdown_watch = asyncio.create_task(on_shutdown())
    try:
        log.info("job %s: start (epoch %s, attempt %s)", job["id"], job["lease_epoch"], job["attempt"])
        await runner.run()
    finally:
        await runner.executor.cancel()
        stop.set()
        hb.cancel()
        shutdown_watch.cancel()
        for task in (hb, shutdown_watch):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        log.info("job %s: done", job["id"])


async def _sandbox_reaper(shutting_down: asyncio.Event):
    if settings.executor != "harbor":
        return
    from .executors.harbor_runner import reap_expired_sandboxes
    while not shutting_down.is_set():
        try:
            await reap_expired_sandboxes()
        except Exception:
            log.exception("managed sandbox cleanup failed")
        try:
            await asyncio.wait_for(shutting_down.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass


async def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker_id = settings.worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    pool = await get_pool()
    shutting_down = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutting_down.set)
    running: set[asyncio.Task] = set()
    reaper = asyncio.create_task(_sandbox_reaper(shutting_down))
    log.info("worker %s up: executor=%s optimizer=%s concurrency=%s", worker_id, settings.executor, settings.optimizer, settings.worker_concurrency)
    try:
        while not shutting_down.is_set():
            for task in tuple(running):
                if task.done():
                    running.remove(task)
                    try:
                        task.result()
                    except Exception:
                        log.exception("job task ended unexpectedly; lease recovery will resume it")
            if len(running) < settings.worker_concurrency:
                try:
                    async with pool.acquire() as c:
                        job = await repo.claim_job(c, worker_id, settings.lease_ttl_sec)
                except Exception:
                    log.exception("claim failed"); job = None
                if job is not None:
                    running.add(asyncio.create_task(_run_one(pool, job, worker_id, shutting_down)))
                    continue
            try:
                await asyncio.wait_for(shutting_down.wait(), timeout=settings.claim_poll_sec)
            except asyncio.TimeoutError:
                pass
        if running:
            log.info("shutdown: cancelling local work for %d job(s); durable leases permit recovery", len(running))
            _, pending = await asyncio.wait(running, timeout=30)
            for task in pending:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
