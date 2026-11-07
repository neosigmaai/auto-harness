"""Worker process: claim jobs from Postgres, run them, heartbeat leases, stop gracefully."""
from __future__ import annotations

import asyncio
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


def make_optimizer():
    if settings.optimizer == "simulated":
        return SimulatedOptimizer()
    if settings.optimizer == "llm":
        from .llm_optimizer import LLMOptimizer
        return LLMOptimizer()
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
                runner.cancel_flag.set()
        except LeaseLost:
            log.warning("heartbeat: lease lost for %s; cancelling local work", lease.job_id)
            runner.cancel_flag.set()
            return
        except Exception:
            log.exception("heartbeat failed (will retry)")


async def _run_one(pool, job, worker_id: str):
    lease = Lease(job["id"], worker_id, job["lease_epoch"])
    runner = JobRunner(pool, job, lease, make_executor(), make_optimizer())
    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat_loop(pool, lease, runner, stop))
    try:
        log.info("job %s: start (epoch %s, attempt %s)", job["id"], job["lease_epoch"], job["attempt"])
        await runner.run()
    finally:
        stop.set()
        await hb
        log.info("job %s: done", job["id"])


async def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker_id = settings.worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    pool = await get_pool()
    shutting_down = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutting_down.set)
    running: set[asyncio.Task] = set()
    log.info("worker %s up: executor=%s optimizer=%s concurrency=%s", worker_id, settings.executor, settings.optimizer, settings.worker_concurrency)
    try:
        while not shutting_down.is_set():
            running = {t for t in running if not t.done()}
            if len(running) < settings.worker_concurrency:
                try:
                    async with pool.acquire() as c:
                        job = await repo.claim_job(c, worker_id, settings.lease_ttl_sec)
                except Exception:
                    log.exception("claim failed"); job = None
                if job is not None:
                    running.add(asyncio.create_task(_run_one(pool, job, worker_id)))
                    continue
            try:
                await asyncio.wait_for(shutting_down.wait(), timeout=settings.claim_poll_sec)
            except asyncio.TimeoutError:
                pass
        if running:
            log.info("shutdown: waiting for %d running job(s) to reach a checkpoint (max 30 s)", len(running))
            await asyncio.wait(running, timeout=30)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
