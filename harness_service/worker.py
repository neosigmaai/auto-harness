"""Background job worker.

M0: lifecycle only — starts/stops with the app and heartbeats, proving the
wiring. It does NOT yet claim jobs; the DB-backed claim (``FOR UPDATE SKIP
LOCKED``) and job processing land in M1/M2. Kept as a class so M1 fills in
``_claim_one`` / ``_process`` without touching the app's lifespan code.
"""

from __future__ import annotations

import asyncio
import logging

from harness_service.config import Settings

logger = logging.getLogger("harness.worker")


class Worker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run_forever(), name="harness-worker")
            logger.info("worker started (poll=%.1fs)", self._settings.worker_poll_interval_s)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("worker stopped")

    async def _run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = await self._claim_one()
                if not claimed:
                    await asyncio.sleep(self._settings.worker_poll_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the loop die on a single bad job
                logger.exception("worker loop iteration failed")
                await asyncio.sleep(self._settings.worker_poll_interval_s)

    async def _claim_one(self) -> bool:
        """Claim + process one queued job. Returns True if one was handled.

        M0 stub: no claiming yet (returns False). M1 implements the
        ``SELECT ... FOR UPDATE SKIP LOCKED`` claim and job processing.
        """
        return False
