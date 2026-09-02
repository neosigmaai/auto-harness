"""Worker process: claim queued runs and execute them."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
import uuid

from api.db import init_db
from api.services.runner import MockBenchmarkRunner
from api.store import PostgresRunStore

logger = logging.getLogger("worker")

_shutdown = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    global _shutdown
    logger.info("received signal %s; shutting down after current job", signum)
    _shutdown = True


def default_worker_id() -> str:
    return os.environ.get("WORKER_ID") or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def process_one(
    store: PostgresRunStore,
    runner: MockBenchmarkRunner,
    *,
    worker_id: str,
    stale_after_sec: int,
) -> bool:
    """Claim and execute one run. Returns True if work was done."""
    run_id = store.claim_next(worker_id, stale_after_sec=stale_after_sec)
    if run_id is None:
        return False
    logger.info("claimed run_id=%s", run_id)
    runner.execute_sync(run_id)
    record = store.get(run_id)
    logger.info(
        "finished run_id=%s status=%s",
        run_id,
        record.status.value if record else "missing",
    )
    return True


def run_loop(
    *,
    poll_interval: float = 1.0,
    stale_after_sec: int = 1800,
    step_delay_sec: float = 0.05,
    max_jobs: int | None = None,
) -> None:
    init_db()
    store = PostgresRunStore()
    runner = MockBenchmarkRunner(store=store, step_delay_sec=step_delay_sec)
    worker_id = default_worker_id()
    logger.info("worker starting id=%s", worker_id)

    jobs_done = 0
    while not _shutdown:
        did_work = process_one(
            store,
            runner,
            worker_id=worker_id,
            stale_after_sec=stale_after_sec,
        )
        if did_work:
            jobs_done += 1
            if max_jobs is not None and jobs_done >= max_jobs:
                logger.info("reached max_jobs=%s; exiting", max_jobs)
                break
            continue
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="auto-harness benchmark worker")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--stale-after-sec", type=int, default=1800)
    parser.add_argument(
        "--step-delay-sec",
        type=float,
        default=float(os.environ.get("MOCK_STEP_DELAY_SEC", "0.05")),
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Exit after processing this many jobs (useful for tests)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_loop(
        poll_interval=args.poll_interval,
        stale_after_sec=args.stale_after_sec,
        step_delay_sec=args.step_delay_sec,
        max_jobs=args.max_jobs,
    )


if __name__ == "__main__":
    main()
