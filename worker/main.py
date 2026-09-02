"""Worker process: claim queued runs and execute them."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
import uuid

from api.config import clear_config_cache, load_config
from api.db import init_db
from api.job_store import PostgresJobStore
from api.schemas import RunError, RunStatus
from api.services.artifacts import create_artifact_store
from api.services.improver import create_improver
from api.services.runner import (
    ExecutionUnavailableError,
    HarborBenchmarkRunner,
    MockBenchmarkRunner,
    create_runner,
)
from api.store import PostgresRunStore, _utcnow
from worker.steps import StepExecutor

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
    runner: MockBenchmarkRunner | HarborBenchmarkRunner,
    *,
    worker_id: str,
    stale_after_sec: int,
    job_store: PostgresJobStore | None = None,
    step_executor: StepExecutor | None = None,
) -> bool:
    """
    Claim and execute one unit of work. Returns True if work was done.

    Job steps take priority; when no step is queued (or the worker was built
    without a job store) this falls back to the legacy standalone-run queue.
    """
    if job_store is not None and step_executor is not None:
        step = job_store.claim_next_step(worker_id)
        if step is not None:
            logger.info(
                "claimed step_id=%s type=%s job_id=%s iteration=%s",
                step.step_id,
                step.type,
                step.job_id,
                step.iteration,
            )
            try:
                step_executor.execute(step)
            except Exception as exc:  # noqa: BLE001
                logger.exception("step executor crashed step_id=%s", step.step_id)
                job_store.fail_step(
                    step.step_id,
                    error_code="internal_error",
                    error_message=str(exc),
                )
            return True

    run_id = store.claim_next(worker_id, stale_after_sec=stale_after_sec)
    if run_id is None:
        return False
    logger.info("claimed run_id=%s", run_id)
    try:
        runner.execute_sync(run_id)
    except ExecutionUnavailableError as exc:
        store.update(
            run_id,
            status=RunStatus.failed,
            finished_at=_utcnow(),
            error=RunError(code="execution_unavailable", message=str(exc)),
        )
        logger.error("execution unavailable run_id=%s: %s", run_id, exc)
    except Exception as exc:  # noqa: BLE001
        store.update(
            run_id,
            status=RunStatus.failed,
            finished_at=_utcnow(),
            error=RunError(code="internal_error", message=str(exc)),
        )
        logger.exception("worker failed run_id=%s", run_id)

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
    clear_config_cache()
    cfg = load_config()
    init_db()
    store = PostgresRunStore()
    runner = create_runner(store, config=cfg, step_delay_sec=step_delay_sec)
    job_store = PostgresJobStore()
    artifacts = create_artifact_store(cfg)
    improver = create_improver(cfg)
    step_executor = StepExecutor(
        job_store,
        store,
        config=cfg,
        improver=improver,
        artifacts=artifacts,
        step_delay_sec=step_delay_sec,
    )
    worker_id = default_worker_id()
    logger.info(
        "worker starting id=%s backend=%s env_provider=%s improver=%s",
        worker_id,
        cfg.execution_backend,
        cfg.env_provider,
        type(improver).__name__,
    )

    jobs_done = 0
    while not _shutdown:
        did_work = process_one(
            store,
            runner,
            worker_id=worker_id,
            stale_after_sec=stale_after_sec,
            job_store=job_store,
            step_executor=step_executor,
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
        help="Delay between mock tasks (mock backend only)",
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
