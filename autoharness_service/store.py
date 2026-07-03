from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import psycopg
from autoharness_service.models import IterationRecord, RunRecord, TaskResultRecord
from autoharness_service.schemas import RunCreateRequest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

RUNS_TABLE = "aos_runs"
TASK_RESULTS_TABLE = "aos_task_results"
ITERATIONS_TABLE = "aos_iterations"
PROPOSAL_CHAR_LIMIT = 20000


class PostgresStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def _connect(self):
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
                  id uuid PRIMARY KEY,
                  org_id text NOT NULL,
                  created_by text NOT NULL,
                  status text NOT NULL,
                  mode text NOT NULL,
                  model text NOT NULL,
                  sandbox_provider text NOT NULL,
                  requested_concurrency integer NOT NULL,
                  max_iterations integer NOT NULL,
                  task_ids jsonb NOT NULL,
                  score double precision,
                  error text,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  started_at timestamptz,
                  completed_at timestamptz
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_aos_runs_org_created_at ON {RUNS_TABLE} (org_id, created_at DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TASK_RESULTS_TABLE} (
                  run_id uuid NOT NULL REFERENCES {RUNS_TABLE}(id) ON DELETE CASCADE,
                  task_id text NOT NULL,
                  status text NOT NULL,
                  reward double precision,
                  failure_type text,
                  error_summary text,
                  trace_path text,
                  result_path text,
                  metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  started_at timestamptz,
                  completed_at timestamptz,
                  PRIMARY KEY (run_id, task_id)
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {TASK_RESULTS_TABLE} ADD COLUMN IF NOT EXISTS started_at timestamptz"
            )
            conn.execute(
                f"ALTER TABLE {TASK_RESULTS_TABLE} ADD COLUMN IF NOT EXISTS completed_at timestamptz"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ITERATIONS_TABLE} (
                  run_id uuid NOT NULL REFERENCES {RUNS_TABLE}(id) ON DELETE CASCADE,
                  iteration_index integer NOT NULL,
                  status text NOT NULL,
                  agent_version text NOT NULL,
                  score double precision,
                  proposal text,
                  accepted boolean,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY (run_id, iteration_index)
                )
                """
            )

    def create_run(
        self,
        request: RunCreateRequest,
        org_id: str,
        created_by: str,
    ) -> RunRecord:
        run_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                f"""
                INSERT INTO {RUNS_TABLE} (
                  id, org_id, created_by, status, mode, model, sandbox_provider,
                  requested_concurrency, max_iterations, task_ids
                )
                VALUES (%s, %s, %s, 'queued', %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    run_id,
                    org_id,
                    created_by,
                    request.mode,
                    request.model,
                    request.sandbox_provider,
                    request.requested_concurrency,
                    request.max_iterations,
                    Jsonb(request.task_ids),
                ),
            ).fetchone()
        return _run_from_row(row)

    def create_task_queue(
        self, run_id: str, org_id: str, task_ids: Iterable[str]
    ) -> None:
        with self._connect() as conn:
            for task_id in task_ids:
                conn.execute(
                    f"""
                    INSERT INTO {TASK_RESULTS_TABLE} (
                      run_id, task_id, status, reward, metadata
                    )
                    SELECT {RUNS_TABLE}.id, %s, 'queued', NULL, %s
                    FROM {RUNS_TABLE}
                    WHERE {RUNS_TABLE}.id = %s AND {RUNS_TABLE}.org_id = %s
                    ON CONFLICT (run_id, task_id) DO NOTHING
                    """,
                    (
                        task_id,
                        Jsonb({"source": "queued"}),
                        run_id,
                        org_id,
                    ),
                )

    def get_run(self, run_id: str, org_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {RUNS_TABLE} WHERE id = %s AND org_id = %s",
                (run_id, org_id),
            ).fetchone()
        return _run_from_row(row) if row else None

    def list_incomplete_runs(self, limit: int = 10) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM {RUNS_TABLE}
                WHERE status IN ('queued', 'running')
                ORDER BY created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def mark_run_running(self, run_id: str, org_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {RUNS_TABLE}
                SET status = 'running', started_at = COALESCE(started_at, now())
                WHERE id = %s AND org_id = %s AND status IN ('queued', 'running')
                """,
                (run_id, org_id),
            )

    def mark_run_succeeded(self, run_id: str, org_id: str, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {RUNS_TABLE}
                SET status = 'succeeded', score = %s, completed_at = now()
                WHERE id = %s AND org_id = %s
                  AND status NOT IN ('succeeded', 'failed', 'timed_out', 'cancelled')
                """,
                (score, run_id, org_id),
            )

    def mark_run_failed(
        self, run_id: str, org_id: str, status: str, error: str
    ) -> None:
        if status not in {"failed", "timed_out", "cancelled"}:
            raise ValueError("terminal failure status expected")
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {RUNS_TABLE}
                SET status = %s, error = %s, completed_at = now()
                WHERE id = %s AND org_id = %s
                  AND status NOT IN ('succeeded', 'failed', 'timed_out', 'cancelled')
                """,
                (status, error[:4000], run_id, org_id),
            )

    def mark_task_running(self, run_id: str, org_id: str, task_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE {task_results_table}
                SET status = 'running',
                    started_at = COALESCE({task_results_table}.started_at, now())
                FROM {runs_table}
                WHERE {task_results_table}.run_id = {runs_table}.id
                  AND {task_results_table}.run_id = %s
                  AND {runs_table}.org_id = %s
                  AND {task_results_table}.task_id = %s
                  AND {task_results_table}.status = 'queued'
                """.format(
                    task_results_table=TASK_RESULTS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (run_id, org_id, task_id),
            )
            return bool(cursor.rowcount)

    def upsert_task_result(
        self, run_id: str, org_id: str, result: TaskResultRecord
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {TASK_RESULTS_TABLE} (
                  run_id, task_id, status, reward, failure_type, error_summary,
                  trace_path, result_path, metadata, started_at, completed_at
                )
                SELECT {RUNS_TABLE}.id, %s, %s, %s, %s, %s, %s, %s, %s, now(), now()
                FROM {RUNS_TABLE}
                WHERE {RUNS_TABLE}.id = %s AND {RUNS_TABLE}.org_id = %s
                ON CONFLICT (run_id, task_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  reward = EXCLUDED.reward,
                  failure_type = EXCLUDED.failure_type,
                  error_summary = EXCLUDED.error_summary,
                  trace_path = EXCLUDED.trace_path,
                  result_path = EXCLUDED.result_path,
                  metadata = EXCLUDED.metadata,
                  started_at = COALESCE({TASK_RESULTS_TABLE}.started_at, now()),
                  completed_at = now()
                """,
                (
                    result.task_id,
                    result.status,
                    result.reward,
                    result.failure_type,
                    result.error_summary,
                    result.trace_path,
                    result.result_path,
                    Jsonb(result.metadata),
                    run_id,
                    org_id,
                ),
            )

    def requeue_running_tasks(self, run_id: str, org_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE {task_results_table}
                SET status = 'queued'
                FROM {runs_table}
                WHERE {task_results_table}.run_id = {runs_table}.id
                  AND {task_results_table}.run_id = %s
                  AND {runs_table}.org_id = %s
                  AND {task_results_table}.status = 'running'
                """.format(
                    task_results_table=TASK_RESULTS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (run_id, org_id),
            )
            return cursor.rowcount or 0

    def reset_task_queue(
        self,
        run_id: str,
        org_id: str,
        task_ids: Iterable[str],
        metadata: dict[str, Any],
    ) -> None:
        selected_task_ids = list(task_ids)
        if not selected_task_ids:
            return

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE {task_results_table}
                SET status = 'queued',
                    reward = NULL,
                    failure_type = NULL,
                    error_summary = NULL,
                    trace_path = NULL,
                    result_path = NULL,
                    metadata = %s,
                    started_at = NULL,
                    completed_at = NULL
                FROM {runs_table}
                WHERE {task_results_table}.run_id = {runs_table}.id
                  AND {task_results_table}.run_id = %s
                  AND {runs_table}.org_id = %s
                  AND {task_results_table}.task_id = ANY(%s)
                """.format(
                    task_results_table=TASK_RESULTS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (Jsonb(metadata), run_id, org_id, selected_task_ids),
            )

    def replace_task_results(
        self,
        run_id: str,
        org_id: str,
        task_results: Iterable[TaskResultRecord],
    ) -> None:
        task_results = list(task_results)
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM {task_results_table}
                USING {runs_table}
                WHERE {task_results_table}.run_id = {runs_table}.id
                  AND {runs_table}.id = %s
                  AND {runs_table}.org_id = %s
                """.format(
                    task_results_table=TASK_RESULTS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (run_id, org_id),
            )
            for result in task_results:
                conn.execute(
                    f"""
                    INSERT INTO {TASK_RESULTS_TABLE} (
                      run_id, task_id, status, reward, failure_type, error_summary,
                      trace_path, result_path, metadata
                    )
                    SELECT {RUNS_TABLE}.id, %s, %s, %s, %s, %s, %s, %s, %s
                    FROM {RUNS_TABLE}
                    WHERE {RUNS_TABLE}.id = %s AND {RUNS_TABLE}.org_id = %s
                    ON CONFLICT (run_id, task_id) DO UPDATE SET
                      status = EXCLUDED.status,
                      reward = EXCLUDED.reward,
                      failure_type = EXCLUDED.failure_type,
                      error_summary = EXCLUDED.error_summary,
                      trace_path = EXCLUDED.trace_path,
                      result_path = EXCLUDED.result_path,
                      metadata = EXCLUDED.metadata,
                      completed_at = now()
                    """,
                    (
                        result.task_id,
                        result.status,
                        result.reward,
                        result.failure_type,
                        result.error_summary,
                        result.trace_path,
                        result.result_path,
                        Jsonb(result.metadata),
                        run_id,
                        org_id,
                    ),
                )

    def list_task_results(self, run_id: str, org_id: str) -> list[TaskResultRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT {task_results_table}.*
                FROM {task_results_table}
                JOIN {runs_table} ON {runs_table}.id = {task_results_table}.run_id
                WHERE {task_results_table}.run_id = %s AND {runs_table}.org_id = %s
                ORDER BY task_id
                """.format(
                    task_results_table=TASK_RESULTS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (run_id, org_id),
            ).fetchall()
        return [_task_result_from_row(row) for row in rows]

    def create_iteration(
        self,
        run_id: str,
        org_id: str,
        iteration_index: int,
        status: str,
        agent_version: str,
        score: float | None = None,
        proposal: str | None = None,
        accepted: bool | None = None,
    ) -> IterationRecord:
        proposal_text = _validate_proposal_text(proposal)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                INSERT INTO {ITERATIONS_TABLE} (
                  run_id, iteration_index, status, agent_version, score, proposal, accepted
                )
                SELECT {RUNS_TABLE}.id, %s, %s, %s, %s, %s, %s
                FROM {RUNS_TABLE}
                WHERE {RUNS_TABLE}.id = %s AND {RUNS_TABLE}.org_id = %s
                ON CONFLICT (run_id, iteration_index) DO UPDATE SET
                  status = EXCLUDED.status,
                  agent_version = EXCLUDED.agent_version,
                  score = EXCLUDED.score,
                  proposal = EXCLUDED.proposal,
                  accepted = EXCLUDED.accepted
                RETURNING *
                """,
                (
                    iteration_index,
                    status,
                    agent_version,
                    score,
                    proposal_text,
                    accepted,
                    run_id,
                    org_id,
                ),
            ).fetchone()
            if row is None:
                raise KeyError("run not found")
        return _iteration_from_row(row)

    def list_iterations(self, run_id: str, org_id: str) -> list[IterationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT {iterations_table}.*
                FROM {iterations_table}
                JOIN {runs_table} ON {runs_table}.id = {iterations_table}.run_id
                WHERE {iterations_table}.run_id = %s AND {runs_table}.org_id = %s
                ORDER BY iteration_index
                """.format(
                    iterations_table=ITERATIONS_TABLE,
                    runs_table=RUNS_TABLE,
                ),
                (run_id, org_id),
            ).fetchall()
        return [_iteration_from_row(row) for row in rows]


def _task_ids_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def _run_from_row(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(row["id"]),
        status=str(row["status"]),
        task_ids=_task_ids_from_json(row["task_ids"]),
        mode=str(row["mode"]),
        model=str(row["model"]),
        sandbox_provider=str(row["sandbox_provider"]),
        requested_concurrency=int(row["requested_concurrency"]),
        max_iterations=int(row["max_iterations"]),
        org_id=str(row["org_id"]),
        created_by=str(row["created_by"]),
        score=row["score"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _task_result_from_row(row: dict[str, Any]) -> TaskResultRecord:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return TaskResultRecord(
        task_id=str(row["task_id"]),
        status=str(row["status"]),
        reward=row["reward"],
        failure_type=row["failure_type"],
        error_summary=row["error_summary"],
        trace_path=row["trace_path"],
        result_path=row["result_path"],
        metadata=dict(metadata or {}),
    )


def _iteration_from_row(row: dict[str, Any]) -> IterationRecord:
    return IterationRecord(
        run_id=str(row["run_id"]),
        iteration_index=int(row["iteration_index"]),
        status=str(row["status"]),
        agent_version=str(row["agent_version"]),
        score=row["score"],
        proposal=row["proposal"],
        accepted=row["accepted"],
    )


def _validate_proposal_text(proposal: str | None) -> str | None:
    if proposal is None:
        return None
    if len(proposal) > PROPOSAL_CHAR_LIMIT:
        raise ValueError(f"proposal exceeds {PROPOSAL_CHAR_LIMIT} character limit")
    return proposal
