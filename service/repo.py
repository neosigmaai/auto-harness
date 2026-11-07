"""Data access. Tenant-facing functions take a Principal and scope every query by org_id
(and by created_by for members). Worker-facing functions take a lease (job_id, owner, epoch)
and refuse to write once the lease has moved on (fencing)."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg

from .auth import Principal, generate_api_key, hash_api_key


class LeaseLost(Exception):
    """Raised when a worker tries to write for a job it no longer owns."""


class JobCancelled(Exception):
    """Cancellation was requested before new work or promotion."""


class BudgetExceeded(Exception):
    """The persisted time or spending budget does not admit more work."""


class IdempotencyConflict(Exception):
    """An idempotency key was reused with a different request."""


class ReservationConflict(Exception):
    """An external call has already been reserved; do not dispatch it twice."""


@dataclass(frozen=True)
class Lease:
    job_id: UUID
    owner: str
    epoch: int


@asynccontextmanager
async def lease_transaction(conn: asyncpg.Connection, lease: Lease, *, active: bool = False):
    """Serialize protected writes with lease transfer. Never hold over external I/O."""
    async with conn.transaction():
        job = await conn.fetchrow(
            "SELECT * FROM jobs WHERE id=$1 FOR UPDATE", lease.job_id)
        # Read time after acquiring the lock: SELECT target expressions may have
        # been evaluated before waiting for another transaction's row lock.
        db_now = await conn.fetchval('SELECT clock_timestamp()')
        if (job is None or job['status'] != 'RUNNING' or job['lease_owner'] != lease.owner
                or job['lease_epoch'] != lease.epoch or job['lease_expires_at'] is None
                or job['lease_expires_at'] <= db_now):
            raise LeaseLost(f"lease for job {lease.job_id} is no longer current")
        if active:
            if job['cancel_requested']:
                raise JobCancelled('job cancellation requested')
            if job['deadline_at'] is not None and job['deadline_at'] <= db_now:
                raise BudgetExceeded('job wall-clock deadline exceeded')
        yield job


async def _require_iteration(conn, lease: Lease, iteration_id: UUID) -> None:
    if not await conn.fetchval("SELECT 1 FROM iterations WHERE id=$1 AND job_id=$2", iteration_id, lease.job_id):
        raise ValueError('iteration does not belong to leased job')


# ── tenancy helpers ────────────────────────────────────────────────────────────
def _scope(p: Principal) -> tuple[str, list[Any]]:
    """WHERE fragment + params restricting jobs to what the principal may see."""
    if p.is_admin:
        return "j.org_id = $1", [p.org_id]
    return "j.org_id = $1 AND j.created_by = $2", [p.org_id, p.user_id]


# ── organisations / users / keys ───────────────────────────────────────────────
async def create_org(conn: asyncpg.Connection, name: str, admin_email: str, max_concurrent_jobs: int) -> dict:
    async with conn.transaction():
        org = await conn.fetchrow(
            "INSERT INTO organizations(name, max_concurrent_jobs) VALUES ($1,$2) RETURNING *", name, max_concurrent_jobs
        )
        return await create_user(conn, org["id"], org["name"], admin_email, "admin")


async def create_user(conn: asyncpg.Connection, org_id: UUID, org_name: str, email: str, role: str) -> dict:
    user = await conn.fetchrow(
        "INSERT INTO users(org_id, email, role) VALUES ($1,$2,$3) RETURNING *", org_id, email, role
    )
    raw = generate_api_key()
    await conn.execute(
        "INSERT INTO api_keys(key_hash, key_prefix, user_id, org_id) VALUES ($1,$2,$3,$4)",
        hash_api_key(raw), raw[:8], user["id"], org_id,
    )
    return {"org_id": org_id, "org_name": org_name, "user_id": user["id"], "email": email, "role": role, "api_key": raw}


async def get_org(conn: asyncpg.Connection, org_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow("SELECT * FROM organizations WHERE id = $1", org_id)


# ── jobs (tenant side) ─────────────────────────────────────────────────────────
async def create_job(conn: asyncpg.Connection, p: Principal, config: dict, idempotency_key: str | None) -> tuple[asyncpg.Record, bool]:
    """Returns (job, created). With an idempotency key, a repeat returns the existing job."""
    fingerprint = sha256_text(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False))
    async with conn.transaction():
        job = await conn.fetchrow(
            """INSERT INTO jobs(org_id, created_by, status, config, idempotency_key, request_fingerprint)
               VALUES ($1,$2,'QUEUED',$3,$4,$5)
               ON CONFLICT (org_id, created_by, idempotency_key) DO NOTHING RETURNING *""",
            p.org_id, p.user_id, config, idempotency_key, fingerprint,
        )
        if job is None:
            # A conflicting INSERT waits for its transaction; this fresh statement
            # then sees the committed row under READ COMMITTED.
            job = await conn.fetchrow(
                'SELECT * FROM jobs WHERE org_id=$1 AND created_by=$2 AND idempotency_key=$3',
                p.org_id, p.user_id, idempotency_key)
            existing_fingerprint = job['request_fingerprint'] or sha256_text(
                json.dumps(job['config'], sort_keys=True, separators=(',', ':'), allow_nan=False))
            if existing_fingerprint != fingerprint:
                raise IdempotencyConflict('Idempotency-Key was already used with a different request')
            return job, False
        await add_event(conn, job["id"], "job.submitted", {"by": str(p.user_id), "n_tasks": len(config["tasks"])})
        return job, True


async def get_job(conn: asyncpg.Connection, p: Principal, job_id: UUID) -> asyncpg.Record | None:
    where, params = _scope(p)
    return await conn.fetchrow(f"""SELECT j.*,
        (SELECT coalesce(sum(actual_usd),0) FROM spend_reservations s WHERE s.job_id=j.id) AS spent_usd,
        (SELECT coalesce(sum(reserved_usd),0) FROM spend_reservations s WHERE s.job_id=j.id AND settled_at IS NULL) AS reserved_usd
        FROM jobs j WHERE j.id = ${len(params)+1} AND {where}""", *params, job_id)


async def list_jobs(conn: asyncpg.Connection, p: Principal, status: str | None, limit: int, before: str | None) -> list[asyncpg.Record]:
    where, params = _scope(p)
    sql = f"""SELECT j.*,
        (SELECT coalesce(sum(actual_usd),0) FROM spend_reservations s WHERE s.job_id=j.id) AS spent_usd,
        (SELECT coalesce(sum(reserved_usd),0) FROM spend_reservations s WHERE s.job_id=j.id AND settled_at IS NULL) AS reserved_usd
        FROM jobs j WHERE {where}"""
    if status:
        params.append(status); sql += f" AND j.status = ${len(params)}"
    if before:
        params.append(before); sql += f" AND j.created_at < ${len(params)}::timestamptz"
    params.append(limit); sql += f" ORDER BY j.created_at DESC LIMIT ${len(params)}"
    return await conn.fetch(sql, *params)


async def request_cancel(conn: asyncpg.Connection, p: Principal, job_id: UUID) -> asyncpg.Record | None:
    async with conn.transaction():
        where, params = _scope(p)
        job = await conn.fetchrow(
            f'SELECT j.* FROM jobs j WHERE j.id=${len(params)+1} AND {where} FOR UPDATE', *params, job_id)
        if job is None or job['status'] in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            return job
        if job["status"] == "QUEUED":
            # Nobody owns it yet: cancel outright (CAS on status so a racing claim loses).
            row = await conn.fetchrow(
                """UPDATE jobs SET status='CANCELLED', cancel_requested=true, stop_reason='cancelled', finished_at=now(), updated_at=now()
                   WHERE id=$1 AND status='QUEUED' RETURNING *""", job_id)
            if row:
                await add_event(conn, job_id, "job.cancelled", {"by": str(p.user_id), "while": "QUEUED"})
                return row
        row = await conn.fetchrow(
            "UPDATE jobs SET cancel_requested=true, updated_at=now() WHERE id=$1 RETURNING *", job_id)
        await add_event(conn, job_id, "job.cancel_requested", {"by": str(p.user_id)})
        return row


async def job_progress(conn: asyncpg.Connection, job_id: UUID) -> dict | None:
    it = await conn.fetchrow(
        "SELECT id, index, phase, status FROM iterations WHERE job_id=$1 ORDER BY index DESC LIMIT 1", job_id)
    if it is None:
        return None
    counts = await conn.fetchrow(
        """SELECT count(*) FILTER (WHERE status NOT IN ('pending','running')) AS done, count(*) AS total
           FROM task_runs WHERE iteration_id=$1""", it["id"])
    return {"iteration": it["index"], "phase": it["phase"], "iteration_status": it["status"],
            "tasks_done": counts["done"], "tasks_total": counts["total"]}


async def list_iterations(conn: asyncpg.Connection, job_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        """SELECT i.*, a.sha256 AS agent_sha256 FROM iterations i JOIN agent_versions a ON a.id=i.agent_version_id
           WHERE i.job_id=$1 ORDER BY i.index""", job_id)


async def get_iteration(conn: asyncpg.Connection, job_id: UUID, index: int) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """SELECT i.*, a.sha256 AS agent_sha256, a.content AS agent_source FROM iterations i
           JOIN agent_versions a ON a.id=i.agent_version_id WHERE i.job_id=$1 AND i.index=$2""", job_id, index)


async def list_task_runs(conn: asyncpg.Connection, iteration_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        """SELECT task_name, status, reward, failure_class, duration_s, n_steps, exit_reason,
                  input_tokens,output_tokens,cost_usd,model_cost_usd,sandbox_cost_usd,cost_known
           FROM task_runs WHERE iteration_id=$1 ORDER BY task_name""", iteration_id)


async def get_task_run(conn: asyncpg.Connection, iteration_id: UUID, task: str) -> asyncpg.Record | None:
    return await conn.fetchrow("SELECT * FROM task_runs WHERE iteration_id=$1 AND task_name=$2", iteration_id, task)


async def list_events(conn: asyncpg.Connection, job_id: UUID, after_id: int, limit: int) -> list[asyncpg.Record]:
    return await conn.fetch(
        "SELECT id, ts, kind, payload FROM job_events WHERE job_id=$1 AND id>$2 ORDER BY id LIMIT $3", job_id, after_id, limit)


async def add_event(conn: asyncpg.Connection, job_id: UUID, kind: str, payload: dict | None = None) -> None:
    await conn.execute("INSERT INTO job_events(job_id, kind, payload) VALUES ($1,$2,$3)", job_id, kind, payload or {})


# ── agent versions ─────────────────────────────────────────────────────────────
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def upsert_agent_version(conn: asyncpg.Connection, org_id: UUID, content: str, parent_id: UUID | None) -> asyncpg.Record:
    sha = sha256_text(content)
    row = await conn.fetchrow(
        """INSERT INTO agent_versions(org_id, sha256, content, parent_id) VALUES ($1,$2,$3,$4)
           ON CONFLICT (org_id, sha256) DO UPDATE SET sha256 = EXCLUDED.sha256 RETURNING *""",
        org_id, sha, content, parent_id)
    return row


async def get_agent_version(conn: asyncpg.Connection, version_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow("SELECT * FROM agent_versions WHERE id=$1", version_id)


# ── worker side: queue + leases ────────────────────────────────────────────────
async def claim_job(conn: asyncpg.Connection, worker_id: str, ttl_sec: int) -> asyncpg.Record | None:
    async with conn.transaction():
        # Admission is serialized per organisation, not merely per job. A second
        # claimant cannot independently consume the same remaining quota slot.
        org = await conn.fetchrow("""
            SELECT o.* FROM organizations o
            WHERE EXISTS (SELECT 1 FROM jobs j WHERE j.org_id=o.id AND
                (j.status='QUEUED' OR (j.status='RUNNING' AND j.lease_expires_at < clock_timestamp())))
              AND (SELECT count(*) FROM jobs r WHERE r.org_id=o.id AND r.status='RUNNING'
                   AND r.lease_expires_at >= clock_timestamp()) < o.max_concurrent_jobs
            ORDER BY o.last_dispatched_at NULLS FIRST, o.created_at, o.id
            FOR UPDATE OF o SKIP LOCKED LIMIT 1
        """)
        if org is None:
            return None
        active = await conn.fetchval("""SELECT count(*) FROM jobs WHERE org_id=$1 AND status='RUNNING'
            AND lease_expires_at >= clock_timestamp()""", org['id'])
        if active >= org['max_concurrent_jobs']:
            return None
        job = await conn.fetchrow("""
            WITH candidate AS (
                SELECT id FROM jobs WHERE org_id=$1 AND
                  (status='QUEUED' OR (status='RUNNING' AND lease_expires_at < clock_timestamp()))
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE jobs SET status='RUNNING',lease_owner=$2,
                lease_expires_at=clock_timestamp()+make_interval(secs=>$3),
                lease_epoch=lease_epoch+1,attempt=attempt+1,
                started_at=coalesce(started_at,clock_timestamp()),
                deadline_at=coalesce(deadline_at,
                    coalesce(started_at,clock_timestamp())+
                    make_interval(mins=>coalesce((config->'budget'->>'max_wall_clock_min')::int,120))),
                updated_at=clock_timestamp()
            FROM candidate WHERE jobs.id=candidate.id RETURNING jobs.*
        """, org['id'], worker_id, float(ttl_sec))
        if job:
            await conn.execute('UPDATE organizations SET last_dispatched_at=clock_timestamp() WHERE id=$1', org['id'])
            await add_event(conn, job["id"], "job.claimed",
                            {"worker": worker_id, "epoch": job["lease_epoch"], "attempt": job["attempt"]})
        return job


async def heartbeat(conn: asyncpg.Connection, lease: Lease, ttl_sec: int) -> bool:
    """Extend the lease. Returns cancel_requested. Raises LeaseLost if we no longer own the job."""
    row = await conn.fetchrow(
        """UPDATE jobs SET lease_expires_at = now() + make_interval(secs => $4), updated_at = now()
           WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING'
             AND lease_expires_at > clock_timestamp() RETURNING cancel_requested""",
        lease.job_id, lease.owner, lease.epoch, float(ttl_sec))
    if row is None:
        raise LeaseLost(f"lease for job {lease.job_id} (epoch {lease.epoch}) no longer held by {lease.owner}")
    return row["cancel_requested"]


async def assert_lease(conn: asyncpg.Connection, lease: Lease) -> None:
    # This is a read-only liveness check. Mutations must use lease_transaction.
    async with lease_transaction(conn, lease):
        pass


async def finish_job(conn: asyncpg.Connection, lease: Lease, status: str, stop_reason: str | None,
                     error: str | None = None, best_iteration_id: UUID | None = None,
                     best_score: float | None = None, baseline_score: float | None = None,
                     final_evaluation: dict | None = None) -> None:
    async with lease_transaction(conn, lease) as job:
        if job['cancel_requested']:
            status, stop_reason = 'CANCELLED', 'cancelled'
        row = await conn.fetchrow(
            """UPDATE jobs SET status=$4, stop_reason=$5, error=$6, finished_at=now(), updated_at=now(),
                   best_iteration_id = coalesce($7, best_iteration_id),
                   best_score = coalesce($8, best_score), baseline_score = coalesce($9, baseline_score),
                   final_evaluation = coalesce($10, final_evaluation),
                   lease_owner=NULL, lease_expires_at=NULL
               WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING' RETURNING id""",
            lease.job_id, lease.owner, lease.epoch, status, stop_reason, error, best_iteration_id, best_score,
            baseline_score, final_evaluation)
        if row is None:
            raise LeaseLost("cannot finish a job we do not own")
        await add_event(conn, lease.job_id, f"job.{status.lower()}", {"stop_reason": stop_reason, "error": error})


async def update_job_progress(conn: asyncpg.Connection, lease: Lease, **fields) -> None:
    allowed = {'current_iteration', 'baseline_score', 'best_score', 'best_iteration_id', 'final_evaluation'}
    if set(fields) - allowed:
        raise ValueError('unsupported job progress fields')
    if not fields:
        return
    sets, params = [], [lease.job_id, lease.owner, lease.epoch]
    for k, v in fields.items():
        params.append(v); sets.append(f"{k} = ${len(params)}")
    sets.append("updated_at = now()")
    async with lease_transaction(conn, lease, active=True):
        row = await conn.fetchrow(
            f"UPDATE jobs SET {', '.join(sets)} WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING' RETURNING id",
            *params)
        if row is None:
            raise LeaseLost("progress update rejected: lease lost")


# ── worker side: iterations / task runs ────────────────────────────────────────
async def latest_iteration(conn: asyncpg.Connection, job_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow("SELECT * FROM iterations WHERE job_id=$1 ORDER BY index DESC LIMIT 1", job_id)


async def create_iteration(conn: asyncpg.Connection, lease: Lease, index: int, agent_version_id: UUID,
                           parent_iteration_id: UUID | None, proposal: dict | None) -> asyncpg.Record:
    async with lease_transaction(conn, lease, active=True) as job:
        if not await conn.fetchval('SELECT 1 FROM agent_versions WHERE id=$1 AND org_id=$2', agent_version_id, job['org_id']):
            raise ValueError('agent version belongs to another organisation')
        if parent_iteration_id is not None:
            await _require_iteration(conn, lease, parent_iteration_id)
        return await conn.fetchrow(
            """INSERT INTO iterations(job_id, index, agent_version_id, parent_iteration_id, phase, status, proposal)
               VALUES ($1,$2,$3,$4,'PROVISION','running',$5) RETURNING *""",
            lease.job_id, index, agent_version_id, parent_iteration_id, proposal)


async def set_phase(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, phase: str) -> None:
    async with lease_transaction(conn, lease, active=True):
        await _require_iteration(conn, lease, iteration_id)
        await conn.execute("UPDATE iterations SET phase=$2 WHERE id=$1", iteration_id, phase)
        await add_event(conn, lease.job_id, "iteration.phase", {"iteration_id": str(iteration_id), "phase": phase})


async def complete_iteration(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, *, score: float | None,
                             n_pass: int, n_fail: int, n_error: int, decision: str, decision_reason: str | None,
                             learnings: str | None = None, status: str = "completed", error: str | None = None,
                             metrics: dict | None = None) -> None:
    promoting = decision in ('baseline', 'accepted')
    async with lease_transaction(conn, lease, active=promoting) as job:
        await _require_iteration(conn, lease, iteration_id)
        it = await conn.fetchrow('SELECT * FROM iterations WHERE id=$1', iteration_id)
        if it['status'] == 'completed':
            if it['decision'] != decision:
                raise ValueError('completed iteration decision is immutable')
            return
        if decision == 'accepted' and job['best_iteration_id'] != it['parent_iteration_id']:
            raise ValueError('candidate parent is no longer the incumbent')
        await conn.execute(
            """UPDATE iterations SET phase='DONE', status=$2, score=$3, n_pass=$4, n_fail=$5, n_error=$6,
                   decision=$7, decision_reason=$8, learnings=coalesce($9, learnings), error=$10,
                   metrics=$11, finished_at=now() WHERE id=$1""",
            iteration_id, status, score, n_pass, n_fail, n_error, decision, decision_reason, learnings, error, metrics or {})
        if promoting:
            await conn.execute("""UPDATE jobs SET best_iteration_id=$2,best_score=$3,
                baseline_score=CASE WHEN $4='baseline' THEN $3 ELSE baseline_score END,
                current_iteration=$5,updated_at=now() WHERE id=$1""",
                lease.job_id, iteration_id, score, decision, it['index'])
        await add_event(conn, lease.job_id, 'iteration.finished',
                        {'iteration': it['index'], 'decision': decision, 'score': score, 'reason': decision_reason})


async def ensure_task_runs(conn: asyncpg.Connection, iteration_id: UUID, tasks: list[str], *, lease: Lease) -> None:
    async with lease_transaction(conn, lease, active=True):
        await _require_iteration(conn, lease, iteration_id)
        await conn.executemany(
            "INSERT INTO task_runs(iteration_id, task_name, status) VALUES ($1,$2,'pending') ON CONFLICT DO NOTHING",
            [(iteration_id, t) for t in tasks])


async def pending_task_names(conn: asyncpg.Connection, iteration_id: UUID) -> list[str]:
    rows = await conn.fetch(
        "SELECT task_name FROM task_runs WHERE iteration_id=$1 AND status IN ('pending','running') ORDER BY task_name",
        iteration_id)
    return [r["task_name"] for r in rows]


async def mark_task_running(conn: asyncpg.Connection, iteration_id: UUID, task: str, sandbox_id: str | None = None,
                            *, lease: Lease) -> None:
    async with lease_transaction(conn, lease, active=True):
        await _require_iteration(conn, lease, iteration_id)
        await conn.execute(
            """UPDATE task_runs SET status='running', started_at=coalesce(started_at, now()), sandbox_id=coalesce($3, sandbox_id)
               WHERE iteration_id=$1 AND task_name=$2 AND status IN ('pending','running','error')""", iteration_id, task, sandbox_id)


async def record_task_result(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, task: str, *, status: str,
                             reward: float | None, failure_class: str | None, duration_s: float | None,
                             n_steps: int | None, exit_reason: str | None, test_stdout: str | None,
                             trace: Any, digest: dict | None, sandbox_id: str | None = None,
                             evaluation_phase: str = 'main', repetition: int = 1, attempt: int = 1,
                             input_tokens: int = 0, output_tokens: int = 0, cost_usd: float | None = None,
                             model_cost_usd: float | None = None, sandbox_cost_usd: float | None = None, cost_known: bool = False,
                             metadata: dict | None = None) -> None:
    values = dict(status=status, reward=reward, failure_class=failure_class, duration_s=duration_s,
                  n_steps=n_steps, exit_reason=exit_reason, test_stdout=test_stdout, trace=trace, digest=digest,
                  sandbox_id=sandbox_id, input_tokens=input_tokens, output_tokens=output_tokens,
                  cost_usd=cost_usd, model_cost_usd=model_cost_usd, sandbox_cost_usd=sandbox_cost_usd,
                  cost_known=cost_known, metadata=metadata or {})
    result_hash = sha256_text(json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False))
    async with lease_transaction(conn, lease):
        await _require_iteration(conn, lease, iteration_id)
        fields = list(values)
        columns = ','.join(fields)
        placeholders = ','.join(f'${n}' for n in range(7, 7 + len(fields)))
        inserted = await conn.fetchval(
            f'''INSERT INTO task_attempts(iteration_id,evaluation_phase,task_name,repetition,attempt,lease_epoch,
                {columns},result_hash) VALUES ($1,$2,$3,$4,$5,$6,{placeholders},${7+len(fields)})
                ON CONFLICT (iteration_id,evaluation_phase,task_name,repetition,attempt) DO NOTHING RETURNING id''',
            iteration_id,evaluation_phase,task,repetition,attempt,lease.epoch,*values.values(),result_hash)
        if inserted is None:
            previous_hash = await conn.fetchval('''SELECT result_hash FROM task_attempts
                WHERE iteration_id=$1 AND evaluation_phase=$2 AND task_name=$3 AND repetition=$4 AND attempt=$5''',
                iteration_id,evaluation_phase,task,repetition,attempt)
            if previous_hash != result_hash:
                raise ValueError('attempt result is immutable; a retry requires a new attempt number')
            return
        if evaluation_phase == 'main' and repetition == 1:
            # Projection is the first non-infrastructure-error result, not the
            # highest reward or last observation. Full evidence remains above.
            chosen = (await attempt_results(conn, iteration_id, evaluation_phase, repetition))[task]
            projection = {k: chosen[k] for k in values if k != 'metadata'}
            projection_columns = ','.join(projection)
            projection_args = ','.join(f'${n}' for n in range(3, 3+len(projection)))
            updates = ','.join(f'{k}=EXCLUDED.{k}' for k in projection)
            await conn.execute(f'''INSERT INTO task_runs(iteration_id,task_name,{projection_columns},started_at,finished_at)
                VALUES ($1,$2,{projection_args},now(),now()) ON CONFLICT(iteration_id,task_name)
                DO UPDATE SET {updates},finished_at=now()''', iteration_id,task,*projection.values())


async def task_results(conn: asyncpg.Connection, iteration_id: UUID) -> dict[str, asyncpg.Record]:
    rows = await conn.fetch("SELECT * FROM task_runs WHERE iteration_id=$1", iteration_id)
    return {r["task_name"]: r for r in rows}


async def attempt_results(conn: asyncpg.Connection, iteration_id: UUID, evaluation_phase: str = 'main',
                          repetition: int = 1) -> dict[str, asyncpg.Record]:
    rows = await conn.fetch('''SELECT DISTINCT ON(task_name) * FROM task_attempts
        WHERE iteration_id=$1 AND evaluation_phase=$2 AND repetition=$3
        ORDER BY task_name,(status='error'),CASE WHEN status <> 'error' THEN attempt END ASC,attempt DESC''',
        iteration_id,evaluation_phase,repetition)
    return {r['task_name']: r for r in rows}


async def list_attempts(conn: asyncpg.Connection, iteration_id: UUID, task: str | None = None) -> list[asyncpg.Record]:
    return await conn.fetch('''SELECT * FROM task_attempts WHERE iteration_id=$1 AND ($2::text IS NULL OR task_name=$2)
        ORDER BY evaluation_phase,repetition,task_name,attempt''',iteration_id,task)


async def next_attempt_number(conn: asyncpg.Connection, iteration_id: UUID, evaluation_phase: str = 'main',
                              repetition: int = 1) -> int:
    return await conn.fetchval('''SELECT coalesce(max(attempt),0)+1 FROM task_attempts
        WHERE iteration_id=$1 AND evaluation_phase=$2 AND repetition=$3''',iteration_id,evaluation_phase,repetition)


async def set_task_digest(conn: asyncpg.Connection, iteration_id: UUID, task: str, digest: dict, *, lease: Lease) -> None:
    async with lease_transaction(conn, lease):
        await _require_iteration(conn, lease, iteration_id)
        await conn.execute("UPDATE task_runs SET digest=$3 WHERE iteration_id=$1 AND task_name=$2", iteration_id, task, digest)


async def spend_summary(conn: asyncpg.Connection, job_id: UUID) -> dict:
    row = await conn.fetchrow('''SELECT coalesce(sum(actual_usd),0) AS spent_usd,
        coalesce(sum(reserved_usd) FILTER(WHERE settled_at IS NULL),0) AS reserved_usd,
        coalesce(sum(input_tokens),0) AS input_tokens,coalesce(sum(output_tokens),0) AS output_tokens
        FROM spend_reservations WHERE job_id=$1''',job_id)
    return {k: float(v) if k.endswith('_usd') else int(v) for k,v in dict(row).items()}


async def reserve_spend(conn: asyncpg.Connection, lease: Lease, reservation_key: str,
                        max_usd: float, kind: str) -> UUID:
    if not math.isfinite(float(max_usd)) or max_usd < 0:
        raise ValueError('reservation amount must be finite and nonnegative')
    async with lease_transaction(conn, lease, active=True) as job:
        if await conn.fetchval('SELECT 1 FROM spend_reservations WHERE job_id=$1 AND reservation_key=$2',
                               lease.job_id,reservation_key):
            raise ReservationConflict('this external call is already reserved')
        committed = await conn.fetchval('''SELECT coalesce(sum(coalesce(actual_usd,reserved_usd)),0)
            FROM spend_reservations WHERE job_id=$1''',lease.job_id)
        limit = Decimal(str((job['config'].get('budget') or {}).get('max_llm_usd',10)))
        amount = Decimal(str(max_usd))
        if committed + amount > limit:
            raise BudgetExceeded('job spending budget cannot cover this reservation')
        return await conn.fetchval('''INSERT INTO spend_reservations(job_id,reservation_key,lease_epoch,kind,reserved_usd)
            VALUES($1,$2,$3,$4,$5) RETURNING id''',lease.job_id,reservation_key,lease.epoch,kind,amount)


async def settle_spend(conn: asyncpg.Connection, reservation_id: UUID, actual_usd: float,
                       input_tokens: int = 0, output_tokens: int = 0, metadata: dict | None = None) -> None:
    """Account for an already-dispatched call even when its worker lost ownership."""
    if not math.isfinite(float(actual_usd)) or actual_usd < 0 or min(input_tokens,output_tokens) < 0:
        raise ValueError('invalid usage accounting')
    async with conn.transaction():
        row = await conn.fetchrow('SELECT * FROM spend_reservations WHERE id=$1 FOR UPDATE',reservation_id)
        if row is None:
            raise ValueError('unknown spending reservation')
        actual = Decimal(str(actual_usd))
        if row['settled_at'] is not None:
            if row['actual_usd'] != actual or row['input_tokens'] != input_tokens or row['output_tokens'] != output_tokens:
                raise ValueError('settled usage is immutable')
            return
        # If a provider reports more than the reservation, retain its actual cost;
        # never hide overspend. The gateway must bound requests before dispatch.
        await conn.execute('''UPDATE spend_reservations SET actual_usd=$2,input_tokens=$3,output_tokens=$4,
            metadata=$5,settled_at=now() WHERE id=$1''',reservation_id,actual,input_tokens,output_tokens,metadata or {})


async def acquire_sandbox_slot(conn: asyncpg.Connection, lease: Lease, slot_key: str, limit: int,
                               ttl_sec: float, provider_id: str | None = None) -> bool:
    if limit < 1 or ttl_sec <= 0:
        raise ValueError('positive slot limit and TTL required')
    async with lease_transaction(conn, lease, active=True):
        # One short lock serializes global admission across all worker replicas.
        await conn.execute('SELECT pg_advisory_xact_lock(781204990)')
        # Expiry alone is not provider cleanup proof. In particular, a worker
        # must not free another Docker daemon's capacity using local absence.
        existing = await conn.fetchrow('SELECT * FROM sandbox_slots WHERE slot_key=$1',slot_key)
        if existing:
            if (existing['job_id'],existing['lease_owner'],existing['lease_epoch']) != (lease.job_id,lease.owner,lease.epoch):
                return False
            if existing['provider_id'] != provider_id:
                return False  # An attempt cannot be rebound to a different daemon.
            await conn.execute('''UPDATE sandbox_slots SET expires_at=clock_timestamp()+make_interval(secs=>$2),
                provider_id=coalesce($3,provider_id) WHERE slot_key=$1''',slot_key,float(ttl_sec),provider_id)
            return True
        if await conn.fetchval('SELECT count(*) FROM sandbox_slots') >= limit:
            return False
        await conn.execute('''INSERT INTO sandbox_slots(slot_key,job_id,lease_owner,lease_epoch,expires_at,provider_id)
            VALUES($1,$2,$3,$4,clock_timestamp()+make_interval(secs=>$5),$6)''',
            slot_key,lease.job_id,lease.owner,lease.epoch,float(ttl_sec),provider_id)
        return True


async def release_sandbox_slot(conn: asyncpg.Connection, lease: Lease, slot_key: str) -> None:
    # Caller has stopped its local runner and confirmed provider cleanup. Exact
    # identity remains safe after lease loss and cannot remove a replacement slot.
    await conn.execute('''DELETE FROM sandbox_slots WHERE slot_key=$1 AND job_id=$2 AND lease_owner=$3 AND lease_epoch=$4''',
                       slot_key,lease.job_id,lease.owner,lease.epoch)


async def list_reconcilable_sandbox_slots(conn: asyncpg.Connection, provider_id: str,
                                         limit: int = 100) -> list[asyncpg.Record]:
    """Expired orphan slots on one identified provider, never legacy NULL scope.

    Waiting for expiry prevents an orphan runner still preparing an image from
    recreating a live task after an early negative provider inspection.
    """
    if not provider_id or not 1 <= limit <= 1000:
        raise ValueError('provider identity and bounded positive limit required')
    return await conn.fetch('''SELECT s.* FROM sandbox_slots s JOIN jobs j ON j.id=s.job_id
        WHERE s.provider_id=$1 AND s.expires_at <= clock_timestamp()
          AND (j.status <> 'RUNNING' OR j.lease_owner IS DISTINCT FROM s.lease_owner
               OR j.lease_epoch <> s.lease_epoch OR j.lease_expires_at IS NULL
               OR j.lease_expires_at <= clock_timestamp())
        ORDER BY s.expires_at,s.slot_key LIMIT $2''', provider_id,limit)


async def reconcile_sandbox_slot(conn: asyncpg.Connection, lease: Lease, slot_key: str,
                                 provider_id: str, expected_expires_at) -> bool:
    """Release after caller confirms the identified provider has no resources.

    No provider I/O belongs in this transaction. Recheck the observed slot and
    job together so a delayed inspection cannot release a renewed/replaced slot.
    """
    if not provider_id:
        raise ValueError('provider identity is required')
    async with conn.transaction():
        job = await conn.fetchrow('SELECT * FROM jobs WHERE id=$1 FOR UPDATE',lease.job_id)
        slot = await conn.fetchrow('SELECT * FROM sandbox_slots WHERE slot_key=$1 FOR UPDATE',slot_key)
        if job is None or slot is None:
            return False
        if (slot['job_id'],slot['lease_owner'],slot['lease_epoch'],slot['provider_id'],slot['expires_at']) != (
                lease.job_id,lease.owner,lease.epoch,provider_id,expected_expires_at):
            return False
        now = await conn.fetchval('SELECT clock_timestamp()')
        stale = (job['status'] != 'RUNNING' or job['lease_owner'] != lease.owner
                 or job['lease_epoch'] != lease.epoch or job['lease_expires_at'] is None
                 or job['lease_expires_at'] <= now)
        if slot['expires_at'] > now or not stale:
            return False
        await conn.execute('DELETE FROM sandbox_slots WHERE slot_key=$1',slot_key)
        await add_event(conn,lease.job_id,'sandbox.slot_reconciled',{
            'slot_key':slot_key,'provider_id':provider_id,'lease_epoch':lease.epoch,
            'provider_absence_confirmed':True,'expired_orphan':True})
        return True
