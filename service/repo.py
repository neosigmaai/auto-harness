"""Data access. Tenant-facing functions take a Principal and scope every query by org_id
(and by created_by for members). Worker-facing functions take a lease (job_id, owner, epoch)
and refuse to write once the lease has moved on (fencing)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from .auth import Principal, generate_api_key, hash_api_key


class LeaseLost(Exception):
    """Raised when a worker tries to write for a job it no longer owns."""


@dataclass(frozen=True)
class Lease:
    job_id: UUID
    owner: str
    epoch: int


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
    async with conn.transaction():
        if idempotency_key:
            existing = await conn.fetchrow(
                "SELECT * FROM jobs WHERE org_id=$1 AND idempotency_key=$2", p.org_id, idempotency_key
            )
            if existing:
                return existing, False
        job = await conn.fetchrow(
            """INSERT INTO jobs(org_id, created_by, status, config, idempotency_key)
               VALUES ($1,$2,'QUEUED',$3,$4) RETURNING *""",
            p.org_id, p.user_id, config, idempotency_key,
        )
        await add_event(conn, job["id"], "job.submitted", {"by": str(p.user_id), "n_tasks": len(config["tasks"])})
        return job, True


async def get_job(conn: asyncpg.Connection, p: Principal, job_id: UUID) -> asyncpg.Record | None:
    where, params = _scope(p)
    return await conn.fetchrow(f"SELECT j.* FROM jobs j WHERE j.id = ${len(params)+1} AND {where}", *params, job_id)


async def list_jobs(conn: asyncpg.Connection, p: Principal, status: str | None, limit: int, before: str | None) -> list[asyncpg.Record]:
    where, params = _scope(p)
    sql = f"SELECT j.* FROM jobs j WHERE {where}"
    if status:
        params.append(status); sql += f" AND j.status = ${len(params)}"
    if before:
        params.append(before); sql += f" AND j.created_at < ${len(params)}::timestamptz"
    params.append(limit); sql += f" ORDER BY j.created_at DESC LIMIT ${len(params)}"
    return await conn.fetch(sql, *params)


async def request_cancel(conn: asyncpg.Connection, p: Principal, job_id: UUID) -> asyncpg.Record | None:
    job = await get_job(conn, p, job_id)
    if job is None:
        return None
    if job["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
        return job
    async with conn.transaction():
        if job["status"] == "QUEUED":
            # Nobody owns it yet: cancel outright (CAS on status so a racing claim loses).
            row = await conn.fetchrow(
                """UPDATE jobs SET status='CANCELLED', stop_reason='cancelled', finished_at=now(), updated_at=now()
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
        """SELECT task_name, status, reward, failure_class, duration_s, n_steps, exit_reason
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
CLAIM_SQL = """
WITH candidate AS (
    SELECT j.id
    FROM jobs j
    JOIN organizations o ON o.id = j.org_id
    WHERE (j.status = 'QUEUED' OR (j.status = 'RUNNING' AND j.lease_expires_at < now()))
      AND (SELECT count(*) FROM jobs r
           WHERE r.org_id = j.org_id AND r.status = 'RUNNING' AND r.lease_expires_at >= now() AND r.id <> j.id)
          < o.max_concurrent_jobs
    ORDER BY j.created_at
    FOR UPDATE OF j SKIP LOCKED
    LIMIT 1
)
UPDATE jobs SET status = 'RUNNING',
                lease_owner = $1,
                lease_expires_at = now() + make_interval(secs => $2),
                lease_epoch = lease_epoch + 1,
                attempt = attempt + 1,
                started_at = coalesce(started_at, now()),
                updated_at = now()
FROM candidate WHERE jobs.id = candidate.id
RETURNING jobs.*
"""


async def claim_job(conn: asyncpg.Connection, worker_id: str, ttl_sec: int) -> asyncpg.Record | None:
    async with conn.transaction():
        job = await conn.fetchrow(CLAIM_SQL, worker_id, float(ttl_sec))
        if job:
            await add_event(conn, job["id"], "job.claimed",
                            {"worker": worker_id, "epoch": job["lease_epoch"], "attempt": job["attempt"]})
        return job


async def heartbeat(conn: asyncpg.Connection, lease: Lease, ttl_sec: int) -> bool:
    """Extend the lease. Returns cancel_requested. Raises LeaseLost if we no longer own the job."""
    row = await conn.fetchrow(
        """UPDATE jobs SET lease_expires_at = now() + make_interval(secs => $4), updated_at = now()
           WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING' RETURNING cancel_requested""",
        lease.job_id, lease.owner, lease.epoch, float(ttl_sec))
    if row is None:
        raise LeaseLost(f"lease for job {lease.job_id} (epoch {lease.epoch}) no longer held by {lease.owner}")
    return row["cancel_requested"]


async def assert_lease(conn: asyncpg.Connection, lease: Lease) -> None:
    ok = await conn.fetchval(
        "SELECT 1 FROM jobs WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING'",
        lease.job_id, lease.owner, lease.epoch)
    if not ok:
        raise LeaseLost(f"lease for job {lease.job_id} epoch {lease.epoch} lost")


async def finish_job(conn: asyncpg.Connection, lease: Lease, status: str, stop_reason: str | None,
                     error: str | None = None, best_iteration_id: UUID | None = None,
                     best_score: float | None = None, baseline_score: float | None = None) -> None:
    async with conn.transaction():
        row = await conn.fetchrow(
            """UPDATE jobs SET status=$4, stop_reason=$5, error=$6, finished_at=now(), updated_at=now(),
                   best_iteration_id = coalesce($7, best_iteration_id),
                   best_score = coalesce($8, best_score), baseline_score = coalesce($9, baseline_score),
                   lease_owner=NULL, lease_expires_at=NULL
               WHERE id=$1 AND lease_owner=$2 AND lease_epoch=$3 AND status='RUNNING' RETURNING id""",
            lease.job_id, lease.owner, lease.epoch, status, stop_reason, error, best_iteration_id, best_score, baseline_score)
        if row is None:
            raise LeaseLost("cannot finish a job we do not own")
        await add_event(conn, lease.job_id, f"job.{status.lower()}", {"stop_reason": stop_reason, "error": error})


async def update_job_progress(conn: asyncpg.Connection, lease: Lease, **fields) -> None:
    sets, params = [], [lease.job_id, lease.owner, lease.epoch]
    for k, v in fields.items():
        params.append(v); sets.append(f"{k} = ${len(params)}")
    sets.append("updated_at = now()")
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
    await assert_lease(conn, lease)
    return await conn.fetchrow(
        """INSERT INTO iterations(job_id, index, agent_version_id, parent_iteration_id, phase, status, proposal)
           VALUES ($1,$2,$3,$4,'PROVISION','running',$5) RETURNING *""",
        lease.job_id, index, agent_version_id, parent_iteration_id, proposal)


async def set_phase(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, phase: str) -> None:
    await assert_lease(conn, lease)
    await conn.execute("UPDATE iterations SET phase=$2 WHERE id=$1", iteration_id, phase)
    await add_event(conn, lease.job_id, "iteration.phase", {"iteration_id": str(iteration_id), "phase": phase})


async def complete_iteration(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, *, score: float | None,
                             n_pass: int, n_fail: int, n_error: int, decision: str, decision_reason: str | None,
                             learnings: str | None = None, status: str = "completed", error: str | None = None) -> None:
    await assert_lease(conn, lease)
    await conn.execute(
        """UPDATE iterations SET phase='DONE', status=$2, score=$3, n_pass=$4, n_fail=$5, n_error=$6,
               decision=$7, decision_reason=$8, learnings=coalesce($9, learnings), error=$10, finished_at=now()
           WHERE id=$1""",
        iteration_id, status, score, n_pass, n_fail, n_error, decision, decision_reason, learnings, error)


async def ensure_task_runs(conn: asyncpg.Connection, iteration_id: UUID, tasks: list[str]) -> None:
    await conn.executemany(
        "INSERT INTO task_runs(iteration_id, task_name, status) VALUES ($1,$2,'pending') ON CONFLICT DO NOTHING",
        [(iteration_id, t) for t in tasks])


async def pending_task_names(conn: asyncpg.Connection, iteration_id: UUID) -> list[str]:
    rows = await conn.fetch(
        "SELECT task_name FROM task_runs WHERE iteration_id=$1 AND status IN ('pending','running') ORDER BY task_name",
        iteration_id)
    return [r["task_name"] for r in rows]


async def mark_task_running(conn: asyncpg.Connection, iteration_id: UUID, task: str, sandbox_id: str | None = None) -> None:
    await conn.execute(
        """UPDATE task_runs SET status='running', started_at=coalesce(started_at, now()), sandbox_id=coalesce($3, sandbox_id)
           WHERE iteration_id=$1 AND task_name=$2""", iteration_id, task, sandbox_id)


async def record_task_result(conn: asyncpg.Connection, lease: Lease, iteration_id: UUID, task: str, *, status: str,
                             reward: float | None, failure_class: str | None, duration_s: float | None,
                             n_steps: int | None, exit_reason: str | None, test_stdout: str | None,
                             trace: Any, digest: dict | None, sandbox_id: str | None = None) -> None:
    await assert_lease(conn, lease)
    await conn.execute(
        """INSERT INTO task_runs(iteration_id, task_name, status, reward, failure_class, duration_s, n_steps, exit_reason,
                                 test_stdout, trace, digest, sandbox_id, started_at, finished_at)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,now(),now())
           ON CONFLICT (iteration_id, task_name) DO UPDATE SET
               status=EXCLUDED.status, reward=EXCLUDED.reward, failure_class=EXCLUDED.failure_class,
               duration_s=EXCLUDED.duration_s, n_steps=EXCLUDED.n_steps, exit_reason=EXCLUDED.exit_reason,
               test_stdout=EXCLUDED.test_stdout, trace=EXCLUDED.trace, digest=EXCLUDED.digest,
               sandbox_id=coalesce(EXCLUDED.sandbox_id, task_runs.sandbox_id), finished_at=now()""",
        iteration_id, task, status, reward, failure_class, duration_s, n_steps, exit_reason, test_stdout, trace, digest, sandbox_id)


async def task_results(conn: asyncpg.Connection, iteration_id: UUID) -> dict[str, asyncpg.Record]:
    rows = await conn.fetch("SELECT * FROM task_runs WHERE iteration_id=$1", iteration_id)
    return {r["task_name"]: r for r in rows}


async def set_task_digest(conn: asyncpg.Connection, iteration_id: UUID, task: str, digest: dict) -> None:
    await conn.execute("UPDATE task_runs SET digest=$3 WHERE iteration_id=$1 AND task_name=$2", iteration_id, task, digest)
