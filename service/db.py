"""Every query, and every one takes org_id first — that convention is the isolation.

claim_job() is the one deliberate exception: the worker legitimately spans tenants.
"""

from __future__ import annotations

import hashlib
import pathlib
import secrets
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

import config

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config.database_url(), kwargs={"row_factory": dict_row},
                               min_size=1, open=True)
        with _pool.connection() as c:
            c.execute(pathlib.Path(__file__).with_name("schema.sql").read_text())
    return _pool


# --- keys -------------------------------------------------------------------------

def principal(api_key: str) -> dict[str, Any] | None:
    """The only place org_id enters the system."""
    with pool().connection() as c:
        return c.execute(
            "select org_id, id as user_id, role from users where api_key_hash = %s",
            (hashlib.sha256(api_key.encode()).digest(),),
        ).fetchone()


# --- orgs and members -------------------------------------------------------------

def create_org(name: str, admin_email: str) -> tuple[str, str]:
    with pool().connection() as c:
        org = c.execute("insert into orgs (name) values (%s) returning id",
                        (name,)).fetchone()["id"]
    return str(org), create_member(org, admin_email, "admin")


def create_member(org_id: str, email: str, role: str) -> str:
    """Returns the plaintext key — the only time it exists. We store the hash."""
    key = "ao_" + secrets.token_urlsafe(32)
    with pool().connection() as c:
        c.execute(
            "insert into users (org_id, email, role, api_key_hash) values (%s,%s,%s,%s)",
            (org_id, email, role, hashlib.sha256(key.encode()).digest()),
        )
    return key


# --- jobs -------------------------------------------------------------------------

def create_job(org_id: str, user_id: str, *, task_ids: list[str],
               holdout_task_ids: list[str], max_iterations: int, mode: str,
               idempotency_key: str | None, request_hash: str | None) -> dict[str, Any]:
    """Insert, or replay the original job when the idempotency key repeats."""
    with pool().connection() as c:
        row = c.execute(
            """insert into jobs (org_id, created_by, task_ids, holdout_task_ids,
                                 max_iterations, mode, idempotency_key, request_hash)
               values (%s,%s,%s,%s,%s,%s,%s,%s)
               on conflict (org_id, idempotency_key) do nothing
               returning id, status, request_hash""",
            (org_id, user_id, task_ids, holdout_task_ids, max_iterations, mode,
             idempotency_key, request_hash),
        ).fetchone()
        if row:
            return row | {"replayed": False}
        prior = c.execute(
            "select id, status, request_hash from jobs where org_id=%s and idempotency_key=%s",
            (org_id, idempotency_key),
        ).fetchone()
        return prior | {"replayed": True}


def get_job(org_id: str, job_id: str, *, only_user: str | None) -> dict[str, Any] | None:
    """only_user set => member scope. Missing and other-org are indistinguishable."""
    with pool().connection() as c:
        return c.execute(
            "select * from jobs where org_id=%s and id=%s "
            "and (%s::uuid is null or created_by=%s)",
            (org_id, job_id, only_user, only_user),
        ).fetchone()


def list_jobs(org_id: str, *, only_user: str | None,
              status: str | None) -> list[dict[str, Any]]:
    with pool().connection() as c:
        return c.execute(
            "select * from jobs where org_id=%s "
            "and (%s::uuid is null or created_by=%s) "
            "and (%s::text is null or status=%s) order by created_at desc",
            (org_id, only_user, only_user, status, status),
        ).fetchall()


def cancel_job(org_id: str, job_id: str, *, only_user: str | None) -> str | None:
    """queued jumps straight to cancelled; running goes via cancelling."""
    with pool().connection() as c:
        row = c.execute(
            """update jobs set status = case when status='queued' then 'cancelled'
                                             else 'cancelling' end,
                               stopped_because='cancelled', failure_reason='cancelled',
                               finished_at = case when status='queued' then now() end
               where org_id=%s and id=%s and status in ('queued','running')
                 and (%s::uuid is null or created_by=%s)
               returning status""",
            (org_id, job_id, only_user, only_user),
        ).fetchone()
        return row["status"] if row else None


# --- worker -----------------------------------------------------------------------

def claim_job() -> dict[str, Any] | None:
    """Cross-org by design. The advisory lock is what makes the counts below safe —
    SKIP LOCKED protects the candidate row, not an aggregate over other rows."""
    with pool().connection() as c:
        c.execute("select pg_advisory_xact_lock(%s)", (config.CLAIM_LOCK,))
        # ponytail: two queries under the lock. Fuse into a CTE if claims ever get hot.
        counts = {r["org_id"]: r["n"] for r in c.execute(
            "select org_id, count(*) n from jobs where status in ('running','cancelling')"
            " group by org_id")}
        if sum(counts.values()) >= config.PLATFORM_MAX_JOBS:
            return None
        for job in c.execute(
            """select j.*, o.max_concurrent_jobs, o.max_job_seconds
               from jobs j join orgs o on o.id = j.org_id
               where j.status='queued' order by j.created_at
               for update of j skip locked"""
        ):
            if counts.get(job["org_id"], 0) >= job["max_concurrent_jobs"]:
                continue
            return c.execute(
                """update jobs set status='running', claimed_at=now(), started_at=now(),
                          deadline_at = now() + %s * interval '1 second'
                   where org_id=%s and id=%s returning *""",
                (job["max_job_seconds"], job["org_id"], job["id"]),
            ).fetchone()
        return None


def reclaim_expired() -> list[dict[str, Any]]:
    """Past-lease rows fail, they do not resume. Caller kills sandbox_id first."""
    with pool().connection() as c:
        return c.execute(
            """update jobs set status='failed', failure_reason='infra',
                      stopped_because='error', finished_at=now(),
                      error_detail='worker lease expired; job abandoned mid-run'
               where status in ('running','cancelling') and deadline_at is not null
                 and now() > deadline_at + (%s || ' seconds')::interval
               returning org_id, id, sandbox_id""",
            (str(config.LEASE_SLACK_SECONDS),),
        ).fetchall()


def capacity_used(org_id: str) -> tuple[int, int, int]:
    """(platform running, org running, org cap) — for queue_reason."""
    with pool().connection() as c:
        r = c.execute(
            """select (select count(*) from jobs
                        where status in ('running','cancelling')) platform,
                      (select count(*) from jobs where org_id=%s
                        and status in ('running','cancelling')) mine,
                      max_concurrent_jobs cap
               from orgs where id=%s""",
            (org_id, org_id),
        ).fetchone()
        return r["platform"], r["mine"], r["cap"]


def finish_job(org_id: str, job_id: str, *, status: str, stopped_because: str,
               failure_reason: str | None = None, error_detail: str | None = None,
               scores: dict[str, float | None] | None = None) -> None:
    s = scores or {}
    with pool().connection() as c:
        c.execute(
            """update jobs set status=%s, stopped_because=%s, failure_reason=%s,
                      error_detail=%s, finished_at=now(), baseline_score=%s,
                      best_visible_score=%s, holdout_score=%s
               where org_id=%s and id=%s""",
            (status, stopped_because, failure_reason, error_detail,
             s.get("baseline"), s.get("best_visible"), s.get("holdout"), org_id, job_id),
        )


def set_sandbox(org_id: str, job_id: str, sandbox_id: str | None) -> None:
    """Written as soon as the sandbox exists — cancel and reclaim both need the handle."""
    with pool().connection() as c:
        c.execute("update jobs set sandbox_id=%s where org_id=%s and id=%s",
                  (sandbox_id, org_id, job_id))


def job_status(org_id: str, job_id: str) -> str | None:
    with pool().connection() as c:
        row = c.execute("select status from jobs where org_id=%s and id=%s",
                        (org_id, job_id)).fetchone()
        return row["status"] if row else None



def add_iteration(org_id: str, job_id: str, n: int, **f: Any) -> None:
    with pool().connection() as c:
        c.execute(
            """insert into iterations (org_id, job_id, n, agent_source, proposal, results,
                    failures, visible_score, holdout_score, outcome, error_detail,
                    accepted, llm_calls, input_tokens, output_tokens, sandboxes_used,
                    sandbox_seconds)
               values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (org_id, job_id, n, f["agent_source"], f.get("proposal"),
             Jsonb(f["results"]), Jsonb(f.get("failures", [])), f.get("visible_score"),
             f.get("holdout_score"), f["outcome"], f.get("error_detail"), f["accepted"],
             f.get("llm_calls"), f.get("input_tokens"), f.get("output_tokens"),
             f.get("sandboxes_used"), f.get("sandbox_seconds")),
        )


def iterations(org_id: str, job_id: str) -> list[dict[str, Any]]:
    with pool().connection() as c:
        return c.execute(
            "select * from iterations where org_id=%s and job_id=%s order by n",
            (org_id, job_id),
        ).fetchall()


def best_accepted(org_id: str, job_id: str) -> dict[str, Any] | None:
    with pool().connection() as c:
        return c.execute(
            """select * from iterations where org_id=%s and job_id=%s and accepted
               order by visible_score desc nulls last, n desc limit 1""",
            (org_id, job_id),
        ).fetchone()


def usage(org_id: str, job_id: str) -> dict[str, Any]:
    with pool().connection() as c:
        return c.execute(
            """select coalesce(sum(llm_calls),0) llm_calls,
                      coalesce(sum(input_tokens),0) input_tokens,
                      coalesce(sum(output_tokens),0) output_tokens,
                      coalesce(sum(sandboxes_used),0) sandboxes_used,
                      coalesce(sum(sandbox_seconds),0) sandbox_seconds
               from iterations where org_id=%s and job_id=%s""",
            (org_id, job_id),
        ).fetchone()
