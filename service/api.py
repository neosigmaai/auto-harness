"""HTTP surface. Stateless — Postgres is the only channel to the worker."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import config
import db
import optimizer

app = FastAPI(title="Agent Optimization Service")



def principal(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "expected 'Authorization: Bearer <api key>'")
    who = db.principal(authorization.removeprefix("Bearer "))
    if not who:
        raise HTTPException(401, "unknown api key")
    return who


Principal = Annotated[dict[str, Any], Depends(principal)]


def owner_filter(who: dict[str, Any]) -> str | None:
    """Extra per-user narrowing on top of the org
    """
    return None if who["role"] == "admin" else who["user_id"]


# request/response models ----------------------------------------------------------------

class JobRequest(BaseModel):
    task_ids: list[str] | None = None          # defaults to the whole allowed subset
    max_iterations: int = Field(default=3, ge=1)
    mode: Literal["real", "mock"] = "real"


class MemberRequest(BaseModel):
    email: str
    role: Literal["admin", "member"] = "member"


def split_holdout(task_ids: list[str]) -> list[str]:
    """~30% held out of the prompt, not the run. Seeded so a set always splits the same."""
    ids = sorted(task_ids)
    k = max(1, round(len(ids) * config.HOLDOUT_FRACTION)) if len(ids) > 2 else 0
    return sorted(random.Random(str(ids)).sample(ids, k)) if k else []


@app.get("/health")
def health() -> dict[str, str]:
    db.pool()
    return {"status": "ok", "database": config.db_label()}


@app.post("/jobs", status_code=202)
def submit(req: JobRequest, who: Principal,
           idempotency_key: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    task_ids = req.task_ids or list(config.TASK_SUBSET)
    unknown = sorted(set(task_ids) - set(config.TASK_SUBSET))
    if unknown:
        raise HTTPException(422, f"unknown task_ids: {unknown}")   # trust boundary
    if not task_ids:
        raise HTTPException(422, "task_ids must not be empty")
    if req.mode == "real" and not config.PLATFORM_OPENAI_KEY:
        raise HTTPException(422, "OPENAI_API_KEY is not configured; use mode='mock'")

    body_hash = hashlib.sha256(
        json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()
    job = db.create_job(
        who["org_id"], who["user_id"], task_ids=task_ids,
        holdout_task_ids=split_holdout(task_ids),
        max_iterations=min(req.max_iterations, config.MAX_ITERATIONS_CAP),  
        mode=req.mode, idempotency_key=idempotency_key, request_hash=body_hash)
    if job["replayed"] and job["request_hash"] != body_hash:
        raise HTTPException(409, "Idempotency-Key reused with a different request body")
    return {"job_id": job["id"], "status": job["status"], "replayed": job["replayed"]}


@app.get("/jobs")
def list_jobs(who: Principal, status: str | None = None) -> list[dict[str, Any]]:
    rows = db.list_jobs(who["org_id"], only_user=owner_filter(who), status=status)
    return [_summary(who["org_id"], j) for j in rows]


@app.get("/jobs/{job_id}")
def get_job(job_id: str, who: Principal) -> dict[str, Any]:
    job = db.get_job(who["org_id"], job_id, only_user=owner_filter(who))
    if not job:
        raise HTTPException(404, "no such job")   
    its = db.iterations(who["org_id"], job_id)
    latest = its[-1] if its else None
    return {
        "job_id": job["id"],
        "status": job["status"],
        "failure_reason": job["failure_reason"],
        "error_detail": job["error_detail"],
        "queue_reason": _queue_reason(who["org_id"]) if job["status"] == "queued" else None,
        "mode": job["mode"],
        "iterations_completed": len(its),
        "progress": _progress(job, its),
        "base_commit": job["base_commit"],
        "task_ids": job["task_ids"],
        "holdout_task_ids": job["holdout_task_ids"],
        "usage": db.usage(who["org_id"], job_id),
        "latest": _latest(latest),
        "final_outcome": {
            "stopped_because": job["stopped_because"],
            "baseline_score": (job["baseline_score"]
                               if job["baseline_score"] is not None
                               else its[0]["visible_score"] if its else None),
            "best_visible_score": (job["best_visible_score"]
                                   if job["best_visible_score"] is not None
                                   else max((i["visible_score"] for i in its
                                             if i["accepted"] and i["visible_score"]
                                             is not None), default=None)),
            "holdout_score": job["holdout_score"],
            "improved": (job["best_visible_score"] is not None
                         and job["baseline_score"] is not None
                         and job["best_visible_score"] > job["baseline_score"]),
        },
    }


@app.get("/jobs/{job_id}/iterations")
def get_iterations(job_id: str, who: Principal) -> list[dict[str, Any]]:
    if not db.get_job(who["org_id"], job_id, only_user=owner_filter(who)):
        raise HTTPException(404, "no such job")
    return db.iterations(who["org_id"], job_id)


@app.get("/jobs/{job_id}/iterations/{n}/optimizer-input")
def optimizer_input(job_id: str, n: int, who: Principal) -> dict[str, Any]:
    """Exactly what the LLM was shown to produce iteration n.

    Rebuilt from the stored rows rather than saved twice — so it tracks whatever the
    prompt builder does today, which is the honest caveat.
    """
    job = db.get_job(who["org_id"], job_id, only_user=owner_filter(who))
    if not job:
        raise HTTPException(404, "no such job")
    its = {it["n"]: it for it in db.iterations(who["org_id"], job_id)}
    if n not in its:
        raise HTTPException(404, f"job has no iteration {n}")
    if n == 0:
        raise HTTPException(422, "iteration 0 is the baseline — no optimizer call made it")

    prior = its[n - 1]
    visible = set(its[n]["results"] or prior["results"]) - set(job["holdout_task_ids"])
    failures = [f for f in prior["failures"] if f["task_id"] in visible]
    ledger = [f"iter {i}: {its[i]['proposal'] or 'baseline'} -> "
              f"{its[i]['visible_score']} "
              f"({'accepted' if its[i]['accepted'] else 'rejected'})"
              for i in sorted(its) if i < n]
    return {
        "iteration": n,
        "produced_proposal": its[n]["proposal"],
        "inputs": {"failures_shown": failures, "ledger": ledger},
        "prompt": optimizer.build_prompt(prior["agent_source"], failures, ledger),
        "system_prompt": optimizer.SYSTEM,
        "note": "reconstructed from stored rows with the current prompt builder",
    }


@app.post("/jobs/{job_id}/cancel", status_code=202)
def cancel(job_id: str, who: Principal) -> dict[str, str]:
    status = db.cancel_job(who["org_id"], job_id, only_user=owner_filter(who))
    if not status:
        raise HTTPException(404, "no such job, or it is already finished")
    return {"status": status}


@app.post("/orgs/members", status_code=201)
def add_member(req: MemberRequest, who: Principal) -> dict[str, str]:
    if who["role"] != "admin":
        raise HTTPException(403, "admin only")
    key = db.create_member(who["org_id"], req.email, req.role)
    return {"email": req.email, "role": req.role, "api_key": key,
            "note": "shown once, never again"}


def _queue_reason(org_id: str) -> str:
    """"Busy" and "stuck" have to stay distinguishable, or you go hunting the wrong thing."""
    platform, mine, cap = db.capacity_used(org_id)
    if platform >= config.PLATFORM_MAX_JOBS:
        return f"waiting on platform job capacity ({platform}/{config.PLATFORM_MAX_JOBS})"
    if mine >= cap:
        return f"waiting on org job capacity ({mine}/{cap})"
    return "capacity is free but nothing has claimed it"


def _progress(job: dict[str, Any], its: list[dict[str, Any]]) -> dict[str, Any]:
    """What is happening now. Rows only land when an iteration finishes, so without this
    a job mid-baseline looks identical to a wedged one.
    """
    now = datetime.now(timezone.utc)
    running = job["status"] in ("running", "cancelling")
    started, last = job["started_at"], job["finished_at"]
    newest = max((it["created_at"] for it in its), default=started)
    return {
        "current_iteration": len(its) if running else None,
        "elapsed_s": int(((last or now) - started).total_seconds()) if started else None,
        "last_progress_age_s": (int((now - newest).total_seconds())
                                if running and newest else None),
        "sandbox_id": job["sandbox_id"],
    }


def _latest(it: dict[str, Any] | None) -> dict[str, Any] | None:
    if not it:
        return None
    results = it["results"]
    return {
        "iteration": it["n"],
        "passed": sorted(t for t, r in results.items() if r is not None and r >= 1.0),
        "failed": sorted(t for t, r in results.items() if r is not None and r < 1.0),
        "errored": sorted(t for t, r in results.items() if r is None),
        "visible_score": it["visible_score"],
        "holdout_score": it["holdout_score"],
        "failures": it["failures"],
    }


def _summary(org_id: str, job: dict[str, Any]) -> dict[str, Any]:
    """Counters ride on the job; no separate rollup route."""
    return {
        "job_id": job["id"], "status": job["status"], "mode": job["mode"],
        "submitted_by": job["created_by"], "created_at": job["created_at"],
        "best_visible_score": job["best_visible_score"],
        "sandboxes_reserved": (1 + min(config.N_CONCURRENT, len(job["task_ids"]))
                               if job["status"] in ("running", "cancelling") else 0),
        **_progress(job, db.iterations(org_id, job["id"])),
        **db.usage(org_id, job["id"]),
    }
