"""Tenant isolation, idempotency, claim, ratchet. Runs against a throwaway database."""

from __future__ import annotations

import os
import tempfile

# Unconditional, never setdefault — this leaves jobs 'running' on purpose, and pointing
# it at a real database fills the platform ceiling and wedges the worker.
os.environ["PGDATA_DIR"] = tempfile.mkdtemp(prefix="aos-test-")

import config  # noqa: E402
import db  # noqa: E402


def main() -> None:
    a_org, a_key = db.create_org("org-a", "admin@a.test")
    b_org, b_key = db.create_org("org-b", "admin@b.test")
    member_key = db.create_member(a_org, "member@a.test", "member")

    a_admin = db.principal(a_key)
    a_member = db.principal(member_key)
    assert a_admin["role"] == "admin" and str(a_admin["org_id"]) == a_org
    assert db.principal("ao_live_nonsense") is None, "unknown key must not authenticate"

    # invisible to another member, and to the other org entirely
    job = db.create_job(a_org, a_member["user_id"], task_ids=["t1", "t2"],
                        holdout_task_ids=["t2"], max_iterations=3, mode="mock",
                        idempotency_key="k1", request_hash="h1")
    assert job["replayed"] is False
    jid = job["id"]
    assert db.get_job(a_org, jid, only_user=None) is not None, "admin sees org job"
    assert db.get_job(a_org, jid, only_user=a_member["user_id"]) is not None, "owner sees own"
    assert db.get_job(a_org, jid, only_user=a_admin["user_id"]) is None, "non-owner member blind"
    assert db.get_job(b_org, jid, only_user=None) is None, "cross-org read must miss"

    # replay, not double-spend
    again = db.create_job(a_org, a_member["user_id"], task_ids=["t1", "t2"],
                          holdout_task_ids=["t2"], max_iterations=3, mode="mock",
                          idempotency_key="k1", request_hash="h1")
    assert again["replayed"] is True and again["id"] == jid, "same key must replay"

    # the platform ceiling holds
    claimed = db.claim_job()
    assert claimed is not None and claimed["id"] == jid
    assert claimed["deadline_at"] is not None, "deadline set at claim"
    for i in range(config.PLATFORM_MAX_JOBS + 1):
        db.create_job(b_org, db.principal(b_key)["user_id"], task_ids=["t1"],
                      holdout_task_ids=[], max_iterations=1, mode="mock",
                      idempotency_key=f"b{i}", request_hash="h")
    got = [db.claim_job() for _ in range(config.PLATFORM_MAX_JOBS + 2)]
    running = 1 + sum(g is not None for g in got)
    assert running <= config.PLATFORM_MAX_JOBS, f"ceiling breached: {running}"

    # best accepted, ignoring rejected higher scores
    for n, (score, accepted) in enumerate([(0.4, True), (0.6, True), (0.9, False)]):
        db.add_iteration(a_org, jid, n, agent_source=f"src{n}", results={"t1": score},
                         outcome="baseline" if n == 0 else "improved",
                         accepted=accepted, visible_score=score, llm_calls=2,
                         input_tokens=10, output_tokens=5)
    best = db.best_accepted(a_org, jid)
    assert best["visible_score"] == 0.6, f"ratchet must ignore rejected: {best}"
    assert len(db.iterations(a_org, jid)) == 3
    assert db.usage(a_org, jid)["llm_calls"] == 6

    # cancel is scoped like every other read
    assert db.cancel_job(b_org, jid, only_user=None) is None, "cross-org cancel must miss"
    assert db.cancel_job(a_org, jid, only_user=None) == "cancelling"

    print("ok — isolation, idempotency, claim ceiling, ratchet, cancel scoping")


if __name__ == "__main__":
    main()
