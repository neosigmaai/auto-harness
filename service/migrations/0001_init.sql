-- Agent Optimization Service: initial schema.
-- Conventions: every tenant-owned row carries org_id; timestamps are timestamptz (UTC).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE organizations (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name                text NOT NULL UNIQUE,
    max_concurrent_jobs int  NOT NULL DEFAULT 2,
    max_iterations      int  NOT NULL DEFAULT 20,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email       text NOT NULL,
    role        text NOT NULL CHECK (role IN ('admin', 'member')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (org_id, email)
);

CREATE TABLE api_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash    text NOT NULL UNIQUE,          -- sha256 of the raw key; raw key is never stored
    key_prefix  text NOT NULL,                 -- first 8 chars, for display only
    user_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id      uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);

-- Content-addressed agent sources. Same text => same row (per org).
CREATE TABLE agent_versions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    sha256      text NOT NULL,
    content     text NOT NULL,
    parent_id   uuid REFERENCES agent_versions(id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (org_id, sha256)
);

CREATE TABLE jobs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    created_by          uuid NOT NULL REFERENCES users(id),
    status              text NOT NULL CHECK (status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','CANCELLED')),
    config              jsonb NOT NULL,
    idempotency_key     text,
    cancel_requested    boolean NOT NULL DEFAULT false,
    -- lease: who is working on it and until when. epoch is a fencing token.
    lease_owner         text,
    lease_expires_at    timestamptz,
    lease_epoch         int NOT NULL DEFAULT 0,
    attempt             int NOT NULL DEFAULT 0,
    -- progress / outcome
    current_iteration   int,
    baseline_score      double precision,
    best_score          double precision,
    best_iteration_id   uuid,
    stop_reason         text,
    error               text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    started_at          timestamptz,
    finished_at         timestamptz,
    updated_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (org_id, idempotency_key)
);
CREATE INDEX jobs_queue_idx    ON jobs (created_at) WHERE status = 'QUEUED';
CREATE INDEX jobs_running_idx  ON jobs (lease_expires_at) WHERE status = 'RUNNING';
CREATE INDEX jobs_org_idx      ON jobs (org_id, created_at DESC);

CREATE TABLE iterations (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id              uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    index               int  NOT NULL,
    agent_version_id    uuid NOT NULL REFERENCES agent_versions(id),
    parent_iteration_id uuid REFERENCES iterations(id),
    -- phase is the checkpoint: PROVISION, BENCHMARK, COLLECT, DIGEST, PROPOSE, VALIDATE, DECIDE, DONE
    phase               text NOT NULL,
    status              text NOT NULL CHECK (status IN ('running','completed','errored')),
    score               double precision,
    n_pass              int,
    n_fail              int,
    n_error             int,
    proposal            jsonb,      -- {hypothesis, targets, rationale, change_summary}
    decision            text CHECK (decision IN ('baseline','accepted','rejected','errored')),
    decision_reason     text,
    learnings           text,
    error               text,
    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    UNIQUE (job_id, index)
);

CREATE TABLE task_runs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    iteration_id    uuid NOT NULL REFERENCES iterations(id) ON DELETE CASCADE,
    task_name       text NOT NULL,
    status          text NOT NULL CHECK (status IN ('pending','running','passed','failed','timeout','error')),
    reward          double precision,
    failure_class   text,
    duration_s      double precision,
    n_steps         int,
    exit_reason     text,
    test_stdout     text,
    trace           jsonb,
    digest          jsonb,
    sandbox_id      text,
    started_at      timestamptz,
    finished_at     timestamptz,
    UNIQUE (iteration_id, task_name)
);
CREATE INDEX task_runs_iter_idx ON task_runs (iteration_id);

CREATE TABLE job_events (
    id       bigserial PRIMARY KEY,
    job_id   uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts       timestamptz NOT NULL DEFAULT now(),
    kind     text NOT NULL,
    payload  jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX job_events_job_idx ON job_events (job_id, id);

-- Sandbox provider readiness per task (template pre-warm bookkeeping).
CREATE TABLE task_catalog_state (
    task_name           text NOT NULL,
    provider            text NOT NULL,
    template_ready_at   timestamptz,
    last_error          text,
    PRIMARY KEY (task_name, provider)
);
