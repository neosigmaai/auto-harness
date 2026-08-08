-- Agent Optimization Service — the whole schema. Applied idempotently at startup.
-- org_id leads every PK and FK, so the service can be multi-tenant with no cross-org leakage.

create table if not exists orgs (
  id                  uuid primary key default gen_random_uuid(),
  name                text not null,
  max_concurrent_jobs int  not null default 3,     -- under PLATFORM_MAX_JOBS
  max_job_seconds     int  not null default 5400,  -- deadline; lease and E2B timeout
                                                   --   are this + slack
  created_at          timestamptz not null default now()
);


create table if not exists users (
  org_id       uuid not null references orgs (id) on delete cascade,
  id           uuid not null default gen_random_uuid(),
  email        text not null,
  role         text not null check (role in ('admin', 'member')),
  api_key_hash bytea not null,
  created_at   timestamptz not null default now(),
  primary key (org_id, id),
  unique (org_id, email)
);


create unique index if not exists users_api_key_hash_idx on users (api_key_hash);

create table if not exists jobs (
  org_id             uuid not null,
  id                 uuid not null default gen_random_uuid(),
  created_by         uuid not null,
  status             text not null default 'queued'
                       check (status in ('queued', 'running', 'cancelling',
                                         'succeeded', 'failed', 'cancelled')),
  failure_reason     text check (failure_reason in ('infra', 'llm', 'cancelled')),
  error_detail       text,                        -- the message, not just the enum
  stopped_because    text check (stopped_because in ('max_iterations', 'no_improvement',
                                                     'overfitting_suspected',
                                                     'time_limit', 'cancelled', 'error')),
  mode               text not null default 'real' check (mode in ('real', 'mock')),
  task_ids           text[] not null,             -- validated against the subset
  holdout_task_ids   text[] not null,             -- never enter the prompt
  max_iterations     int  not null,               -- server-capped
  deadline_at        timestamptz,                 -- claimed_at + org.max_job_seconds
  base_commit        text,                        -- provenance
  idempotency_key    text,
  request_hash       text,                        -- same key + different body -> 409
  sandbox_id         text,                        -- cancel + reclaim kill
  claimed_at         timestamptz,                 -- lease = deadline + slack
  created_at         timestamptz not null default now(),
  started_at         timestamptz,
  finished_at        timestamptz,
  baseline_score     double precision,
  best_visible_score double precision,
  holdout_score      double precision,
  primary key (org_id, id),
  foreign key (org_id, created_by) references users (org_id, id),
  unique (org_id, idempotency_key)                -- replay-safe submit
);

create index if not exists jobs_queued_idx on jobs (created_at) where status = 'queued';
create index if not exists jobs_lease_idx  on jobs (deadline_at)
  where status in ('running', 'cancelling');
create index if not exists jobs_owner_idx  on jobs (org_id, created_by, created_at desc);

-- Append-only
create table if not exists iterations (
  org_id          uuid not null,
  job_id          uuid not null,
  n               int  not null,
  agent_source    text not null,        -- M4: "state of the agent at each step"
  proposal        text,                 -- null at n = 0
  results         jsonb not null,       -- {task_id: reward | null}
  failures        jsonb not null default '[]',   -- distilled records
  visible_score   double precision,
  holdout_score   double precision,
  outcome         text not null check (outcome in ('baseline', 'improved',
                                                   'regressed', 'infra_failed')),
  error_detail    text,                 -- why, when infra_failed
  accepted        boolean not null,     -- the ratchet reads this
  llm_calls       int,                  -- counters, not costs
  input_tokens    bigint,
  output_tokens   bigint,
  sandboxes_used  int,
  sandbox_seconds int,
  created_at      timestamptz not null default now(),
  primary key (org_id, job_id, n),
  foreign key (org_id, job_id) references jobs (org_id, id) on delete cascade
);
