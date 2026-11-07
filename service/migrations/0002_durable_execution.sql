-- Keep existing rows; introduce durable execution identities and accounting.
ALTER TABLE organizations ADD COLUMN last_dispatched_at timestamptz;
ALTER TABLE jobs ADD COLUMN request_fingerprint text;
ALTER TABLE jobs ADD COLUMN deadline_at timestamptz;
ALTER TABLE jobs ADD COLUMN final_evaluation jsonb;
ALTER TABLE jobs DROP CONSTRAINT jobs_org_id_idempotency_key_key;
ALTER TABLE jobs ADD CONSTRAINT jobs_principal_idempotency_key UNIQUE (org_id, created_by, idempotency_key);
ALTER TABLE iterations ADD COLUMN metrics jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE task_runs ADD COLUMN input_tokens bigint NOT NULL DEFAULT 0;
ALTER TABLE task_runs ADD COLUMN output_tokens bigint NOT NULL DEFAULT 0;
ALTER TABLE task_runs ADD COLUMN cost_usd double precision NOT NULL DEFAULT 0;
ALTER TABLE task_runs ADD COLUMN model_cost_usd double precision NOT NULL DEFAULT 0;
ALTER TABLE task_runs ADD COLUMN sandbox_cost_usd double precision NOT NULL DEFAULT 0;
ALTER TABLE task_runs ADD COLUMN cost_known boolean NOT NULL DEFAULT false;

-- Only completed attempts are inserted here. They are append-only: a retry or
-- repetition gets another identity; it never replaces an earlier observation.
CREATE TABLE task_attempts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    iteration_id uuid NOT NULL REFERENCES iterations(id) ON DELETE CASCADE,
    evaluation_phase text NOT NULL,
    task_name text NOT NULL,
    repetition int NOT NULL CHECK (repetition > 0),
    attempt int NOT NULL CHECK (attempt > 0),
    lease_epoch int NOT NULL,
    status text NOT NULL CHECK (status IN ('passed','failed','timeout','error')),
    reward double precision,
    failure_class text,
    duration_s double precision,
    n_steps int,
    exit_reason text,
    test_stdout text,
    trace jsonb,
    digest jsonb,
    sandbox_id text,
    input_tokens bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    cost_usd double precision NOT NULL DEFAULT 0,
    model_cost_usd double precision NOT NULL DEFAULT 0,
    sandbox_cost_usd double precision NOT NULL DEFAULT 0,
    cost_known boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    result_hash text NOT NULL,
    finished_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (iteration_id, evaluation_phase, task_name, repetition, attempt)
);
CREATE INDEX task_attempts_iteration ON task_attempts(iteration_id, evaluation_phase, repetition);

-- Preserve pre-migration evidence as main phase, first repetition/attempt.
INSERT INTO task_attempts(iteration_id,evaluation_phase,task_name,repetition,attempt,lease_epoch,
    status,reward,failure_class,duration_s,n_steps,exit_reason,test_stdout,trace,digest,sandbox_id,result_hash,finished_at)
SELECT t.iteration_id,'main',t.task_name,1,1,j.lease_epoch,t.status,t.reward,t.failure_class,
    t.duration_s,t.n_steps,t.exit_reason,t.test_stdout,t.trace,t.digest,t.sandbox_id,
    encode(digest(t.id::text,'sha256'),'hex'),coalesce(t.finished_at,now())
FROM task_runs t JOIN iterations i ON i.id=t.iteration_id JOIN jobs j ON j.id=i.job_id
WHERE t.status NOT IN ('pending','running');

CREATE TABLE spend_reservations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    reservation_key text NOT NULL,
    lease_epoch int NOT NULL,
    kind text NOT NULL,
    reserved_usd numeric NOT NULL CHECK (reserved_usd >= 0),
    actual_usd numeric CHECK (actual_usd >= 0),
    input_tokens bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    settled_at timestamptz,
    UNIQUE(job_id, reservation_key)
);
CREATE INDEX spend_reservations_job ON spend_reservations(job_id);

CREATE TABLE sandbox_slots (
    slot_key text PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    lease_owner text NOT NULL,
    lease_epoch int NOT NULL,
    expires_at timestamptz NOT NULL,
    provider_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX sandbox_slots_expiry ON sandbox_slots(expires_at);
