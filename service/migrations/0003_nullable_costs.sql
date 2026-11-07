-- Unknown monetary amounts must stay unknown, rather than being reported as zero.
ALTER TABLE task_runs ALTER COLUMN cost_usd DROP NOT NULL;
ALTER TABLE task_runs ALTER COLUMN cost_usd DROP DEFAULT;
ALTER TABLE task_runs ALTER COLUMN model_cost_usd DROP NOT NULL;
ALTER TABLE task_runs ALTER COLUMN model_cost_usd DROP DEFAULT;
ALTER TABLE task_runs ALTER COLUMN sandbox_cost_usd DROP NOT NULL;
ALTER TABLE task_runs ALTER COLUMN sandbox_cost_usd DROP DEFAULT;
ALTER TABLE task_attempts ALTER COLUMN cost_usd DROP NOT NULL;
ALTER TABLE task_attempts ALTER COLUMN cost_usd DROP DEFAULT;
ALTER TABLE task_attempts ALTER COLUMN model_cost_usd DROP NOT NULL;
ALTER TABLE task_attempts ALTER COLUMN model_cost_usd DROP DEFAULT;
ALTER TABLE task_attempts ALTER COLUMN sandbox_cost_usd DROP NOT NULL;
ALTER TABLE task_attempts ALTER COLUMN sandbox_cost_usd DROP DEFAULT;

-- Avoid rounding away sub-cent usage in databases that ran an early 0002.
ALTER TABLE spend_reservations ALTER COLUMN reserved_usd TYPE numeric;
ALTER TABLE spend_reservations ALTER COLUMN actual_usd TYPE numeric;
