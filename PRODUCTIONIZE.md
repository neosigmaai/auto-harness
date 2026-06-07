# Prepare /auto-harness API

HTTP API for accepting a benchmark task list, running the benchmark on those
tasks, and returning task-level results via job status polling.

The production goal is:

1. Receive Terminal-Bench task IDs from the client.
2. Run the benchmark for those tasks.
3. Return a job ID immediately.
4. Let clients poll for pass/fail results and summary counts.
5. Queue the auto-harness agent for follow-up optimization work.

This version implements **only the benchmark part**. It does not queue or run the
auto-harness agent yet.

Benchmark configuration still comes from `experiment_config.yaml`.

## Endpoint

```http
POST /auto-harness
Content-Type: application/json
```

Starts a benchmark job and returns immediately.

```http
GET /auto-harness/{job_id}
```

Returns the status of a running or finished benchmark job.

## Input Schema

```json
{
  "tasks": ["string"]
}
```

Example:

```json
{
  "tasks": [
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "build-cython-ext"
  ]
}
```

## Start Job Output Schema

```json
{
  "job_id": "string",
  "status": "running",
  "tasks": ["string"],
  "started_at": "string"
}
```

Example:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "tasks": ["adaptive-rejection-sampler", "bn-fit-modify", "build-cython-ext"],
  "started_at": "2026-06-07T16:30:00+00:00"
}
```

## Status Output Schema

While the job is running:

```json
{
  "job_id": "string",
  "status": "running",
  "tasks": ["string"],
  "started_at": "string"
}
```

When the job completes:

```json
{
  "job_id": "string",
  "status": "completed",
  "tasks": ["string"],
  "started_at": "string",
  "finished_at": "string",
  "result": {
    "results": [
      {
        "task_id": "string",
        "status": "passed | failed"
      }
    ],
    "summary": {
      "val_score": "number",
      "passed": "number",
      "failed": "number",
      "total": "number"
    }
  }
}
```

When the job fails:

```json
{
  "job_id": "string",
  "status": "failed",
  "tasks": ["string"],
  "started_at": "string",
  "finished_at": "string",
  "error": "string"
}
```

## Behavior

- Server accepts a list of Terminal-Bench task IDs.
- Server creates an in-memory job record and returns its `job_id` immediately.
- Server clears `workspace/results.tsv` and `tbench_data/task_split.json` in the
  background job.
- Server calls the existing `prepare.py` benchmark path for those task IDs in a
  background thread.
- `prepare.py` runs the supplied tasks, generates the train/test split, records
  the baseline row, and returns task-level benchmark results plus summary counts.
- `summary.total` is the number of submitted task results counted by the API.
- `summary.val_score` is still the validation score from the generated test
  split.
- Future version: after the benchmark result is available, queue the auto-harness
  agent. This is intentionally out of scope for the current implementation.

## Error Cases

- Empty `tasks` list returns `422`.
- Concurrent `POST /auto-harness` while one request is already running returns `409`.
- Unknown `GET /auto-harness/{job_id}` returns `404`.
- `prepare.py` failures are stored on the job as `status: "failed"` with an
  `error` message.

## Example Curl

```bash
start_response="$(curl -s -X POST localhost:8800/auto-harness \
  -H 'content-type: application/json' \
  -d '{"tasks":["adaptive-rejection-sampler","bn-fit-modify","build-cython-ext"]}')"

echo "$start_response" | jq

job_id="$(echo "$start_response" | jq -r '.job_id')"
curl -s "localhost:8800/auto-harness/$job_id" | jq
```
