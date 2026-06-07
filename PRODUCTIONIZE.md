# Prepare /auto-harness API

HTTP API for accepting a benchmark task list, running the benchmark on those
tasks, and returning task-level results.

The production goal is:

1. Receive Terminal-Bench task IDs from the client.
2. Run the benchmark for those tasks.
3. Return pass/fail results and summary counts.
4. Queue the auto-harness agent for follow-up optimization work.

This version runs the benchmark synchronously, returns the benchmark response,
and starts the Cursor agent in the background after a successful benchmark.

Benchmark configuration still comes from `experiment_config.yaml`.

## Endpoint

```http
POST /auto-harness
Content-Type: application/json
```

The request blocks until the benchmark finishes.

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

## Output Schema

```json
{
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
  },
  "agent": {
    "status": "idle | queued | running | finished | error | failed_to_start",
    "prompt": "string",
    "run_id": "string | null",
    "error": "string | null",
    "started_at": "string | null",
    "finished_at": "string | null"
  }
}
```

Example:

```json
{
  "results": [
    {
      "task_id": "adaptive-rejection-sampler",
      "status": "passed"
    },
    {
      "task_id": "bn-fit-modify",
      "status": "passed"
    },
    {
      "task_id": "build-cython-ext",
      "status": "failed"
    }
  ],
  "summary": {
    "val_score": 0.5,
    "passed": 2,
    "failed": 1,
    "total": 3
  },
  "agent": {
    "status": "queued",
    "prompt": "Refer to PROGRAM.md and perform the tasks",
    "run_id": null,
    "error": null,
    "started_at": "2026-06-07T13:13:00.000000+00:00",
    "finished_at": null
  }
}
```

## Behavior

- Server accepts a list of Terminal-Bench task IDs.
- Server clears `workspace/results.tsv` and `tbench_data/task_split.json`.
- Server calls the existing `prepare.py` benchmark path for those task IDs.
- `prepare.py` runs the supplied tasks, generates the train/test split, records
  the baseline row, and returns task-level benchmark results plus summary counts.
- `summary.total` is the number of submitted task results counted by the API.
- `summary.val_score` is still the validation score from the generated test
  split.
- After the benchmark result is available, the server queues a local Cursor SDK
  agent with prompt `Refer to PROGRAM.md and perform the tasks`.
- The HTTP response is returned immediately after the agent is queued. Clients
  can poll `GET /health` for the latest `agent` state.
- A second `POST /auto-harness` is rejected while either the benchmark or
  background agent is active because both share the same repo checkout and
  workspace files.

## Error Cases

- Empty `tasks` list returns `422`.
- Concurrent `POST /auto-harness` while one benchmark or optimization agent is
  already running returns `409`.
- `prepare.py` failures return `500`.
- If `CURSOR_API_KEY` is missing, the benchmark still returns `200` and the
  response includes `agent.status: failed_to_start`.

## Example Curl

```bash
curl -s -X POST localhost:8800/auto-harness \
  -H 'content-type: application/json' \
  -d '{"tasks":["adaptive-rejection-sampler","bn-fit-modify","build-cython-ext"]}' | jq
```
