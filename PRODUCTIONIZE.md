# Prepare /auto-harness API

HTTP API for accepting a benchmark task list, running the benchmark on those
tasks, and returning task-level results.

The production goal is:

1. Receive Terminal-Bench task IDs from the client.
2. Run the benchmark for those tasks.
3. Return pass/fail results and summary counts.
4. Queue the auto-harness agent for follow-up optimization work.

This version implements **only the benchmark part**. It does not queue or run the
auto-harness agent yet.

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
- Future version: after the benchmark result is available, queue the auto-harness
  agent. This is intentionally out of scope for the current implementation.

## Error Cases

- Empty `tasks` list returns `422`.
- Concurrent `POST /auto-harness` while one request is already running returns `409`.
- `prepare.py` failures return `500`.

## Example Curl

```bash
curl -s -X POST localhost:8800/auto-harness \
  -H 'content-type: application/json' \
  -d '{"tasks":["adaptive-rejection-sampler","bn-fit-modify","build-cython-ext"]}' | jq
```
