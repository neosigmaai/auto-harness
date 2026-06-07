# Prepare /auto-harness API

HTTP API for accepting a benchmark task list, running the benchmark on those
tasks, and returning task-level results via job status polling.

The production goal is:

1. Receive Terminal-Bench task IDs from the client.
2. Send a coding-agent instruction to the worker sandbox to run the benchmark
   for those tasks.
3. Return a job ID immediately.
4. Let clients poll for pass/fail results and summary counts.
5. Queue the auto-harness agent for follow-up optimization work.

This version implements the benchmark bootstrap through the worker coding-agent
API. It does not queue the follow-up optimization loop yet.

Benchmark configuration still comes from `experiment_config.yaml`.

## Architecture

The runtime is split into two containers:

- `orchestrator` exposes the public `/auto-harness` API on port `8800`. It does
  not mount or import the benchmark code. It builds a deterministic instruction
  that tells the worker coding agent to run `prepare.py` for the requested task
  IDs and write JSON to `workspace/coding_agent_result.json`.
- `worker` exposes internal `POST /coding_agent`, `GET /coding_agent/{job_id}`,
  and `GET /health` routes on port `8810`. For each coding-agent job, it creates
  an E2B sandbox, uploads the repo snapshot, runs the Cursor SDK agent inside
  that sandbox, and copies `workspace/coding_agent_result.json` back into the
  worker job result.

`/auto-harness` is the only public entry point. The worker's `/coding_agent`
API is internal to the Compose network.

## Setup instructions

**Requirements:** Docker, `CURSOR_API_KEY`, `E2B_API_KEY`, and an LLM key for
`prepare.py` / Terminal-Bench (typically `OPENAI_API_KEY`).

From the repo root:

```bash
cd auto-harness
cp .env.example .env
```

Edit `.env` and set at least:

```bash
CURSOR_API_KEY=...
E2B_API_KEY=...
OPENAI_API_KEY=...
CURSOR_AGENT_MODEL=composer-2.5

ORCHESTRATOR_PORT=8800
WORKER_PORT=8810
WORKER_BASE_URL=http://worker:8810
```

Ensure `experiment_config.yaml` is configured for Terminal-Bench (see
`experiment_config.yaml.template`).

Build and start both containers:
```bash
docker compose up --build
```

In another terminal, verify the public API and worker reachability:

```bash
curl -s localhost:8800/health | jq
```

Start a benchmark job:

```bash
start_response="$(curl -s -X POST localhost:8800/auto-harness \
  -H 'content-type: application/json' \
  -d '{"tasks":["pypi-server", "kv-store-grpc"]}')"

echo "$start_response" | jq
job_id="$(echo "$start_response" | jq -r '.job_id')"
```

Poll until `status` is `completed` or `failed`:

```bash
curl -s "localhost:8800/auto-harness/$job_id" | jq
```

Notes:

- Only `POST /auto-harness` and `GET /auto-harness/{job_id}` are public. The
  worker `/coding_agent` API is internal to the Compose network.
- Each job runs a fresh E2B sandbox. The Cursor agent executes inside E2B; the
  worker copies `workspace/coding_agent_result.json` back after the run.
- Worker job state is in-memory. Restarting containers clears active/historical
  job IDs — start a new job after rebuild.

### Which Terminal-Bench tasks you selected and why?

Selected 10 tasks from the following buckets: systems, coding and debugging.
The aim is to have similar tasks for each buckets in order to have higher transferability of Agents' improvement.

System:
1. kv-store-grpc
2. pypi-server
3. torch-pipeline-parallelism
4. torch-tensor-parallelism
Coding:
5. polyglot-c-py
6. polyglot-rust-c
7. cobol-modernization
8. build-cython-ext
Debugging:
9. build-cython-ext
10. merge-diff-arc-agi-task


### Key design decisions and why you made them?

1. Two containers: Orchestrator: Public API and Worker: Triggers E2B sandbox
Orchestrator acts a reverse-proxy, controlling the input to Worker's agent. This provides flexibility to build a generalized worker layer ,i.e., worker's function can guided with a prompt

2. Reused `prepare.py` instead of creating a new handler for benchmarking. Updated in `prepare.py`: added `--tasks` CLI for subset benchmarking.

### What would you do differently with more time?

1. Currently the state of API is pushed to `coding_agent_result.json`, the final state of the API could be moved to a database instead for longer persistence. Currently, a new call wipese the older state.

2. Would go a multi-threaded setup at the Worker container, especially because E2B setup is in place and we could make parallel calls to E2B server. 


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
- Orchestrator builds a coding-agent instruction and forwards it to the worker's
  `/coding_agent` endpoint.
- Worker creates an in-memory job record and returns its `job_id` immediately.
- The worker launches an E2B sandbox and runs the Cursor agent there. The agent
  runs the existing `prepare.py` benchmark path for those task IDs.
- `prepare.py` runs the supplied tasks, generates the train/test split, records
  the baseline row, and the agent writes task-level benchmark results plus
  summary counts to `workspace/coding_agent_result.json`.
- `summary.total` is the number of submitted task results counted by the API.
- `summary.val_score` is still the validation score from the generated test
  split.
- Future version: after the benchmark result is available, queue a second
  coding-agent instruction to run the optimization loop.

## Internal Worker Coding-Agent API

The worker endpoint is generic and instruction-driven, but it is not exposed to
clients directly:

```http
POST /coding_agent
Content-Type: application/json
```

```json
{
  "instruction": "Write {\"ok\": true} as JSON to workspace/coding_agent_result.json",
  "model": "composer-2.5"
}
```

The worker returns a running job immediately and stores the final structured
response by parsing `workspace/coding_agent_result.json`.

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
