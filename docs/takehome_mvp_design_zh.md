# Take-home MVP Design 中文方案

> Branch context:
>
> - Heavy reference branch: `archive` at `cf78c72` (`archive: preserve heavy runtime design`)
> - Clean implementation branch: `takehome/mvp` from `origin/main` at `de6b3ed`
> - Goal: deliver a small, runnable Agent Optimization Service for the take-home assignment without carrying over the full production runtime design.

## 0. 当前判断

这次 take-home 的核心不是“把 agent 分数优化到最高”，也不是“实现完整 AutoHarness 生产系统”。它考的是一个 backend service 能否把 agent evaluation 这件事服务化：

```text
HTTP request
  -> create optimization job
  -> async benchmark execution
  -> sandboxed agent run
  -> persist status / results / traces
  -> expose polling and structured results
  -> optionally run one improvement iteration
```

旧方案里的 `EvaluationEngine` 思想是最有价值的，但原来的 `GateEngine`、`SuiteStore`、CandidateGraph、Beam Search、Kubernetes/VPC 设计都不应该进入一天 MVP 的核心实现。它们适合放在 README 的 "what I would do with more time" 和面试扩展讨论里。

## 1. Assignment 要求拆解

### Milestone 1: API Design

要求：

```text
accept a request to run the agent against benchmark tasks
return structured result: passed, failed, observed failure summary
execution can be simulated
focus on clear inputs, output schema, sensible error handling
```

MVP 必须交付：

```text
POST /runs
GET /runs/{run_id}
GET /runs/{run_id}/results
GET /tasks
```

如果做 Milestone 4，再加：

```text
GET /runs/{run_id}/iterations
```

### Milestone 2: Asynchronous Processing

要求：

```text
submit returns immediately
processing happens in background
caller polls result
job lifecycle is tracked
API and processing layer communicate cleanly
```

MVP 必须交付：

```text
POST /runs returns 202 + run_id
background worker starts benchmark
GET /runs/{run_id} returns status and progress
Postgres persists run/task status
```

第一版可以用 FastAPI `BackgroundTasks` 或一个 in-process worker。README 中说明生产版会换成 Redis Streams/Kafka + worker pool。

### Milestone 3: Sandbox Execution

要求：

```text
agent should run inside isolated sandbox
not in API process or worker process itself
capture real task output
use captured output as context for LLM improvements
handle sandbox lifecycle and unexpected failure
```

MVP 最好真实调用现有 TerminalBench/Harbor/Daytona 路径，而不是只 mock：

```text
BenchmarkWorker
  -> BenchmarkRunnerAdapter
  -> TerminalBenchRunner
  -> harbor run --env daytona
  -> Harbor/Daytona sandbox
  -> workspace/tbench_jobs/.../result.json + trace.json
```

如果真实 Daytona 太慢，保留 `simulate=true` 只用于 Milestone 1/2 demo，但 README 要明确：Milestone 3 real mode 使用 Harbor/Daytona。

### Milestone 4: Iterative Optimization Loop

要求：

```text
observe failures
LLM proposes improvement
apply improvement
rerun benchmark
continue until no improvement or max iterations
persist full iteration history
```

一天 MVP 建议做最小闭环：

```text
iteration 0: baseline benchmark
if max_iterations > 0:
  summarize failed tasks
  ask LLM for an improvement proposal
  persist proposal
  optionally apply patch only if safe
  rerun benchmark
  compare score
```

如果时间不足，先只实现：

```text
failure summary + LLM proposal + persisted iteration history
```

不强行做自动 patch。这样仍然可以解释完整生产方案，但降低代码风险。

### Milestone 5: Multi-Tenancy

要求：

```text
multiple organizations
roles
API-level access control
```

一天 MVP 不建议做完整 auth。可以做轻量设计或 mock header：

```text
X-Org-Id
X-User-Id
X-Role
```

表里保留：

```text
org_id
created_by
```

查询时加 org/user filter。README 里说明生产版会接 JWT/OAuth、RBAC、audit logs。

## 2. 从 archive 可复用的关键点

### 2.1 保留：EvaluationEngine 思想

旧方案中最重要的抽象：

```text
EvaluationEngine = how a batch actually runs
EvaluationOrchestrator = when to submit the next batch
```

MVP 不直接搬完整 `EvaluationEngine/BatchStore/RedisQueue/WorkerPool`，但保留边界：

```text
API 不直接运行 benchmark
RunService 创建 run
Worker/Executor 执行 run
ResultNormalizer 归一化输出
Store 持久化状态
```

简化后的 MVP：

```text
RunService
  -> create BenchmarkRun row
  -> schedule background task

BenchmarkExecutor
  -> run selected tasks
  -> capture raw outputs
  -> normalize results
  -> update rows
```

### 2.2 保留：ResultNormalizer 思想

旧 `ResultNormalizer` 的价值在于把 Harbor/TerminalBench 的 raw result 转成稳定状态：

```text
reward >= 0.5          -> passed
reward < 0.5           -> failed
missing result.json    -> infra_failed
runner timeout         -> timed_out
EnvironmentStartTimeoutError -> infra timeout
invalid reward         -> infra_failed
missing verifier       -> infra_failed or verifier_failed
```

MVP 必须有这个分类，否则无法回答“这是 agent 错了还是基础设施失败了”。

### 2.3 保留：Harbor/Daytona sandbox 边界

旧方案中的关系应该继续沿用：

```text
Our code -> Harbor CLI -> --env daytona -> Daytona sandbox
```

关键判断：

```text
Daytona 是 sandbox provider，不是顶层 scheduler
HarborRunner/BenchmarkRunnerAdapter 是我们的 adapter
API 层不直接知道 Daytona 细节
```

### 2.4 保留：Postgres 状态机思想

旧方案强调：

```text
所有跨进程状态进入 Postgres
terminal status sticky
worker 可重复运行
```

MVP 可以不用完整 `eval_batches` 表，但 `runs` 和 `task_results` 必须有明确状态。

### 2.5 暂缓：GateEngine / SuiteStore

旧 `GateEngine` 四步：

```text
suite
regression_suite
test_split
suite_promotion
```

这不是 take-home Milestone 1-3 的重点。MVP 中只保留最简单的 score comparison：

```text
baseline_score
new_score
improved = new_score > baseline_score
```

GateEngine 放入 future work：

```text
production regression protection
candidate promotion
suite management
```

### 2.6 暂缓：CandidateGraph / Beam Search / Merge

这些属于“并行优化多个候选版本”的生产级系统，不进入 take-home MVP。

README 可写：

```text
With more time, I would add CandidateGraphManager to track parent/child agent versions,
parallel candidate evaluation, gated promotion, and merge evaluation.
```

## 3. 新 MVP 目标架构

```text
test_client.py
  -> FastAPI Service
      -> RunService
          -> PostgresStore
          -> BackgroundTask / LocalWorker
              -> BenchmarkExecutor
                  -> TerminalBenchRunner / SimulatedRunner
                  -> Harbor/Daytona sandbox in real mode
                  -> ResultNormalizer
                  -> Artifact capture
          -> optional OptimizerService
              -> FailureSummary
              -> LLM proposal
              -> Iteration history
```

### Component 划分

```text
app.py
  FastAPI app, route registration, dependency wiring.

schemas.py
  Pydantic request/response models.

store.py
  Postgres persistence for runs, task results, iterations.

service.py
  RunService: create run, read status/result, start background work.

executor.py
  BenchmarkExecutor: execute selected tasks through real or simulated runner.

runner.py
  Runner abstraction plus TerminalBenchRunnerAdapter and SimulatedRunner.

normalizer.py
  Convert raw reward/artifacts/errors into stable TaskResult.

optimizer.py
  Minimal failure summary and optional LLM proposal.

test_client.py
  End-to-end script: submit, poll, print summary.
```

如果时间紧，可以把 `service.py/executor.py/runner.py/normalizer.py` 合并为更少文件，但边界要在 README 中讲清楚。

## 4. API Design

### POST /runs

Request:

```json
{
  "task_ids": ["break-filter-js-from-html"],
  "max_iterations": 0,
  "sandbox_provider": "daytona",
  "model": "gpt-5.4",
  "mode": "real"
}
```

Fields:

```text
task_ids: required non-empty list for MVP
max_iterations: default 0; 0 means benchmark only
sandbox_provider: daytona | e2b | docker | simulated
model: LLM model used by agent and optional optimizer
mode: real | simulated
```

Response:

```json
{
  "run_id": "uuid",
  "status": "queued",
  "created_at": "2026-07-02T00:00:00Z",
  "status_url": "/runs/uuid",
  "result_url": "/runs/uuid/results"
}
```

### GET /runs/{run_id}

Response:

```json
{
  "run_id": "uuid",
  "status": "running",
  "progress": {
    "total": 10,
    "queued": 0,
    "running": 2,
    "completed": 8
  },
  "score": null,
  "error": null,
  "created_at": "...",
  "started_at": "...",
  "completed_at": null
}
```

### GET /runs/{run_id}/results

Response:

```json
{
  "run_id": "uuid",
  "status": "succeeded",
  "score": 0.7,
  "tasks_total": 10,
  "tasks_passed": 7,
  "tasks_failed": 2,
  "tasks_infra_failed": 1,
  "task_results": [
    {
      "task_id": "break-filter-js-from-html",
      "status": "failed",
      "reward": 0.0,
      "failure_type": "agent_failed",
      "error_summary": "Verifier reward below threshold",
      "trace_path": "workspace/traces/latest/break-filter-js-from-html/trace.json",
      "result_path": "workspace/traces/latest/break-filter-js-from-html/result.json"
    }
  ],
  "failure_summary": {
    "agent_failures": 2,
    "infra_failures": 1,
    "top_failure_modes": [
      "reward_below_threshold",
      "environment_start_timeout"
    ]
  }
}
```

### GET /runs/{run_id}/iterations

Only needed if Milestone 4 is implemented.

```json
{
  "run_id": "uuid",
  "iterations": [
    {
      "iteration": 0,
      "agent_version": "initial",
      "score": 0.4,
      "proposal": null,
      "status": "completed"
    },
    {
      "iteration": 1,
      "agent_version": "proposal-1",
      "score": 0.5,
      "proposal": "Tighten bash command planning and verify outputs before final answer.",
      "status": "completed",
      "accepted": true
    }
  ]
}
```

## 5. 状态机设计

### Run Status

```text
queued
running
succeeded
failed
timed_out
cancelled
```

规则：

```text
terminal states are sticky
queued -> running -> succeeded
queued -> running -> failed
queued -> running -> timed_out
```

### Task Status

```text
queued
running
passed
failed
infra_failed
timed_out
```

### Failure Type

```text
agent_failed
verifier_failed
infra_failed
sandbox_timeout
runner_timeout
missing_result
invalid_result
```

## 6. Postgres Data Model

MVP 表可以很少：

```text
runs
  id uuid primary key
  status text
  mode text
  model text
  sandbox_provider text
  max_iterations int
  task_ids jsonb
  score double precision null
  error text null
  created_at timestamptz
  started_at timestamptz null
  completed_at timestamptz null

task_results
  id uuid primary key
  run_id uuid references runs(id)
  task_id text
  status text
  reward double precision null
  failure_type text null
  error_summary text null
  trace_path text null
  result_path text null
  raw_metadata jsonb
  started_at timestamptz null
  completed_at timestamptz null

iterations
  id uuid primary key
  run_id uuid references runs(id)
  iteration_index int
  agent_version text
  proposal text null
  score double precision null
  accepted boolean null
  status text
  created_at timestamptz
```

如果要轻量 mock multi-tenancy：

```text
runs.org_id text
runs.created_by text
```

## 7. Execution Path

### Real Mode

```text
POST /runs
  -> RunService.create_run()
  -> insert runs(status=queued)
  -> background task starts
  -> update runs(status=running)
  -> for selected task_ids:
       TerminalBenchRunnerAdapter.run(task_ids)
       Harbor creates Daytona sandbox
       agent runs in sandbox
       Harbor writes jobs_dir
  -> parse result.json / trace.json
  -> normalize task statuses
  -> update task_results
  -> compute score
  -> update runs(status=succeeded)
```

### Simulated Mode

Used for quick API demo and tests:

```text
POST /runs mode=simulated
  -> create deterministic fake task results
  -> still writes the same DB rows
  -> still exposes the same API responses
```

This keeps Milestone 1-2 testable even when Daytona is slow or unavailable.

## 8. Failure Handling

Important cases:

```text
Invalid request:
  400 with clear error.

Unknown run_id:
  404.

Run not terminal but results requested:
  409 or return partial=false with current status.

Harbor command fails before output:
  run/task infra_failed.

Daytona environment start timeout:
  task timed_out or infra_failed with failure_type=sandbox_timeout.

Verifier reward 0.0:
  task failed with failure_type=agent_failed.

No result.json:
  task infra_failed with failure_type=missing_result.

LLM proposal failure:
  iteration failed; benchmark result remains available.
```

## 9. Test Client Behavior

`test_client.py` should:

```text
1. POST /runs with selected task_ids
2. print run_id
3. poll GET /runs/{run_id}
4. stop when status terminal
5. GET /runs/{run_id}/results
6. print structured summary:
   - score
   - pass/fail/infra counts
   - failed task ids
   - failure summary
   - iteration history if available
```

This is important because assignment says they will run `test_client.py` against the service.

## 10. Task Selection Strategy

README must state which 10-20 TerminalBench tasks were selected and why.

Selection criteria:

```text
fast enough to finish in a reasonable time
representative of bash/file/code-debug/verifier-style failures
mix of pass and fail tasks for useful failure summary
avoid tasks with known repeated sandbox startup failures
```

Implementation path:

```text
start with 1-3 known tasks for dev loop
then expand to 10-20 after timing
record task ids and rationale in README
```

## 11. What To Cut From MVP

Do not implement in one-day version:

```text
GateEngine four-stage regression suite
Suite promotion
CandidateGraphManager
Beam Search
MergeEngine
Kubernetes worker pods
Redis/Kafka queue
Full RBAC auth
Direct Daytona API integration
Automatic git patch application with rollback
```

These are good interview discussion points, but they make the submitted service harder to review.

## 12. Recommended Afternoon Implementation Order

### Phase 1: API + Models

```text
Add FastAPI dependencies.
Create request/response schemas.
Create POST /runs and GET /runs/{id}.
Use simulated executor first.
```

Success:

```text
test_client.py can submit and poll a simulated run.
```

### Phase 2: Postgres Persistence

```text
Add simple schema init.
Implement Store with runs/task_results/iterations.
Persist status transitions.
```

Success:

```text
service restart does not lose completed run history.
```

### Phase 3: Real Benchmark Adapter

```text
Wrap existing TerminalBenchRunner.
Run selected task_ids.
Capture trace/result paths.
Normalize rewards and missing outputs.
```

Success:

```text
one real task can run through Harbor/Daytona and appear in /results.
```

### Phase 4: Minimal Optimization History

```text
Build failure_summary.
Optionally call LLM for proposal.
Persist iteration 0 and proposal iteration.
```

Success:

```text
GET /runs/{id}/iterations returns useful history.
```

### Phase 5: README Polish

```text
Setup and run instructions.
Selected tasks and why.
API examples.
Design decisions.
Known limitations.
What I would do with more time.
```

## 13. README Key Design Decisions To State

Suggested language:

```text
I intentionally focused the implementation on Milestones 1-3 because those are
the concrete 6-8 hour requirements. Milestone 4 is represented by a minimal
iteration/proposal history, and Milestone 5 is designed through org/user fields
and documented RBAC rather than a full auth system.
```

```text
The service treats benchmark execution as an asynchronous job. The API layer
never runs the agent directly; it creates a run, persists state, and starts a
background worker. Real TerminalBench execution goes through Harbor with the
configured sandbox provider, so the agent execution boundary remains outside
the API process.
```

```text
The older production design had EvaluationEngine, GateEngine, suite promotion,
and distributed workers. For this take-home I kept only the execution boundary
and result-normalization ideas, because the reviewer needs a small service that
is easy to run and inspect.
```

## 14. Future Production Architecture

If asked in interview:

```text
Replace BackgroundTasks with Redis Streams or Kafka.
Make workers stateless and horizontally scalable.
Persist attempt leases and heartbeats.
Store trace/result artifacts in S3 or MinIO.
Add GateEngine for regression suite and score-gated promotion.
Add CandidateGraphManager for parallel candidate optimization.
Add org/user RBAC with JWT/OAuth and audit logs.
Deploy API/worker/store in Kubernetes private subnets.
Use NAT only for outbound LLM/sandbox API calls.
```

This maps directly to the archived design but avoids shipping it as the MVP.

## 15. Decision Summary

```text
Use archive as reference, not base branch.
Start MVP from origin/main.
Focus on service API, async state, sandbox boundary, result persistence.
Do not carry over GateEngine/CandidateGraph into implementation.
Keep real-mode Harbor/Daytona path, with simulated mode for reliable API tests.
Persist enough iteration history to satisfy Milestone 4 if time allows.
Document multi-tenancy as API-level design unless time remains for mock headers.
```
