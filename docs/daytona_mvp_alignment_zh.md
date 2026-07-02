# Daytona MVP Alignment 中文核对

> 目的：基于 Daytona 官方文档，核对当前 `docs/takehome_mvp_design_zh.md` 的 sandbox / Harbor / async / result collection 设计是否合理，并明确 MVP 该怎么取舍。
>
> 主要参考：
>
> - Daytona docs: https://www.daytona.io/docs/
> - Python SDK: https://www.daytona.io/docs/python-sdk/
> - Process and Code Execution: https://www.daytona.io/docs/en/process-code-execution/
> - File System Operations: https://www.daytona.io/docs/en/file-system-operations/
> - Python FileSystem SDK: https://www.daytona.io/docs/en/python-sdk/sync/file-system/
> - Webhooks: https://www.daytona.io/docs/en/webhooks/
> - Volumes: https://www.daytona.io/docs/en/volumes/
> - Mount External Storage: https://www.daytona.io/docs/en/mount-external-storage/
> - Network Limits: https://www.daytona.io/docs/en/network-limits/
> - API Keys: https://www.daytona.io/docs/en/api-keys/

## 1. 官方文档确认的事实

### 1.1 Daytona 是 sandbox provider，不是我们的调度系统

Daytona 的核心是 secure / elastic sandbox infrastructure。官方文档说 sandbox 有独立 kernel、filesystem、network stack、vCPU、RAM、disk，并通过 SDK/API/CLI 进行 lifecycle、filesystem、process/code execution 等操作。

对我们的结论：

```text
Daytona 负责 sandbox execution lifecycle。
AutoHarness service 负责 API、job lifecycle、batch/task state、result persistence、optimizer。
```

这和当前 `takehome_mvp_design_zh.md` 的判断一致。

### 1.2 基础配置需要 DAYTONA_API_KEY

官方 Python SDK 支持通过环境变量配置：

```text
DAYTONA_API_KEY
DAYTONA_API_URL
DAYTONA_TARGET
```

当前 repo 的 `prepare.py` 里，当 `env_provider == "daytona"` 时会检查 `DAYTONA_API_KEY`，并且 Terminal-Bench runner 通过 Harbor CLI 使用 `--env daytona`。

对 MVP 的结论：

```text
必需：
  DAYTONA_API_KEY
  OPENAI_API_KEY 或对应 LLM key
  harbor CLI

暂不必需：
  direct Daytona Python SDK dependency
```

因为当前 MVP 优先复用：

```text
FastAPI service -> existing TerminalBenchRunner -> harbor run --env daytona
```

而不是直接写 Daytona SDK runner。

### 1.3 Daytona 支持直接执行命令、session、日志、文件上传下载

官方能力包括：

```text
sandbox.process.exec(...)
sandbox.process.create_session(...)
sandbox.process.execute_session_command(...)
sandbox.process.get_session(...)
sandbox.process.list_sessions(...)
sandbox.process.get_session_command_logs(...)
sandbox.fs.upload_file(...)
sandbox.fs.download_file(...)
```

这说明如果未来绕过 Harbor 直接接 Daytona SDK，我们可以：

```text
create sandbox
upload repo/task files
execute command
poll session/command status
stream or fetch logs
download result artifacts
delete sandbox
```

但在当前 MVP 中，Harbor 已经包装了这些细节。因此我们不直接实现 Daytona SDK runner。

### 1.4 Webhook 是生命周期辅助，不适合作为本地 MVP 的唯一完成信号

官方 Webhooks 文档主要列的是 lifecycle 事件：

```text
sandbox.created
sandbox.state.updated
snapshot.created
snapshot.state.updated
volume.created
volume.state.updated
```

这不是一个天然的 per-command completion API。命令级完成更适合通过：

```text
session_id
command_id / cmd_id
exit_code
logs
```

来判断。

对 MVP 的结论：

```text
本地 MVP 不做 webhook。
使用 polling / subprocess completion / Harbor output directory 作为完成信号。
生产版可以加 webhook 作为 sandbox lifecycle reconcile signal。
```

### 1.5 文件回收比挂载本地 PostgreSQL 更合理

Daytona 支持：

```text
fs upload/download
Daytona Volumes
external object storage mount: S3, R2, GCS, Azure Blob
```

但文档没有把“把本地 PostgreSQL 直接挂进 Daytona sandbox”作为常规模式。对本地 MVP 而言，sandbox 在远端/隔离环境里，直接访问你本机 Postgres 还会遇到：

```text
localhost 不同义：sandbox 里的 localhost 不是你的 Mac
网络入站不可达：本机 Postgres 默认不暴露公网
认证和安全风险：不应把 DB 写权限放进 task sandbox
稳定性差：本地网络、NAT、防火墙会影响运行
```

对 MVP 的结论：

```text
sandbox/Harbor 只产出文件：result.json、trace.json、stdout/stderr/logs。
本地 service/worker 把这些文件取回或读取 Harbor jobs_dir。
本地 service 再写入本地 Postgres。
```

不要让 sandbox 直接写本地 Postgres。

## 2. 与当前 MVP 设计的核对结果

### 一致的部分

当前设计里这些是正确的：

```text
1. 继续把 Daytona 视为 sandbox provider。
2. API 层不直接调用 Daytona。
3. 当前真实执行路径先走 Harbor:
   TerminalBenchRunner -> harbor run --env daytona -> Daytona sandbox.
4. 本地 MVP 用 polling/subprocess completion，而不是 webhook。
5. logs/traces/results 回收到本地后，再写 Postgres。
6. simulated mode 保留，用于快速验证 API/async/result schema。
```

### 需要补充/修正的部分

当前设计文档应该补充以下细节：

```text
1. "fetch results" 优先级：
   MVP: read Harbor jobs_dir on local filesystem.
   Future direct Daytona SDK: use sandbox.fs.download_file / logs API.

2. "async" 的准确边界：
   HTTP API async 是我们自己的 job async。
   Daytona session async 是 sandbox 内命令执行 async。
   MVP 不一定直接使用 Daytona async session；Harbor subprocess already blocks until benchmark completes.

3. "Batch Manager"：
   MVP 可以把 batch manager 简化为 RunService + task_results aggregation。
   不需要单独 eval_batches 表。

4. "concurrency"：
   MVP 不直接按 Daytona SDK threads 发 task。
   先把 task_ids 交给 Harbor `-n <concurrency>`。
   我们的 API 只限制 requested_concurrency，避免超过 Daytona/Harbor quota。

5. "webhook"：
   本地不使用；生产版才作为 lifecycle reconcile。
```

## 3. 推荐 MVP 执行/回收路径

### 3.1 Real mode: Harbor-first path

这是当前最现实、最稳的路径：

```text
POST /runs
  -> insert runs(status=queued)
  -> background task starts
  -> update runs(status=running)
  -> call TerminalBenchRunnerAdapter
       -> harbor run -d terminal-bench@2.0
       -> --env daytona
       -> --jobs-dir workspace/tbench_jobs/<run_id or timestamp>
       -> -i task_id ...
       -> -n requested_concurrency
  -> Harbor/Daytona run tasks in sandboxes
  -> Harbor writes local jobs_dir
  -> service reads local jobs_dir result.json / trace.json
  -> normalize per task
  -> insert task_results rows
  -> compute run score
  -> update runs terminal status
```

这里的“结果回收”不是从 Daytona 直接 fetch，而是读 Harbor 已经落在本地的 artifact。

优点：

```text
最少代码
满足 sandbox requirement
复用 repo 已有 TerminalBenchRunner
不需要理解 Daytona SDK 每个细节
```

风险：

```text
Harbor subprocess 是粗粒度的：一个 batch 完成后才能统一 parse。
无法非常细粒度地在每个 task 完成瞬间更新 DB。
```

MVP 可以接受这个风险。API 的 run 状态仍然是 async；task-level progress 可以先粗略显示。

### 3.2 Direct Daytona SDK path: future only

未来如果不用 Harbor，可以变成：

```text
for each task:
  sandbox = daytona.create(...)
  sandbox.fs.upload_files(...)
  session_id = f"task-{attempt_id}"
  sandbox.process.create_session(session_id)
  command = sandbox.process.execute_session_command(..., run_async=True)
  persist sandbox_id / session_id / cmd_id

poller:
  sandbox.process.get_session(session_id)
  inspect command.exit_code
  fetch logs
  sandbox.fs.download_file(result.json)
  sandbox.fs.download_file(trace.json)
  update task_results
  sandbox.delete()
```

这个路径更生产化，但不适合下午 MVP。

## 4. 执行反馈与状态判断

### 4.1 MVP 状态来源

MVP 状态来源按优先级：

```text
1. Background worker process state
2. Harbor subprocess return / timeout
3. Harbor jobs_dir existence
4. Per-trial result.json verifier_result.rewards.reward
5. trace.json / exception metadata
```

### 4.2 状态映射

```text
Harbor run completed + reward >= 0.5
  -> task passed

Harbor run completed + reward < 0.5
  -> task failed / agent_failed

result.json exists but verifier_result missing
  -> infra_failed or verifier_failed

result.json missing for requested task
  -> infra_failed / missing_result

Harbor subprocess timeout
  -> timed_out

EnvironmentStartTimeoutError in exception metadata
  -> sandbox_timeout

No job output directory
  -> infra_failed / runner_failed
```

### 4.3 Batch 完成判断

MVP 里不需要单独 BatchManager 类。`RunService` 足够：

```text
expected_task_count = len(request.task_ids)
finished_task_count = count(task_results where run_id = ...)

if worker returns normally:
  if all requested task_ids have task_results:
    mark run succeeded
  else:
    synthesize missing task_results as infra_failed
    mark run succeeded with infra failures

if worker raises controlled exception:
  mark run failed

if worker timeout:
  mark run timed_out
```

也就是说：

```text
runs = business-level batch
task_results = per-task attempt result
```

不需要 `eval_batches` 表。

## 5. 数据回收方案评估

### 方案 A: sandbox 直接写本地 Postgres

不推荐。

问题：

```text
sandbox 无法天然访问你的本机 localhost
暴露本地 DB 到公网不适合 take-home
sandbox 拿 DB credentials 有安全风险
任务失败时 DB 状态可能半写入
```

### 方案 B: 共享文件夹

在 Harbor 当前实现中已经近似存在：

```text
--jobs-dir workspace/tbench_jobs
```

Harbor 把结果落回本地 jobs_dir。对 MVP 来说，这就是最好的共享 artifact path。

### 方案 C: fetch/download

适合未来 direct Daytona SDK：

```text
sandbox.fs.download_file(...)
sandbox.process.get_session_command_logs(...)
```

如果未来直接 Daytona SDK，这是标准路径。但当前 Harbor-first MVP 不需要。

### 方案 D: Daytona Volume / external storage

适合生产化：

```text
Daytona Volume for shared sandbox-persistent files
External S3/R2/GCS/Azure mount for large artifacts/datasets
```

不适合 MVP，因为配置和调试成本高。

### MVP 决策

```text
Use Harbor jobs_dir as artifact handoff.
Parse local files.
Store normalized records in local Postgres.
Document that direct Daytona download/volumes are future extensions.
```

## 6. Harbor 封装与异步流程

### 6.1 MVP HarborRunner / BenchmarkExecutor

建议封装成：

```text
BenchmarkExecutor.run(run_id, task_ids, config)
  -> TerminalBenchRunnerAdapter.run(task_ids)
  -> returns RawBenchmarkResult
  -> ResultNormalizer.normalize(...)
```

不建议在 MVP 中自己拆成每个 task 一个 Daytona sandbox，因为 Harbor 已经负责。

### 6.2 并发控制

当前 TerminalBenchRunner 支持：

```text
harbor run ... -n <n> -i task1 -i task2 ...
```

MVP API 中 `requested_concurrency` 应该被限制：

```text
min(requested_concurrency, len(task_ids), MAX_LOCAL_CONCURRENCY)
```

建议默认：

```text
MAX_LOCAL_CONCURRENCY=4 或 8
```

原因：

```text
Daytona/Harbor 可能有 quota 和 rate limit
官方 limits 文档建议用 request queue 防止 burst
本地 take-home 追求可跑完，而不是 32 并发压测
```

### 6.3 Polling / listening

MVP：

```text
API polling = client polls our service /runs/{run_id}
Worker does not poll Daytona directly
Worker waits for Harbor subprocess, then parses output
```

生产版 direct Daytona：

```text
store sandbox_id/session_id/cmd_id
poll Daytona get_session(...)
fetch command exit_code and logs
webhook handles sandbox lifecycle fallback
```

## 7. Optimize 步骤设计

### 7.1 MVP 最小优化闭环

推荐不要自动修改 `agent/agent.py`，先实现：

```text
1. baseline run finishes
2. collect failed task summaries
3. build optimizer context:
   - task_id
   - reward
   - failure_type
   - error_summary
   - relevant trace excerpt or trace path
4. call LLM to propose improvement
5. persist proposal in iterations
6. return proposal via API
```

这样可以满足“uses an LLM to propose improvements”的一部分，同时避免自动 patch 带来的文件安全、git rollback、broken code 风险。

### 7.2 如果时间允许，再加一轮 rerun

可选增强：

```text
LLM outputs patch suggestion
system stores patch as proposal
human/manual mode or safe apply mode applies patch
rerun same task_ids
compare new_score vs baseline_score
mark accepted if new_score > baseline_score
```

不建议下午做复杂自动 patch。

### 7.3 LLM context

传给 LLM 的上下文应控制大小：

```text
System:
  You are improving a TerminalBench agent. Propose one focused improvement.

User:
  Selected tasks
  Aggregate score
  Failed tasks
  Failure type counts
  Per failed task:
    task_id
    reward
    error_summary
    trace excerpt or trace_path

Output schema:
  hypothesis
  proposed_change
  expected_effect
  risks
  tasks_to_rerun
```

MVP 只持久化 proposal，不需要把完整 trace 全塞进 DB。DB 保存 trace path 和短摘要即可。

## 8. 需要更新到主设计文档的结论

建议把 `docs/takehome_mvp_design_zh.md` 中的 Daytona 相关部分补充为：

```text
For MVP, real mode uses Harbor as the Daytona adapter. We do not call the Daytona SDK directly.
The service reads Harbor's local jobs_dir output and stores normalized results in Postgres.
Local Postgres is not mounted into Daytona sandboxes.
Webhooks are not used locally; client polls our service, and the worker waits for Harbor output.
Future direct Daytona integration would persist sandbox_id/session_id/cmd_id, poll command exit_code, stream logs, and download artifacts through Daytona fs/process APIs.
```

## 9. 最终建议

下午 MVP 采用：

```text
Harbor-first real execution
Simulated mode for fast API tests
Local Postgres only written by service
Harbor jobs_dir as artifact handoff
RunService as batch manager
No Daytona webhook
No direct Daytona SDK yet
No sandbox-to-DB mount
LLM optimizer stores proposal first; automatic patch is optional future work
```

这个方案和 Daytona 官方文档一致，也最符合 take-home 的时间约束。
