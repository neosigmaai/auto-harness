# Agent Optimization Service MVP System Design

This diagram shows how the MVP satisfies the take-home milestones:

- Milestone 1: HTTP API with structured run, status, result, and iteration resources.
- Milestone 2: asynchronous job lifecycle with polling.
- Milestone 3: real sandbox execution through Harbor and Daytona.
- Milestone 4: single-optimized-version loop with restricted instruction patching.
- Milestone 5-lite: API-level org/user/role scoping through demo headers.

```mermaid
flowchart TD
    U["User / test_client.py"] -->|"HTTP + X-Org-Id / X-User-Id / X-Role"| API["FastAPI Service"]

    subgraph API_LAYER["Milestone 1: API Layer"]
        API --> POST["POST /runs"]
        API --> STATUS["GET /runs/{run_id}"]
        API --> RESULTS["GET /runs/{run_id}/results"]
        API --> ITERS["GET /runs/{run_id}/iterations"]
        API --> TASKLIST["GET /tasks"]
    end

    POST -->|"validate request, org scope, role"| RS["RunService"]
    RS -->|"insert run: queued"| PG[("Postgres")]
    PG --> RUNS["aos_runs"]
    PG --> TASK_RESULTS["aos_task_results"]
    PG --> ITERATIONS["aos_iterations"]

    RS -->|"return 202 + run_id immediately"| U

    subgraph ASYNC["Milestone 2: Async Processing"]
        RS -->|"start daemon background thread"| BG["Background Run Executor"]
        BG -->|"mark run: running"| PG
        BG -->|"mode = simulated"| SIM["SimulatedBenchmarkRunner"]
        BG -->|"mode = real"| LOCK["Real-run Semaphore"]
    end

    LOCK -->|"serialize real Harbor/Daytona runs"| ADAPTER["TerminalBenchRunnerAdapter"]
    ADAPTER -->|"build command args, no shell=True"| HARBOR_CLI["harbor run"]
    HARBOR_CLI -->|"--env daytona"| HARBOR_RUNTIME["Harbor Trial Runtime"]

    subgraph HARBOR["Milestone 3: Harbor-managed Benchmark Lifecycle"]
        HARBOR_RUNTIME --> LOAD_TASK["load Terminal-Bench task"]
        LOAD_TASK --> CREATE_ENV["create Environment"]
        CREATE_ENV --> AGENT_SETUP["agent setup"]
        AGENT_SETUP --> AGENT_RUN["agent run"]
        AGENT_RUN --> VERIFY["verifier run"]
        VERIFY --> WRITE_ARTIFACTS["write result.json / trace.json / logs"]
        WRITE_ARTIFACTS --> HARBOR_CLEANUP["teardown / cleanup"]
    end

    subgraph DAYTONA["Daytona Sandbox Lifecycle Through Harbor"]
        CREATE_ENV --> PRECHECK["preflight Daytona auth"]
        PRECHECK --> CREATE_SANDBOX["daytona.create sandbox"]
        CREATE_SANDBOX --> SETUP_SANDBOX["upload env / build image or compose / start services"]
        SETUP_SANDBOX --> EXEC_CMD["create process session + execute command"]
        EXEC_CMD --> POLL_CMD["poll cmd_id until exit/logs"]
        POLL_CMD --> DOWNLOAD["download artifacts"]
        DOWNLOAD --> DELETE_SANDBOX["compose down + sandbox delete"]
    end

    WRITE_ARTIFACTS --> NORMALIZER["ResultNormalizer"]
    SIM --> NORMALIZER

    NORMALIZER -->|"passed / failed / infra_failed / timed_out"| TASK_RESULTS
    NORMALIZER -->|"score + failure summary"| RUNS

    subgraph OPT["Milestone 4: Single Optimized Version Loop"]
        NORMALIZER --> FAILURE_SUMMARY["Failure Summary"]
        FAILURE_SUMMARY --> LLM["Structured LLM JSON Proposal"]
        LLM --> PATCH["AgentPatchService"]
        PATCH -->|"replace AGENT_INSTRUCTION only"| AGENT_FILE["agent/agent.py"]
        PATCH -->|"reset same task IDs"| TASK_RESULTS
        PATCH -->|"rerun proposal-1"| BG
        PATCH -->|"accept if rerun score improves; otherwise revert"| ITERATIONS
    end

    STATUS -->|"poll progress"| PG
    RESULTS -->|"read terminal result"| PG
    ITERS -->|"read baseline and proposal-1 history"| PG
```

## Request Flow

The user enters the system only through HTTP. A run is created with `POST /runs`.
The API validates task IDs, mode, sandbox provider, org scope, and role. It writes
a queued run to Postgres and returns `202 Accepted` with a `run_id` immediately.
The user then polls `GET /runs/{run_id}` and reads terminal output through
`GET /runs/{run_id}/results`.

## Milestone 4 Optimization Loop

When `max_iterations=1`, the service executes exactly one optimized version named
`proposal-1`:

```text
baseline -> failure summary -> structured LLM proposal -> restricted patch -> rerun -> score compare -> accept/revert
```

The patch boundary is intentionally small. `AgentPatchService` can only replace
the top-level `AGENT_INSTRUCTION` assignment in `agent/agent.py`, then validates
the file with `py_compile`. The rerun uses the same task IDs, mode, sandbox
provider, model, and requested concurrency. A patch is accepted only when
`rerun_score > baseline_score`; otherwise the service restores the original
agent file and restores the baseline task rows so final results match the final
agent state.

The MVP does not implement multi-round optimization, multiple optimized versions,
candidate graphs, beam search, merging, GateEngine, or suite promotion.

## Async Harbor Flow

The API process starts a background executor for the submitted run. In simulated
mode, the executor uses deterministic fake rewards so reviewers can validate the
API lifecycle without external credentials. In real mode, the executor calls
`TerminalBenchRunnerAdapter`, which invokes `harbor run --env daytona` through
argument lists, not `shell=True`.

Real Harbor/Daytona runs are serialized in the MVP with a process-local semaphore.
This keeps the one-day version simple and avoids shared artifact races and
Daytona quota spikes. A production version would replace this with a durable queue,
worker leases, heartbeats, and reconciliation.

## Daytona Lifecycle Boundary

Our service does not call the Daytona SDK directly. Harbor owns the Daytona-facing
lifecycle for the benchmark trial:

1. Check Daytona credentials.
2. Create the sandbox.
3. Upload environment files and build/start the task environment.
4. Execute agent and verifier commands through Daytona process sessions.
5. Poll command status and collect logs.
6. Download result and trace artifacts.
7. Tear down compose services and delete the sandbox.

This is why the MVP treats Harbor as the sandbox adapter. Daytona provides the
isolated execution substrate; Harbor adds benchmark semantics, task setup,
verification, artifact collection, timeout handling, and cleanup.
