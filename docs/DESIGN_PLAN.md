# Agent Optimization Service: implemented design and tradeoffs

The editable diagram is [`final_architecture.excalidraw`](../final_architecture.excalidraw);
[`architecture.png`](architecture.png) is its preview. Test evidence and measured results live
in [`VALIDATION.md`](VALIDATION.md). This describes the implementation, not a promise that
an optimizer will discover a better agent on every run.

## Architecture and priorities

Milestones 3 and 4 drive the design: the **entire generated Python agent runs inside a fresh
Docker task container**, and improvements are chosen from durable, measured experiments.
FastAPI, PostgreSQL, async jobs, organization/member authorization and `test_client.py` provide
the supporting milestones. The original standalone benchmark scripts remain independent.

FastAPI authenticates, validates and commits a job before returning HTTP 202. Workers claim
jobs using PostgreSQL `FOR UPDATE SKIP LOCKED`. A fixed Harbor 0.20.0 installed-agent adapter
uploads candidate source as data and starts it inside the task container. Neither API nor
worker imports candidate code. A standard-library compatibility runtime supports the existing
`HarnessAgent` template, avoiding a full Harbor/LiteLLM install in every task image.

Each task has its own Harbor subprocess and result callback. Results checkpoint as individual
tasks finish. This adds CLI startup overhead but allows independent cancellation, deadlines,
retries and progress. Images are cached; every attempt gets a fresh writable container.
Pre-pulling reduces cold starts but must be reported separately from warm measurements.

Submission is synchronous through authentication and the database commit: returning 202
before persistence would acknowledge work that could disappear. Long execution is asynchronous.
Within a task, model and shell steps remain sequential because later steps depend on prior
output and filesystem changes. Independent tasks run concurrently within resource limits.
Promotion is a synchronization point: all required evidence must be durable before one
transaction records the decision and changes the selected version.

Before paid agent execution, the service checks that audited verifier tools are ready.
The default cold mode installs them in the attempt container using a checksum-pinned `uv`,
declared Python/package versions, bounded preparation attempts and a setup deadline. No task
assertions run during preparation. A failed prerequisite check blocks paid agent calls,
at the cost of repeated installation time. The original verifier's later bootstrap can
still fail, including from network errors.

An opt-in **prepared-tools-v1** mode now builds a clean image once from the exact benchmark
base image. The builder receives no candidate source, instructions, assertions or credentials.
It adds only the same trusted prerequisites, restores the base runtime configuration and
sanitizes transient builder labels. A separate network-disabled container checks tool readiness
before an image is published. The API freezes the image IDs, base identities, recipe hashes
and Docker daemon identity in the job; each attempt gets its own fresh writable layer and
performs an offline readiness check. A poisoned/deleted tool in one trial cannot alter the
next container's image. There is no shared writable package-cache mount and no trial snapshot.

Missing, stale or wrong-daemon images fail before model calls; a job never silently switches
between cold and prepared modes. This implementation targets one Docker daemon, with multiple
worker processes if desired. Cross-host prepared-image routing and recovery need registry and
placement work. Preparation costs wall time and disk once per image; report that cost separately
and amortize it across repeated trials. Trial duration still includes its own readiness,
execution, verification and cleanup. The original verifier bootstrap may still use the network.
Image IDs make a job repeatable; live OS/transitive dependency resolution means rebuilding an
image is not yet bit-for-bit reproducible. These are explicit limits of the local cache.

Scoring supports the pinned suite's pytest/CTRF contract explicitly. The trusted verifier
clears old reports, records its invocation, and requires a fresh pytest completion marker,
complete individual-case outcomes and a matching binary reward. Its pytest hook counts
setup/call/teardown outcomes independently: the pinned CTRF plugin groups parameterized
cases, so report groups must not be equated with collected test cases. Deselected,
interrupted or unexecuted suites cannot qualify as complete evidence. A dependency failure that
writes reward zero without running tests is an infrastructure error with no score. The
original benchmark assertions and reward rule remain unchanged. This validates operational
completion; it does not remove the shared-filesystem tampering limitation below.

## Sandbox and credentials

The candidate container has no host bind mounts, Docker socket, database credentials or real
provider key. Its only network is a Docker internal network. A fixed non-root relay connects
that network to an egress network. The relay has a read-only mount of its own source. It
forwards model calls to a fixed trusted gateway and proxies public HTTP(S) downloads, resolving
and pinning destination IPs, rejecting non-public addresses, and allowing only ports 80/443.

The gateway accepts a revocable trial capability, fixes the model and token limits, validates
text/tool request shapes, reserves budget in PostgreSQL, calls OpenAI, then settles token usage.
Only `NEOSIGMA_OPEN_API_KEY` is accepted; there is no personal-key fallback. Even the trusted
Harbor subprocess gets an allowlisted environment without the company key or database DSN.

Tasks receive 1 CPU, 2 GiB memory and bounded process counts. The default global limit is two
tasks: the tested Docker VM has about 7.65 GiB RAM, and four 2-GiB tasks would exceed it before
relay/database overhead. Raise concurrency only after sizing the worker. The compatibility
tool wrapper applies command timeouts and closes stdin; arbitrary candidate Python remains
bounded by the whole-trial deadline and container limits. Cancellation kills process groups
and removes attempt-labelled containers/networks.
A bounded keepalive and periodic reaper clean abandoned attempts. Global admission slots bound
concurrent tasks. Logs are copied out with size and path checks rather than writable host mounts.

Docker shares a kernel. The worker and Docker daemon remain trusted. Harbor installs the pinned
verifier after agent execution, but verifier and agent share task filesystem state: malicious
code could alter binaries or leave processes that interfere. This is **not a tamper-proof reward
oracle or hardened public-code platform**. Stronger isolation needs independent verification
over allowlisted artifacts, an external watchdog and isolated worker machines/microVMs.
E2B is a future adapter, not an implemented/tested provider. No EC2 deployment is claimed.

## Consistency, availability and recovery

| Decision | Benefit | Tradeoff / boundary |
|---|---|---|
| PostgreSQL queue + state | Atomic submission; no DB/broker dual write | Poll latency; DB required for dispatch |
| Short job-row transactions | Ownership, epoch, cancellation and writes checked atomically | Per-job contention; no lock held during inference |
| Organization lock on admission | Concurrent claims respect quota | Briefly serializes tenant admission |
| Principal-scoped idempotency + payload fingerprint | Retry returns same job; changed payload gets 409 | External work still is not exactly once |
| Immutable attempts + source hashes | Retry, confirmation and rejection evidence survives | Storage and retention overhead |
| Original durable deadline | Recovery cannot reset budget clock | Outage may consume the remaining deadline |
| Fail closed on lost/unverifiable ownership | No stale promotion or new unaccounted inference | DB outage reduces availability |

External execution is **at least once**. A crash after provider completion but before checkpoint
can repeat work. Fencing protects DB writes, not external side effects. Every inference dispatch
has a durable reservation; ambiguous requests retain their reservation. Settlement is idempotent
and can complete after lease loss. Slot expiry depends on old-container termination/reaping;
it is not proof that adversarial code has stopped. Completed attempts are reused on recovery;
infrastructure errors get a bounded retry. Confirmation phase/repetition identities cannot
overwrite main results. A single PostgreSQL instance is durable storage, **not high availability**.
Production needs managed failover, backups/PITR, TLS, isolated workers and measured capacity.

## Optimization and evaluation

The task plan freezes three disjoint sets. **Feedback** supplies score history, compact failure
dossiers and rejection reasons to the optimizer. Logs are untrusted data. **Selection** decides
promotion; detailed selection traces are withheld from proposal prompts. Repeated selection
still creates adaptive bias. **Holdout** is evaluated only after selection ends, comparing the
original and final versions; its results never return to the optimizer. Budget interruption
produces an explicitly incomplete final report, never inherited or invented passes.

The baseline runs feedback plus selection. An LLM proposes one hypothesis, rationale, targets
and complete source. Syntax/interface checks and a durable SHA-256 history reject invalid or
duplicate source before containers start. Full source and computed diffs remain inspectable.
Here, self-improvement means searching over the agent's Python source; provider model weights
remain fixed. Successful proposal generation alone does not establish a better agent.
The template is small enough for full-file output. Search-and-replace edits are a later token
optimization requiring exact-match validation; they are not inherently superior for every file.

Quality proposals first run failed feedback tasks plus one passing control. With no repair,
the candidate is screened out; unexecuted tasks stay unmeasured. Efficiency proposals bypass
this repair condition. Survivors run all feedback/selection tasks. Promising candidates receive
fresh matched parent/candidate selection trials (two repetitions by default, alternating order).

Promotion requires complete verifier evidence, no observed mean-reward regression, and no more
than 10% increase in either mean model cost or p95 task duration. It then requires improved
reward or at least 10% improvement in one efficiency metric with positive reward. Missing
timing/price evidence cannot pass. Failed attempts and retry cost remain charged; step count
is only diagnostic. These tolerances are configurable **before** the experiment.
The 10% thresholds are explicit engineering tolerances, not statistically calibrated
confidence bounds; they are not loosened after seeing a candidate's results.
An ungraded baseline, including a crash that prevents verification, can therefore block
promotion until valid baseline evidence is established. This sacrifices search progress
to avoid assigning an invented benchmark score to missing evidence.

Those promotion metrics refer to the selection set. Feedback-task spending, optimizer calls
and rejected trials are additionally reported in total search cost; a selection-set efficiency
gain does not imply that every feedback task became cheaper. The three calibrated selection
tasks (four in stress16) are a small workload sample, not a claim about production-wide cost
or latency.

This policy avoids hiding quality loss behind an arbitrary dollar/second
weight. It can reject a worthwhile but expensive quality gain. Small-sample p95 is noisy, and
two repeats are not statistical significance. The configured seed controls the simulator;
live model calls currently do not pass a provider seed and remain stochastic.
Preliminary confirmation eligibility is now separate from final acceptance. Complete
preliminary evidence with no quality regression can trigger confirmation on a configured
quality or efficiency signal, even if preliminary cost/time exceeds the final tolerance.
The final decision still applies the unchanged policy to fresh matched trials. The earlier
implementation filtered too early on a noisy single sample; the independent audit exposed
substantial baseline cost variation. Each additionally confirmed candidate consumes twelve
selection trials in calibrated12, before retries, so job budgets and deadlines remain essential.
Skipped/invalid proposals retain an explicit missing observation in each feedback scoreboard;
pre-test optimizer notes are labelled unverified hypotheses with their eventual decision.
Stop conditions include iterations, patience,
cancellation, original deadline and model budget. A perfect observed score does not prevent
efficiency work. A compatible authorized terminal experiment can seed rejection memory into a
new job and supply an interrupted candidate for fresh evaluation. These snapshots contain source
and feedback evidence, never inherited scores. Interrupted candidates are not labelled failures.
Hash deduplication detects identical code, not paraphrased mechanisms; the optimizer must explain
any substantive reason to revisit a failed mechanism. Archive crossover is still deferred.

A later feedback-only audit found that command-prefix truncation hid shell redirections
and rejected-candidate aggregates omitted useful failure mechanisms. The current proposer
therefore retains bounded command/output beginnings and endings, feedback instructions up
to 8,192 characters, and at most six failed-feedback dossiers from the latest two rejected
iterations. Compatible authorized prior jobs can seed the same bounded context, with source
and decision labels and no inherited scores. Selection and held-back traces remain excluded.
The prompt describes fresh-shell behavior and unavailable hidden verifier files, and separates
observed facts from speculative causes. This is improved evidence and guidance, not a guarantee
that the LLM follows every instruction or chooses an effective change. More context costs
optimizer tokens; bounded retention trades completeness for manageable prompt size.

## Task subset and model choice

New jobs default to **calibrated12**, at Terminal-Bench 2.0 commit
`69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`. The earlier **stress16** profile remains
available explicitly. This is a new cohort after pilot observations; its scores must not be
compared with old 16-task projections as if an agent had improved. Existing job configurations
and results are unchanged. The 900-second agent limit is a ceiling, not a runtime measurement.

| Included tasks | Reason |
|---|---|
| fix-git, sanitize-git-repo | Recovery, multi-step history changes and early stopping |
| nginx-request-logging, openssl-selfsigned-cert | Service/certificate setup and end-to-end checks |
| regex-log, sparql-university | Exact text/query specifications |
| largest-eigenval, tune-mjcf | Precision and parameter search |
| extract-elf | Binary inspection and tool selection |
| cancel-async-tasks, polyglot-c-py | Concurrency and constrained code generation |
| sqlite-db-truncate | Database-file diagnosis/recovery |

Four tasks remain in stress16 but are excluded from the development profile for explicit
operational reasons, without removing their historical results:

| Excluded from calibrated12 | Reason |
|---|---|
| chess-best-move | Board-image interpretation is a mismatch for the current text/function gateway |
| gcode-to-text | Geometry reconstruction is a representation-heavy stress case |
| adaptive-rejection-sampler | Repeated pre-agent Python installation timeouts and a long scientific setup/execution path |
| configure-git-webserver | Multi-service configuration couples service lifecycle, setup and agent diagnosis |

The remaining task roles stay unchanged: **five feedback, three selection, four holdout**.
The four original holdout tasks stayed sealed until the calibrated search ended;
previously exposed tasks were not reassigned into them. That run's final observations
are reported separately in the validation report. New comparisons must measure fresh
baseline/candidate trials rather than import old scores.
Removing observed infrastructure/representation outliers improves evaluation relevance, but
does not by itself establish either a faster complete suite or better agent quality.
The offline image inventory found missing system Python in three holdout images.
Separate complete setup probes passed for all three in about 94–140 seconds, with
zero model calls and no task instructions/assertions accessed. That establishes observed
setup viability, not guaranteed future network reliability or benchmark success.

Custom task lists use a deterministic allocation across historical difficulty ranks. Those
mixed-agent solve-rate proxies came from the earlier proposal, not measurements of our model.
Neither profile is a random sample of all Terminal-Bench tasks. GPU/training and long-build
workloads are outside this development profile. Report all errors and keep each experiment's
plan fixed; do not silently replace disappointing tasks after scores arrive.

Task execution uses `gpt-5.4-mini`, fixed for baseline and candidate. The new calibrated
experiment uses `gpt-5.4` only for the relatively few optimizer proposals: earlier small-model
proposals repeated generic verification mechanisms, while code-level changes require more
analysis. This is a testable allocation hypothesis, not evidence that the larger model is
better. The deployment's optimizer default remains mini pending evidence. A live JSON-format
check confirmed access to the larger optimizer at an estimated $0.0004325.

Actual API checks required `reasoning_effort=none` for tool-bearing Chat Completions; tool-free
optimizer calls use `low`. The gateway enforces this outside candidate control. Model listing
alone does not prove inference permission. Standard text prices are estimates, with cached
input conservatively charged at full rate, not provider invoices. Local Docker dollars remain
unknown. Task-attempt wall time includes admission, setup, execution, verification and cleanup;
it is neither billed container time nor measured CPU consumption. Total search spend includes
optimizer calls and rejected/failed candidates.

## Alternatives considered

| Suggestion | Assessment / action |
|---|---|
| Keep queue/lease implementation as-is | Direction sound; atomic fencing, quota and idempotency fixes were needed |
| Rejection history was absent | Earlier design described it; durable reasons and bounded prompt memory now implemented |
| Environment scrubbing isolates Python | False; full candidate moved into task container |
| A subprocess prevents early stop | False; scheduling boundaries matter; one subprocess per task implemented |
| Blind split blocks reward hacking | Overstated; reduces adaptive leakage, not filesystem/verifier tampering |
| Full-file output must become patches | Conditional; complete source retained for small agents |
| Screening always improves optimization | Saves work but can discard general improvements; efficiency path bypasses it |
| Verification / middle-out truncation are new | Both already exist in the baseline; improvements need actual failure evidence |

Prepared tooling images were prioritized after the longest candidate trial spent 131.118 of
186.482 seconds in dependency preflight and 29.098 seconds executing the agent. The cache is
now implemented and tested independently as infrastructure; it is not an LLM-generated agent
improvement. Next, lock the complete dependency closure and measure cold preparation plus
amortized full trials. Original verifier bootstrap still uses the network, so prepared images
do not guarantee offline or uniformly fast grading. Keep old cold-runtime results attached to
their original harness and do not subtract setup estimates from them to invent a new score.

Then measure concurrency against CPU/memory/provider limits, cache prompt prefixes, compact
context while retaining evidence, use targeted edits for larger files, and explore diverse
candidate archives. Parallelism can worsen tail latency and cost. None is called a measured
gain without an ablation. The final validation report records the observed cost reduction and
latency regression separately; the default agent remains the original baseline.

Primary references: [Harbor](https://harborframework.com/docs),
[PostgreSQL locking](https://www.postgresql.org/docs/current/sql-select.html),
[Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/),
[GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
