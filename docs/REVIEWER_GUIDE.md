# Reviewer guide

This guide separates the implemented service, measured agent behavior, and remaining proof.
The completed corrected prepared-mode study ended at **2026-11-07 13:24:00 UTC** with all
40 observations valid, the baseline retained, and resource cleanup independently checked.
The earlier cold-mode results retain their **2026-11-07 12:13 UTC** historical cutoff.
A later evidence-guided experiment stopped at its unchanged **14:06:29 UTC** deadline.
Its job is `CANCELLED`, its controller report is `INCOMPLETE`, and cleanup is verified.
It provides additional valid comparisons but an incomplete final held-back evaluation.

## The honest opening

> I built a Python/FastAPI and PostgreSQL service that runs the complete agent inside Docker,
> captures real benchmark failures, asks an LLM for a code change, and evaluates the new source.
> Jobs, versions, attempts, decisions and spending survive worker restarts. In the completed study,
> the LLM generated three candidate changes and the service automatically ran fresh matched
> confirmation for the promising one. That candidate was cheaper and faster in the observed
> sample, but passed fewer trials, so it was rejected. The baseline remains selected.
> Manual preparation work also reduced repeated tool-readiness time. That engineering result
> is separate from agent self-improvement; no accepted balanced agent gain is demonstrated.

The assignment prioritizes the infrastructure around the LLM, state, isolation and reasoning.
Milestone 4 asks for proposal, application, re-evaluation, stopping and accessible history;
it does not guarantee that a particular search will find a better agent. Nevertheless,
**a real, automatically accepted improvement is the strongest missing performance proof**.
This is worth presenting as a working, measured optimization service with a candid result,
not as a success-rate breakthrough or a guarantee of an interview outcome.

## Implemented architecture and deployment scope

The **full-agent sandbox architecture** is the implemented base. Generated Python executes
inside a fresh Docker task container; the host worker supplies a fixed adapter and collects
results. A scoped gateway keeps the provider key outside candidate execution.

E2B is a possible future provider beneath this same boundary. Remote microVMs could add
isolation and capacity, while the current Docker implementation shares a kernel and prepared
mode has single-daemon affinity. That adapter requires implementation and fresh verification;
no E2B deployment or cross-provider performance comparison is claimed here.

## Five-minute architecture story

1. **0:00–0:45 — Request and durable ownership.** Open the
   [editable diagram](../final_architecture.excalidraw) or [preview](architecture.png).
   Authentication, validation and the PostgreSQL commit finish before HTTP 202. Workers
   claim jobs with `FOR UPDATE SKIP LOCKED`; a lease and increasing epoch fence stale writes.
   Long benchmark work is asynchronous. Model/tool steps inside one task remain sequential;
   independent tasks run concurrently within global and organization limits.
2. **0:45–1:45 — Milestone 3: the actual execution boundary.** The worker uploads source
   as data. It never imports candidate Python. A fixed Harbor adapter starts the complete
   agent inside a fresh Docker task container, with no host bind mounts, Docker socket,
   database credentials or provider key. A scoped gateway meters calls using only
   `NEOSIGMA_OPEN_API_KEY`. Deadlines, cancellation and verified resource cleanup bound
   the lifecycle. Show the [sandbox evidence](sandbox_smoke.json) and
   [PostgreSQL/Docker cancellation proof](sandbox_slot_lifecycle_smoke.json).
3. **1:45–3:00 — Milestone 4: a measured search.** The fixed twelve-task profile has five
   feedback, three selection and four final holdout tasks. Only feedback dossiers enter
   proposal prompts. Each proposal includes a hypothesis and full source; syntax/interface
   checks and source hashes run before sandbox dispatch. Rejected ideas remain in durable
   history. Promising candidates need fresh matched selection evidence before promotion.
   Completion and the selected-version pointer change atomically. Open the
   [autonomously triggered matched confirmation](live_prepared_verified_confirmation_start.json)
   and [exact candidate source versions](prepared_verified_candidates/index.json).
4. **3:00–4:00 — Show results, including rejection.** Walk through the table below. A
   promising single-run quality increase did not survive the full balanced decision.
   Missing verifier output is an infrastructure failure, not an invented score of zero.
   Local Docker dollars are unknown, not free; reported dollar comparisons are model usage
   estimates. The [validation report](VALIDATION.md) separates simulated, container and
   paid-provider evidence.
5. **4:00–5:00 — Explain the separate infrastructure result.** Prepared-tool readiness was
   measured on clean container clones and is much shorter than preparing those tools from
   scratch. Upfront image construction and disk storage pay for that reuse. The same images
   are frozen for baseline and candidate. Six-sample p95 is still just the worst trial;
   faster setup and real autonomous confirmation do not establish an accepted agent gain.

## What is proved, milestone by milestone

| Requirement | Implemented evidence | Boundary |
|---|---|---|
| M1: structured HTTP results | Submission, status, per-task outcomes, failures, versions and events; OpenAPI | `SUCCEEDED` means lifecycle completion, not an improvement |
| M2: background execution | Durable queue, polling, cancellation, fenced takeover; actual API/worker/CLI integration | PostgreSQL availability is required; external execution is at least once |
| M3: isolated real execution | Docker/Harbor fixtures and 40 valid observations in the completed 13:24 study; actual cleanup tests | Docker shares a kernel; no EC2/E2B deployment or tamper-proof grader is claimed |
| M4: iterative optimization | Completed 13:24 study: three paid proposals, two feedback screen-outs, one autonomously scheduled fresh matched comparison, persisted rejection and complete final evaluation | No accepted real candidate; held-back tasks had prior development exposure |
| M5: organizations and roles | Admin/member API permissions, scoped idempotency and cross-tenant database/API tests | This is a take-home access model, not a complete enterprise identity product |
| Deliverables | Python service, PostgreSQL migrations, `test_client.py`, setup/rationale/tradeoff docs, Excalidraw | Private credentials and the local evidence database are excluded; saved reports remain inspectable |

## Evidence-guided follow-up — deadline-limited, cleaned up

The [follow-up experiment](live_evidence_guided_review.json), job prefix `1e3b888c`, started again
from the original baseline and performed fresh evaluations. It retained only authorized feedback evidence
and rejected-hypothesis context from the earlier compatible study; no previous scores or candidate
versions counted as new observations. The runtime, prepared images and evaluation policy stayed fixed.

| Source | Observed preliminary result | Durable decision |
|---|---|---|
| Fresh baseline | Feedback 1/5; selection 2/3 | Baseline |
| Candidate 1 | Feedback 3/5; selection 2/3; model cost −6.81%, p95 +12.19% | Rejected: no qualifying selection improvement; cost reduction is below the declared 10% signal |
| Candidate 2 | Selection 1/3 | Rejected: quality regression |
| Candidate 3 | Qualified for fresh matched selection confirmation | Rejected: confirmed model cost exceeded the 10% guard |

Feedback repairs are useful diagnostic progress, but they do not substitute for selection
evidence. Candidate 3 then produced the following **fresh matched**, six-trial-per-source result:

| Fresh matched metric | Incumbent | Candidate 3 | Observed change |
|---|---:|---:|---:|
| Passed trials | 2/6 | 3/6 | One additional pass |
| Mean model cost/task | $0.013355875 | $0.015754250 | +17.96% |
| p95 complete attempt | 74.77 s | 81.69 s | +9.25% |

This is an **observed quality gain with higher cost**, not an accepted balanced improvement.
The service rejected it under the unchanged 10% cost guard and retained the original baseline.
Six trials are not a statistically established gain; the p95 values are the worst observed trials.
The job reached its fixed 25-minute deadline during final held-back evaluation, so its controller
correctly reports `INCOMPLETE` and its job is `CANCELLED`, rather than claiming lifecycle success.

The [terminal audit](live_evidence_guided_audit.json) records **45 durable completed observations**,
all valid under verifier v2: 32 main, 12 confirmation and one held-back observation, with zero
infrastructure errors. The final held-back evaluation is **incomplete** and must not be scored
as a complete four-task result. The experiment cost **$1.2527385** in estimated model usage;
cumulative session spending reached **$6.71844375**, with zero unresolved reservations.
All owned processes, slots, job containers and networks were cleaned up. Exact source versions
are in the [candidate index](evidence_guided_candidates/index.json).

Use the completed 40-observation study below as the main demo for a complete lifecycle. This
follow-up is a useful optional demonstration of a genuine quality/cost tradeoff and deadline handling.

## What the evidence actually says about the bottlenecks

The earlier candidate-3 [mechanism audit](live_prepared_verified_candidate3_mechanism.json)
showed that its proposed pytest check never executed its intended verification branch in the
four inspected feedback/control traces. Hidden verifier files are installed after the agent,
so a proposal depending on them during execution is not a usable mechanism. The saved evidence
does not distinguish which prerequisite caused its skip. This supports better capability guidance,
not the conclusion that all verification is ineffective.

A feedback-only review also found that command-prefix summaries could hide shell redirections
and that aggregate rejection notes omitted specific failure mechanisms. The current proposer
therefore receives bounded command/output beginnings and endings, more complete feedback
instructions, and at most six failed-feedback dossiers from the last two rejected iterations.
It explicitly describes fresh-shell behavior and unavailable hidden tests. Selection and held-back
traces remain excluded. These are manual improvements to evidence delivery. The follow-up produced
a candidate with an observed matched quality gain, but no accepted balanced gain; it does not
isolate a causal benefit from the evidence-delivery changes.

Other measured limits are independent: a promising single-run signal can disappear under fresh
matched confirmation; dependency setup can dominate an observed tail; and a small repeated
development set cannot establish generalization. The preparation results and confirmation policy
address different parts of that problem. Neither guarantees an effective generated change.

## Historical corrected prepared-mode results — completed at 13:24 UTC

The [completed experiment](live_prepared_verified_review.json) uses the corrected verifier,
frozen prepared images, a fresh original baseline, `gpt-5.4-mini` execution and `gpt-5.4`
proposals. It inherited no previous candidates, scores or rejection memory. Its initial
selection score was 1/3. Its frozen service manifest begins `99ead2b2`; the later proposer
changes do not rewrite this study. All three proposal decisions are durable:

| Candidate | Observed evaluation | Service decision |
|---|---|---|
| 1 | No previously failing feedback task repaired in its screen | Screened out; selection unmeasured |
| 2: compact long tool output to first/last 60 lines | Preliminary selection 2/3; fresh matched comparison below | Rejected: quality regression |
| 3: conditional pytest verification | No feedback repair; intended verification branch was skipped | Screened out; selection unmeasured |

Candidate 2's preliminary p95 was 12.6% higher, but its quality signal correctly triggered
fresh confirmation rather than automatic rejection from a single noisy comparison. The
service ran B/C/C/B, two repetitions of three selection tasks per source. All twelve
confirmation observations were valid, with no retries:

| Fresh matched metric | Incumbent | Candidate 2 | Observed change |
|---|---:|---:|---:|
| Passed trials | 3/6 | 2/6 | One fewer pass |
| Mean model cost/task | $0.009667875 | $0.009306250 | −3.74% |
| p95 complete attempt | 58.08 s | 51.31 s | −11.66% |

The service **automatically rejected candidate 2 because quality regressed**. This proves
actual LLM proposal, application, autonomous matched confirmation and decision persistence.
The CLI's headline baseline score still represents its initial 1/3 observation. The rejection
compares candidate 2 against the **fresh incumbent 3/6** stored in its `metrics.incumbent`,
not against that earlier headline; compare equal-sized contemporaneous samples.
It does not prove balanced self-improvement. The [terminal audit](live_prepared_verified_audit.json)
confirms **40/40 valid observations**, no infrastructure errors, timeouts or retries, and
completion in **20 minutes 26.6 seconds**. The baseline remained selected. Its final held-back
score was **1/4**; the identical selected source reused those same observations rather than
claiming an independent second evaluation. These task roles were previously observed, so this
is repeated development validation, not an untouched generalization test.

The job used **$0.8211915** in estimated model charges. Cumulative session model spending at
that completed-study cutoff, including diagnostics and interrupted experiments, was
**$5.46570525**, with **zero unresolved
reservations**. The audit found no owned API/worker processes, active jobs, sandbox slots or
job containers/networks remaining. Local Docker compute is unpriced, not zero-cost.

The [candidate-3 mechanism audit](live_prepared_verified_candidate3_mechanism.json) found that
all four feedback/control traces took the conditional verification skip path. It does not
establish whether the missing prerequisite was the test directory or pytest. Consequently,
the screen-out cannot show that successful execution of the intended verification mechanism
would be ineffective. The subsequent experiment adds that capability guidance and evaluates it
separately; its effect was not established by this completed study.

## Historical cold-mode results — cutoff 12:13 UTC

The search used `gpt-5.4-mini` for execution and `gpt-5.4` for code proposals, with the
same executor model, limits and task roles for baseline and candidates.

| Search source | Selection passes | Mean model cost/task | p95 attempt duration | Automatic decision |
|---|---:|---:|---:|---|
| Original baseline | 1/3 | $0.0069525 | 126.39 s | Retained |
| Candidate 1: one extra verification round | 1/3 | $0.0204005 | 131.39 s | Rejected: cost |
| Candidate 2: path-aware output compaction | 1/3 | $0.0091910 | 135.25 s | Rejected: cost |
| Candidate 3: dependency-portability guidance | 2/3 | $0.01301425 | 120.35 s | Rejected: cost |

The [original run](live_calibrated_benchmark_audit.json) completed **36 valid observations**
in about 46 minutes, cost **$0.9720645** in estimated model usage, and retained the original
source. Final baseline holdout was **1/4**. Because the selected source was unchanged, its
holdout results were reused, not presented as a second independent sample.

A separately declared [matched audit](live_calibrated_candidate3_paired_audit.json) then ran
candidate 3 against the baseline in B/C/C/B order: **12/12 valid observations**, **3/6 passes
for each source**, model cost **16.8% lower**, but end-to-end p95 **53.2% higher** for the
candidate. It therefore still failed the unchanged latency guard. The longest candidate
attempt included **131.12 seconds of dependency preparation**; that observation was retained.
It does not establish that the candidate caused the delay. The audit cost **$0.17102475**.

Across the completed search and audit: **48 valid observations, no accepted balanced gain**.
Cumulative session model spending at that cutoff, including earlier diagnostics and aborted
experiments, was **$4.067705**, with zero unresolved reservations. Earlier infrastructure-invalid
runs are preserved and excluded from performance conclusions. Six observations per source are
not statistical significance or a representative full Terminal-Bench estimate.
With six trials, the implementation's nearest-rank **p95 is simply the worst observed trial**.
The 10% guard is therefore sensitive to a single slow setup; it is an engineering tolerance,
not a confidence interval or a reliable estimate of the population tail.

## Autonomous changes versus our engineering work

The candidate source changes in these searches came from the optimizer LLM using feedback evidence;
the service applied, ran and decided on them. The sandbox, gateway, verifier-completion checks,
budget ledger, confirmation scheduling and any preparation/cache changes were implemented by
the engineering team. **A manually improved execution engine is not agent self-improvement.**
The historical external candidate-3 audit was manually scheduled and did not change that job's
decisions. In the completed 13:24 study, candidate-2 confirmation was scheduled autonomously by
the service. The newer proposer evidence changes are our engineering work, not an LLM-discovered
agent improvement.

## Manual setup improvement: measured separately

Opt-in prepared-tool images have infrastructure evidence: all twelve image entries passed
provenance checks, and [four task images each passed three clean-container probes](prepared_image_validation.json)
without candidate execution, original assertions or provider calls. These checks validate
tool readiness and fresh-layer isolation; they do not establish an agent performance gain.
The measured setup-stage comparison is:

| Task | Cold preparation: one observation | Prepared readiness: median of three |
|---|---:|---:|
| fix-git | 74.56 s | 1.097 s |
| nginx-request-logging | 13.16 s | 1.024 s |
| largest-eigenval | 86.52 s | 0.535 s |
| regex-log | 110.96 s | 1.378 s |

These are preparation/readiness measurements, not a randomized end-to-end agent speedup.
Build/commit/probe work across all twelve images totaled **1,139.29 aggregate seconds**,
excluding base-image pulls and final builder cleanup; this is summed work, not wall-clock
duration or a cloud bill.

Each job freezes immutable image IDs and the Docker daemon identity. Every trial still gets
a fresh writable layer. Images are built from clean task bases with trusted tooling only;
no candidate state or provider credentials are snapshotted. This is **manual infrastructure
work**, applied equally to baseline and candidate, not an autonomous agent improvement.

The tradeoff is upfront build time and retained disk usage versus repeated setup work and
variance. The original verifier may still access the network, so preparation does not make
execution fully offline or deterministic. The current implementation has **single-Docker-daemon
affinity**: several worker processes on the same host can use it, but cross-host prepared-cache
recovery and registry-based routing are not implemented.

The first prepared-mode experiment was interrupted and preserved after finding a verifier
accounting defect: the CTRF reporter collapsed parameterized cases into groups, and comparing
its group count directly with pytest's collected-case count could reject an otherwise valid
execution. That interrupted experiment is **not a valid performance comparison**. The corrected
verifier passed separate 27-case successful, failing-assertion and missing-dependency fixtures
before the completed 13:24 study started. Those fixtures made no provider calls. Historical cold-mode
evidence remains unchanged; the new agent comparison is reported separately above.

## Working demo paths

Run commands from the repository root. Use a local terminal for credential setup; never show
provider keys on a shared screen. A service API key (`AOS_API_KEY`) is different from the
provider credential. The simulator requires no provider credential.

### Fast, no-provider-cost service demonstration

Prerequisites: Docker/Compose, Python 3.12 and installed service dependencies. On a fresh demo
database, with ports 8000 and 5433 available:

```sh
python3.12 -m venv .venv-service
.venv-service/bin/pip install -r service/requirements.txt
docker compose -f compose.service.yml up --build -d
docker compose -f compose.service.yml run --rm api python -m scripts.bootstrap_demo
# Privately save one generated service key in AOS_API_KEY, then:
.venv-service/bin/python test_client.py --max-iterations 2 --patience 2
```

Bootstrap creates organizations and prints service keys; run it once for a fresh database.
If this database already exists, reuse its service key. Compose explicitly selects the
simulated executor and optimizer. Say that before showing its scores. The actual API, worker,
PostgreSQL persistence and full-history client are exercised; its benchmark outcomes are fake.
Use `/docs` for API exploration and `/v1/readyz` for database readiness.

### Inspect the completed real job without paying for another run

On the original development machine, the recorded report and its ignored private control
file support a single-command replay:

```sh
.venv-service/bin/python -m scripts.demo_recorded_review \
  --report docs/live_prepared_verified_review.json
```

This starts a temporary API on its own ephemeral localhost port, rejects HTTP writes, checks
that the authorized database job matches the terminal report, and invokes `test_client.py`
with `--job-id`. It starts no worker, submits nothing and removes provider credentials from
its subprocess environments. It stops only its own API afterward. The compact replay was
actually exercised against the saved job: all four iterations were displayed, the baseline
was retained and cleanup completed; the [redacted replay proof](recorded_demo_validation.json)
records those checks. Add `--json` only when detailed private traces are needed.
The final-job replay succeeded in 1.174 seconds; this is demo responsiveness, not benchmark
latency. The earlier historical replay also remains available by passing
`docs/live_calibrated_benchmark_review.json` to the same command.
For the optional deadline-limited follow-up, use:

```sh
.venv-service/bin/python -m scripts.demo_recorded_review \
  --report docs/live_evidence_guided_review.json
```

This replay was also tested: all four iterations and the `CANCELLED` state were displayed,
the baseline remained selected, and the temporary API stopped. It completed in 1.541 seconds
with no worker, submission or provider calls. Its incomplete final evaluation is part of the
demonstration, not a hidden failure. The same [redacted proof](recorded_demo_validation.json)
preserves all three replay checks.
The private control file and evidence database are required; neither should be published
with the repository. A reviewer without them can use the simulator and saved evidence above.

For manual attachment to an existing inspection API, use the following alternative.
For read-only inspection, point an API process at the preserved **evidence database**, using its existing
authorized service key. The simulator database above is a different database and will not
contain the recorded job. No worker is needed for terminal-job inspection:

```sh
# DATABASE_URL and AOS_API_KEY must already refer to that evidence environment.
EXECUTOR=harbor OPTIMIZER=llm .venv-service/bin/uvicorn \
  service.api.app:app --host 127.0.0.1 --port 8001
```

In a second configured terminal:

```sh
.venv-service/bin/python test_client.py --base-url http://127.0.0.1:8001 \
  --job-id 253886be-b646-4176-ac23-7e6990e71755
```

`--job-id` attaches; it does not submit new work or make model calls. Start with this compact
screen view. Add `--json` to the same attachment command afterward when full iteration details
and immutable attempts are useful; that output includes task traces and should be handled
privately. If an authorized API is already running against the evidence database, use its
current base URL instead of starting another server. The retained agent has no diff from baseline; inspect candidate versions
through `/v1/jobs/{id}/iterations/{index}` or the exported text files instead:

```sh
diff -u docs/prepared_verified_candidates/baseline.py.txt \
  docs/prepared_verified_candidates/candidate2.py.txt
```

A nonzero `diff` exit code simply indicates differences. Do not import candidate files on the
host. If the private evidence database is unavailable, show the saved audit and source index;
describe them as archived evidence, not a running service.

For a new real execution, follow [README setup](../README.md#real-sandboxed-execution), stop
the simulator worker on that queue, and use an explicit model budget and deadline. Preparing
benchmark images and tools can take minutes; a full optimization is not a five-minute demo.
Do not start an unplanned paid run merely to fill interview time.

## Five design tradeoffs to defend

1. **PostgreSQL queue and state together.** Atomic submission and decisions avoid a broker/DB
   dual-write problem. Polling adds latency, and a single PostgreSQL instance is not HA.
2. **Fail closed on ownership or budget uncertainty.** Fencing and durable reservations prevent
   stale promotion and unaccounted dispatch. An outage can consume the deadline or temporarily
   strand capacity; availability is sacrificed for consistent decisions and bounded spend.
3. **Fresh containers with a trusted fixed runner.** Candidate code stays outside the API/worker
   processes. Docker remains a shared-kernel boundary, and verifier/agent share task filesystem
   state. This is appropriate for the demonstrated assignment, not hostile public multi-tenancy.
4. **Quality plus explicit efficiency guardrails.** Promotion requires complete evidence, no
   quality regression, no greater than 10% increase in model cost or p95, and meaningful quality
   or efficiency gain. This is explainable but can reject an expensive quality win. At six
   trials, nearest-rank p95 is the maximum, making a 10% guard brittle under variable setup.
   It is not a confidence interval. Thresholds are declared before a run, not loosened afterward.
5. **Small fixed, separated task roles.** Twelve tasks cover Git, services, certificates, data,
   binary inspection, numerical work and concurrency at a manageable scale. They are a curated
   development set, not a random benchmark sample. Withholding detailed selection traces and
   final holdout feedback reduces adaptation but does not eliminate overfitting or reward hacking.

## Questions to be ready for

**Does the LLM actually work?** Yes: the paid provider executed task conversations and generated
distinct candidate files in the searches. The completed 13:24 study also automatically scheduled
fresh matched confirmation. Source versions, the usage ledger and verifier results substantiate
that. “The integration works” does not mean its proposed fix works.

**Did the agent improve itself?** It autonomously proposed and tested changes. The completed
searches have not selected a better balanced agent. The completed prepared study's candidate 2
initially showed 2/3 versus 1/3, then achieved 2/6 versus the incumbent's 3/6 under autonomous
confirmation. The historical
candidate 3 likewise failed its separate matched audit. Initial gains are not confirmed wins.
The latest follow-up's candidate 3 did pass one additional matched trial, 3/6 versus 2/6, but
cost 17.96% more and was rejected by the declared balanced policy. That observed quality signal
is worth showing honestly; it is not an automatically accepted or statistically established gain.

**Why not just accept the cheaper candidate?** The completed prepared study's candidate 2 had
fewer confirmed passes, so accepting it would violate the declared no-quality-regression policy. The historical
candidate 3 was rejected on latency instead. Neither lower cost nor one favorable preliminary
score overrides the complete balanced rule. Prepared tooling is identical for baseline and
candidate, with build/storage overhead disclosed separately.

**What happens after a crash or provider failure?** Lease epochs fence writes, completed attempts
can be reused, and infrastructure retries are bounded. External work can repeat after an ambiguous
crash. Budget reservations stay conservative, and a missing verifier result cannot qualify for
promotion. Resource cleanup is provider-scoped; local absence is not proof about another host.

**What remains before a stronger performance claim?** A real service job must discover and
automatically promote a candidate from fresh matched evidence. Then an appropriately sealed final
evaluation must support its generalization claim. The four published holdout outcomes are now
observed; reuse can be a disclosed regression check, not a newly untouched test set. More repetitions
and stable preparation would improve confidence, but no experiment guarantees a gain.

**What would you do with more time?** Use the now-measured preparation and autonomous confirmation
path to test more focused changes, improve optimizer capability awareness, and increase repeated
evaluation coverage while tracking total search cost. Managed PostgreSQL failover,
stronger external verification and remote sandbox providers are deployment work, not evidence
that the current optimizer has improved.
