# Validation evidence

Last updated: 2026-11-07. These checks distinguish control-plane correctness,
container execution, live model access, and actual optimization performance.
Passing one layer does not establish the next layer.

## Automated service tests

**355 tests and 12 subtests passed; ten tests were skipped** with the isolated
PostgreSQL test server enabled. Nine skipped cases require the optional Harbor
environment and passed separately there. The tenth is the opt-in actual
Docker/PostgreSQL lifecycle fixture, previously executed successfully as described below.
The service test command made no paid model calls.
This pass includes the recorded-demo helper and feedback-context continuation
regressions. The earlier 347-test pass took 12.73 seconds; the subsequent
proposer-evidence changes were tested before the separate experiment below.

```sh
AOS_TEST_DATABASE_URL='postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test' \
  .venv-service/bin/python -m pytest service/tests -q --tb=short
```

The PostgreSQL integration tests create randomly named disposable databases,
apply all migrations, and drop only those databases afterward. They never
truncate the existing development database at port 5433. Other environments
should supply their own explicitly designated test-server DSN with permission
to create databases.

Verified properties include:

- Concurrent organization admission cannot consume the same quota slot twice;
  global trial-slot admission is also serialized.
- Reusing an idempotency key for the same principal and payload returns the same
  job; a changed payload conflicts; another member cannot use the key to access
  the first member's job.
- Lease takeover fences stale writes. A worker whose lease expires while
  waiting for a row lock cannot write afterward. Recovery preserves the original
  wall-clock deadline.
- Cancellation prevents promotion. Iteration completion and the best-version
  pointer change in one transaction. A stopped job records an incomplete final
  evaluation rather than claiming a completed holdout.
- Retry and confirmation attempts are immutable, separately identified, and do
  not overwrite the original observation. Missing results do not shrink the
  evaluation denominator or inherit a parent's passes.
- Budget reservations are atomic; ambiguous calls remain charged at their
  reserved upper estimate. Fractional token costs survive settlement without
  rounding away. Non-finite budgets are rejected with HTTP 422 before dispatch.
  Unknown sandbox or task costs remain null rather than zero.
- Gateway tests use mocked HTTP transports to check the company-key-only
  policy, fixed upstream destination, scoped authorization and revocation,
  trusted model/output limits, conservative malformed-usage accounting, and
  durable budget/cancellation/lease stop signals. Database failures during pool
  acquisition, reservation, or settlement now set a trusted failure signal and
  unknown usage rather than allowing a swallowed HTTP error to appear free.
- Legacy database rows and results survive the added migrations. Cross-tenant
  API reads and cancellation return the same not-found response as absent jobs.
- Workflow tests exercise screening, fixed split boundaries, paired
  confirmation, quality/cost/latency selection, bounded optimizer repair,
  cancellation, and recovery. Sandbox unit tests inspect the fixed adapter and
  network policy and exercise bounded output and command termination.
- A terminal status alone cannot substitute for a numeric reward. The actual
  outer-deadline executor branch is covered by a regression test: its ungraded
  timeout remains incomplete while preserving cost and cleanup. A timeout with
  executed verifier evidence and an explicit reward remains scorable.
- Prepared-image admission freezes the image mapping; a changed mapping conflicts
  with an existing idempotency key. Missing provenance is rejected before creating
  a job. Legacy cold jobs remain compatible with cold mode, but cannot silently
  continue into a different prepared environment.
- Preliminary confirmation eligibility is separate from final promotion. Tests
  cover a promising candidate reaching fresh confirmation despite noisy preliminary
  efficiency, while preserving the final quality and 10% cost/p95 guardrails.
  Duplicate/invalid proposals retain scoreboard alignment; generated rationales
  remain explicitly unverified hypotheses.
- Independent pytest case accounting covers parameterized pass/fail, deselection,
  collection/setup/teardown errors, early termination, skipped-only suites and
  false rewards. Evidence readers preserve the v2 case counts and explicitly
  distinguish older v1 observations without inventing missing case metadata.

## Actual API, worker, and CLI run using simulated outcomes

The smoke launched real FastAPI and worker subprocesses, invoked `test_client.py`,
and used a disposable PostgreSQL database. Both the executor and optimizer were
simulated; provider credentials were removed from subprocess environments.

```sh
AOS_TEST_DATABASE_URL='postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test' \
  .venv-service/bin/python -m service.tests.smoke_simulated
```

The earlier [calibrated-profile smoke](calibrated_simulated_smoke.json) recorded:

```json
{
  "status": "passed",
  "mode": "simulated_only",
  "cli_iterations": 3,
  "holdout": "complete",
  "cancel_latency_s": 0.275,
  "recovered_lease_epoch": 2,
  "durable_deadline_preserved": true
}
```

This verifies integration, cancellation, and recovery behavior. Its scores,
costs, and durations cannot support a real benchmark performance claim. The
smoke stopped its subprocesses and removed its temporary database.

The later [confirmation-workflow validation](workflow_confirmation_validation.json)
records another actual API/worker/CLI simulation: three iterations, completed final
evaluation, cancellation in 0.082 seconds, recovery at lease epoch 2 and preservation
of the original deadline. It also records 189 focused passing tests. These are
control-flow observations, not real agent-quality improvements.

The [recorded-history demo proof](recorded_demo_validation.json) verifies an actual
attachment to the preserved evidence database: four historical iterations displayed,
no worker, no job submission, no model calls and cleanup of its temporary API.
Four focused helper tests cover terminal-job identity, private-control path bounds,
credential environments and rejection of mutating HTTP methods. This makes existing
evidence convenient to inspect; it is not a new benchmark repetition.

## Actual Docker and Harbor execution with scripted model replies

[The sandbox smoke report](sandbox_smoke.json) records a successful Docker/Harbor
run of the unchanged starting template with two scripted model responses and
zero real provider calls. The observed verifier reward was 1.0 and the trial
took 14.83 seconds in the cold-setup configuration. This is an infrastructure check,
not an optimization result. The newer prepared-mode fixture is described below.

```sh
.venv-service/bin/python -m service.sandbox.smoke
```

The run inspected the candidate container and verified no host bind mounts and
one internal network. Its checks found no host marker or provider/database
credential environment variables, rejected private and metadata destinations
through the proxy, rejected direct host access, and successfully fetched public
HTTPS content. Cleanup ran after the smoke.

The [negative verifier fixture](sandbox_invalid_verifier_smoke.json) reproduced
the important failure mode: a shell script exited zero and wrote reward zero
although a test dependency was missing. The corrected path returned
`status=error`, `reward=null`, `failure_class=infra_error`; no test outcome was
invented. This full sandbox check took 103.83 seconds, including 92.17 seconds of
successful dependency setup, and made zero provider calls. Setup time varied substantially
between fixtures; these timings are not a controlled performance comparison.

The [audited sampler setup](sandbox_verifier_preflight_smoke.json) separately
prepared the actual `amd64` benchmark image through the isolated relay. It
verified the pinned `uv` archive checksum and prepared Python 3.13.9, pytest
8.4.1, CTRF 0.3.5, NumPy 2.3.3 and SciPy 1.16.2 before any agent execution.
Its first attempt passed in 114.89 seconds, with no original assertions executed
and zero provider calls. This probe called the tool preflight directly; it did
not exercise the installed adapter's separate system-Python prerequisite.
Later live sampler attempts timed out at that earlier prerequisite. This is a
measured tool-setup cost, not proof of complete sampler readiness or a speedup. It makes
prebuilt, versioned tool images a concrete optimization to measure; the later
prepared-mode measurements below address that setup path.

The [offline prerequisite inventory](sandbox_prerequisite_inventory.json) checked
all twelve development images without network, candidate execution or task-content
access. Three original holdout images lacked system Python. The subsequent
[complete setup probes](sandbox_complete_setup_smoke.json) exercised both the
Python prerequisite and verifier-tool preparation: regex-log took 135.74 seconds,
sparql-university 140.46 seconds and polyglot-c-py 93.91 seconds. All passed within
the existing limits, with no candidate, task instructions, original assertions or
model calls. These observations show setup viability only; the later calibrated
run's actual holdout results are reported below. Future network failures remain possible.

[Public egress probes](sandbox_egress_smoke.json) reached the required PyPI and
GitHub endpoints. Astral returned HTTP 403 for the Python-urllib user agent and
200 for curl/uv user agents; this does not establish the cause of earlier
intermittent benchmark failures. Relay regression tests also verify public-IP
fallback without relaxing private-address checks and correct failure handling
after a TLS tunnel is established.

A separate [actual Docker cleanup smoke](sandbox_cleanup_smoke.json) stopped its
runner, container, and network in 0.316 seconds. It also verified that reaping
removed an expired fixture while preserving an unexpired control. No model
calls were made. This is a targeted cancellation/reaping check, not a sustained
load or provider-failure stress test.

```sh
.venv-service/bin/python -m service.sandbox.cleanup_smoke
```

Container lifetime now uses an absolute expiry and calculates the remaining
seconds when the container starts. Image preparation or a delayed startup can
therefore no longer extend a relative keepalive past the planned expiry. The
runner deadline and labeled-container reaper provide additional cleanup paths.

The [actual Docker/PostgreSQL lifecycle proof](sandbox_slot_lifecycle_smoke.json)
also paused database release and cancelled repeatedly. Runner/container cleanup
finished before the reservation was released, and the release still completed.
Reconciliation removed an expired slot only for the same Docker daemon after
provider cleanup; foreign, legacy, active and unexpired reservations remained.
Expiry alone no longer deletes reservations. This is deliberately conservative:
unknown remote/legacy resources require verified owner/operator cleanup.
The latest lifecycle check passed in 4.09 seconds. The
[historical cold source manifest](cold_source_manifest.json) contains 38 service production
Python, SQL and runtime JSON files and was unchanged before and after both cold
Docker/Harbor fixtures. It is distinct from the newer prepared-mode source manifest.

```sh
AOS_RUN_DOCKER_TESTS=1 \
  AOS_TEST_DATABASE_URL='postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test' \
  .venv-service/bin/python -m pytest service/tests/test_sandbox_database_lifecycle.py -q
```

A previous fixture took 29.15 seconds before shortening container cleanup
grace; intermediate observations were 13.56 and 11.48 seconds. The current
14.83-second fixture additionally prepares pytest and checks fresh grading
completion. These configurations differ and are not a controlled speed
comparison or evidence of a benchmark-quality gain.

Nine separate Harbor-environment checks passed, covering normal artifact
copies, unsafe or oversized entries, and the adapter's setup/config boundary:

```sh
"$(uv tool dir)/harbor/bin/python" \
  -m unittest service.tests.test_sandbox_artifacts
```

That command uses the optional Harbor environment; a service-only environment
without Harbor skips those artifact tests.

## Prepared tooling: measured infrastructure improvement

The opt-in prepared mode is implemented and tested on the local Docker daemon.
It prepares the existing audited tools in clean images, freezes immutable image IDs
at job admission, and creates a fresh writable layer for every attempt. Baseline
and candidates use the same frozen mapping. No candidate, task instructions,
original assertions or provider credentials enter the image-building process.

The [image validation report](prepared_image_validation.json) records one cold
preparation and three fresh, network-disabled readiness probes for each of four
tasks. These measure the setup stage, not whole-job speedup or agent quality:

| Task | Cold preparation, one observation | Prepared readiness, median of three |
|---|---:|---:|
| fix-git | 74.56 s | 1.097 s |
| nginx-request-logging | 13.16 s | 1.024 s |
| largest-eigenval | 86.52 s | 0.535 s |
| regex-log | 110.96 s | 1.378 s |

Each first disposable clone was then deliberately changed, including deletion of
tools; later fresh clones retained intact tools and no contamination marker.
All twelve published image entries passed provenance validation and an offline
fresh-clone probe. One-time build/commit/probe work across the twelve images totaled
1,139.29 aggregate seconds, excluding final builder cleanup and base-image pulls.
That is summed work across tasks, not elapsed wall-clock time or a cloud bill.
This mode trades upfront build time and disk space for cheaper repeated readiness.

The final [prepared sandbox fixture](prepared_sandbox_smoke.json) ran the unchanged
baseline with scripted replies and 27 real synthetic parameterized test cases in
15.61 seconds. All 27 passed; the grouped CTRF report contained one entry and the
corrected verifier returned reward 1. A [matched failing fixture](prepared_sandbox_assertion_failure_smoke.json)
had 26 passing cases and one failure before later passes; it returned reward 0 in
16.00 seconds. The [negative bootstrap fixture](prepared_sandbox_invalid_verifier_smoke.json)
returned an infrastructure error and null reward for a missing verifier dependency,
in 14.12 seconds. All made zero provider calls and used the unchanged
[verified source manifest](prepared_verified_source_manifest.json), 39 files with
aggregate hash `99ead2b2fc81d10797661254f4f92d980b3732d3f5921a1b1cec49b4f492d2f4`.
Older fixtures are preserved with the `_before_case_count_fix` suffix.

These are **manual execution-engine improvements, not LLM self-improvement**.
The verifier remains unchanged and may still bootstrap over the network. Images
are local to one Docker daemon; they are not a distributed registry or a bit-for-bit
rebuild guarantee. Measurements include amd64 emulation on an aarch64 host.

A [separate development-repeat experiment](live_prepared_development_review_ctrf_grouping_failure.json)
started with a fresh original-agent baseline and the prepared mapping. It was stopped
after discovering a verifier-contract defect: pinned `pytest-json-ctrf` 0.3.5 groups
parameterized cases into function entries. One valid 27-case passing pytest session
therefore produced three passing report groups, which our count-equality guard
incorrectly rejected. The retry cost and all original observations remain recorded;
no historical score was rewritten.

The [interruption audit](live_prepared_development_ctrf_grouping_failure_audit.json)
marks this comparison invalid, preserves 21 attempts and three generated sources,
and verifies complete cleanup. It spent **$0.57680875**, bringing session model
spending to **$4.64451375**, with zero unresolved reservations. Its
[source exports](prepared_interrupted_candidates/index.json) demonstrate real
autonomous proposals but cannot establish superiority or inferiority under the
defective scoring contract. No new held-back outcomes completed. Those task roles
already had four prior observations, so any follow-up is repeated development
validation, not a new blind generalization test.

### Corrected prepared-mode experiment

The [fresh corrected experiment](live_prepared_verified_review.json) started at
13:03:32 UTC on the v2 source manifest above, with a fresh original-agent baseline,
no inherited candidates, scores or rejection memory, a $4 model ceiling and a
25-minute deadline. Its task roles and final promotion thresholds were fixed.
Prior held-back exposure remains disclosed.

Candidate 2 autonomously reduced long tool outputs to the first and last 60 lines
before applying the existing character limit. Its preliminary selection result
was 2/3 versus the baseline's 1/3, despite 12.6% higher p95. The
[recorded confirmation trigger](live_prepared_verified_confirmation_start.json)
shows that the service scheduled fresh matched trials on the quality signal.
It did not promote the candidate from that preliminary result.

The fresh B/C/C/B confirmation completed with two repetitions of three selection
tasks per source, all twelve observations valid and no retries:

| Fresh matched metric | Incumbent | Candidate 2 | Observed change |
|---|---:|---:|---:|
| Passed trials | 3/6 | 2/6 | One fewer pass |
| Mean model cost/task | $0.009667875 | $0.009306250 | −3.74% |
| p95 complete attempt | 58.08 s | 51.31 s | −11.66% |

The service **automatically rejected** this candidate because quality regressed.
The lower observed cost and latency do not make this a balanced improvement.
With six observations per source, nearest-rank p95 is the maximum observation;
these estimates do not establish statistical significance. This is real evidence
of autonomous proposal, application, matched confirmation and a persisted decision.

The [terminal audit](live_prepared_verified_audit.json) confirms completion at
13:24:00 UTC, after **20 minutes 26.6 seconds**, with **40/40 valid observations**,
zero infrastructure errors, timeouts or retries, and all 40 runtime image IDs
matching the frozen mapping. Candidate 1's additional Git-diff feedback and
candidate 3's conditional pytest mechanism were screened out for no diagnostic
repair. A [mechanism audit](live_prepared_verified_candidate3_mechanism.json)
shows candidate 3 took its skip path: the intended automatic pytest verification
did not run. Its code checked both `/tests` and pytest availability; retained
evidence does not distinguish which prerequisite was absent. The generated
rationale is not credited as an implemented performance gain.

All three proposals were rejected and the original source remained selected.
Final held-back development evaluation completed at **1/4 passed** (regex-log).
Baseline and final source were identical, so these same four observations were
explicitly reused, not counted as two independent evaluations. The tasks had
already been observed in an earlier study, so this is not a fresh blind test.
The [source/proposal index](prepared_verified_candidates/index.json) preserves
the baseline and all three exact generated sources as text.

This experiment cost **$0.8211915** in estimated model usage. Cumulative session
spending, including interrupted runs and diagnostics, is **$5.46570525**, with
**$0 unresolved reservations**. All calls used `NEOSIGMA_OPEN_API_KEY`; local
Docker dollars remain unpriced. Cleanup verified stopped owned API/worker
processes, no active jobs or slots, and no owned task containers/networks.
The evidence database and prepared images are intentionally retained.

**Conclusion:** measured preparation work improved, and the real autonomous
optimization/confirmation loop completed. No balanced agent improvement was
accepted. The observed efficiency reductions accompanied lower quality and
must not be presented as a benchmark-quality gain.

### Follow-up with improved diagnostic evidence

The [feedback-only failure audit](optimizer_failure_analysis.json) identified
information loss in the optimizer input: command prefixes hid trailing heredoc
redirections, while rejected candidates supplied aggregates rather than detailed
failure dossiers. It also distinguished actual shell-I/O mistakes from unsupported
causal stories and proposals whose intended mechanism never ran.

The proposer now receives bounded command/output beginnings and endings, fuller
feedback instructions, recent rejected feedback dossiers with provenance, and an
explicit runtime contract. Related-job snapshots retain only authorized compatible
feedback evidence. They inherit neither candidates nor scores. These changes
improve the optimizer's inputs; no candidate was manually patched.

A [separate evidence-guided experiment](live_evidence_guided_review.json) started
at 13:41:29 UTC. Compared with the preceding request, **only `prior_job_id` changed**.
The twelve tasks, split, model/reasoning settings, resource limits, three proposals,
two confirmation repetitions, $4 cap, 25-minute deadline and final quality/cost/p95
policy remain identical. No selection or held-back traces entered the added memory.
Previously observed held-back tasks remain development validation.

The [new source/controller manifest](evidence_guided_controller_manifest.json)
records production hash `a9c1a5ced409f3f08006ae12ab72070ea736846ef6b372972909ed64d6834012`
and an archived source snapshot. Its [real sandbox fixture](evidence_guided_sandbox_smoke.json)
passed 27 parameterized synthetic cases in 18.64 seconds with scripted replies,
zero provider calls and unchanged source. The verifier and evaluation rules were
not modified.

The [terminal audit](live_evidence_guided_audit.json) records **45 valid completed
observations**: 32 initial trials, twelve fresh confirmation trials and one final
held-back observation. Two additional attempts were interrupted and remain unscored;
their settled inference charges are included. There were no infrastructure errors
among completed observations. The original 25-minute deadline was reached at
14:06:29 UTC; the job was cancelled and **final held-back evaluation is incomplete**.
The persisted `budget_exhausted` reason refers to wall-clock time, not the $4 ceiling.
No deadline or scoring threshold was relaxed, and no missing result became a zero
or an inherited pass.

The first new proposal's completeness warning improved observed diagnostic results
from 1/5 to 3/5, but selection stayed 2/3. A 6.81% cost reduction was below the fixed
10% efficiency signal, and p95 was 12.19% higher. The second proposal's artifact and
fresh-shell instructions produced a 1/3 selection result and were rejected for
quality regression. These observations are not accepted performance gains.

Candidate 3 saved captured tool output to sandbox files for later rereading.
Its preliminary 2/3 result matched the baseline with 11.09% lower model cost,
triggering the unchanged fresh B/C/C/B confirmation:

| Fresh matched metric | Incumbent | Candidate 3 | Observed change |
|---|---:|---:|---:|
| Passed trials | 2/6 | 3/6 | One additional pass |
| Mean model cost/task | $0.013355875 | $0.015754250 | +17.96% |
| p95 complete attempt | 74.77 s | 81.69 s | +9.25% |

The service rejected the candidate because model cost exceeded the unchanged 10%
allowance. This is an observed quality gain in a small matched sample, **not an
accepted balanced improvement or statistically established generalization**.
The archive preserves only output already captured by the runtime; it cannot
restore text discarded by shell pipelines or the runtime's own output cap. It
adds recoverability, not context pruning, and rereading can increase cost.

The [exact sources and proposals](evidence_guided_candidates/index.json) are
preserved. All three candidates were rejected; the original baseline remains
selected. Experiment model spending was **$1.2527385**, with **$0 unresolved
reservations**; cumulative spending across our recorded experiments is now
**$6.71844375**. This includes interrupted work and excludes unpriced local Docker
compute and unrelated E2B-branch experiments. All requests used the company key.
Source/controller hashes stayed frozen, and owned processes, task containers,
networks and admission slots were cleaned up. No further paid experiment ran.

## Real metered model completions

[The redacted model smoke report](live_model_smoke.json) records one successful
`gpt-5.4-mini` request to `https://api.openai.com/v1/chat/completions`, using only
`NEOSIGMA_OPEN_API_KEY`. No fallback to personal credentials was enabled. The
report excludes credentials, prompts, completion text, and raw provider bodies.

| Measurement | Observed value |
| --- | ---: |
| Input tokens | 14 |
| Output tokens reported by the provider | 14 |
| Completion latency | 2.025 seconds |
| Reserved upper cost estimate | $0.003945 |
| Settled model cost estimate | $0.0000735 |
| Unresolved reservation afterward | $0 |
| Number of dispatched requests | 1 |

The request had a $0.05 job cap and no automatic retries. The durable ledger
contains exactly one reservation and one settlement. Cost is the configured
standard uncached text-token estimate, not an invoice from the provider.

Reproduction makes a paid request and therefore requires an explicit switch:

```sh
AOS_TEST_DATABASE_URL='postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test' \
  .venv-service/bin/python -m scripts.smoke_company_model --allow-paid
```

The script creates and drops its own test database. It requires the company
credential in the environment and never prints it. Re-running it overwrites the
model smoke report with the new observation.

A separate [tool-bearing diagnostic](live_model_tool_smoke.json) used the
unchanged template's function schema and system/user messages. It returned a
function call with 248 input tokens and 21 output tokens, in 1.214 seconds, at an
estimated $0.0002805. The requested function was not executed by this diagnostic.

This caught a real API compatibility defect: GPT-5.4-mini Chat Completions
rejected `tools` combined with `reasoning_effort=low`. The
[rejected request report](live_model_tool_smoke_rejected.json) preserves the
provider's structural error fields and zero settled spend. The gateway now
explicitly fixes tool-bearing requests to `reasoning_effort=none`; tool-free
optimizer requests use `low`. The successful diagnostic verifies that exact
configuration. Both baseline and candidates use the same configuration. The
[model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
lists `none` as the default; supporting reasoning with tools through a Responses
adapter is future work, rather than an untested feature of this implementation.

```sh
AOS_TEST_DATABASE_URL='postgresql://aos_test:aos_test_local@127.0.0.1:55433/aos_test' \
  .venv-service/bin/python -m scripts.smoke_company_model --allow-paid --with-tools
```

The tool diagnostic has a $0.01 cap. Across the plain request, rejected tool
request, and corrected tool request, the measured model-cost estimate was
$0.000354.

An additional [optimizer-contract preflight](live_optimizer_contract_smoke.json)
verified `reasoning_effort=low` with `response_format=json_object` and no tools.
It returned the expected single-key JSON object with finish reason `stop`, using
31 input tokens and 28 output tokens in 1.235 seconds. It settled an estimated
$0.00014925 against a $0.01 cap, with no unresolved reservation. This confirms
the response-format path; it does not prove that a full agent-source proposal
will be valid or improve performance. Production code remained unchanged during
that preflight.

Across the first four dispatched diagnostics, the measured model-cost estimate was
**$0.00050325**. Each dispatch had its own budget reservation; there were no
automatic retries. This spend is separate from the live benchmark experiment.

A fifth [optimizer-format diagnostic](live_optimizer_large_model_smoke.json)
verified GPT-5.4 with tool-free JSON and low reasoning effort. It used 17 input
and 26 output tokens, took 1.288 seconds, and settled **$0.0004325** with no
outstanding reservation. The total for all five diagnostics is **$0.00093575**.
This proves access and response format, not better optimization performance.

The compatibility failure also exposed a reporting issue: a candidate can catch
a model error and return normally. Trusted gateway state now carries budget,
cancellation, lease-loss, and inference-failure stop reasons back to the executor
so an infrastructure failure cannot be silently treated as cheap successful agent
execution.

## Real benchmark optimization

**No balanced benchmark improvement was established in the completed cold-setup experiments below.** The
[calibrated12 experiment](live_calibrated_benchmark_review.json) completed with a
fresh baseline, the preserved 5/3/4 task roles, mini task execution and GPT-5.4
optimizer proposals. All **36/36 observations had valid executed-test evidence**;
there were no infrastructure errors or timeouts. It finished in 45 minutes 56 seconds.
The [terminal audit](live_calibrated_benchmark_audit.json) verifies the outcome and
cleanup independently: no owned API/worker processes, containers, networks, active
jobs, admission slots or outstanding model reservations remained.

The calibrated run used the recorded `e67f2d81...` service manifest. After it
finished, a narrow scoring correction removed status-to-reward fallback for
ungraded timeouts. An [offline replay](calibrated_metric_replay.json) under the
corrected scorer verified all 36 observations and reproduced every recorded
selection/holdout metric unchanged. This replay made no model calls and is not
a new benchmark repetition. The subsequent paired audit uses the corrected,
separately tested source manifest.

The single-repetition selection observations were:

| Source / change | Passed | Mean model cost/task | p95 attempt time | Automatic decision |
|---|---:|---:|---:|---|
| Original baseline | 1/3 | $0.0069525 | 126.39 s | Retained |
| Candidate 1: one additional verification-request turn | 1/3 | $0.0204005 | 131.39 s | Rejected: cost |
| Candidate 2: path-aware output compaction | 1/3 | $0.0091910 | 135.25 s | Rejected: cost |
| Candidate 3: dependency-portability guidance | 2/3 | $0.01301425 | 120.35 s | Rejected: cost |

These are screening observations, **not fresh paired confirmation**. Candidate 3's
observed quality signal came with 87.2% higher mean model cost and 4.8% lower p95
time; it failed the predeclared 10% cost tolerance. Its cost per observed solved task
was about 6.4% lower, but that ratio uses only one versus two successes and was not
the acceptance metric. No gate was changed afterward. Candidate 1 requests
verification but does not enforce that a verification command actually runs.
The [source/proposal index](calibrated_candidates/index.json) preserves the exact
baseline and all three generated sources as text. Generated rationales are hypotheses,
not independently established causal explanations.

The selected source remained the original baseline. It passed **1/4 holdout tasks**
(tune-mjcf); the other three failed valid tests. Because baseline and selected source
were identical, the same four observations were reused explicitly, not described as
two independent evaluations. No holdout result was returned to the optimizer.

A separate [fixed-source follow-up protocol](calibrated_candidate3_followup_declaration.json)
was declared for candidate 3: two fresh repetitions of each of the same three
selection tasks, alternating B/C/C/B, with the same policy and a $2 / 45-minute cap.
It makes no new proposals and cannot retroactively promote a source in the completed
job. Its source hashes and plan were fixed before the parent coordinator inspected
holdout outcomes; the declaration file accurately records that some holdout results
already existed before the file itself was written. The
[paired follow-up](live_calibrated_candidate3_paired_review.json) completed on the
then-current `7bc8de27...` source manifest; the complete service suite and both full Docker
fixtures passed before it started. All twelve observations had valid verifier
evidence, with no infrastructure errors or timeouts.

| Fresh matched selection metric | Baseline | Candidate 3 | Observed change |
|---|---:|---:|---:|
| Passed trials | 3/6 | 3/6 | Equal |
| Mean model cost/task | $0.015560875 | $0.01294325 | −16.82% |
| Mean end-to-end attempt time | 87.68 s | 96.61 s | +10.19% |
| p95 end-to-end attempt time | 121.74 s | 186.48 s | +53.18% |

The predeclared policy rejected the candidate on p95 latency. The preliminary
quality increase did not repeat across the full matched comparison. Lower observed
model cost is a useful signal, but it is not a balanced performance win or a broad
generalization claim from three tasks and two repetitions. The original job's
decisions remain unchanged, and the audit generated no proposals or holdout trials.

The [paired audit and stage timings](live_calibrated_candidate3_paired_audit.json)
show that the longest candidate attempt included 131.118 seconds of dependency
preflight, 29.098 seconds of agent execution and 14.599 seconds of verification.
Most of that tail observation was setup. This does not establish that candidate
code caused the delay. The full 186.482-second attempt remains in the scoring
metric; no slow observation was removed afterward. This motivated the separately
implemented and measured prepared-tooling mode described above; it does not change
the original experiment's result.

The completed calibrated experiment cost **$0.9720645** and the matched follow-up
cost **$0.17102475**. At that historical cutoff, session model spending was estimated
at **$4.067705**, with **$0 outstanding reservations**. The user subsequently lifted
the original $15 overall cap; the new experiment still has its own $6 ceiling.
All model requests used
`NEOSIGMA_OPEN_API_KEY`. Final cleanup verified no owned API/worker processes,
task containers/networks, active jobs or admission slots. The evidence database is
retained for inspection. Local Docker compute remains unpriced.

The earlier total was **$2.92461575**:

| Recorded experiment | Outcome | Model estimate |
|---|---|---:|
| [Preliminary optimization](live_benchmark_review_diversity.json) | Candidate with two feedback repairs interrupted before selection confirmation; unconfirmed | $1.06154475 |
| [Fresh continuation](live_benchmark_review_verifier_failure.json) | Stopped after detecting invalid verifier bootstrap results | $1.6303005 |
| [Corrected paired audit](live_paired_selection_review_python_bootstrap_failure.json) | Incomplete after two zero-model Python prerequisite timeouts on candidate sampler | $0.23183475 |
| Five model/format diagnostics | Access/format checks only | $0.00093575 |

The original Harbor parser treated a shell-written reward zero as an ordinary
failure even when pytest never started. The [validity audit](live_benchmark_review_validity.json)
found invalid baseline and candidate selection observations. The raw displayed
baseline score therefore cannot support a complete comparison. The corrected
verifier's positive and negative tests above demonstrate the operational fix;
they do not retroactively repair old observations or delete their costs.

In the corrected paired audit, the first baseline repetition completed with
**2/4 passed**, all backed by fresh executed-test evidence, at $0.15938625.
Its p95 task-attempt duration was 857.47 seconds, including the sampler's 184.46
seconds of preparation. Candidate repetition one had three valid results and
two pre-agent infrastructure failures on the remaining task. On the three common
valid tasks only, baseline passed 2/3 at $0.026865 and candidate passed 1/3 at
$0.0724485. This is not the planned complete paired comparison and cannot justify
promotion or a general quality/cost conclusion.

[The infrastructure analysis](live_paired_selection_review_python_bootstrap_analysis.json)
records both 180-second system-Python prerequisite timeouts. They occurred before
agent/model/verifier execution and consumed zero model tokens; their 191.26 and
192.86 seconds of full-attempt time remain recorded. The exact apt/mirror/network
cause was not established because the timed-out command buffer was unavailable.
The audit stopped before repetition two and any holdout. All owned processes,
containers, networks and admission slots were cleaned up without manual recovery.

The new twelve-task cohort is explicitly distinct from these sixteen-task plans.
[Its selection rationale](DESIGN_PLAN.md#task-subset-and-model-choice) explains the
representation and operational exclusions. No old task score is imported into its
baseline. Its results must stand on their own; a changed task mix is not an agent
performance gain. All real timings so far include amd64-on-aarch64 Docker emulation
and substantial preparation variance, limiting transfer to native production workers.

The API also supports related fresh experiments only when benchmark revision,
tasks, split, agent model and original baseline match. It copies bounded rejection
context and optional unconfirmed source as data, never old scores. The calibrated
profile starts a separate experiment rather than bypassing those compatibility rules.

## Architecture artifact

The final [Excalidraw file](../final_architecture.excalidraw) was parsed as valid
JSON and contains 69 native diagram elements. Its rendered
[PNG preview](architecture.png) was visually reviewed. This validates the
deliverable's structure and legibility; runtime properties rely on the separate
tests above.

## Limits of this evidence

- Docker resource limits, network isolation, and credential separation are not
  equivalent to a microVM boundary or proof against kernel vulnerabilities.
- Candidate code runs in the mutable task container later used by the benchmark
  verifier. The current design does not prove resistance to intentional verifier
  tampering. A blind holdout reduces feedback leakage; independent trusted
  verification would be a further boundary.
- Trial-slot expiry depends on sandbox lifetime enforcement and cleanup. The
  database alone cannot prove that an unreachable provider has terminated work.
- A small fixed task subset and repeated trials support observations about that
  experiment; they do not establish broad Terminal-Bench generalization.
- Sandbox monetary cost is unpriced locally. Runtime and model cost are measured
  separately, and unknown totals are not presented as free execution.
