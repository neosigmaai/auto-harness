# Agent Optimization Service

A backend service that runs the auto-harness Terminal-Bench agent against a task subset,
mines the failures, asks an LLM for an improvement, applies it, and re-runs — tracking the
full iteration history per job, across multiple organisations.

FastAPI + PostgreSQL. The API and the worker never talk to each other: Postgres is the
entire channel, so every handoff is a transaction and there is no state in flight to lose.

> **Status.** All five milestones are implemented and `mode: "real"` has been run end to
> end against live Terminal-Bench tasks in E2B. 

---

## Run it

Everything below is verified from a clean database on macOS (Apple silicon). **The only
prerequisite is `uv`** — no Docker, no container runtime, no system Postgres, no
`brew install` beyond uv itself. PostgreSQL 16 comes from the `pgserver` Python package as
a real server binary, so `DATABASE_URL` is optional and there is no migration step.

```bash
# 0. once
brew install uv

cd service
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pgserver
```

```bash
# 1. create an org and its first admin key — printed once, stored only as a hash
export PGDATA_DIR=/tmp/aos-dev
.venv/bin/python seed.py demo
```

```bash
.venv/bin/uvicorn api:app --port 8000
.venv/bin/python worker.py
```

```bash
cd ..
python3 test_client.py --key ao_XXXXXXXXXXXXXXXX
```

`test_client.py` is stdlib-only on purpose — no virtualenv, no `pip install`, any Python
3.9+ will run it.



### Running it for real

`mode: "real"` runs the agent inside an E2B sandbox against live Terminal-Bench tasks and
calls an LLM for each improvement. It costs real money and real time — budget roughly two
to four minutes per task per iteration.

```bash
set -a; . ../.env; set +a          
.venv/bin/python worker.py
```

```bash
python3 test_client.py --key ao_XXXX --real --max-iterations 2
```

Or aim it at a few named tasks to keep the bill small:

```bash
curl -X POST localhost:8000/jobs -H "Authorization: Bearer ao_XXXX" \
  -H 'content-type: application/json' \
  -d '{"mode":"real","max_iterations":1,
       "task_ids":["fix-git","prove-plus-comm","cobol-modernization"]}'
```

One sandbox per job, reused across iterations and killed once. Nothing needs installing
locally for this: harbor lives inside the sandbox, not on your machine.


### Seeing what the LLM was given

The failure evidence and the assembled prompt are both readable over the API, which is the
fastest way to tell a bad proposal from a bad prompt:

```bash
curl -s localhost:8000/jobs/$JOB/iterations -H "Authorization: Bearer ao_XXXX"
curl -s localhost:8000/jobs/$JOB/iterations/1/optimizer-input -H "Authorization: Bearer ao_XXXX"
```

---

## API

Auth is `Authorization: Bearer <key>` on everything. There is no signup endpoint: the seed
script mints the first admin, and admins mint members.

| | | |
|---|---|---|
| `POST` | `/jobs` | 202 + `job_id`; `Idempotency-Key` header is honoured |
| `GET` | `/jobs/{id}` | status, usage counters, latest results, final outcome |
| `GET` | `/jobs/{id}/iterations` | the full history — agent source, proposal, results, failures |
| `GET` | `/jobs/{id}/iterations/{n}/optimizer-input` | the exact prompt that produced iteration n |
| `GET` | `/jobs?status=running` | member: own jobs · admin: the whole org |
| `POST` | `/jobs/{id}/cancel` | cooperative; checked at iteration boundaries |
| `POST` | `/orgs/members` | admin only; returns the new key once, ever |
| `GET` | `/health` | |

```jsonc
POST /jobs
{ "task_ids": ["hello-world", "regex-log"],   // optional; defaults to the pinned subset
  "max_iterations": 5,                        // server-capped
  "mode": "mock" }                            // "mock" runs the whole lifecycle free
```

Interactive docs at `http://localhost:8000/docs`.

---

## 1. Diagram



```
┌────────────────────────────────────────────────────────────────────────────────┐
│  TRUSTED  —  our processes, our secrets, no untrusted code executes here       │
│                                                                                │
│   ┌────────────┐   Authorization: Bearer ao_…         ┌──────────────────┐     │
│   │  client    │ ────────────────────────────────────>│   FastAPI        │     │
│   │ test_client│ <────── 202 + job_id, then polls ────│   (api process)  │     │
│   └────────────┘                                      └────────┬─────────┘     │
│                                                                │               │
│                                        INSERT job / SELECT job │               │
│                                                                ▼               │
│                                                    ┌───────────────────────┐   │
│                                                    │     PostgreSQL        │   │
│                                                    │  orgs users jobs      │   │
│                                                    │       iterations      │   │
│                                                    └───────────┬───────────┘   │
│                                    SELECT … FOR UPDATE SKIP LOCKED │           │
│                                                                ▼               │
│                                                    ┌───────────────────────┐   │
│                                                    │      worker           │   │
│                                                    │  ├ claim loop         │   │
│                                                    │  ├ optimizer (1 LLM   │──────> OpenAI
│                                                    │  │   call, no tools)  │   │  (proposals)
│                                                    │  ├ canary (1 task)    │   │
│                                                    │  ├ benchmark.py source│   │
│                                                    │  └ PLATFORM_OPENAI_KEY│   │
│                                                    │    E2B_API_KEY        │   │
│                                                    └───────────┬───────────┘   │
└────────────────────────────────────────────────────────────────┼───────────────┘
                                                                 │
              e2b.Sandbox.create() / files.write() / commands.run()
                                                                 │
   ══════════════════════════════ TRUST BOUNDARY ════════════════╪═══════════════
                                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  UNTRUSTED  —  LLM-authored code executes here                                 │
│                                                                                │
│   ┌──────────────────────────────────────────┐                                 │
│   │  OUTER SANDBOX — one per JOB, reused     │                                 │
│   │  across iterations. /home/user/harness   │                                 │
│   │  (runs as `user`, not root)              │                                 │
│   │                                          │                                 │
│   │   uv → harbor[e2b]   ~8s, once per job   │                                 │
│   │   benchmark.py       written by worker   │                                 │
│   │   agent/agent.py  ← rewritten each iter  │ ───────────────────────────────────> OpenAI
│   │                      THE AGENT'S LOOP    │                                 │  (agent's own
│   │                      LIVES HERE          │                                 │   LLM calls,
│   │   workspace/  wiped between iterations   │                                 │   hundreds)
│   │   env: E2B_API_KEY, PLATFORM_OPENAI_KEY  │                                 │
│   └────────────────────┬─────────────────────┘                                 │
│                        │  E2B API — siblings                                   │
│         ┌──────────────┼──────────────┬───────────────┐                        │
│         ▼              ▼              ▼               ▼                        │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│   │ task 1   │   │ task 2   │   │ task 3   │ … │ task N   │  one per task, per │
│   │ container│   │ container│   │ container│   │ container│  iteration; from   │
│   │          │   │          │   │          │   │          │  the dataset image │
│   │ bash     │   │ bash     │   │ bash     │   │ bash     │  ← ONLY the command│
│   │ strings  │   │ strings  │   │ strings  │   │ strings  │    strings arrive  │
│   │ ───────  │   │ ───────  │   │ ───────  │   │ ───────  │    here, never the │
│   │ VERIFIER │   │ VERIFIER │   │ VERIFIER │   │ VERIFIER │    agent itself    │
│   │ (hidden  │   │          │   │          │   │          │                    │
│   │  pytest) │   │          │   │          │   │          │  injected after the│
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘  agent stops       │
└────────────────────────────────────────────────────────────────────────────────┘
```


Correction:

So part that I misunderstood was the harbor. As harbor was a CL I thought it is executing bash commands only with the subprocess but with the terminal bench runner creating that harbor for me runs that agent and sandbox task for the task, so yes harbor being ran in outer sandbox agent.py is running in outer sandbox and the sandbox it creates runs the bash command which it gives reposne back to harbor to go to agent.py for multiturn as well.