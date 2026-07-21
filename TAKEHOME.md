# Take-home: Terminal-Bench test environment

This branch pins a **12-task Terminal-Bench 2.0 subset** so the agent-optimization
service can be reviewed without running the full 89-task benchmark.

## Selected tasks

| Split | Tasks |
|-------|-------|
| **train** (8) | `regex-log`, `extract-elf`, `log-summary-date-ranges`, `openssl-selfsigned-cert`, `sqlite-db-truncate`, `fix-code-vulnerability`, `fix-git`, `filter-js-from-html` |
| **test** (4) | `cancel-async-tasks`, `gcode-to-text`, `git-leak-recovery`, `cobol-modernization` |

Pinned in [`tbench_data/task_split.json`](tbench_data/task_split.json).

### Why these tasks

- **Fast enough to iterate**: shorter wall-clock than heavy tasks (`qemu-alpine`,
  `extract-moves-from-video`, `hf-model-inference`, etc.).
- **Representative domains**: logs/text, crypto/certs, sqlite, security fixes,
  git recovery, async Python, gcode, COBOL modernization.
- **Optimization signal**: baseline agent does not solve all of them, so the
  failure → propose → re-run loop has room to improve.

A smaller default used during service smoke tests is the first three train tasks
(`regex-log`, `extract-elf`, `log-summary-date-ranges`).

## How to use this branch

```bash
cp experiment_config.takehome.yaml experiment_config.yaml
# set OPENAI_API_KEY + DAYTONA_API_KEY in .env
python prepare.py   # uses the committed split; does not re-run all 89 tasks
```

Or pass task IDs explicitly:

```bash
python benchmark.py --task-ids regex-log extract-elf log-summary-date-ranges
```

## Updating `agent/agent.py` from the optimization service

During a job, the service **does not commit** to this repo. It:

1. Clones this branch into a per-job workspace
2. Overwrites `agent/agent.py` with each `AgentVersion` before `harbor run`
3. Stores every proposal (accepted and rejected) in Postgres

To promote the best agent back onto this branch after a job completes:

```bash
# From the optimization service API:
#   GET /jobs/{job_id}                         → best_agent_version_no
#   GET /jobs/{job_id}/agent-versions/{n}       → .content is full agent.py

curl -s -H "Authorization: Bearer $TOKEN" \
  "$API/jobs/$JOB_ID/agent-versions/$BEST_NO" \
  | jq -r .content > agent/agent.py

# Optional: also refresh the starting template so prepare.py starts from it
cp agent/agent.py agent/templates/terminal_bench.py

git add agent/agent.py agent/templates/terminal_bench.py
git commit -m "Promote optimized Terminal-Bench agent from job $JOB_ID"
```

Diff against the stock template anytime:

```bash
diff agent/templates/terminal_bench.py agent/agent.py
```
