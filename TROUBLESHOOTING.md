# Auto-Harness Troubleshooting Guide

Common issues and solutions when setting up and running auto-harness.

---

## Setup Issues

### Docker daemon not running

**Symptom:**
```
failed to connect to the docker API at unix:///Users/bobshah/.docker/run/docker.sock
```

**Solution:**
1. Start Docker Desktop: `open -a Docker` (macOS) or start from Applications
2. Wait 10-15 seconds for Docker to fully start
3. Verify with: `docker ps`
4. Retry: `docker compose build`

---

### Missing OPENAI_API_KEY

**Symptom:**
```
[prepare] ERROR: missing env vars: OPENAI_API_KEY
          Copy .env.example to .env and fill in the values.
```

**Solution:**
1. Copy the template: `cp .env.example .env`
2. Get an OpenAI API key from https://platform.openai.com/api-keys
3. Edit `.env` and add your key:
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
4. Retry: `docker compose run autoeval python prepare.py`

**Note:** Auto-harness uses OpenAI's API for the agent under optimization. You'll need credits in your OpenAI account to run benchmarks.

---

### What is TAU2_DATA_DIR?

**Question:** The `.env.example` file has `TAU2_DATA_DIR=` but the README says it's "auto-cloned by prepare.py". Can this be left empty?

**Answer:** Yes, leave it empty for first-time setup. The `prepare.py` script will:
1. Check if `TAU2_DATA_DIR` is set in `.env`
2. If empty, it will clone tau2-bench data to a default location
3. The data will be stored inside the Docker container at `/app/workspace/tau2_data`

**Optional:** If you want to reuse tau2 data across runs or share it with other projects:
1. Clone tau2-bench data locally: `git clone https://github.com/sierra-research/tau2-bench-data ~/tau2-data`
2. Set in `.env`: `TAU2_DATA_DIR=/path/to/tau2-data`
3. Update `docker-compose.yml` to mount this directory

---

## Model Configuration Issues

### Model "gpt-5.4" doesn't exist

**Symptom:**
The `experiment_config.yaml` template uses `agent_model: "gpt-5.4"`, but this model doesn't exist yet.

**Solution:**
Edit `experiment_config.yaml` and use a real OpenAI model:

```yaml
# For best performance (expensive):
agent_model: "gpt-4o"

# For balanced cost/performance:
agent_model: "gpt-4o-mini"

# For budget testing:
agent_model: "gpt-4-turbo"
```

**Supported models:** Any model supported by OpenAI's API. See https://platform.openai.com/docs/models

---

## Benchmark Issues

### Benchmark runs slowly

**Symptom:**
`python benchmark.py` takes a very long time (>10 minutes)

**Causes & Solutions:**

1. **Too many tasks:** The default "retail" domain might have many tasks
   - **Solution:** Start with a few tasks: `python benchmark.py --task-ids 0 1 2`
   
2. **Low concurrency:** Default `max_concurrency: 3` in `experiment_config.yaml`
   - **Solution:** Increase if you have API rate limit headroom: `max_concurrency: 10`
   
3. **Slow model:** Complex models take longer per task
   - **Solution:** Use `gpt-4o-mini` for faster iterations

---

### Out of OpenAI credits

**Symptom:**
```
Error: insufficient_quota
```

**Solution:**
1. Check your OpenAI account balance: https://platform.openai.com/usage
2. Add credits if needed
3. Consider using `gpt-4o-mini` to reduce costs

**Cost estimate:**
- One full benchmark run (e.g., retail domain, ~20 tasks): $0.50-$2.00 depending on model
- One iteration with gating (full test + regression suite): $1.00-$5.00
- Overnight optimization (10-20 iterations): $10-$100

---

## Docker Issues

### Container takes too much disk space

**Symptom:**
Docker images and volumes accumulate over time

**Solution:**
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Full cleanup (WARNING: removes all unused Docker resources)
docker system prune -a --volumes -f
```

---

### Cannot mount workspace directory

**Symptom:**
```
Error: cannot mount /path/to/auto-harness/workspace
```

**Solution:**
1. Ensure the `workspace/` directory exists: `mkdir -p workspace`
2. Check Docker Desktop → Settings → Resources → File Sharing
3. Ensure your project directory is in an allowed path
4. On macOS: `/Users/` should be allowed by default

---

## Runtime Issues

### Agent keeps failing the gate

**Symptom:**
```
[gating] FAILED: eval suite pass rate 65% < threshold 80%
```

**Understanding:**
The agent's changes made some previously-passing tasks fail. This is the gating mechanism working correctly.

**Solutions:**

1. **Lower threshold temporarily** (for initial testing):
   ```yaml
   # In experiment_config.yaml
   threshold: 0.5  # Allow more regressions during early exploration
   ```

2. **Investigate failures:**
   ```bash
   # Run specific failing tasks
   docker compose run autoeval python benchmark.py --task-ids 3 7 12
   ```

3. **Review workspace/learnings.md:**
   The agent logs what it tried and why it failed there.

---

### How long should one iteration take?

**Expected times:**

- **prepare.py:** 30-60 seconds (clones tau2 data, initializes workspace)
- **benchmark.py (full):** 3-10 minutes depending on domain size and model
- **benchmark.py (3 tasks):** 30-90 seconds
- **gating.py:** 5-15 minutes (runs eval suite + full test)
- **One optimization iteration:** 10-20 minutes (analyze → improve → gate → record)

**Rule of thumb:** If any step takes >30 minutes, something is wrong (check API rate limits, model timeouts, network issues).

---

## Agent Behavior Issues

### Agent doesn't follow PROGRAM.md

**Symptom:**
You prompted the coding agent (Claude Code, Codex) with "Read PROGRAM.md and start the optimization loop", but it's not following the instructions.

**Solutions:**

1. **Be more explicit:**
   ```
   Read PROGRAM.md carefully. This defines your optimization loop. 
   Your goal is to improve agent/agent.py to score higher on the benchmark.
   
   Step 1: Run `python benchmark.py` and analyze the failures.
   Step 2: Edit agent/agent.py to fix one failure pattern.
   Step 3: Run `python gating.py` to verify your change doesn't break existing tests.
   Step 4: If gating passes, run `python record.py` to log the result.
   Step 5: Repeat.
   
   Start now with Step 1.
   ```

2. **Use a stronger model:**
   - Claude 3.5 Sonnet (via Claude Code) works well
   - GPT-4o (for your own agent runtime) also works
   - Weaker models may not follow complex multi-step instructions

3. **Check workspace structure:**
   - Ensure `PROGRAM.md` is in the repo root
   - Ensure `agent/agent.py` exists

---

### Agent makes changes to wrong files

**Symptom:**
The agent edits `benchmark.py`, `gating.py`, or other infrastructure files instead of just `agent/agent.py`.

**Solution:**
Add this to your prompt:
```
IMPORTANT: You may ONLY edit agent/agent.py. 
All other files (benchmark.py, gating.py, prepare.py, record.py, PROGRAM.md, workspace/*) are read-only infrastructure. 
Do not modify them under any circumstances.
```

---

## Platform-Specific Issues

### macOS: Permission denied on workspace/

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/app/workspace/...'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 workspace/

# If using Docker Desktop with file sharing, ensure workspace/ is in a shared location
```

---

### Linux: Docker requires sudo

**Symptom:**
```
permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and back in for the change to take effect
# Or use: newgrp docker

# Verify
docker ps
```

---

## Getting Help

### Before asking for help, provide:

1. **Your platform:** macOS / Linux / Windows + version
2. **Docker version:** `docker --version`
3. **Python version:** `python --version` (should be 3.12 in container)
4. **The command that failed** (exact command)
5. **The full error message** (not just the last line)
6. **Your .env settings** (with API keys redacted)
7. **Your experiment_config.yaml** (full file)

### Where to ask:

- **GitHub Issues:** https://github.com/neosigmaai/auto-harness/issues
- **Pull Requests:** If you found a bug and have a fix, submit a PR!

---

## Tips for Success

### Start small

Don't try to optimize on the full benchmark immediately:

```bash
# Test with 3 tasks first
docker compose run autoeval python benchmark.py --task-ids 0 1 2

# Once working, scale up
docker compose run autoeval python benchmark.py
```

### Use cheaper models for testing

While developing/debugging, use `gpt-4o-mini` to save costs. Switch to `gpt-4o` or better for serious optimization runs.

### Monitor costs

Check your OpenAI usage frequently: https://platform.openai.com/usage

Set a monthly budget limit in your OpenAI account to avoid surprises.

### Keep learnings.md

The `workspace/learnings.md` file is gold. Read it after each iteration to understand what the agent learned.

---

## Additional Resources

- **tau-bench documentation:** https://github.com/sierra-research/tau2-bench
- **OpenAI API docs:** https://platform.openai.com/docs
- **Docker Compose reference:** https://docs.docker.com/compose/
- **uv (Python package manager):** https://github.com/astral-sh/uv

---

*Found an issue not covered here? Please open a GitHub issue or submit a PR to improve this guide!*
