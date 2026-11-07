"""Curated Terminal-Bench 2.0 task subset.

Generated from each task's task.toml (timeouts, resources, pinned image) and the public
leaderboard dataset (harborframework/terminal-bench-2-leaderboard; mixed-strength average solve
rate across 27 submissions). Rationale for the selection lives in docs/DESIGN_PLAN.md §10.

role:
  canary    - near-certain pass for the baseline agent; detects regressions, no learning signal
  signal    - 25-80 % average solve rate; where an optimisation loop can win tasks
  alternate - same constraints; swapped in if a signal task proves degenerate in calibration
"""
from __future__ import annotations

TASKS: list[dict] = [
    {
        "name": "fix-git",
        "role": "canary",
        "category": "software-engineering",
        "difficulty": "easy",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/fix-git:20251031",
        "leaderboard_solve_rate": 95.9,
        "why": "git state recovery; regression detector"
    },
    {
        "name": "regex-log",
        "role": "canary",
        "category": "data-processing",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/regex-log:20251031",
        "leaderboard_solve_rate": 88.6,
        "why": "exact-spec reading, self-verification"
    },
    {
        "name": "nginx-request-logging",
        "role": "canary",
        "category": "system-administration",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/nginx-request-logging:20251031",
        "leaderboard_solve_rate": 95.9,
        "why": "service config + verify loop"
    },
    {
        "name": "largest-eigenval",
        "role": "signal",
        "category": "mathematics",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/largest-eigenval:20251031",
        "leaderboard_solve_rate": 69.9,
        "why": "numeric precision, output-format discipline"
    },
    {
        "name": "sparql-university",
        "role": "signal",
        "category": "data-querying",
        "difficulty": "hard",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/sparql-university:20251031",
        "leaderboard_solve_rate": 69.1,
        "why": "query construction, iterative checking"
    },
    {
        "name": "extract-elf",
        "role": "signal",
        "category": "file-operations",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/extract-elf:20251031",
        "leaderboard_solve_rate": 57.0,
        "why": "binary tooling; needs env bootstrapping"
    },
    {
        "name": "sanitize-git-repo",
        "role": "signal",
        "category": "security",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/sanitize-git-repo:20251031",
        "leaderboard_solve_rate": 56.6,
        "why": "multi-step history rewriting; easy to stop early"
    },
    {
        "name": "cancel-async-tasks",
        "role": "signal",
        "category": "software-engineering",
        "difficulty": "hard",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/cancel-async-tasks:20251031",
        "leaderboard_solve_rate": 55.3,
        "why": "concurrency semantics; strict tests"
    },
    {
        "name": "tune-mjcf",
        "role": "signal",
        "category": "scientific-computing",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/tune-mjcf:20251031",
        "leaderboard_solve_rate": 53.7,
        "why": "parameter search against a simulator"
    },
    {
        "name": "configure-git-webserver",
        "role": "signal",
        "category": "system-administration",
        "difficulty": "hard",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/configure-git-webserver:20251031",
        "leaderboard_solve_rate": 52.9,
        "why": "multi-service setup; verification matters"
    },
    {
        "name": "chess-best-move",
        "role": "signal",
        "category": "games",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/chess-best-move:20251031",
        "leaderboard_solve_rate": 47.2,
        "why": "reasoning + installing/using a tool"
    },
    {
        "name": "adaptive-rejection-sampler",
        "role": "signal",
        "category": "scientific-computing",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/adaptive-rejection-sampler:20251031",
        "leaderboard_solve_rate": 41.0,
        "why": "algorithm implementation from spec"
    },
    {
        "name": "polyglot-c-py",
        "role": "signal",
        "category": "software-engineering",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/polyglot-c-py:20251031",
        "leaderboard_solve_rate": 35.0,
        "why": "constrained code generation; headroom"
    },
    {
        "name": "gcode-to-text",
        "role": "signal",
        "category": "file-operations",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/gcode-to-text:20251031",
        "leaderboard_solve_rate": 25.4,
        "why": "format reverse-engineering; headroom"
    },
    {
        "name": "sqlite-db-truncate",
        "role": "signal",
        "category": "debugging",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/sqlite-db-truncate:20251031",
        "leaderboard_solve_rate": 77.1,
        "why": "forensic recovery; careful inspection"
    },
    {
        "name": "openssl-selfsigned-cert",
        "role": "signal",
        "category": "security",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/openssl-selfsigned-cert:20251031",
        "leaderboard_solve_rate": 80.5,
        "why": "exact-spec command crafting"
    },
    {
        "name": "model-extraction-relu-logits",
        "role": "alternate",
        "category": "mathematics",
        "difficulty": "hard",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/model-extraction-relu-logits:20251031",
        "leaderboard_solve_rate": 24.2,
        "why": ""
    },
    {
        "name": "build-pmars",
        "role": "alternate",
        "category": "software-engineering",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/build-pmars:20251031",
        "leaderboard_solve_rate": 70.7,
        "why": ""
    },
    {
        "name": "count-dataset-tokens",
        "role": "alternate",
        "category": "model-training",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/count-dataset-tokens:20251031",
        "leaderboard_solve_rate": 76.9,
        "why": ""
    },
    {
        "name": "build-cython-ext",
        "role": "alternate",
        "category": "debugging",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/build-cython-ext:20251031",
        "leaderboard_solve_rate": 76.4,
        "why": ""
    },
    {
        "name": "password-recovery",
        "role": "alternate",
        "category": "security",
        "difficulty": "hard",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/password-recovery:20251031",
        "leaderboard_solve_rate": 76.4,
        "why": ""
    },
    {
        "name": "kv-store-grpc",
        "role": "alternate",
        "category": "software-engineering",
        "difficulty": "medium",
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 900,
        "cpus": 1,
        "memory_mb": 2048,
        "docker_image": "alexgshaw/kv-store-grpc:20251031",
        "leaderboard_solve_rate": 84.6,
        "why": ""
    }
]

DEFAULT_TASKS: list[str] = [t["name"] for t in TASKS if t["role"] in ("canary", "signal")]
ALL_TASK_NAMES: set[str] = {t["name"] for t in TASKS}
BY_NAME: dict[str, dict] = {t["name"]: t for t in TASKS}
