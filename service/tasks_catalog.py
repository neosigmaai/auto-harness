"""Curated Terminal-Bench 2.0 task subset.

Task metadata and solve-rate proxies were inherited from the proposed design.
Solve rates are historical mixed-agent heuristics, not measurements of our model.
The pinned task revision is authoritative; measured runtime is reported separately.
Selection rationale and limitations live in docs/DESIGN_PLAN.md.

role:
  canary    - easier historical proxy; regression detector, baseline pass not guaranteed
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

STRESS_TASKS: list[str] = [t["name"] for t in TASKS if t["role"] in ("canary", "signal")]
STRESS_SPLIT: dict[str, list[str]] = {
    "feedback": ["fix-git", "openssl-selfsigned-cert", "sqlite-db-truncate", "extract-elf",
                 "sanitize-git-repo", "configure-git-webserver", "chess-best-move", "gcode-to-text"],
    "selection": ["nginx-request-logging", "largest-eigenval", "cancel-async-tasks",
                  "adaptive-rejection-sampler"],
    "holdout": ["regex-log", "sparql-university", "tune-mjcf", "polyglot-c-py"],
}
CALIBRATED_SPLIT: dict[str, list[str]] = {
    "feedback": ["fix-git", "openssl-selfsigned-cert", "sqlite-db-truncate", "extract-elf",
                 "sanitize-git-repo"],
    "selection": ["nginx-request-logging", "largest-eigenval", "cancel-async-tasks"],
    "holdout": ["regex-log", "sparql-university", "tune-mjcf", "polyglot-c-py"],
}
CALIBRATED_TASKS: list[str] = [task for group in CALIBRATED_SPLIT.values() for task in group]
CALIBRATION_EXCLUSIONS: dict[str, str] = {
    "chess-best-move": "Representation mismatch for the current text-only agent interface.",
    "gcode-to-text": "Geometry reconstruction is outside this small text-only execution-loop calibration.",
    "adaptive-rejection-sampler": "Repeated pre-agent dependency bootstrap failures and large observed setup/runtime.",
    "configure-git-webserver": "Coupled multi-service setup obscures execution-loop changes in a small calibration budget.",
}
DEFAULT_PROFILE = "calibrated12"
PROFILES: dict[str, dict] = {
    "calibrated12": {
        "tasks": CALIBRATED_TASKS,
        "evaluation_split": CALIBRATED_SPLIT,
        "why": "A budget-conscious calibration of the current text-only execution loop across five feedback, "
               "three selection and four untouched holdout tasks. Exclusions are operational/interface choices, "
               "not evidence of better benchmark performance; this is not a representative Terminal-Bench score.",
        "exclusions": CALIBRATION_EXCLUSIONS,
    },
    "stress16": {
        "tasks": STRESS_TASKS,
        "evaluation_split": STRESS_SPLIT,
        "why": "The original wider stress set, retaining representation, dependency and multi-service challenges.",
        "exclusions": {},
    },
}
DEFAULT_TASKS: list[str] = list(CALIBRATED_TASKS)
ALL_TASK_NAMES: set[str] = {t["name"] for t in TASKS}
BY_NAME: dict[str, dict] = {t["name"]: t for t in TASKS}

for task in TASKS:
    task["profiles"] = [name for name, profile in PROFILES.items() if task["name"] in profile["tasks"]]
    task["evaluation_roles"] = {
        name: role for name, profile in PROFILES.items()
        for role, names in profile["evaluation_split"].items() if task["name"] in names
    }
