"""Auto-Harness optimization service.

A FastAPI + Postgres backend that runs the TerminalBench agent against a task
subset, observes failures, and (M4) uses an LLM to iteratively improve the agent.

Layering (see PLAN.md §3):
    constants   closed vocabularies + tunable defaults
    domain/     pure in-memory state model (no DB, no HTTP)
    db/         SQLAlchemy persistence, 1:1 with domain
    executors/  pluggable agent execution (simulated | harbor sandbox)
    api/        FastAPI app, schemas, routes, auth deps
    worker.py   background job processor (claims queued jobs from the DB)
"""

__version__ = "0.1.0"
