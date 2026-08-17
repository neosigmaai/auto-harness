"""Pydantic wire schemas (server side).

These serialize the domain model to/from JSON. Kept deliberately small; the
job/iteration read schemas grow in M1/M4 as the endpoints are built out.
"""

from __future__ import annotations

from pydantic import BaseModel

from harness_service import __version__


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    db: str  # "ok" | "error: <msg>"
    worker_enabled: bool
