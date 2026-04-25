"""System-agent service for the auto-harness BIRD-Interact integration."""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.helpers.bird_interact.bird_adk_runtime import AdkRuntime
# Provided by the external BIRD-Interact-ADK repo. BirdInteractRunner starts this
# service with cwd=self.adk_dir so the ADK's shared/ package is importable.
from shared.config import settings

logger = logging.getLogger(__name__)
app = FastAPI(title="auto-harness BIRD System Agent", version="1.0.0")
runtime = AdkRuntime()


class SessionInitRequest(BaseModel):
    task_id: str
    mode: str = "a-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class SessionRunRequest(BaseModel):
    task_id: str
    message: str
    mode: str = "a-interact"


@app.post("/init_session")
async def init_session(req: SessionInitRequest):
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.init_session(
        task_id=req.task_id,
        mode=req.mode,
        state=req.state,
        reset=req.reset,
    )


@app.post("/run_session")
async def run_session(req: SessionRunRequest):
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.run_turn(
        task_id=req.task_id,
        mode=req.mode,
        message=req.message,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "auto_harness_bird_system_agent",
        "model": settings.system_agent_model,
        "adk_available": runtime.available,
        "adk_error": runtime.error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.system_agent_port)
