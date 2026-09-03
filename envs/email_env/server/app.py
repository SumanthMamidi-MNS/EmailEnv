"""
envs/email_env/server/app.py

FastAPI OpenEnv Server exposing standard OpenEnv HTTP endpoints for EmailEnvironment.
Supports isolated containerized execution, OpenAPI discovery, and remote client connections.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from envs.email_env.models import EmailAction, EmailObservation, EmailState
from envs.email_env.server.environment import EmailEnvironment

app = FastAPI(
    title="OpenEnv Email Environment Server",
    description="Isolated environment server for evaluating autonomous email triage agents.",
    version="1.0.0",
)

# Global in-memory environment instance for the server session
env = EmailEnvironment()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: str = "STARTER"
    count: Optional[int] = None


class StepResponse(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Returns standard OpenEnv healthy status."""
    return {"status": "healthy", "environment": "email_env"}


@app.get("/metadata")
def get_metadata() -> Dict[str, Any]:
    """Returns standard OpenEnv environment metadata."""
    return {
        "name": "email_env",
        "description": "A Flight Simulator for Autonomous Email Agents",
        "version": "1.0.0",
        "author": "Antigravity Team",
    }


@app.get("/schema")
def get_schema() -> Dict[str, Any]:
    """Returns action, observation, and state JSON schemas."""
    return {
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
        "state": EmailState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> Dict[str, Any]:
    """Standard OpenEnv MCP / JSON-RPC endpoint."""
    try:
        payload = await request.json()
        req_id = payload.get("id", 1) if isinstance(payload, dict) else 1
    except Exception:
        req_id = 1
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "email_env",
            "status": "ready",
            "tools": [
                {"name": "read_email", "description": "Opens and reads an email."},
                {"name": "classify_email", "description": "Assigns classification label."},
                {"name": "reply_email", "description": "Sends a reply to an email."},
                {"name": "archive_email", "description": "Archives an email."},
                {"name": "escalate_email", "description": "Escalates a critical incident."},
            ],
        },
    }


@app.post("/reset", response_model=EmailObservation)
def reset(req: ResetRequest = ResetRequest()) -> EmailObservation:
    try:
        return env.reset(seed=req.seed, difficulty=req.difficulty, count=req.count)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/step", response_model=StepResponse)
def step(action: EmailAction) -> StepResponse:
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.get("/state", response_model=EmailState)
def get_state() -> EmailState:
    try:
        return env.state()
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
