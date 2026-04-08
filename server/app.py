"""
app.py — FastAPI server for Email Triage Environment
Exposes the EmailTriageEnv as a REST API so any agent/client
can interact with it over HTTP.

Endpoints:
  POST /env/create        → create a new env session
  POST /env/{id}/reset    → reset env, get first observation
  POST /env/{id}/step     → submit an action, get obs + reward
  GET  /env/{id}/state    → inspect current env state
  POST /env/{id}/close    → close/cleanup the env session
  GET  /health            → health check
"""

import uuid
import asyncio
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from email_env import EmailTriageEnv
from model import Action, Observation, Reward

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Email Triage Environment API",
    description="OpenEnv-compatible REST API for the email-triage benchmark.",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────
# { session_id: EmailTriageEnv }
sessions: Dict[str, EmailTriageEnv] = {}

# ── Request / Response Schemas ────────────────────────────

class CreateEnvRequest(BaseModel):
    task_level: str = "easy"      # "easy" | "medium" | "hard"
    max_emails: int = 10

class CreateEnvResponse(BaseModel):
    session_id: str
    task_level: str
    max_emails: int
    message: str

class StepRequest(BaseModel):
    relevance: str                 # "relevant" | "not_relevant"
    priority: Optional[str] = None
    reason: Optional[str] = None

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: dict
    message: str

# ── Helper ────────────────────────────────────────────────

def get_session(session_id: str) -> EmailTriageEnv:
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /env/create first."
        )
    return env

def obs_to_dict(obs: Observation) -> dict:
    return obs.model_dump()

# ── Routes ────────────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with simple instructions."""
    return "<h1>Email Triage API is Running!</h1><p>Visit <a href='/docs'>/docs</a> to see the interactive API documentation and test endpoints.</p>"

@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/env/create", response_model=CreateEnvResponse)
async def create_env(req: CreateEnvRequest):
    """
    Create and initialize a new environment session.
    Returns a session_id to use in all subsequent calls.
    """
    valid_levels = {"easy", "medium", "hard"}
    if req.task_level not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"task_level must be one of {valid_levels}"
        )

    session_id = str(uuid.uuid4())
    env = await EmailTriageEnv.from_env(
        task_level=req.task_level,
        max_emails=req.max_emails
    )
    sessions[session_id] = env

    return CreateEnvResponse(
        session_id=session_id,
        task_level=req.task_level,
        max_emails=req.max_emails,
        message="Environment created. Call POST /env/{session_id}/reset to begin."
    )


@app.post("/env/{session_id}/reset", response_model=ResetResponse)
async def reset_env(session_id: str):
    """
    Reset the environment and return the first observation.
    Must be called before the first step.
    """
    env = get_session(session_id)
    obs = await env.reset()
    return ResetResponse(
        observation=obs_to_dict(obs),
        message="Environment reset. Submit actions to POST /env/{session_id}/step."
    )


@app.post("/env/{session_id}/step", response_model=StepResponse)
async def step_env(session_id: str, req: StepRequest):
    """
    Submit an action and receive the next observation + reward.

    Action fields:
      - relevance  : "relevant" | "not_relevant"  (required)
      - priority   : "urgent" | "normal" | "low"  (medium + hard)
      - reason     : string explanation            (hard only)
    """
    env = get_session(session_id)

    action = Action(
        relevance=req.relevance,
        priority=req.priority,
        reason=req.reason,
    )

    obs, reward, done, info = await env.step(action)

    # Auto-cleanup when episode ends
    if done:
        await env.close()
        sessions.pop(session_id, None)

    return StepResponse(
        observation=obs_to_dict(obs),
        reward=reward.value,
        done=done,
        info=info,
    )

# ── Global Fallback Endpoints for OpenEnv Validator ──────

DEFAULT_SESSION_ID = "default"

@app.post("/reset", response_model=ResetResponse)
@app.post("/env/reset", response_model=ResetResponse)
async def reset_env_global(req: Optional[Dict] = None):
    """Fallback for standard OpenEnv validation which hits /reset directly."""
    task_level = "easy"
    max_emails = 10
    if req:
        task_level = req.get("task_level", "easy")
        max_emails = req.get("max_emails", 10)
    
    env = await EmailTriageEnv.from_env(task_level=task_level, max_emails=max_emails)
    sessions[DEFAULT_SESSION_ID] = env
    obs = await env.reset()
    return ResetResponse(
        observation=obs_to_dict(obs),
        message="Environment reset."
    )

@app.post("/step", response_model=StepResponse)
@app.post("/env/step", response_model=StepResponse)
async def step_env_global(req: StepRequest):
    """Fallback for standard OpenEnv validation which hits /step directly."""
    if DEFAULT_SESSION_ID not in sessions:
         # Initialize fallback environment
         env = await EmailTriageEnv.from_env(task_level="easy", max_emails=10)
         sessions[DEFAULT_SESSION_ID] = env
    return await step_env(DEFAULT_SESSION_ID, req)


@app.get("/env/{session_id}/state")
async def get_state(session_id: str):
    """
    Inspect the current state of a running environment session.
    Useful for debugging.
    """
    env = get_session(session_id)
    state = await env.state()
    return state


@app.post("/env/{session_id}/close")
async def close_env(session_id: str):
    """
    Explicitly close and remove an environment session.
    """
    env = get_session(session_id)
    await env.close()
    sessions.pop(session_id, None)
    return {"message": f"Session '{session_id}' closed successfully."}


# ── Entry point ───────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()