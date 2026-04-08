"""
app.py — FastAPI server for the Email Triage OpenEnv environment.

Endpoints:
  POST /reset       — Start a new episode
  POST /step        — Take one action
  GET  /state       — Get current internal state
  GET  /health      — Health check
  GET  /schema      — Return action/observation JSON schema
  GET  /metadata    — Return environment metadata
"""

import os
import sys

# Ensure parent directory is on path so 'models' and 'server.environment' import cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from environment import EmailTriageEnvironment

# ─── App setup ───────────────────────────────

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv environment for email triage — classify, prioritize, and route emails.",
    version="1.0.0",
)

# One global environment instance per process (stateless per-reset design)
TASK = os.getenv("EMAIL_TRIAGE_TASK", "classify_email")
env = EmailTriageEnvironment(task=TASK)


# ─── Request/Response wrappers ───────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None  # Optionally override task at reset time
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ─── Routes ──────────────────────────────────

@app.get("/")
def root():
    return {"message": "Email Triage API is running 🚀"}
@app.get("/health")
def health():
    return {"status": "ok", "task": TASK}


@app.get("/metadata")
def metadata():
    return {
        "name": "email_triage_env",
        "version": "1.0.0",
        "tasks": [
            {"name": "classify_email", "difficulty": "easy"},
            {"name": "prioritize_inbox", "difficulty": "medium"},
            {"name": "route_and_respond", "difficulty": "hard"},
        ],
        "action_fields": ["category", "priority", "department", "reply_draft"],
        "valid_categories": ["spam", "urgent", "billing", "support", "general"],
        "valid_departments": ["sales", "engineering", "billing", "hr", "support"],
    }


@app.get("/schema")
def schema():
    return {
        "action": EmailTriageAction.model_json_schema(),
        "observation": EmailTriageObservation.model_json_schema(),
        "state": EmailTriageState.model_json_schema(),
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env, TASK
    task = req.task or TASK
    env = EmailTriageEnvironment(task=task, seed=req.seed)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        action = EmailTriageAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    try:
        s = env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Entry point ─────────────────────────────
def main():
    uvicorn.run(
        "app:app",   # IMPORTANT for your Docker structure
        host="0.0.0.0",
        port=7860
    )

if __name__ == "__main__":
    main()