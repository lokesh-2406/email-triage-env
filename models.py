"""
models.py — Typed Action, Observation, and State models for the Email Triage environment.
All models use Pydantic for validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Action Models
# ─────────────────────────────────────────────

class EmailTriageAction(BaseModel):
    """
    The agent's action for any email triage task.

    Fields (use only what the task requires):
      - category:     One of [spam, urgent, billing, support, general]
      - priority:     A list of email IDs in order from highest to lowest priority
                      (used for prioritize_inbox task)
      - department:   One of [sales, engineering, billing, hr, support]
      - reply_draft:  A short acknowledgment reply (1-3 sentences)
    """
    category: Optional[str] = Field(
        default=None,
        description="Email category: spam | urgent | billing | support | general"
    )
    priority: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of email IDs from highest to lowest priority"
    )
    department: Optional[str] = Field(
        default=None,
        description="Routing department: sales | engineering | billing | hr | support"
    )
    reply_draft: Optional[str] = Field(
        default=None,
        description="Brief acknowledgment reply (1-3 sentences)"
    )


# ─────────────────────────────────────────────
# Observation Models
# ─────────────────────────────────────────────

class Email(BaseModel):
    """A single email in the inbox."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class EmailTriageObservation(BaseModel):
    """
    Observation returned by the environment after reset() or step().
    """
    task: str = Field(description="Active task name")
    emails: List[Email] = Field(description="Email(s) to process")
    instructions: str = Field(description="Human-readable task instructions")
    reward: float = Field(default=0.0, description="Reward from last action")
    feedback: str = Field(default="", description="Grader feedback for the last action")
    done: bool = Field(default=False, description="Whether the episode is complete")


# ─────────────────────────────────────────────
# State Model
# ─────────────────────────────────────────────

class EmailTriageState(BaseModel):
    """Internal state of the environment."""
    task: str
    step: int = 0
    emails: List[Email]
    expected: Dict[str, Any] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    done: bool = False
