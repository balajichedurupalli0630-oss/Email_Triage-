from pydantic import BaseModel, Field
from typing import Optional 

# ── Pydantic Models

class Observation(BaseModel):
    email_id: str
    subject: str
    body: str
    persona: str
    context: dict
    task_level: str
    step: int = 0

class Action(BaseModel):
    relevance: str           # "relevant" or "not_relevant"
    priority: Optional[str] = None   # "urgent", "normal", "low"
    reason: Optional[str] = None     # why agent made this decision

class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)