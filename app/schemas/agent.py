"""Pydantic schemas for the FinOps optimization agent API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.db.models import ActionStatus, ActionType


class AgentOptimizeRequest(BaseModel):
    instance_id: Optional[str] = Field(
        default=None, description="Specific instance_id to evaluate, or omit to evaluate all instances."
    )
    dry_run: bool = Field(default=True, description="If true, actions are proposed but not executed.")


class AgentActionResult(BaseModel):
    instance_id: str
    action_type: ActionType
    previous_instance_type: str
    new_instance_type: Optional[str]
    reasoning: str
    forecasted_avg_cpu: Optional[float]
    estimated_monthly_savings: float
    dry_run: bool
    status: ActionStatus
    agent_source: str
    created_at: datetime


class AgentOptimizeResponse(BaseModel):
    agent_source: str
    evaluated_instances: int
    actions: list[AgentActionResult]
    total_estimated_monthly_savings: float