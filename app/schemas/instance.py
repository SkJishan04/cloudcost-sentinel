"""Pydantic schemas for cloud instance resources."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import InstanceStatus


class InstanceCreate(BaseModel):
    instance_id: str = Field(..., examples=["i-0a1b2c3d4e5f"])
    name: str
    instance_type: str
    region: str = "us-east-1"
    provider: str = "aws-simulated"
    tags: dict = Field(default_factory=dict)


class InstanceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    instance_id: str
    name: str
    instance_type: str
    region: str
    provider: str
    status: InstanceStatus
    tags: dict
    created_at: datetime


class InstanceCostSummary(BaseModel):
    instance_id: str
    instance_type: str
    hourly_cost_usd: float
    estimated_monthly_cost_usd: float
    status: InstanceStatus