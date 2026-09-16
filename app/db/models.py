"""ORM models for cloud instances, usage telemetry and agent actions."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Boolean, Enum, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InstanceStatus(str, enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"


class ActionType(str, enum.Enum):
    RESIZE = "resize"
    SHUTDOWN = "shutdown"
    NO_ACTION = "no_action"


class ActionStatus(str, enum.Enum):
    PROPOSED = "proposed"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class CloudInstance(Base):
    __tablename__ = "cloud_instances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    instance_type: Mapped[str] = mapped_column(String(32), nullable=False)
    region: Mapped[str] = mapped_column(String(32), default="us-east-1")
    provider: Mapped[str] = mapped_column(String(32), default="aws-simulated")
    status: Mapped[InstanceStatus] = mapped_column(
        Enum(InstanceStatus), default=InstanceStatus.RUNNING, nullable=False
    )
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    usage_metrics: Mapped[list["UsageMetric"]] = relationship(
        back_populates="instance", cascade="all, delete-orphan"
    )
    actions: Mapped[list["AgentAction"]] = relationship(
        back_populates="instance", cascade="all, delete-orphan"
    )

    @property
    def criticality(self) -> str:
        return self.tags.get("criticality", "non-production")


class UsageMetric(Base):
    __tablename__ = "usage_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[int] = mapped_column(ForeignKey("cloud_instances.id"), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    cpu_utilization_pct: Mapped[float] = mapped_column(Float, nullable=False)
    memory_utilization_pct: Mapped[float] = mapped_column(Float, nullable=False)
    network_in_mb: Mapped[float] = mapped_column(Float, default=0.0)
    network_out_mb: Mapped[float] = mapped_column(Float, default=0.0)

    instance: Mapped["CloudInstance"] = relationship(back_populates="usage_metrics")


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[int] = mapped_column(ForeignKey("cloud_instances.id"), index=True, nullable=False)
    action_type: Mapped[ActionType] = mapped_column(Enum(ActionType), nullable=False)
    previous_instance_type: Mapped[str] = mapped_column(String(32), nullable=False)
    new_instance_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    forecasted_avg_cpu: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_monthly_savings: Mapped[float] = mapped_column(Float, default=0.0)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[ActionStatus] = mapped_column(Enum(ActionStatus), default=ActionStatus.PROPOSED)
    agent_source: Mapped[str] = mapped_column(String(32), default="rule_based")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    instance: Mapped["CloudInstance"] = relationship(back_populates="actions")
