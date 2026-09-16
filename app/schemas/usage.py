"""Pydantic schemas for usage metrics and forecasts."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UsageMetricRead(BaseModel):
    timestamp: datetime
    cpu_utilization_pct: float
    memory_utilization_pct: float
    network_in_mb: float
    network_out_mb: float


class UsageSummary(BaseModel):
    instance_id: str
    window_days: int
    avg_cpu_pct: float
    max_cpu_pct: float
    min_cpu_pct: float
    avg_memory_pct: float
    sample_count: int


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_cpu_pct: float


class ForecastResult(BaseModel):
    instance_id: str
    method: str
    horizon_hours: int
    backtest_mae: Optional[float] = None
    forecast_avg_cpu_pct: float
    points: list[ForecastPoint]