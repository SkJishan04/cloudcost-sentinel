"""Endpoints for CPU utilization forecasting."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import get_settings
from app.schemas.usage import ForecastResult
from app.services.forecasting_service import forecast_cpu_utilization

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


@router.get("/{instance_id}", response_model=ForecastResult)
def get_forecast(
    instance_id: str,
    horizon_hours: int = Query(default=None, ge=1, le=720),
    db: Session = Depends(get_db),
) -> ForecastResult:
    settings = get_settings()
    return forecast_cpu_utilization(
        db,
        instance_id,
        horizon_hours=horizon_hours or settings.FORECAST_HORIZON_HOURS,
        history_days=settings.HISTORY_LOOKBACK_DAYS,
    )
