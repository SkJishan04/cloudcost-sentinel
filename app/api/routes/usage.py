"""Endpoints for retrieving usage telemetry and summaries."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import get_settings
from app.schemas.usage import UsageMetricRead, UsageSummary
from app.services.billing_service import get_usage_history, summarize_usage

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/{instance_id}", response_model=list[UsageMetricRead])
def get_usage(
    instance_id: str,
    days: int = Query(default=None, ge=1, le=90),
    db: Session = Depends(get_db),
) -> list:
    settings = get_settings()
    return get_usage_history(db, instance_id, days or settings.HISTORY_LOOKBACK_DAYS)


@router.get("/{instance_id}/summary", response_model=UsageSummary)
def get_usage_summary(
    instance_id: str,
    days: int = Query(default=None, ge=1, le=90),
    db: Session = Depends(get_db),
) -> UsageSummary:
    settings = get_settings()
    return summarize_usage(db, instance_id, days or settings.HISTORY_LOOKBACK_DAYS)
