"""Tests for the forecasting service, including the insufficient-data edge case."""

import pytest

from app.core.exceptions import InsufficientUsageDataError
from app.services.forecasting_service import forecast_cpu_utilization


def test_forecast_returns_points_within_valid_range(db_session, idle_prod_instance):
    result = forecast_cpu_utilization(db_session, idle_prod_instance.instance_id, horizon_hours=48, history_days=14)
    assert len(result.points) == 48
    assert all(0.0 <= p.predicted_cpu_pct <= 100.0 for p in result.points)
    assert result.method in {"holt_winters_seasonal", "moving_average_fallback", "moving_average_fallback_after_error"}


def test_forecast_uses_seasonal_model_with_enough_history(db_session, idle_prod_instance):
    result = forecast_cpu_utilization(db_session, idle_prod_instance.instance_id, horizon_hours=24, history_days=14)
    assert result.method == "holt_winters_seasonal"
    assert result.backtest_mae is not None
    assert result.backtest_mae >= 0


def test_forecast_raises_on_insufficient_data(db_session):
    from app.db.models import CloudInstance

    sparse = CloudInstance(instance_id="i-sparse", name="sparse", instance_type="t3.micro", tags={})
    db_session.add(sparse)
    db_session.commit()

    with pytest.raises(InsufficientUsageDataError):
        forecast_cpu_utilization(db_session, "i-sparse", horizon_hours=24, history_days=14)

