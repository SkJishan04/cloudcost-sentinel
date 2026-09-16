"""
Time-series forecasting for instance CPU utilization.

Design decision: Holt-Winters exponential smoothing (statsmodels) is used
instead of Prophet. Prophet pulls in a heavy native Stan toolchain (cmdstanpy)
that materially slows cold starts and container builds, which is disproportionate
for hourly-cadence, short-horizon (7-day) operational forecasting. Holt-Winters
with a 24-hour seasonal period captures the same daily seasonality relevant to
FinOps right-sizing decisions, is pure-Python, and is trivially swappable behind
this module's function boundary (e.g. for an LSTM-based ForecastingStrategy)
without touching callers.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.core.exceptions import ForecastingError, InsufficientUsageDataError
from app.logging_config import get_logger
from app.schemas.usage import ForecastPoint, ForecastResult
from app.services.billing_service import get_usage_history

logger = get_logger(__name__)

_MIN_SAMPLES_FOR_SEASONAL_MODEL = 48  # >= 2 full daily cycles
_SEASONAL_PERIOD_HOURS = 24
_BACKTEST_HOLDOUT_HOURS = 24


@dataclass
class _SeriesBundle:
    series: pd.Series


def _to_hourly_series(history) -> _SeriesBundle:
    df = pd.DataFrame(
        {
            "timestamp": [m.timestamp for m in history],
            "cpu": [m.cpu_utilization_pct for m in history],
        }
    )
    df = df.set_index("timestamp").sort_index()
    # Resample to a strict hourly grid, interpolating any gaps.
    series = df["cpu"].resample("1h").mean().interpolate(limit_direction="both")
    return _SeriesBundle(series=series)


def _moving_average_forecast(series: pd.Series, horizon_hours: int, window: int = 24) -> np.ndarray:
    avg = float(series.tail(window).mean())
    return np.full(horizon_hours, avg)


def _fit_holt_winters(series: pd.Series):
    return ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=_SEASONAL_PERIOD_HOURS,
        initialization_method="estimated",
    ).fit(optimized=True)


def _backtest_mae(series: pd.Series) -> float | None:
    """Hold out the last _BACKTEST_HOLDOUT_HOURS and measure MAE against it."""
    if len(series) < _MIN_SAMPLES_FOR_SEASONAL_MODEL + _BACKTEST_HOLDOUT_HOURS:
        return None
    train, test = series[:-_BACKTEST_HOLDOUT_HOURS], series[-_BACKTEST_HOLDOUT_HOURS:]
    try:
        model = _fit_holt_winters(train)
        preds = model.forecast(len(test))
        return round(float(np.mean(np.abs(preds.values - test.values))), 3)
    except Exception as exc:  # noqa: BLE001 - backtest is best-effort diagnostics
        logger.warning("Backtest failed, continuing without MAE: %s", exc)
        return None


def forecast_cpu_utilization(db: Session, instance_id: str, horizon_hours: int, history_days: int) -> ForecastResult:
    history = get_usage_history(db, instance_id, history_days)
    if len(history) < 6:
        raise InsufficientUsageDataError(
            f"Instance '{instance_id}' has only {len(history)} usage samples; at least 6 are required to forecast."
        )

    bundle = _to_hourly_series(history)
    series = bundle.series

    try:
        if len(series) >= _MIN_SAMPLES_FOR_SEASONAL_MODEL:
            model = _fit_holt_winters(series)
            forecast_values = model.forecast(horizon_hours).values
            method = "holt_winters_seasonal"
        else:
            forecast_values = _moving_average_forecast(series, horizon_hours)
            method = "moving_average_fallback"
    except Exception as exc:  # noqa: BLE001
        logger.error("Holt-Winters fit failed for %s, falling back to moving average: %s", instance_id, exc)
        forecast_values = _moving_average_forecast(series, horizon_hours)
        method = "moving_average_fallback_after_error"

    forecast_values = np.clip(forecast_values, 0.0, 100.0)
    mae = _backtest_mae(series) if method == "holt_winters_seasonal" else None

    last_ts = series.index[-1]
    points = [
        ForecastPoint(timestamp=last_ts + timedelta(hours=i + 1), predicted_cpu_pct=round(float(v), 2))
        for i, v in enumerate(forecast_values)
    ]

    if not points:
        raise ForecastingError(f"Forecast produced no points for instance '{instance_id}'")

    return ForecastResult(
        instance_id=instance_id,
        method=method,
        horizon_hours=horizon_hours,
        backtest_mae=mae,
        forecast_avg_cpu_pct=round(float(np.mean(forecast_values)), 2),
        points=points,
    )