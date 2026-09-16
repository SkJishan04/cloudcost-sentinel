"""
Synthetic billing/usage data generation and retrieval.

Stands in for a real billing/metrics ingestion pipeline (e.g. CloudWatch,
Azure Monitor) so the forecasting and agent layers have realistic, varied
hourly telemetry to reason over without any external dependency.
"""

import math
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core.exceptions import InstanceNotFoundError
from app.db.models import CloudInstance, UsageMetric
from app.logging_config import get_logger
from app.schemas.usage import UsageSummary

logger = get_logger(__name__)

# Named usage profiles used to synthesize realistic, differentiated telemetry.
_PROFILES = {
    "idle": dict(base=4.0, amplitude=3.0, noise=1.5, business_hours_boost=0.0),
    "business_hours": dict(base=12.0, amplitude=5.0, noise=3.0, business_hours_boost=35.0),
    "steady_high": dict(base=55.0, amplitude=8.0, noise=5.0, business_hours_boost=5.0),
    "spiky": dict(base=20.0, amplitude=25.0, noise=10.0, business_hours_boost=15.0),
}


def _is_business_hour(ts: datetime) -> bool:
    return ts.weekday() < 5 and 9 <= ts.hour < 18


def generate_synthetic_history(
    db: Session, instance: CloudInstance, days: int, profile: str = "idle", seed: int | None = None
) -> int:
    """Populate `days` of hourly UsageMetric rows for `instance` using `profile`.

    Returns the number of rows created.
    """
    if profile not in _PROFILES:
        raise ValueError(f"Unknown usage profile: {profile}")
    params = _PROFILES[profile]
    rng = random.Random(seed)

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    hours = int((end - start).total_seconds() // 3600)

    rows = []
    for h in range(hours):
        ts = start + timedelta(hours=h)
        daily_cycle = math.sin((ts.hour / 24.0) * 2 * math.pi - math.pi / 2)  # peaks midday
        value = params["base"] + params["amplitude"] * max(daily_cycle, 0)
        if _is_business_hour(ts):
            value += params["business_hours_boost"]
        value += rng.gauss(0, params["noise"])
        cpu = max(0.5, min(99.0, value))
        mem = max(1.0, min(99.0, cpu * rng.uniform(0.8, 1.1)))

        rows.append(
            UsageMetric(
                instance_id=instance.id,
                timestamp=ts,
                cpu_utilization_pct=round(cpu, 2),
                memory_utilization_pct=round(mem, 2),
                network_in_mb=round(max(0.0, rng.gauss(cpu * 2, 5)), 2),
                network_out_mb=round(max(0.0, rng.gauss(cpu * 1.5, 4)), 2),
            )
        )

    db.bulk_save_objects(rows)
    db.commit()
    logger.info("Generated %d synthetic usage rows for %s (profile=%s)", len(rows), instance.instance_id, profile)
    return len(rows)


def get_usage_history(db: Session, instance_id: str, days: int) -> list[UsageMetric]:
    instance = db.query(CloudInstance).filter(CloudInstance.instance_id == instance_id).first()
    if instance is None:
        raise InstanceNotFoundError(f"Instance '{instance_id}' was not found")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return (
        db.query(UsageMetric)
        .filter(UsageMetric.instance_id == instance.id, UsageMetric.timestamp >= cutoff)
        .order_by(UsageMetric.timestamp.asc())
        .all()
    )


def summarize_usage(db: Session, instance_id: str, days: int) -> UsageSummary:
    history = get_usage_history(db, instance_id, days)
    if not history:
        return UsageSummary(
            instance_id=instance_id,
            window_days=days,
            avg_cpu_pct=0.0,
            max_cpu_pct=0.0,
            min_cpu_pct=0.0,
            avg_memory_pct=0.0,
            sample_count=0,
        )
    cpu_values = [m.cpu_utilization_pct for m in history]
    mem_values = [m.memory_utilization_pct for m in history]
    return UsageSummary(
        instance_id=instance_id,
        window_days=days,
        avg_cpu_pct=round(sum(cpu_values) / len(cpu_values), 2),
        max_cpu_pct=round(max(cpu_values), 2),
        min_cpu_pct=round(min(cpu_values), 2),
        avg_memory_pct=round(sum(mem_values) / len(mem_values), 2),
        sample_count=len(history),
    )