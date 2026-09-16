"""
Cost modeling: an instance-type pricing catalog and cost/savings arithmetic.

Prices are illustrative, on-demand-style hourly rates (USD) loosely modeled on
common AWS EC2 general-purpose tiers, ordered from smallest to largest so the
optimizer can walk the catalog to find the next cheaper/larger tier.
"""

from dataclasses import dataclass

HOURS_PER_MONTH = 730  # standard FinOps convention (365.25 * 24 / 12)


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    vcpu: int
    memory_gb: float
    hourly_cost_usd: float


# Ordered smallest -> largest.
INSTANCE_CATALOG: list[InstanceSpec] = [
    InstanceSpec("t3.nano", 2, 0.5, 0.0052),
    InstanceSpec("t3.micro", 2, 1.0, 0.0104),
    InstanceSpec("t3.small", 2, 2.0, 0.0208),
    InstanceSpec("t3.medium", 2, 4.0, 0.0416),
    InstanceSpec("t3.large", 2, 8.0, 0.0832),
    InstanceSpec("t3.xlarge", 4, 16.0, 0.1664),
    InstanceSpec("m5.xlarge", 4, 16.0, 0.192),
    InstanceSpec("m5.2xlarge", 8, 32.0, 0.384),
    InstanceSpec("m5.4xlarge", 16, 64.0, 0.768),
]

_CATALOG_BY_NAME = {spec.name: spec for spec in INSTANCE_CATALOG}


def get_spec(instance_type: str) -> InstanceSpec:
    if instance_type not in _CATALOG_BY_NAME:
        raise ValueError(f"Unknown instance type: {instance_type}")
    return _CATALOG_BY_NAME[instance_type]


def get_hourly_cost(instance_type: str) -> float:
    return get_spec(instance_type).hourly_cost_usd


def estimate_monthly_cost(instance_type: str) -> float:
    return round(get_hourly_cost(instance_type) * HOURS_PER_MONTH, 2)


def next_smaller_tier(instance_type: str) -> str | None:
    idx = INSTANCE_CATALOG.index(get_spec(instance_type))
    return INSTANCE_CATALOG[idx - 1].name if idx > 0 else None


def next_larger_tier(instance_type: str) -> str | None:
    idx = INSTANCE_CATALOG.index(get_spec(instance_type))
    return INSTANCE_CATALOG[idx + 1].name if idx < len(INSTANCE_CATALOG) - 1 else None


def estimate_resize_savings(current_type: str, target_type: str) -> float:
    """Positive value = monthly savings; negative = additional monthly spend."""
    return round(estimate_monthly_cost(current_type) - estimate_monthly_cost(target_type), 2)


def estimate_shutdown_savings(current_type: str) -> float:
    return estimate_monthly_cost(current_type)