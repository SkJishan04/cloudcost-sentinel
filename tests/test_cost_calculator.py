"""Unit tests for pricing catalog and cost/savings math."""

import pytest

from app.services import cost_calculator


def test_estimate_monthly_cost_matches_hourly_rate():
    hourly = cost_calculator.get_hourly_cost("t3.medium")
    monthly = cost_calculator.estimate_monthly_cost("t3.medium")
    assert monthly == round(hourly * cost_calculator.HOURS_PER_MONTH, 2)


def test_next_smaller_tier_walks_catalog_down():
    assert cost_calculator.next_smaller_tier("t3.medium") == "t3.small"


def test_next_smaller_tier_none_at_bottom():
    assert cost_calculator.next_smaller_tier("t3.nano") is None


def test_next_larger_tier_none_at_top():
    assert cost_calculator.next_larger_tier("m5.4xlarge") is None


def test_resize_savings_positive_when_downsizing():
    savings = cost_calculator.estimate_resize_savings("m5.2xlarge", "m5.xlarge")
    assert savings > 0


def test_resize_savings_negative_when_upsizing():
    savings = cost_calculator.estimate_resize_savings("t3.small", "t3.large")
    assert savings < 0


def test_shutdown_savings_equals_full_monthly_cost():
    assert cost_calculator.estimate_shutdown_savings("t3.large") == cost_calculator.estimate_monthly_cost(
        "t3.large"
    )


def test_unknown_instance_type_raises():
    with pytest.raises(ValueError):
        cost_calculator.get_spec("not-a-real-type")
