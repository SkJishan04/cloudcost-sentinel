"""
Deterministic rule-based policy engine.

Implements the same FinOps policy described in app/agent/prompts.py as plain
Python, without any LLM call. This serves three purposes:
  1. A zero-dependency fallback used automatically when no LLM API key is
     configured (LLM_PROVIDER=none), so the project is fully runnable and
     reproducible out of the box.
  2. A deterministic baseline the LLM agent's decisions can be evaluated
     against in evaluation/evaluate_agent.py.
  3. A safety net: the LLM agent's tools enforce the same guardrails
     (see app/agent/tools.py), so behavior is consistent regardless of which
     policy engine is active.
"""

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ActionStatus, ActionType, AgentAction, CloudInstance
from app.logging_config import get_logger
from app.services import cost_calculator
from app.services.billing_service import summarize_usage
from app.services.cloud_provider import get_cloud_provider
from app.services.forecasting_service import forecast_cpu_utilization

logger = get_logger(__name__)


def evaluate_instance(db: Session, instance: CloudInstance, dry_run: bool) -> AgentAction:
    settings = get_settings()
    provider = get_cloud_provider()

    summary = summarize_usage(db, instance.instance_id, days=settings.HISTORY_LOOKBACK_DAYS)
    try:
        forecast = forecast_cpu_utilization(
            db,
            instance.instance_id,
            horizon_hours=settings.FORECAST_HORIZON_HOURS,
            history_days=settings.HISTORY_LOOKBACK_DAYS,
        )
        forecast_avg = forecast.forecast_avg_cpu_pct
    except Exception as exc:  # noqa: BLE001 - insufficient data is a valid, handled outcome
        logger.warning("Forecast unavailable for %s: %s", instance.instance_id, exc)
        forecast_avg = summary.avg_cpu_pct  # conservative fallback: assume steady state

    threshold = settings.UNDERUTILIZED_CPU_THRESHOLD
    underutilized = summary.avg_cpu_pct < threshold and forecast_avg < threshold
    overloaded = forecast_avg > 80.0  # nearing saturation regardless of tag

    action_type = ActionType.NO_ACTION
    new_type: str | None = None
    savings = 0.0
    status = ActionStatus.PROPOSED
    reasoning = (
        f"avg_cpu={summary.avg_cpu_pct}%, forecast_avg_cpu={forecast_avg}%, "
        f"threshold={threshold}%, criticality={instance.criticality}"
    )

    if overloaded:
        larger = cost_calculator.next_larger_tier(instance.instance_type)
        if larger:
            action_type = ActionType.RESIZE
            new_type = larger
            savings = cost_calculator.estimate_resize_savings(instance.instance_type, larger)
            reasoning = f"Forecasted CPU {forecast_avg}% nears saturation; upsizing to {larger}. " + reasoning
    elif underutilized:
        if instance.criticality == "production":
            smaller = cost_calculator.next_smaller_tier(instance.instance_type)
            if smaller:
                action_type = ActionType.RESIZE
                new_type = smaller
                savings = cost_calculator.estimate_resize_savings(instance.instance_type, smaller)
                reasoning = f"Underutilized production instance; downsizing to {smaller}. " + reasoning
        else:
            action_type = ActionType.SHUTDOWN
            savings = cost_calculator.estimate_shutdown_savings(instance.instance_type)
            reasoning = "Underutilized non-production instance; recommending shutdown. " + reasoning

    if not dry_run and action_type != ActionType.NO_ACTION:
        if action_type == ActionType.RESIZE and new_type:
            provider.resize_instance(db, instance.instance_id, new_type)
            status = ActionStatus.EXECUTED
        elif action_type == ActionType.SHUTDOWN:
            provider.shutdown_instance(db, instance.instance_id)
            status = ActionStatus.EXECUTED

    action = AgentAction(
        instance_id=instance.id,
        action_type=action_type,
        previous_instance_type=instance.instance_type,
        new_instance_type=new_type,
        reasoning=reasoning,
        forecasted_avg_cpu=forecast_avg,
        estimated_monthly_savings=savings,
        dry_run=dry_run,
        status=status,
        agent_source="rule_based",
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def run_rule_based_policy(db: Session, instance_id: str | None, dry_run: bool) -> list[AgentAction]:
    query = db.query(CloudInstance)
    if instance_id:
        query = query.filter(CloudInstance.instance_id == instance_id)
    instances = query.all()
    return [evaluate_instance(db, instance, dry_run) for instance in instances]
