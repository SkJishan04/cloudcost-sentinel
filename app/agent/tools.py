
"""
LangChain tools exposed to the LLM agent.

Each tool opens its own short-lived DB session because tool invocation happens
outside the FastAPI request lifecycle. Tools are intentionally thin wrappers
around the services layer so that the same business logic backs both the LLM
agent and the deterministic rule-based fallback (app/agent/rule_based.py).
"""

import json

from langchain_core.tools import tool

from app.config import get_settings
from app.core.exceptions import GuardrailViolationError
from app.db.base import SessionLocal
from app.db.models import ActionStatus, ActionType, AgentAction, CloudInstance
from app.logging_config import get_logger
from app.services import cost_calculator
from app.services.billing_service import summarize_usage
from app.services.cloud_provider import get_cloud_provider
from app.services.forecasting_service import forecast_cpu_utilization

logger = get_logger(__name__)
settings = get_settings()


@tool
def list_all_instances() -> str:
    """List all cloud instances with their id, type, status and criticality tag, as JSON."""
    with SessionLocal() as db:
        instances = db.query(CloudInstance).all()
        return json.dumps(
            [
                {
                    "instance_id": i.instance_id,
                    "name": i.name,
                    "instance_type": i.instance_type,
                    "status": i.status.value,
                    "criticality": i.criticality,
                }
                for i in instances
            ]
        )


@tool
def get_instance_usage_summary(instance_id: str) -> str:
    """Get trailing usage statistics (avg/max/min CPU, avg memory) for an instance, as JSON."""
    with SessionLocal() as db:
        summary = summarize_usage(db, instance_id, days=settings.HISTORY_LOOKBACK_DAYS)
        return summary.model_dump_json()


@tool
def get_forecast(instance_id: str) -> str:
    """Get a forecasted average CPU utilization for the next 7 days for an instance, as JSON."""
    with SessionLocal() as db:
        result = forecast_cpu_utilization(
            db,
            instance_id,
            horizon_hours=settings.FORECAST_HORIZON_HOURS,
            history_days=settings.HISTORY_LOOKBACK_DAYS,
        )
        return json.dumps(
            {
                "instance_id": result.instance_id,
                "method": result.method,
                "forecast_avg_cpu_pct": result.forecast_avg_cpu_pct,
                "backtest_mae": result.backtest_mae,
            }
        )


def _log_action(
    db,
    instance: CloudInstance,
    action_type: ActionType,
    new_type: str | None,
    reasoning: str,
    forecasted_avg_cpu: float | None,
    savings: float,
    dry_run: bool,
    status: ActionStatus,
) -> AgentAction:
    action = AgentAction(
        instance_id=instance.id,
        action_type=action_type,
        previous_instance_type=instance.instance_type,
        new_instance_type=new_type,
        reasoning=reasoning,
        forecasted_avg_cpu=forecasted_avg_cpu,
        estimated_monthly_savings=savings,
        dry_run=dry_run,
        status=status,
        agent_source="llm_react_agent",
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


@tool
def propose_resize(instance_id: str, new_instance_type: str, reasoning: str, dry_run: bool = True) -> str:
    """Propose (and, if dry_run=False, execute) resizing an instance to new_instance_type.

    Always safe for both production and non-production instances. Returns a JSON
    string describing the resulting AgentAction record.
    """
    with SessionLocal() as db:
        provider = get_cloud_provider()
        instance = provider.get_instance(db, instance_id)
        savings = cost_calculator.estimate_resize_savings(instance.instance_type, new_instance_type)

        status = ActionStatus.PROPOSED
        if not dry_run:
            provider.resize_instance(db, instance_id, new_instance_type)
            status = ActionStatus.EXECUTED

        action = _log_action(
            db, instance, ActionType.RESIZE, new_instance_type, reasoning, None, savings, dry_run, status
        )
        return json.dumps(
            {
                "instance_id": instance_id,
                "action": "resize",
                "new_instance_type": new_instance_type,
                "status": action.status.value,
                "estimated_monthly_savings": savings,
            }
        )


@tool
def propose_shutdown(instance_id: str, reasoning: str, dry_run: bool = True) -> str:
    """Propose (and, if dry_run=False, execute) shutting down an instance.

    GUARDRAIL: refuses to execute (raises) against instances tagged
    criticality="production", even when dry_run=False. Production instances
    may only ever be resized, never shut down, by policy.
    """
    with SessionLocal() as db:
        provider = get_cloud_provider()
        instance = provider.get_instance(db, instance_id)

        if instance.criticality == "production":
            action = _log_action(
                db,
                instance,
                ActionType.SHUTDOWN,
                None,
                f"REJECTED by guardrail: cannot shut down a production instance. Original reasoning: {reasoning}",
                None,
                0.0,
                dry_run,
                ActionStatus.REJECTED,
            )
            raise GuardrailViolationError(
                f"Refusing to shut down production instance '{instance_id}'. "
                "Use propose_resize instead for production workloads."
            )

        savings = cost_calculator.estimate_shutdown_savings(instance.instance_type)
        status = ActionStatus.PROPOSED
        if not dry_run:
            provider.shutdown_instance(db, instance_id)
            status = ActionStatus.EXECUTED

        action = _log_action(
            db, instance, ActionType.SHUTDOWN, None, reasoning, None, savings, dry_run, status
        )
        return json.dumps(
            {
                "instance_id": instance_id,
                "action": "shutdown",
                "status": action.status.value,
                "estimated_monthly_savings": savings,
            }
        )


AGENT_TOOLS = [list_all_instances, get_instance_usage_summary, get_forecast, propose_resize, propose_shutdown]

