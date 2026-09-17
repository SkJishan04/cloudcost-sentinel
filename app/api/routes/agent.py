"""Endpoints exposing the FinOps optimization agent."""

from fastapi import APIRouter

from app.agent.graph import run_agent_optimization
from app.config import get_settings
from app.schemas.agent import AgentActionResult, AgentOptimizeRequest, AgentOptimizeResponse

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/optimize", response_model=AgentOptimizeResponse)
def optimize(payload: AgentOptimizeRequest) -> AgentOptimizeResponse:
    settings = get_settings()
    dry_run = payload.dry_run if payload.dry_run is not None else settings.DRY_RUN_DEFAULT

    actions, agent_source = run_agent_optimization(payload.instance_id, dry_run)

    results = [
        AgentActionResult(
            instance_id=action.instance.instance_id,
            action_type=action.action_type,
            previous_instance_type=action.previous_instance_type,
            new_instance_type=action.new_instance_type,
            reasoning=action.reasoning,
            forecasted_avg_cpu=action.forecasted_avg_cpu,
            estimated_monthly_savings=action.estimated_monthly_savings,
            dry_run=action.dry_run,
            status=action.status,
            agent_source=action.agent_source,
            created_at=action.created_at,
        )
        for action in actions
    ]

    return AgentOptimizeResponse(
        agent_source=agent_source,
        evaluated_instances=len(results),
        actions=results,
        total_estimated_monthly_savings=round(sum(r.estimated_monthly_savings for r in results), 2),
    )