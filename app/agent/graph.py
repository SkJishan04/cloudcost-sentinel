
"""
LangGraph ReAct agent orchestration.

If a usable LLM provider/key is configured, a tool-calling ReAct agent (built
with langgraph.prebuilt.create_react_agent) evaluates instances by reasoning
over the tools in app/agent/tools.py. Otherwise, the deterministic
app/agent/rule_based policy is used transparently, so /api/v1/agent/optimize
always returns a consistent response shape regardless of configuration.
"""

from tenacity import retry, stop_after_attempt, wait_exponential

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.rule_based import run_rule_based_policy
from app.agent.tools import AGENT_TOOLS
from app.config import get_settings
from app.db.base import SessionLocal
from app.db.models import AgentAction, CloudInstance
from app.logging_config import get_logger

logger = get_logger(__name__)


def _build_chat_model():
    settings = get_settings()
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.LLM_MODEL, api_key=settings.ANTHROPIC_API_KEY, temperature=0)
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0)
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def _build_react_agent():
    from langgraph.prebuilt import create_react_agent

    model = _build_chat_model()
    return create_react_agent(model=model, tools=AGENT_TOOLS, prompt=SYSTEM_PROMPT)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_agent_with_retry(agent_executor, user_message: str):
    return agent_executor.invoke({"messages": [("user", user_message)]})


def _run_llm_agent(instance_id: str | None, dry_run: bool) -> list[AgentAction]:
    agent_executor = _build_react_agent()

    with SessionLocal() as db:
        target_ids = (
            [instance_id]
            if instance_id
            else [row.instance_id for row in db.query(CloudInstance.instance_id).all()]
        )

    actions: list[AgentAction] = []
    for target in target_ids:
        user_message = (
            f"Evaluate instance '{target}' against FinOps policy and propose the appropriate "
            f"action. dry_run={dry_run}. Call propose_resize or propose_shutdown with dry_run={dry_run} "
            "if an action is warranted; otherwise state no_action is required and take no tool action."
        )
        try:
            _invoke_agent_with_retry(agent_executor, user_message)
        except Exception as exc:  # noqa: BLE001 - guardrail rejections and LLM errors both land here
            logger.warning("LLM agent run failed for %s, falling back to rule-based: %s", target, exc)
            with SessionLocal() as db:
                instance = db.query(CloudInstance).filter(CloudInstance.instance_id == target).first()
                if instance:
                    from app.agent.rule_based import evaluate_instance

                    actions.append(evaluate_instance(db, instance, dry_run))
            continue

    # The tools persist AgentAction rows directly; read back the ones created for this run.
    with SessionLocal() as db:
        rows = (
            db.query(AgentAction)
            .join(CloudInstance)
            .filter(CloudInstance.instance_id.in_(target_ids))
            .order_by(AgentAction.created_at.desc())
            .limit(len(target_ids) * 3)
            .all()
        )
        db.expunge_all()
    return actions + rows[: len(target_ids)]


def run_agent_optimization(instance_id: str | None, dry_run: bool) -> tuple[list[AgentAction], str]:
    """Entry point used by the API layer.

    Returns (actions, agent_source). Falls back to the deterministic rule-based
    policy whenever no LLM provider is configured, or if the LLM path raises
    an unrecoverable error.
    """
    settings = get_settings()

    if not settings.llm_enabled:
        logger.info("LLM not configured (LLM_PROVIDER=%s); using rule-based policy", settings.LLM_PROVIDER)
        with SessionLocal() as db:
            actions = run_rule_based_policy(db, instance_id, dry_run)
            db.expunge_all()
        return actions, "rule_based"

    try:
        actions = _run_llm_agent(instance_id, dry_run)
        return actions, "llm_react_agent"
    except Exception as exc:  # noqa: BLE001 - any agent-construction failure falls back safely
        logger.error("LLM agent unavailable, falling back to rule-based policy: %s", exc)
        with SessionLocal() as db:
            actions = run_rule_based_policy(db, instance_id, dry_run)
            db.expunge_all()
        return actions, "rule_based_fallback"