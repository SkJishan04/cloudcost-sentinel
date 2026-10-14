"""
LangGraph ReAct agent orchestration.

If a usable LLM provider/key is configured, a tool-calling ReAct agent (built
with langgraph.prebuilt.create_react_agent) evaluates instances by reasoning
over the tools in app/agent/tools.py. Otherwise, the deterministic
app/agent/rule_based policy is used transparently, so /api/v1/agent/optimize
always returns a consistent response shape regardless of configuration.
"""

import inspect

from sqlalchemy import func
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.rule_based import evaluate_instance, run_rule_based_policy
from app.agent.tools import AGENT_TOOLS
from app.config import get_settings
from app.db.base import SessionLocal
from app.db.models import ActionStatus, ActionType, AgentAction, CloudInstance
from app.logging_config import get_logger

logger = get_logger(__name__)

_MAX_REASONING_CHARS = 1000


def _detach_with_instance_loaded(db, actions: list[AgentAction]) -> list[AgentAction]:
    """Force-load each action's `instance` relationship before expunging.

    AgentAction.instance is lazy-loaded by default. The API layer reads
    `action.instance.instance_id` after this function returns, by which point
    the session is closed — accessing a lazy relationship on a detached
    object raises sqlalchemy.orm.exc.DetachedInstanceError. Touching the
    attribute here, while the session is still open, caches it on the
    instance so it is safe to read after detaching.
    """
    for action in actions:
        _ = action.instance.instance_id
    db.expunge_all()
    return actions


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

    # The system-prompt argument was renamed across LangGraph releases
    # (`state_modifier` in 0.2.x, `prompt` in newer versions). Detect which one
    # the installed version supports instead of hardcoding a name.
    params = inspect.signature(create_react_agent).parameters
    prompt_kwarg = "prompt" if "prompt" in params else "state_modifier"
    return create_react_agent(model=model, tools=AGENT_TOOLS, **{prompt_kwarg: SYSTEM_PROMPT})


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_agent_with_retry(agent_executor, user_message: str):
    return agent_executor.invoke({"messages": [("user", user_message)]})


def _final_message_text(result) -> str:
    """Extract the agent's final natural-language message as plain text."""
    try:
        content = result["messages"][-1].content
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    if isinstance(content, list):  # some providers return a list of content blocks
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content).strip()[:_MAX_REASONING_CHARS]


def _latest_action_id() -> int:
    with SessionLocal() as db:
        return db.query(func.max(AgentAction.id)).scalar() or 0


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
        # Anything the tools persist after this id belongs to this instance's run.
        baseline_id = _latest_action_id()
        user_message = (
            f"Evaluate instance '{target}' against FinOps policy and propose the appropriate "
            f"action. dry_run={dry_run}. Call propose_resize or propose_shutdown with dry_run={dry_run} "
            "if an action is warranted; otherwise state no_action is required and take no tool action."
        )
        try:
            result = _invoke_agent_with_retry(agent_executor, user_message)
        except Exception as exc:  # noqa: BLE001 - LLM/network errors land here
            logger.warning("LLM agent run failed for %s, falling back to rule-based: %s", target, exc)
            with SessionLocal() as db:
                instance = db.query(CloudInstance).filter(CloudInstance.instance_id == target).first()
                if instance:
                    action = evaluate_instance(db, instance, dry_run)
                    actions.extend(_detach_with_instance_loaded(db, [action]))
            continue

        final_text = _final_message_text(result)

        with SessionLocal() as db:
            # The tools persist AgentAction rows directly; use the most recent one for this instance.
            action = (
                db.query(AgentAction)
                .join(CloudInstance)
                .filter(CloudInstance.instance_id == target, AgentAction.id > baseline_id)
                .order_by(AgentAction.id.desc())
                .first()
            )
            if action is None:
                # The agent decided no tool action was needed: record that decision so
                # every evaluated instance has an auditable row and the API shape stays stable.
                instance = db.query(CloudInstance).filter(CloudInstance.instance_id == target).first()
                if instance is None:
                    continue
                action = AgentAction(
                    instance_id=instance.id,
                    action_type=ActionType.NO_ACTION,
                    previous_instance_type=instance.instance_type,
                    new_instance_type=None,
                    reasoning=final_text or "LLM agent determined no action is required.",
                    forecasted_avg_cpu=None,
                    estimated_monthly_savings=0.0,
                    dry_run=dry_run,
                    status=ActionStatus.PROPOSED,
                    agent_source="llm_react_agent",
                )
                db.add(action)
                db.commit()
                db.refresh(action)
            actions.extend(_detach_with_instance_loaded(db, [action]))

    return actions


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
            actions = _detach_with_instance_loaded(db, actions)
        return actions, "rule_based"

    try:
        actions = _run_llm_agent(instance_id, dry_run)
        return actions, "llm_react_agent"
    except Exception as exc:  # noqa: BLE001 - any agent-construction failure falls back safely
        logger.error("LLM agent unavailable, falling back to rule-based policy: %s", exc)
        with SessionLocal() as db:
            actions = run_rule_based_policy(db, instance_id, dry_run)
            actions = _detach_with_instance_loaded(db, actions)
        return actions, "rule_based_fallback"