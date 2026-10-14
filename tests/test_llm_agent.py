"""
Offline tests for the LangGraph ReAct agent path.

A scripted fake chat model replaces the real LLM, so the full loop
(model -> tool calls -> guardrails -> database -> API shape) runs
deterministically with no network access or API key.
"""

from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy.orm import sessionmaker

from app.db.models import ActionStatus, ActionType, AgentAction


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model that replays scripted messages and tolerates tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture()
def llm_env(db_session, monkeypatch):
    """Point every module-level SessionLocal at the test DB and enable the LLM path."""
    factory = sessionmaker(bind=db_session.get_bind(), autoflush=False, autocommit=False)
    monkeypatch.setattr("app.db.base.SessionLocal", factory)
    monkeypatch.setattr("app.agent.tools.SessionLocal", factory)
    monkeypatch.setattr("app.agent.graph.SessionLocal", factory)
    monkeypatch.setattr(
        "app.agent.graph.get_settings",
        lambda: SimpleNamespace(llm_enabled=True, LLM_PROVIDER="fake"),
    )

    def install_script(messages: list[AIMessage]) -> None:
        monkeypatch.setattr(
            "app.agent.graph._build_chat_model",
            lambda: ScriptedChatModel(messages=iter(messages)),
        )

    return install_script


def test_llm_agent_proposes_shutdown_for_idle_nonprod(llm_env, db_session, idle_nonprod_instance):
    from app.agent.graph import run_agent_optimization

    iid = idle_nonprod_instance.instance_id
    llm_env(
        [
            _tool_call("get_instance_usage_summary", {"instance_id": iid}, "c1"),
            _tool_call(
                "propose_shutdown",
                {"instance_id": iid, "reasoning": "avg_cpu=5%, non-production", "dry_run": True},
                "c2",
            ),
            AIMessage(content="Proposed shutdown."),
        ]
    )

    actions, source = run_agent_optimization(iid, dry_run=True)

    assert source == "llm_react_agent"
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.SHUTDOWN
    assert actions[0].status == ActionStatus.PROPOSED
    assert actions[0].agent_source == "llm_react_agent"
    assert actions[0].instance.instance_id == iid  # relationship usable after detaching


def test_llm_guardrail_blocks_production_shutdown_then_agent_resizes(llm_env, db_session, idle_prod_instance):
    from app.agent.graph import run_agent_optimization

    iid = idle_prod_instance.instance_id
    llm_env(
        [
            _tool_call(
                "propose_shutdown",
                {"instance_id": iid, "reasoning": "looks idle", "dry_run": True},
                "c1",
            ),
            _tool_call(
                "propose_resize",
                {"instance_id": iid, "new_instance_type": "t3.xlarge", "reasoning": "downsize", "dry_run": True},
                "c2",
            ),
            AIMessage(content="Resized instead."),
        ]
    )

    actions, source = run_agent_optimization(iid, dry_run=True)

    assert source == "llm_react_agent"
    assert actions[0].action_type == ActionType.RESIZE

    rejected = (
        db_session.query(AgentAction)
        .filter(AgentAction.status == ActionStatus.REJECTED, AgentAction.action_type == ActionType.SHUTDOWN)
        .all()
    )
    assert len(rejected) == 1, "the guardrail rejection must be recorded in the audit trail"


def test_llm_agent_records_no_action_decision(llm_env, db_session, idle_nonprod_instance):
    from app.agent.graph import run_agent_optimization

    llm_env([AIMessage(content="No action required: utilization is healthy.")])

    actions, source = run_agent_optimization(idle_nonprod_instance.instance_id, dry_run=True)

    assert source == "llm_react_agent"
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.NO_ACTION
    assert "No action required" in actions[0].reasoning


def test_falls_back_to_rule_based_when_agent_cannot_be_built(llm_env, db_session, idle_nonprod_instance, monkeypatch):
    from app.agent.graph import run_agent_optimization

    def _boom():
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr("app.agent.graph._build_react_agent", _boom)

    actions, source = run_agent_optimization(idle_nonprod_instance.instance_id, dry_run=True)

    assert source == "rule_based_fallback"
    assert actions[0].agent_source == "rule_based"