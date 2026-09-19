"""Tests for the rule-based policy engine and tool-level guardrails."""

import pytest

from app.core.exceptions import GuardrailViolationError
from app.db.models import ActionType
from app.services.cloud_provider import SimulatedCloudProvider


def test_underutilized_production_instance_is_resized_not_shutdown(db_session, idle_prod_instance):
    from app.agent.rule_based import evaluate_instance

    action = evaluate_instance(db_session, idle_prod_instance, dry_run=True)
    assert action.action_type == ActionType.RESIZE
    assert action.new_instance_type is not None


def test_underutilized_nonprod_instance_is_shutdown(db_session, idle_nonprod_instance):
    from app.agent.rule_based import evaluate_instance

    action = evaluate_instance(db_session, idle_nonprod_instance, dry_run=True)
    assert action.action_type == ActionType.SHUTDOWN


def test_dry_run_does_not_mutate_instance_state(db_session, idle_nonprod_instance):
    from app.agent.rule_based import evaluate_instance
    from app.db.models import InstanceStatus

    evaluate_instance(db_session, idle_nonprod_instance, dry_run=True)
    db_session.refresh(idle_nonprod_instance)
    assert idle_nonprod_instance.status == InstanceStatus.RUNNING


def test_execution_mutates_instance_state(db_session, idle_nonprod_instance):
    from app.agent.rule_based import evaluate_instance
    from app.db.models import InstanceStatus

    evaluate_instance(db_session, idle_nonprod_instance, dry_run=False)
    db_session.refresh(idle_nonprod_instance)
    assert idle_nonprod_instance.status == InstanceStatus.STOPPED


def test_shutdown_guardrail_blocks_production_instance(db_session, idle_prod_instance, monkeypatch):
    # Exercise the tool-level guardrail directly against the production fixture.
    from app.db.base import SessionLocal

    monkeypatch.setattr("app.agent.tools.SessionLocal", lambda: db_session)
    # Prevent context-manager close from tearing down the shared test session.
    monkeypatch.setattr(db_session, "close", lambda: None)

    from app.agent.tools import propose_shutdown

    with pytest.raises(GuardrailViolationError):
        propose_shutdown.func(idle_prod_instance.instance_id, "test forced shutdown attempt", True)

