"""
Evaluation harness for the FinOps optimization agent.

Seeds an isolated SQLite database with the golden dataset, runs the active
policy engine (LLM ReAct agent if configured, else rule-based) against every
case, and reports accuracy plus a per-case breakdown. This mirrors how a
retrieval/RAG project would report Recall@K or answer-correctness against a
labeled eval set, adapted to a decision-classification task.

Usage:
    python -m evaluation.evaluate_agent
"""

import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.graph import run_agent_optimization
from app.db.base import Base
from app.db.models import CloudInstance
from app.logging_config import configure_logging, get_logger
from app.services.billing_service import generate_synthetic_history
from evaluation.dataset import GOLDEN_DATASET

configure_logging()
logger = get_logger(__name__)


def _build_eval_db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def run_evaluation() -> dict:
    SessionLocal = _build_eval_db()

    # Monkeypatch the app-wide session factory so agent tools/rule-engine use this isolated DB.
    import app.db.base as db_base
    import app.agent.tools as tools_module

    original_session_local = db_base.SessionLocal
    db_base.SessionLocal = SessionLocal
    tools_module.SessionLocal = SessionLocal

    results = []
    try:
        with SessionLocal() as db:
            for case in GOLDEN_DATASET:
                instance = CloudInstance(
                    instance_id=case.instance_id,
                    name=case.name,
                    instance_type=case.instance_type,
                    tags={"criticality": case.criticality},
                )
                db.add(instance)
                db.commit()
                db.refresh(instance)
                generate_synthetic_history(db, instance, days=14, profile=case.usage_profile, seed=case.seed)

        correct = 0
        for case in GOLDEN_DATASET:
            actions, agent_source = run_agent_optimization(case.instance_id, dry_run=True)
            predicted = actions[0].action_type.value if actions else "no_action"
            is_correct = predicted == case.expected_action
            correct += int(is_correct)
            results.append(
                {
                    "instance_id": case.instance_id,
                    "expected": case.expected_action,
                    "predicted": predicted,
                    "correct": is_correct,
                    "agent_source": agent_source,
                    "reasoning": actions[0].reasoning if actions else None,
                }
            )
    finally:
        db_base.SessionLocal = original_session_local
        tools_module.SessionLocal = original_session_local

    accuracy = round(correct / len(GOLDEN_DATASET), 4)
    report = {"accuracy": accuracy, "total_cases": len(GOLDEN_DATASET), "correct": correct, "cases": results}

    out_path = Path(__file__).parent / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Evaluation accuracy: %.2f%% (%d/%d). Report written to %s",
                accuracy * 100, correct, len(GOLDEN_DATASET), out_path)
    return report


if __name__ == "__main__":
    run_evaluation()

