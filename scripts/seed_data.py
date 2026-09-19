"""Seed the database with a believable fleet of instances and synthetic usage history."""

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import SessionLocal
from app.db.init_db import init_db
from app.db.models import CloudInstance
from app.logging_config import configure_logging, get_logger
from app.services.billing_service import generate_synthetic_history

configure_logging()
logger = get_logger(__name__)

FLEET = [
    dict(
        instance_id="i-prod-api-01",
        name="prod-api-gateway",
        instance_type="m5.2xlarge",
        tags={"criticality": "production", "team": "platform"},
        profile="business_hours",
        seed=1,
    ),
    dict(
        instance_id="i-prod-db-01",
        name="prod-analytics-db",
        instance_type="m5.4xlarge",
        tags={"criticality": "production", "team": "data"},
        profile="steady_high",
        seed=2,
    ),
    dict(
        instance_id="i-staging-web-01",
        name="staging-web-frontend",
        instance_type="t3.xlarge",
        tags={"criticality": "non-production", "team": "web"},
        profile="idle",
        seed=3,
    ),
    dict(
        instance_id="i-dev-batch-01",
        name="dev-batch-worker",
        instance_type="t3.large",
        tags={"criticality": "non-production", "team": "ml"},
        profile="idle",
        seed=4,
    ),
    dict(
        instance_id="i-prod-worker-01",
        name="prod-video-encoder",
        instance_type="t3.medium",
        tags={"criticality": "production", "team": "media"},
        profile="spiky",
        seed=5,
    ),
    dict(
        instance_id="i-qa-loadtest-01",
        name="qa-loadtest-runner",
        instance_type="m5.xlarge",
        tags={"criticality": "non-production", "team": "qa"},
        profile="idle",
        seed=6,
    ),
]


def seed(db: Session) -> None:
    settings = get_settings()
    for spec in FLEET:
        instance = CloudInstance(
            instance_id=spec["instance_id"],
            name=spec["name"],
            instance_type=spec["instance_type"],
            region="us-east-1",
            provider="aws-simulated",
            tags=spec["tags"],
        )
        db.add(instance)
        db.commit()
        db.refresh(instance)
        generate_synthetic_history(
            db, instance, days=settings.HISTORY_LOOKBACK_DAYS, profile=spec["profile"], seed=spec["seed"]
        )
        logger.info("Seeded %s (%s)", instance.instance_id, spec["profile"])


if __name__ == "__main__":
    init_db()
    with SessionLocal() as session:
        seed(session)
    logger.info("Seed complete.")