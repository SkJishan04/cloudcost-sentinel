"""Shared pytest fixtures: isolated in-memory DB and FastAPI test client."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import CloudInstance
from app.services.billing_service import generate_synthetic_history


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def app_client(db_session, monkeypatch):
    """FastAPI TestClient wired to the isolated in-memory session via dependency override."""
    from app.api.deps import get_db
    from app.main import app

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def idle_prod_instance(db_session):
    instance = CloudInstance(
        instance_id="i-test-idle-prod",
        name="test-idle-prod",
        instance_type="m5.xlarge",
        tags={"criticality": "production"},
    )
    db_session.add(instance)
    db_session.commit()
    db_session.refresh(instance)
    generate_synthetic_history(db_session, instance, days=14, profile="idle", seed=42)
    return instance


@pytest.fixture()
def idle_nonprod_instance(db_session):
    instance = CloudInstance(
        instance_id="i-test-idle-nonprod",
        name="test-idle-nonprod",
        instance_type="t3.large",
        tags={"criticality": "non-production"},
    )
    db_session.add(instance)
    db_session.commit()
    db_session.refresh(instance)
    generate_synthetic_history(db_session, instance, days=14, profile="idle", seed=43)
    return instance
