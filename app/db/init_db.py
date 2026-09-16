"""Database initialization helpers."""

from app.db.base import Base, engine
from app.logging_config import get_logger

logger = get_logger(__name__)


def init_db() -> None:
    """Create all tables if they do not already exist.

    Uses SQLAlchemy's metadata.create_all rather than a migrations framework
    (e.g. Alembic) because this project ships with SQLite for local/demo
    reproducibility. A production deployment against Postgres would introduce
    Alembic migrations at this integration point without touching callers.
    """
    logger.info("Initializing database schema")
    Base.metadata.create_all(bind=engine)
