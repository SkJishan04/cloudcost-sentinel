"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, forecast, instances, usage
from app.config import get_settings
from app.core.exceptions import FinOpsError, finops_exception_handler
from app.db.init_db import init_db
from app.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s, llm_provider=%s, llm_enabled=%s)",
                settings.APP_NAME, settings.ENV, settings.LLM_PROVIDER, settings.llm_enabled)
    init_db()

    if settings.AUTO_SEED:
        from app.db.base import SessionLocal
        from app.db.models import CloudInstance
        with SessionLocal() as db:
            if db.query(CloudInstance).count() == 0:
                logger.info("No instances found; auto-seeding demo dataset")
                from scripts.seed_data import seed
                seed(db)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Autonomous FinOps agent that forecasts cloud compute utilization and "
        "proposes/executes cost-optimizing resize and shutdown actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(FinOpsError, finops_exception_handler)

app.include_router(instances.router)
app.include_router(usage.router)
app.include_router(forecast.router)
app.include_router(agent.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "env": settings.ENV, "llm_enabled": settings.llm_enabled}

