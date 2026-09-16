"""
Centralized, environment-driven configuration.

All runtime configuration flows through this module so that no secret or
environment-specific value is hardcoded elsewhere in the codebase.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    APP_NAME: str = "FinOps Optimization Agent"
    ENV: Literal["development", "test", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite:///./finops.db"

    # LLM provider
    LLM_PROVIDER: Literal["anthropic", "openai", "none"] = "none"
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "claude-3-5-sonnet-20241022"

    # FinOps policy thresholds
    UNDERUTILIZED_CPU_THRESHOLD: float = 15.0
    FORECAST_HORIZON_HOURS: int = 168
    HISTORY_LOOKBACK_DAYS: int = 14
    DRY_RUN_DEFAULT: bool = True

    # Bootstrap behavior
    AUTO_SEED: bool = True
    CORS_ORIGINS: str = "*"

    @property
    def llm_enabled(self) -> bool:
        """True only when a provider is selected AND its API key is present."""
        if self.LLM_PROVIDER == "anthropic":
            return bool(self.ANTHROPIC_API_KEY)
        if self.LLM_PROVIDER == "openai":
            return bool(self.OPENAI_API_KEY)
        return False


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor; avoids re-parsing the environment on every call."""
    return Settings()
