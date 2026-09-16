"""Structured logging configuration shared across the application."""

import logging
import sys

from app.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    if root.handlers:
        # Avoid duplicate handlers on reload (uvicorn --reload re-imports modules).
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet down noisy third-party loggers while keeping our own at the configured level.
    for noisy in ("httpx", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)