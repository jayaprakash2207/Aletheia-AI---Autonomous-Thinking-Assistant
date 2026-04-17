"""Centralized logging setup."""

from __future__ import annotations

import logging
import logging.config


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structured",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
            "loggers": {
                "httpx": {"level": "WARNING", "propagate": True},
                "google_genai.models": {"level": "WARNING", "propagate": True},
            },
        }
    )

    logging.getLogger(__name__).info("Logging configured", extra={"level": level})
