"""Application configuration and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from aletheia_ai.core.exceptions import ConfigurationError


@dataclass(slots=True)
class AppConfig:
    gemini_api_key: str
    gemini_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    local_llm_provider: str
    local_llm_base_url: str
    local_llm_model: str
    local_llm_api_key: str | None
    log_level: str
    max_retries: int
    max_replans: int
    retry_backoff_seconds: float
    screenshot_path: str
    browser_headless: bool
    keep_browser_open: bool
    selenium_driver: str
    selenium_base_url: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY is required.")

        return cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip(),
            openrouter_api_key=_safe_optional_str("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip(),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            local_llm_provider=os.getenv("LOCAL_LLM_PROVIDER", "ollama").strip().lower(),
            local_llm_base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434").strip(),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b").strip(),
            local_llm_api_key=_safe_optional_str("LOCAL_LLM_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            max_retries=_safe_int("MAX_RETRIES", default=3),
            max_replans=_safe_int("MAX_REPLANS", default=2),
            retry_backoff_seconds=_safe_float("RETRY_BACKOFF_SECONDS", default=1.5),
            screenshot_path=os.getenv("SCREENSHOT_PATH", "./runtime/latest_screen.png").strip(),
            browser_headless=_safe_bool("BROWSER_HEADLESS", default=False),
            keep_browser_open=_safe_bool("KEEP_BROWSER_OPEN", default=True),
            selenium_driver=os.getenv("SELENIUM_DRIVER", "edge").strip().lower(),
            selenium_base_url=os.getenv("SELENIUM_BASE_URL", "https://www.google.com").strip(),
        )


def _safe_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if parsed < 0:
        raise ConfigurationError(f"{name} must be non-negative.")
    return parsed


def _safe_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if parsed < 0:
        raise ConfigurationError(f"{name} must be non-negative.")
    return parsed


def _safe_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean-like value.")


def _safe_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
