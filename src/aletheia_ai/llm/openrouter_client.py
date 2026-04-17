"""Minimal OpenRouter chat-completions client for backup inference."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aletheia_ai.config import AppConfig

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, config: AppConfig) -> None:
        self._api_key = config.openrouter_api_key
        self._model = config.openrouter_model
        self._base_url = config.openrouter_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("OpenRouter API key is not configured")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aletheia-ai.local",
            "X-Title": "Aletheia AI",
        }

        with httpx.Client(timeout=45.0) as client:
            response = client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise RuntimeError("OpenRouter response did not include choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            merged = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    merged.append(str(part.get("text", "")))
            content = "\n".join(merged)

        text = str(content).strip()
        if not text:
            raise RuntimeError("OpenRouter returned empty content")

        logger.info("OpenRouter response received", extra={"model": self._model})
        return text
