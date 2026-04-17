"""Local Ollama client for offline or on-device inference."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aletheia_ai.config import AppConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.local_llm_base_url.rstrip("/")
        self._model = config.local_llm_model
        self._api_key = config.local_llm_api_key

    @property
    def enabled(self) -> bool:
        return True

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        endpoint = self._resolve_generate_endpoint()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(timeout=90.0) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        text = str(data.get("response", "")).strip()
        if not text:
            raise RuntimeError("Ollama returned empty content")

        logger.info("Ollama response received", extra={"model": self._model})
        return text

    def _resolve_generate_endpoint(self) -> str:
        if self._base_url.endswith("/api"):
            return f"{self._base_url}/generate"
        return f"{self._base_url}/api/generate"
