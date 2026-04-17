"""Central LLM router that prefers local inference, then cloud backups."""

from __future__ import annotations

from dataclasses import dataclass

from aletheia_ai.config import AppConfig
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.llm.openrouter_client import OpenRouterClient


@dataclass(slots=True)
class LLMRouter:
    local: OllamaClient | None
    openrouter: OpenRouterClient | None

    @classmethod
    def from_config(cls, config: AppConfig) -> "LLMRouter":
        local = OllamaClient(config) if config.local_llm_provider == "ollama" else None
        openrouter = OpenRouterClient(config)
        return cls(local=local, openrouter=openrouter)
