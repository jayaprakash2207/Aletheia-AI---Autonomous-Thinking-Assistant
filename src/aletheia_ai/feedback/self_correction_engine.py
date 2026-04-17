"""LLM-based self-correction strategy generator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai

from aletheia_ai.config import AppConfig
from aletheia_ai.core.models import PlanStep, SelfCorrectionOutput
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


class SelfCorrectionEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._router = LLMRouter.from_config(config)
        self._openrouter = OpenRouterClient(config)
        self._local = OllamaClient(config)
        self._retry_policy = RetryPolicy(
            max_attempts=max(1, config.max_retries),
            backoff_seconds=config.retry_backoff_seconds,
        )

    def self_correct(self, step: PlanStep, reason: str) -> SelfCorrectionOutput:
        local_correction = self._self_correct_via_local(step=step, reason=reason)
        if local_correction is not None:
            return local_correction

        openrouter_correction = self._self_correct_via_openrouter(step=step, reason=reason)
        if openrouter_correction is not None:
            return openrouter_correction

        def _op() -> SelfCorrectionOutput:
            prompt = self._build_prompt(step=step, reason=reason)
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise ValueError("Self-correction output is empty")

            payload = _parse_json_object(text)
            return SelfCorrectionOutput(
                new_strategy=_require_str(payload, "new_strategy"),
                updated_action=_require_str(payload, "updated_action"),
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="self_correction")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini self-correction failed; using heuristic fallback", extra={"error": str(exc)})
            return SelfCorrectionOutput(
                new_strategy="Use a simpler alternative interaction and add wait before next action.",
                updated_action=f"Retry carefully: {step.description}",
            )
    def _self_correct_via_local(self, step: PlanStep, reason: str) -> SelfCorrectionOutput | None:
        if self._router.local is None:
            return None

        def _op() -> SelfCorrectionOutput:
            prompt = self._build_prompt(step=step, reason=reason)
            text = self._router.local.generate_text(prompt)
            payload = _parse_json_object(text)
            return SelfCorrectionOutput(
                new_strategy=_require_str(payload, "new_strategy"),
                updated_action=_require_str(payload, "updated_action"),
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_self_correction")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local self-correction failed", extra={"error": str(exc)})
            return None

    def _self_correct_via_openrouter(self, step: PlanStep, reason: str) -> SelfCorrectionOutput | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> SelfCorrectionOutput:
            prompt = self._build_prompt(step=step, reason=reason)
            text = self._openrouter.generate_text(prompt)
            payload = _parse_json_object(text)
            return SelfCorrectionOutput(
                new_strategy=_require_str(payload, "new_strategy"),
                updated_action=_require_str(payload, "updated_action"),
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_self_correction")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter self-correction failed", extra={"error": str(exc)})
            return None

    def _build_prompt(self, step: PlanStep, reason: str) -> str:
        return (
            "You are a self-correcting AI.\n\n"
            "The previous action failed.\n\n"
            f"Goal: {step.description}\n"
            f"Error: {reason}\n\n"
            "Instructions:\n"
            "- Analyze what went wrong\n"
            "- Suggest alternative approach\n"
            "- Modify plan\n\n"
            "Output:\n"
            "{\n"
            '  "new_strategy": "...",\n'
            '  "updated_action": "..."\n'
            "}\n"
            "Output only valid JSON, no markdown, no extra keys."
        )


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\\s*(.*?)\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1)


def _extract_json_fragment(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)
        text = _extract_json_fragment(text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Self-correction response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Self-correction response root must be an object")
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{key}' must be a non-empty string")
    return value.strip()
