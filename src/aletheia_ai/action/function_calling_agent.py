"""Gemini function-calling style planner for tool invocation decisions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai

from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import ActionExecutionError
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


class FunctionCallingAgent:
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

    def decide_function(self, user_input: str) -> dict[str, Any]:
        local_payload = self._decide_with_local(user_input)
        if local_payload is not None:
            return local_payload

        openrouter_payload = self._decide_with_openrouter(user_input)
        if openrouter_payload is not None:
            return openrouter_payload

        def _op() -> dict[str, Any]:
            prompt = self._build_prompt(user_input)
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )
            raw_text = getattr(response, "text", "") or ""
            if not raw_text.strip():
                raise ActionExecutionError("Function-calling planner returned empty output.")

            payload = _parse_json_object(raw_text)
            if "function" not in payload:
                payload["function"] = "none"
            if "arguments" not in payload or not isinstance(payload["arguments"], dict):
                payload["arguments"] = {}
            return payload

        try:
            return with_retry(_op, self._retry_policy, op_name="function_calling_decision")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini function-calling failed; falling back to no-function decision", extra={"error": str(exc)})
            return {"function": "none", "arguments": {}}

    def _decide_with_openrouter(self, user_input: str) -> dict[str, Any] | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> dict[str, Any]:
            prompt = self._build_prompt(user_input)
            raw_text = self._openrouter.generate_text(prompt)
            payload = _parse_json_object(raw_text)
            if "function" not in payload:
                payload["function"] = "none"
            if "arguments" not in payload or not isinstance(payload["arguments"], dict):
                payload["arguments"] = {}
            return payload

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_function_calling")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter function-calling failed", extra={"error": str(exc)})
            return None

    def _decide_with_local(self, user_input: str) -> dict[str, Any] | None:
        if self._router.local is None:
            return None

        def _op() -> dict[str, Any]:
            prompt = self._build_prompt(user_input)
            raw_text = self._local.generate_text(prompt)
            payload = _parse_json_object(raw_text)
            if "function" not in payload:
                payload["function"] = "none"
            if "arguments" not in payload or not isinstance(payload["arguments"], dict):
                payload["arguments"] = {}
            return payload

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_function_calling")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local function-calling failed", extra={"error": str(exc)})
            return None

    def _build_prompt(self, user_input: str) -> str:
        return (
            "You are an AI agent that can call tools.\n\n"
            "Available functions:\n"
            "- open_app(name)\n"
            "- click(x, y)\n"
            "- type(text)\n"
            "- open_url(url)\n\n"
            f"Task: {user_input}\n\n"
            "Decide:\n"
            "- Whether to call a function\n"
            "- Which function + arguments\n\n"
            "Return JSON only.\n"
            "Schema:\n"
            "{\n"
            '  "function": "open_app | click | type | open_url | none",\n'
            '  "arguments": {}\n'
            "}"
        )


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\\s*(.*?)\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid function-calling JSON", extra={"payload": text})
        raise ActionExecutionError("Function-calling response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ActionExecutionError("Function-calling response root must be an object.")
    return payload
