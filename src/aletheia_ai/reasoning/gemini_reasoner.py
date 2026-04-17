"""Gemini-backed reasoning engine with strict structured output parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai

from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import ReasoningError
from aletheia_ai.core.models import ReasoningOutput
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


class GeminiReasoner:
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

    def reason(self, task: str, context: str | None = None) -> ReasoningOutput:
        local_result = self._reason_via_local(task=task, context=context)
        if local_result is not None:
            return local_result

        openrouter_result = self._reason_via_openrouter(task=task, context=context)
        if openrouter_result is not None:
            return openrouter_result

        def _op() -> ReasoningOutput:
            prompt = self._build_prompt(task=task, context=context)
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )

            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise ReasoningError("Gemini returned empty reasoning output.")

            parsed = self._parse_response_json(text)
            goal = _require_str(parsed, "goal")
            steps = _require_str_list(parsed, "steps")
            risks = _require_str_list(parsed, "risks")

            strategy_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(steps)]
            strategy = "\n".join(strategy_lines)

            return ReasoningOutput(
                intent=goal,
                constraints=risks,
                strategy=strategy,
                success_criteria=[
                    "All planned steps are executed in order without critical failures.",
                    "Task goal is achieved and verifiable from current system state.",
                ],
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="gemini_reasoning")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini reasoning failed; using deterministic fallback", extra={"error": str(exc)})
            return self._fallback_reasoning(task=task, context=context)
    def _reason_via_local(self, task: str, context: str | None) -> ReasoningOutput | None:
        if self._router.local is None:
            return None

        def _op() -> ReasoningOutput:
            prompt = self._build_prompt(task=task, context=context)
            text = self._router.local.generate_text(prompt)
            parsed = self._parse_response_json(text)
            goal = _require_str(parsed, "goal")
            steps = _require_str_list(parsed, "steps")
            risks = _require_str_list(parsed, "risks")

            strategy_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(steps)]
            strategy = "\n".join(strategy_lines)
            return ReasoningOutput(
                intent=goal,
                constraints=risks,
                strategy=strategy,
                success_criteria=[
                    "All planned steps are executed in order without critical failures.",
                    "Task goal is achieved and verifiable from current system state.",
                ],
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_reasoning")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local reasoning failed", extra={"error": str(exc)})
            return None

    def _reason_via_openrouter(self, task: str, context: str | None) -> ReasoningOutput | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> ReasoningOutput:
            prompt = self._build_prompt(task=task, context=context)
            text = self._openrouter.generate_text(prompt)
            parsed = self._parse_response_json(text)
            goal = _require_str(parsed, "goal")
            steps = _require_str_list(parsed, "steps")
            risks = _require_str_list(parsed, "risks")

            strategy_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(steps)]
            strategy = "\n".join(strategy_lines)
            return ReasoningOutput(
                intent=goal,
                constraints=risks,
                strategy=strategy,
                success_criteria=[
                    "All planned steps are executed in order without critical failures.",
                    "Task goal is achieved and verifiable from current system state.",
                ],
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_reasoning")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter reasoning failed", extra={"error": str(exc)})
            return None

    def _fallback_reasoning(self, task: str, context: str | None = None) -> ReasoningOutput:
        constraints = [
            "LLM quota may be unavailable; use deterministic fallback behavior.",
            "Validate state after every action before continuing.",
        ]
        if context:
            constraints.append(f"Context: {context}")

        strategy = "\n".join(
            [
                "1. Open required application or browser.",
                "2. Navigate to target interface.",
                "3. Input required text or command.",
                "4. Validate visible result and retry safely if needed.",
            ]
        )

        return ReasoningOutput(
            intent=task,
            constraints=constraints,
            strategy=strategy,
            success_criteria=[
                "Executed actions complete without unrecovered failure.",
                "Visible screen state indicates expected progress.",
            ],
        )

    def _build_prompt(self, task: str, context: str | None) -> str:
        context_section = context or "No additional context provided."
        return (
            "You are an AI reasoning engine. Think before acting. "
            "Analyze the task deeply and return only strict JSON with the required schema.\n"
            "Task to reason about: {user_input}\n"
            "Instructions:\n"
            "1. Understand the goal\n"
            "2. Break it into logical steps\n"
            "3. Identify dependencies\n"
            "4. Predict possible failures\n\n"
            "Output format:\n"
            "{\n"
            '  "goal": "...",\n'
            '  "steps": ["step1", "step2", "step3"],\n'
            '  "risks": ["possible failure 1", "possible failure 2"]\n'
            "}\n\n"
            f"Task: {task}\n"
            f"Context: {context_section}\n"
            "Dependency identification requirement: encode dependencies directly inside each step when needed. "
            "Example: 'Ensure ChromeDriver is installed before launching browser'.\n"
            "Output only valid JSON, no markdown, no extra keys."
        )

    def _parse_response_json(self, text: str) -> dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = _strip_markdown_fence(raw)
        raw = _extract_json_fragment(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse reasoning JSON", extra={"payload": raw})
            raise ReasoningError("Reasoning response is not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise ReasoningError("Reasoning JSON root must be an object.")
        return parsed


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1)


def _extract_json_fragment(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReasoningError(f"Field '{key}' must be a non-empty string.")
    return value.strip()


def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ReasoningError(f"Field '{key}' must be a non-empty list.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ReasoningError(f"Field '{key}' contains invalid list items.")
        normalized.append(item.strip())
    return normalized
