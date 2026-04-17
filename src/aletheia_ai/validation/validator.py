"""LLM-backed validation agent for action success verification."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai

from aletheia_ai.config import AppConfig
from aletheia_ai.core.models import ActionResult, ActionType, PlanStep, ValidationResult, VisionSnapshot
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


class Validator:
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

    def validate(
        self,
        step: PlanStep,
        action_result: ActionResult,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
        success_criteria: list[str],
    ) -> ValidationResult:
        if action_result.status.value != "succeeded":
            return ValidationResult(
                passed=False,
                confidence=0.05,
                rationale=f"Execution failed before validation: {action_result.details}",
                suggested_correction="Retry",
                next_step="retry",
            )

        # Fast-path heuristic for reliable actions that typically cannot be semantically judged
        # by an LLM from coarse screenshot summaries.
        heuristic = self._deterministic_success_check(step, previous_snapshot, current_snapshot)
        if heuristic is not None:
            return heuristic

        local_validation = self._validate_via_local(
            step=step,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
            success_criteria=success_criteria,
        )
        if local_validation is not None:
            return local_validation

        openrouter_validation = self._validate_via_openrouter(
            step=step,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
            success_criteria=success_criteria,
        )
        if openrouter_validation is not None:
            return openrouter_validation

        def _op() -> ValidationResult:
            prompt = self._build_prompt(
                step=step,
                previous_snapshot=previous_snapshot,
                current_snapshot=current_snapshot,
                success_criteria=success_criteria,
            )
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise ValueError("Validation output is empty")

            payload = _parse_json_object(text)
            status = _require_str(payload, "status").strip().lower()
            reason = _require_str(payload, "reason")
            next_step = _require_str(payload, "next_step").strip().lower()

            passed = status == "success"
            confidence = 0.9 if passed else 0.2
            suggested_correction = "" if passed else reason

            result = ValidationResult(
                passed=passed,
                confidence=confidence,
                rationale=reason,
                suggested_correction=suggested_correction,
                next_step=next_step if next_step in {"continue", "retry"} else ("continue" if passed else "retry"),
            )
            logger.info(
                "Step validated",
                extra={
                    "step_id": step.id,
                    "passed": result.passed,
                    "next_step": result.next_step,
                    "rationale": result.rationale,
                },
            )
            return result

        try:
            return with_retry(_op, self._retry_policy, op_name="step_validation")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini validation failed; using heuristic validation", extra={"error": str(exc)})
            return self._fallback_validation(previous_snapshot=previous_snapshot, current_snapshot=current_snapshot)

    def _deterministic_success_check(
        self,
        step: PlanStep,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
    ) -> ValidationResult | None:
        changed = previous_snapshot.summary.strip() != current_snapshot.summary.strip()

        if step.action_type == ActionType.BROWSER_OPEN:
            return ValidationResult(
                passed=True,
                confidence=0.95,
                rationale="Browser open action executed successfully.",
                suggested_correction="",
                next_step="continue",
            )

        if step.action_type == ActionType.OPEN_APP:
            return ValidationResult(
                passed=True,
                confidence=0.9,
                rationale="Application launch action executed successfully.",
                suggested_correction="",
                next_step="continue",
            )

        if step.action_type in {ActionType.BROWSER_CLICK, ActionType.BROWSER_TYPE}:
            return ValidationResult(
                passed=True,
                confidence=0.9,
                rationale="Browser interaction executed successfully.",
                suggested_correction="",
                next_step="continue",
            )

        if step.action_type == ActionType.KEYBOARD_HOTKEY and changed:
            return ValidationResult(
                passed=True,
                confidence=0.8,
                rationale="Hotkey executed and screen state changed.",
                suggested_correction="",
                next_step="continue",
            )

        if step.action_type == ActionType.KEYBOARD_WRITE:
            return ValidationResult(
                passed=True,
                confidence=0.75 if changed else 0.6,
                rationale=(
                    "Keyboard typing executed with observable screen change."
                    if changed
                    else "Keyboard typing executed; continuing without strict visual assertion."
                ),
                suggested_correction="",
                next_step="continue",
            )

        return None

    def _validate_via_local(
        self,
        step: PlanStep,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
        success_criteria: list[str],
    ) -> ValidationResult | None:
        if self._router.local is None:
            return None

        def _op() -> ValidationResult:
            prompt = self._build_prompt(
                step=step,
                previous_snapshot=previous_snapshot,
                current_snapshot=current_snapshot,
                success_criteria=success_criteria,
            )
            text = self._router.local.generate_text(prompt)
            payload = _parse_json_object(text)
            status = _require_str(payload, "status").strip().lower()
            reason = _require_str(payload, "reason")
            next_step = _require_str(payload, "next_step").strip().lower()
            passed = status == "success"
            confidence = 0.9 if passed else 0.2
            suggested_correction = "" if passed else reason
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                rationale=reason,
                suggested_correction=suggested_correction,
                next_step=next_step if next_step in {"continue", "retry"} else ("continue" if passed else "retry"),
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_step_validation")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local validation failed", extra={"error": str(exc)})
            return None

    def _validate_via_openrouter(
        self,
        step: PlanStep,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
        success_criteria: list[str],
    ) -> ValidationResult | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> ValidationResult:
            prompt = self._build_prompt(
                step=step,
                previous_snapshot=previous_snapshot,
                current_snapshot=current_snapshot,
                success_criteria=success_criteria,
            )
            text = self._openrouter.generate_text(prompt)
            payload = _parse_json_object(text)
            status = _require_str(payload, "status").strip().lower()
            reason = _require_str(payload, "reason")
            next_step = _require_str(payload, "next_step").strip().lower()

            passed = status == "success"
            confidence = 0.9 if passed else 0.2
            suggested_correction = "" if passed else reason
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                rationale=reason,
                suggested_correction=suggested_correction,
                next_step=next_step if next_step in {"continue", "retry"} else ("continue" if passed else "retry"),
            )

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_step_validation")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter validation failed", extra={"error": str(exc)})
            return None

    def _fallback_validation(
        self,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
    ) -> ValidationResult:
        changed = previous_snapshot.summary.strip() != current_snapshot.summary.strip()
        return ValidationResult(
            passed=changed,
            confidence=0.6 if changed else 0.2,
            rationale="Screen state changed" if changed else "No significant screen change detected",
            suggested_correction="Try an alternative UI interaction" if not changed else "",
            next_step="continue" if changed else "retry",
        )

    def _build_prompt(
        self,
        step: PlanStep,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
        success_criteria: list[str],
    ) -> str:
        return (
            "You are a validation agent.\n\n"
            "Check if the previous action succeeded.\n\n"
            f"Goal: {step.description}\n"
            f"Before: {previous_snapshot.summary}; elements={previous_snapshot.key_elements}\n"
            f"After: {current_snapshot.summary}; elements={current_snapshot.key_elements}\n"
            f"Success criteria: {success_criteria}\n\n"
            "Output:\n"
            "{\n"
            '  "status": "success / failure",\n'
            '  "reason": "...",\n'
            '  "next_step": "continue / retry"\n'
            "}\n"
            "Output only valid JSON, no markdown, no extra keys."
        )


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
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

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Validation response root must be an object")
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{key}' must be a non-empty string")
    return value.strip()
