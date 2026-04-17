"""Task planner that transforms reasoning into executable typed steps."""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote_plus
from typing import Any

from google import genai

from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import PlanningError
from aletheia_ai.core.models import ActionType, PlanStep, ReasoningOutput, TaskPlan
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)
MAX_PLAN_STEPS = 6


class TaskPlanner:
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

    def create_plan(self, task: str, reasoning: ReasoningOutput) -> TaskPlan:
        normalized_task = task.lower()
        # Prefer deterministic plans for highly-structured automation intents.
        if "calculator" in normalized_task or "calc" in normalized_task or "search" in normalized_task:
            return self._fallback_plan(task)

        local_plan = self._plan_via_local(task=task, reasoning=reasoning)
        if local_plan is not None:
            return local_plan

        openrouter_plan = self._plan_via_openrouter(task=task, reasoning=reasoning)
        if openrouter_plan is not None:
            return openrouter_plan

        def _op() -> TaskPlan:
            prompt = self._build_prompt(task=task, reasoning=reasoning)
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise PlanningError("Planner received empty output from Gemini.")

            steps_payload = self._parse_plan_json(text)

            steps: list[PlanStep] = []
            for idx, item in enumerate(steps_payload, start=1):
                if not isinstance(item, dict):
                    raise PlanningError("Each step must be a JSON object.")

                step_number = _require_int(item, "step")
                action_text = _require_str(item, "action")
                action_type, parameters = _infer_execution_spec(action_text)

                steps.append(
                    PlanStep(
                        id=step_number if step_number > 0 else idx,
                        description=action_text,
                        action_type=action_type,
                        parameters=parameters,
                        validation_hint=action_text,
                    )
                )

            steps.sort(key=lambda s: s.id)
            if len(steps) > MAX_PLAN_STEPS:
                logger.warning("Planner produced too many steps; truncating", extra={"count": len(steps)})
                steps = steps[:MAX_PLAN_STEPS]
            if not _is_plan_sane(task=task, steps=steps):
                logger.warning("Planner output appears off-task; using deterministic fallback")
                return self._fallback_plan(task)
            return TaskPlan(goal=task, steps=steps)

        try:
            return with_retry(_op, self._retry_policy, op_name="task_planning")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini planning failed; using deterministic fallback", extra={"error": str(exc)})
            return self._fallback_plan(task)
    def _plan_via_local(self, task: str, reasoning: ReasoningOutput) -> TaskPlan | None:
        if self._router.local is None:
            return None

        def _op() -> TaskPlan:
            prompt = self._build_prompt(task=task, reasoning=reasoning)
            text = self._router.local.generate_text(prompt)
            steps_payload = self._parse_plan_json(text)
            steps: list[PlanStep] = []
            for idx, item in enumerate(steps_payload, start=1):
                if not isinstance(item, dict):
                    raise PlanningError("Each step must be a JSON object.")
                step_number = _require_int(item, "step")
                action_text = _require_str(item, "action")
                action_type, parameters = _infer_execution_spec(action_text)
                steps.append(
                    PlanStep(
                        id=step_number if step_number > 0 else idx,
                        description=action_text,
                        action_type=action_type,
                        parameters=parameters,
                        validation_hint=action_text,
                    )
                )
            steps.sort(key=lambda s: s.id)
            if len(steps) > MAX_PLAN_STEPS:
                logger.warning("Local planner produced too many steps; truncating", extra={"count": len(steps)})
                steps = steps[:MAX_PLAN_STEPS]
            if not _is_plan_sane(task=task, steps=steps):
                logger.warning("Local planner output appears off-task; using deterministic fallback")
                return self._fallback_plan(task)
            return TaskPlan(goal=task, steps=steps)

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_planning")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local planning failed", extra={"error": str(exc)})
            return None

    def _plan_via_openrouter(self, task: str, reasoning: ReasoningOutput) -> TaskPlan | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> TaskPlan:
            prompt = self._build_prompt(task=task, reasoning=reasoning)
            text = self._openrouter.generate_text(prompt)
            steps_payload = self._parse_plan_json(text)

            steps: list[PlanStep] = []
            for idx, item in enumerate(steps_payload, start=1):
                if not isinstance(item, dict):
                    raise PlanningError("Each step must be a JSON object.")
                step_number = _require_int(item, "step")
                action_text = _require_str(item, "action")
                action_type, parameters = _infer_execution_spec(action_text)
                steps.append(
                    PlanStep(
                        id=step_number if step_number > 0 else idx,
                        description=action_text,
                        action_type=action_type,
                        parameters=parameters,
                        validation_hint=action_text,
                    )
                )

            steps.sort(key=lambda s: s.id)
            if len(steps) > MAX_PLAN_STEPS:
                logger.warning("OpenRouter planner produced too many steps; truncating", extra={"count": len(steps)})
                steps = steps[:MAX_PLAN_STEPS]
            if not _is_plan_sane(task=task, steps=steps):
                logger.warning("OpenRouter planner output appears off-task; using deterministic fallback")
                return self._fallback_plan(task)
            return TaskPlan(goal=task, steps=steps)

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_planning")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter planning failed", extra={"error": str(exc)})
            return None

    def _fallback_plan(self, task: str) -> TaskPlan:
        normalized = task.lower()
        steps: list[PlanStep] = []

        if "calculator" in normalized or "calc" in normalized:
            expression = _extract_calculator_expression(task)
            steps.append(
                PlanStep(
                    id=1,
                    description="Open calculator app",
                    action_type=ActionType.OPEN_APP,
                    parameters={"name": "calculator"},
                    validation_hint="calculator opened",
                )
            )
            steps.append(
                PlanStep(
                    id=2,
                    description="Wait for calculator to open",
                    action_type=ActionType.WAIT,
                    parameters={"seconds": 2.5},
                    validation_hint="calculator ready",
                )
            )
            steps.append(
                PlanStep(
                    id=3,
                    description="Type calculator expression",
                    action_type=ActionType.KEYBOARD_WRITE,
                    parameters={"text": expression},
                    validation_hint="expression entered",
                )
            )
            steps.append(
                PlanStep(
                    id=4,
                    description="Evaluate expression",
                    action_type=ActionType.KEYBOARD_HOTKEY,
                    parameters={"keys": ["enter"]},
                    validation_hint="result visible",
                )
            )
            return TaskPlan(goal=task, steps=steps)

        if "search" in normalized:
            search_query = _extract_search_query(task)
            search_url = "https://duckduckgo.com/"
            steps.append(
                PlanStep(
                    id=1,
                    description="Open browser",
                    action_type=ActionType.BROWSER_OPEN,
                    parameters={"url": search_url},
                    validation_hint="search page visible",
                )
            )
            steps.append(
                PlanStep(
                    id=2,
                    description="Focus search bar",
                    action_type=ActionType.BROWSER_CLICK,
                    parameters={"css_selector": "input[name='q']"},
                    validation_hint="search bar focused",
                )
            )
            steps.append(
                PlanStep(
                    id=3,
                    description="Type search query",
                    action_type=ActionType.BROWSER_TYPE,
                    parameters={"css_selector": "input[name='q']", "text": search_query, "clear_first": True},
                    validation_hint="query visible in search bar",
                )
            )
            steps.append(
                PlanStep(
                    id=4,
                    description="Click search button",
                    action_type=ActionType.BROWSER_CLICK,
                    parameters={"css_selector": "button[type='submit']"},
                    validation_hint="search results visible",
                )
            )
            return TaskPlan(goal=task, steps=steps)

        steps.append(
            PlanStep(
                id=1,
                description="Open browser",
                action_type=ActionType.BROWSER_OPEN,
                parameters={},
                validation_hint="browser opened",
            )
        )

        return TaskPlan(goal=task, steps=steps)

    def _build_prompt(self, task: str, reasoning: ReasoningOutput) -> str:
        reasoning_json = json.dumps(
            {
                "intent": reasoning.intent,
                "constraints": reasoning.constraints,
                "strategy": reasoning.strategy,
                "success_criteria": reasoning.success_criteria,
            },
            ensure_ascii=True,
        )
        return (
            "You are a task planner AI.\n\n"
            "Convert the given goal into clear executable steps.\n\n"
            "Rules:\n"
            "- Each step must be atomic (single action)\n"
            "- No ambiguity\n"
            "- Must be executable on a computer\n\n"
            f"Task: {task}\n"
            f"Reasoning context: {reasoning_json}\n\n"
            "Output:\n"
            "[\n"
            '  {"step": 1, "action": "..."},\n'
            '  {"step": 2, "action": "..."}\n'
            "]\n"
            "Output only valid JSON array, no markdown, no extra keys."
        )

    def _parse_plan_json(self, raw_text: str) -> list[dict[str, Any]]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = _strip_markdown_fence(text)
        text = _extract_json_fragment(text)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Planner returned invalid JSON", extra={"payload": text})
            raise PlanningError("Planner response is not valid JSON.") from exc

        if not isinstance(payload, list) or not payload:
            raise PlanningError("Planner response root must be a non-empty array.")

        normalized: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise PlanningError("Each planner list item must be an object.")
            normalized.append(item)
        return normalized


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1)


def _extract_json_fragment(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _extract_search_query(task: str) -> str:
    normalized = task.strip()
    patterns = [
        r"^open browser and search for\s+(.+)$",
        r"^search for\s+(.+)$",
        r"^search\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return normalized


def _extract_calculator_expression(task: str) -> str:
    normalized = task.strip().lower()

    add_match = re.search(r"add\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)", normalized)
    if add_match:
        return f"{add_match.group(1)}+{add_match.group(2)}"

    sub_match = re.search(
        r"subtract\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)",
        normalized,
    )
    if sub_match:
        return f"{sub_match.group(2)}-{sub_match.group(1)}"

    mul_match = re.search(r"multiply\s+(-?\d+(?:\.\d+)?)\s+(?:and|by)\s+(-?\d+(?:\.\d+)?)", normalized)
    if mul_match:
        return f"{mul_match.group(1)}*{mul_match.group(2)}"

    div_match = re.search(r"divide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", normalized)
    if div_match:
        return f"{div_match.group(1)}/{div_match.group(2)}"

    generic_expr = re.search(r"(-?\d+(?:\.\d+)?\s*[+\-*/]\s*-?\d+(?:\.\d+)?)", normalized)
    if generic_expr:
        return generic_expr.group(1).replace(" ", "")

    return "5+3"


def _parse_action_type(raw: Any) -> ActionType:
    if not isinstance(raw, str):
        raise PlanningError("Step action_type must be a string.")
    normalized = raw.strip().lower()
    try:
        return ActionType(normalized)
    except ValueError as exc:
        raise PlanningError(f"Unsupported action_type: {normalized}") from exc


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanningError(f"Field '{key}' must be a non-empty string.")
    return value.strip()


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PlanningError(f"Field '{key}' must be an object.")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PlanningError(f"Field '{key}' must be a string when provided.")
    return value.strip()


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise PlanningError(f"Field '{key}' must be an integer.")
    return value


def _infer_execution_spec(action_text: str) -> tuple[ActionType, dict[str, Any]]:
    action = action_text.lower().strip()

    if action.startswith("open browser") or action.startswith("open the browser"):
        return ActionType.BROWSER_OPEN, {}

    if action.startswith("open url"):
        url = action_text.split(":", maxsplit=1)[1].strip() if ":" in action_text else ""
        return ActionType.BROWSER_OPEN, {"url": url} if url else {}

    if action.startswith("click") and "selector" in action:
        selector = _extract_after_keyword(action_text, "selector")
        return ActionType.BROWSER_CLICK, {"css_selector": selector}

    if action.startswith("type") and "selector" in action:
        selector = _extract_after_keyword(action_text, "selector")
        return ActionType.BROWSER_TYPE, {"css_selector": selector, "text": ""}

    if action.startswith("press"):
        keys = [part.strip() for part in re.split(r"\+", action_text.replace("Press", "").replace("press", "")) if part.strip()]
        if keys:
            return ActionType.KEYBOARD_HOTKEY, {"keys": keys}

    if action.startswith("wait"):
        seconds_match = re.search(r"(\d+(?:\.\d+)?)", action)
        seconds = float(seconds_match.group(1)) if seconds_match else 1.0
        return ActionType.WAIT, {"seconds": seconds}

    return ActionType.KEYBOARD_WRITE, {"text": action_text}


def _extract_after_keyword(text: str, keyword: str) -> str:
    pattern = re.compile(rf"{keyword}\s*[:=]\s*(.+)$", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        raise PlanningError(f"Could not extract value for keyword '{keyword}' from action: {text}")
    return match.group(1).strip()


def _is_plan_sane(task: str, steps: list[PlanStep]) -> bool:
    if not steps:
        return False

    task_text = task.lower()
    all_descriptions = " ".join(step.description.lower() for step in steps)

    off_topic_tokens = {"chromedriver", "webdriver", "binary", "executable", "system path"}
    if any(token in all_descriptions for token in off_topic_tokens) and not any(
        token in task_text for token in off_topic_tokens
    ):
        return False

    if "search" in task_text:
        has_open = any("open" in step.description.lower() and "browser" in step.description.lower() for step in steps)
        has_searchish = any(
            keyword in step.description.lower() for step in steps for keyword in ("search", "query", "type", "enter")
        )
        if not (has_open and has_searchish):
            return False

    return True
