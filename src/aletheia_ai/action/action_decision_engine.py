"""LLM-driven action decision engine that chooses one precise executable action."""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

from google import genai

from aletheia_ai.action.function_calling_agent import FunctionCallingAgent
from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import ActionExecutionError
from aletheia_ai.core.models import ActionDecision, ActionType, PlanStep, VisionSnapshot
from aletheia_ai.llm.llm_router import LLMRouter
from aletheia_ai.llm.openrouter_client import OpenRouterClient
from aletheia_ai.llm.ollama_client import OllamaClient
from aletheia_ai.utils.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


class ActionDecisionEngine:
    def __init__(self, config: AppConfig, function_agent: FunctionCallingAgent) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._router = LLMRouter.from_config(config)
        self._openrouter = OpenRouterClient(config)
        self._local = OllamaClient(config)
        self._function_agent = function_agent
        self._retry_policy = RetryPolicy(
            max_attempts=max(1, config.max_retries),
            backoff_seconds=config.retry_backoff_seconds,
        )

    def decide(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> ActionDecision:
        local_decision = self._decide_via_local(step=step, screen_snapshot=screen_snapshot)
        if local_decision is not None:
            return local_decision

        openrouter_decision = self._decide_via_openrouter(step=step, screen_snapshot=screen_snapshot)
        if openrouter_decision is not None:
            return openrouter_decision

        def _op() -> ActionDecision:
            prompt = self._build_prompt(step=step, screen_snapshot=screen_snapshot)
            response = self._client.models.generate_content(
                model=self._config.gemini_model,
                contents=prompt,
            )
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise ActionExecutionError("Action decision returned empty output.")

            payload = _parse_json_object(text)
            decision = ActionDecision(
                action_type=_require_str(payload, "action_type").lower(),
                target=_require_str(payload, "target"),
                input_text=_optional_str(payload, "input"),
            )

            # Secondary tool-choice normalization path using function-calling prompt.
            function_payload = self._function_agent.decide_function(
                user_input=f"Step: {step.description}; Recommended: {decision.action_type} {decision.target}"
            )
            return self._normalize_with_function_call(decision, function_payload)

        try:
            return with_retry(_op, self._retry_policy, op_name="action_decision")
        except Exception as exc:  # noqa: BLE001 - graceful degradation path
            logger.warning("Gemini action decision failed; using heuristic action decision", extra={"error": str(exc)})
            return self._fallback_decision(step=step, screen_snapshot=screen_snapshot)

    def _decide_via_local(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> ActionDecision | None:
        if self._router.local is None:
            return None

        def _op() -> ActionDecision:
            prompt = self._build_prompt(step=step, screen_snapshot=screen_snapshot)
            text = self._router.local.generate_text(prompt)
            payload = _parse_json_object(text)
            decision = ActionDecision(
                action_type=_require_str(payload, "action_type").lower(),
                target=_require_str(payload, "target"),
                input_text=_optional_str(payload, "input"),
            )
            function_payload = self._function_agent.decide_function(
                user_input=f"Step: {step.description}; Recommended: {decision.action_type} {decision.target}"
            )
            return self._normalize_with_function_call(decision, function_payload)

        try:
            return with_retry(_op, self._retry_policy, op_name="ollama_action_decision")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local action decision failed", extra={"error": str(exc)})
            return None

    def _decide_via_openrouter(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> ActionDecision | None:
        if not self._openrouter.enabled:
            return None

        def _op() -> ActionDecision:
            prompt = self._build_prompt(step=step, screen_snapshot=screen_snapshot)
            text = self._openrouter.generate_text(prompt)
            payload = _parse_json_object(text)
            decision = ActionDecision(
                action_type=_require_str(payload, "action_type").lower(),
                target=_require_str(payload, "target"),
                input_text=_optional_str(payload, "input"),
            )
            function_payload = self._function_agent.decide_function(
                user_input=f"Step: {step.description}; Recommended: {decision.action_type} {decision.target}"
            )
            return self._normalize_with_function_call(decision, function_payload)

        try:
            return with_retry(_op, self._retry_policy, op_name="openrouter_action_decision")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter action decision failed", extra={"error": str(exc)})
            return None

    def _fallback_decision(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> ActionDecision:
        recommendation = screen_snapshot.recommended_action.lower().strip()
        description = step.description.lower().strip()

        if step.action_type == ActionType.KEYBOARD_HOTKEY:
            keys = step.parameters.get("keys", ["enter"])
            if isinstance(keys, list) and keys:
                hotkey = "+".join(str(k).strip() for k in keys if str(k).strip())
                return ActionDecision(action_type="hotkey", target="keyboard", input_text=hotkey or "enter")
            return ActionDecision(action_type="hotkey", target="keyboard", input_text="enter")

        if description.startswith("press "):
            hotkey = description.replace("press", "", 1).strip() or "enter"
            return ActionDecision(action_type="hotkey", target="keyboard", input_text=hotkey)

        if "type" in description:
            text = str(step.parameters.get("text", step.description))
            return ActionDecision(action_type="type", target="keyboard", input_text=text)

        if "click" in description and "css_selector" in step.parameters:
            return ActionDecision(action_type="click", target=str(step.parameters["css_selector"]), input_text="")

        if "open" in description:
            target = str(step.parameters.get("url", "browser"))
            return ActionDecision(action_type="open", target=target, input_text="")

        if "click" in recommendation:
            return ActionDecision(action_type="click", target="100,100", input_text="")

        return ActionDecision(action_type="type", target="keyboard", input_text=step.description)

    def to_executable_step(self, source_step: PlanStep, decision: ActionDecision) -> PlanStep:
        executable = copy.deepcopy(source_step)

        action_type = decision.action_type.strip().lower()
        browser_source_actions = {ActionType.BROWSER_OPEN, ActionType.BROWSER_CLICK, ActionType.BROWSER_TYPE}
        source_is_browser = source_step.action_type in browser_source_actions

        if source_step.action_type == ActionType.OPEN_APP:
            app_name = str(source_step.parameters.get("name", decision.target or "calculator")).strip()
            if not app_name:
                app_name = "calculator"
            executable.action_type = ActionType.OPEN_APP
            executable.parameters = {"name": app_name}
            executable.description = f"open -> {app_name}"
            executable.validation_hint = executable.description
            return executable

        if source_step.action_type == ActionType.WAIT:
            seconds = float(source_step.parameters.get("seconds", 1.0))
            executable.action_type = ActionType.WAIT
            executable.parameters = {"seconds": seconds}
            executable.description = f"wait -> {seconds}"
            executable.validation_hint = executable.description
            return executable

        if source_step.action_type == ActionType.BROWSER_OPEN:
            url = str(source_step.parameters.get("url", decision.target)).strip()
            if not url.startswith("http"):
                url = str(decision.target).strip()
            if not url.startswith("http"):
                url = "https://www.google.com"
            executable.action_type = ActionType.BROWSER_OPEN
            executable.parameters = {"url": url}
            executable.description = f"open -> {url}"
            executable.validation_hint = executable.description
            return executable

        if source_step.action_type == ActionType.BROWSER_CLICK:
            selector = str(source_step.parameters.get("css_selector", decision.target)).strip()
            if not selector:
                selector = "button[type='submit']"
            executable.action_type = ActionType.BROWSER_CLICK
            executable.parameters = {"css_selector": selector}
            executable.description = f"click -> {selector}"
            executable.validation_hint = executable.description
            return executable

        if source_step.action_type == ActionType.BROWSER_TYPE:
            selector = str(source_step.parameters.get("css_selector", decision.target)).strip()
            text = str(source_step.parameters.get("text", decision.input_text or "")).strip()
            executable.action_type = ActionType.BROWSER_TYPE
            executable.parameters = {
                "css_selector": selector,
                "text": text,
                "clear_first": bool(source_step.parameters.get("clear_first", True)),
            }
            executable.description = f"type -> {selector}"
            executable.validation_hint = executable.description
            return executable

        if action_type == "click":
            maybe_xy = _parse_coordinates(decision.target)
            if maybe_xy is not None:
                executable.action_type = ActionType.MOUSE_CLICK
                executable.parameters = {"x": maybe_xy[0], "y": maybe_xy[1], "button": "left"}
            elif source_is_browser and _is_safe_css_selector(decision.target):
                executable.action_type = ActionType.BROWSER_CLICK
                executable.parameters = {"css_selector": decision.target}
            else:
                executable.action_type = ActionType.KEYBOARD_HOTKEY
                executable.parameters = {"keys": ["enter"]}

        elif action_type == "type":
            if source_is_browser and decision.target and _is_safe_css_selector(decision.target):
                executable.action_type = ActionType.BROWSER_TYPE
                executable.parameters = {
                    "css_selector": decision.target,
                    "text": decision.input_text or str(source_step.parameters.get("text", "")),
                    "clear_first": True,
                }
            else:
                executable.action_type = ActionType.KEYBOARD_WRITE
                fallback_text = decision.input_text or str(source_step.parameters.get("text", source_step.description))
                executable.parameters = {"text": fallback_text}

        elif action_type == "hotkey":
            raw = decision.input_text.strip()
            keys = [part.strip() for part in raw.split("+") if part.strip()]
            executable.action_type = ActionType.KEYBOARD_HOTKEY
            executable.parameters = {"keys": keys if keys else ["enter"]}

        elif action_type == "open":
            if decision.target.startswith("http"):
                executable.action_type = ActionType.BROWSER_OPEN
                executable.parameters = {"url": decision.target}
            else:
                executable.action_type = ActionType.OPEN_APP
                executable.parameters = {"name": decision.target}

        elif action_type == "wait":
            seconds = _parse_wait_seconds(decision.input_text or decision.target)
            executable.action_type = ActionType.WAIT
            executable.parameters = {"seconds": seconds}

        else:
            logger.warning(
                "Unsupported decided action_type; falling back to source step",
                extra={"action_type": decision.action_type, "step_id": source_step.id},
            )
            executable = copy.deepcopy(source_step)

        executable.description = f"{decision.action_type} -> {decision.target}"
        executable.validation_hint = executable.description
        return executable

    def _build_prompt(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> str:
        return (
            "You are an execution planner.\n\n"
            "Given:\n"
            "- Current step\n"
            "- Screen analysis\n\n"
            "Decide the exact action.\n\n"
            "Rules:\n"
            "- Be precise (coordinates or UI element)\n"
            "- Only one action\n\n"
            "Output:\n"
            "{\n"
            '  "action_type": "click / type / open",\n'
            '  "target": "...",\n'
            '  "input": "..."\n'
            "}\n\n"
            f"Current step: {step.description}\n"
            f"Screen analysis: {screen_snapshot.summary}; elements={screen_snapshot.key_elements}; "
            f"recommended={screen_snapshot.recommended_action}\n"
            "Output only valid JSON, no markdown, no extra keys."
        )

    def _normalize_with_function_call(
        self,
        decision: ActionDecision,
        function_payload: dict[str, Any],
    ) -> ActionDecision:
        function_name = str(function_payload.get("function", "none")).strip().lower()
        arguments = function_payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        if function_name == "click":
            x = arguments.get("x")
            y = arguments.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return ActionDecision(action_type="click", target=f"{int(x)},{int(y)}", input_text="")

        if function_name == "type":
            text = str(arguments.get("text", decision.input_text))
            return ActionDecision(action_type="type", target="keyboard", input_text=text)

        if function_name == "open_url":
            url = str(arguments.get("url", decision.target))
            return ActionDecision(action_type="open", target=url, input_text="")

        if function_name == "open_app":
            # Action engine currently supports browser open directly; map app open to open intent.
            app_name = str(arguments.get("name", decision.target))
            return ActionDecision(action_type="open", target=app_name, input_text="")

        return decision


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
        raise ActionExecutionError("Action decision response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ActionExecutionError("Action decision response root must be an object.")
    return payload


def _parse_coordinates(target: str) -> tuple[int, int] | None:
    match = re.match(r"^\\s*(\\d{1,5})\\s*,\\s*(\\d{1,5})\\s*$", target)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionExecutionError(f"Field '{key}' must be a non-empty string.")
    return value.strip()


def _optional_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ActionExecutionError(f"Field '{key}' must be a string when provided.")
    return value.strip()


def _parse_wait_seconds(value: str) -> float:
    text = value.strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 1.0
    seconds = float(match.group(1))
    return max(0.1, min(seconds, 30.0))


def _is_safe_css_selector(value: str) -> bool:
    selector = value.strip().lower()
    if not selector:
        return False

    invalid = {
        "keyboard",
        "active_element",
        "terminal",
        "terminal_prompt",
        "prompt",
        "search",
    }
    if selector in invalid:
        return False

    if re.fullmatch(r"[a-z_][a-z0-9_\-]{0,40}", selector) and selector not in {
        "input",
        "textarea",
        "button",
        "select",
        "body",
        "a",
        "form",
        "div",
        "span",
    }:
        return False

    return True
