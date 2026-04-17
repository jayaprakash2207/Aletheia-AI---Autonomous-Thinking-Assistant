"""Vision analysis using screenshots and OpenCV feature extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from google import genai
from google.genai import types

from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import VisionError
from aletheia_ai.core.models import VisionSnapshot
from aletheia_ai.utils.retry import RetryPolicy, with_retry
from aletheia_ai.vision.screenshot_provider import ScreenshotProvider

logger = logging.getLogger(__name__)


class OpenCVVisionAnalyzer:
    def __init__(self, config: AppConfig, screenshot_provider: ScreenshotProvider) -> None:
        self._config = config
        self._screenshot_provider = screenshot_provider
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._retry_policy = RetryPolicy(
            max_attempts=max(1, config.max_retries),
            backoff_seconds=config.retry_backoff_seconds,
        )

    def capture_and_analyze(self, prompt: str) -> VisionSnapshot:
        image_path = self._screenshot_provider.capture(self._config.screenshot_path)

        def _op() -> VisionSnapshot:
            return self._analyze_with_gemini(image_path=image_path, prompt=prompt)

        try:
            return with_retry(_op, self._retry_policy, op_name="vision_gemini_analysis")
        except Exception as exc:  # noqa: BLE001 - fallback boundary
            logger.warning("Gemini vision analysis failed, falling back to OpenCV only", extra={"error": str(exc)})
            return self._fallback_analyze(image_path=image_path, prompt=prompt)

    def _analyze_with_gemini(self, image_path: str, prompt: str) -> VisionSnapshot:
        image = cv2.imread(image_path)
        if image is None:
            raise VisionError(f"OpenCV could not read screenshot at {image_path}")

        height, width, _channels = image.shape
        image_bytes = Path(image_path).read_bytes()

        vision_prompt = self._build_vlam_prompt(current_step=prompt)
        response = self._client.models.generate_content(
            model=self._config.gemini_model,
            contents=[
                vision_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        raw_text = getattr(response, "text", "") or ""
        if not raw_text.strip():
            raise VisionError("Gemini vision response is empty.")

        payload = _parse_json_object(raw_text)
        screen_state = _require_str(payload, "screen_state")
        elements = _require_str_list(payload, "elements")
        recommended_action = _require_str(payload, "recommended_action")

        summary = f"{screen_state} | Recommended action: {recommended_action}"
        return VisionSnapshot(
            image_path=str(Path(image_path)),
            width=width,
            height=height,
            summary=summary,
            key_elements=elements,
            recommended_action=recommended_action,
        )

    def _fallback_analyze(self, image_path: str, prompt: str) -> VisionSnapshot:
        image = cv2.imread(image_path)
        if image is None:
            raise VisionError(f"OpenCV could not read screenshot at {image_path}")

        height, width, _channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, threshold1=75, threshold2=150)
        edge_density = float(np.count_nonzero(edges)) / float(width * height)

        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        key_elements = _infer_key_elements(edge_density=edge_density, brightness=mean_brightness)
        summary = (
            f"Prompt focus: {prompt}. "
            f"Screen {width}x{height}, edge_density={edge_density:.4f}, "
            f"brightness_mean={mean_brightness:.2f}, brightness_std={std_brightness:.2f}."
        )

        logger.info(
            "Vision analysis complete",
            extra={
                "image_path": image_path,
                "width": width,
                "height": height,
                "edge_density": edge_density,
            },
        )

        return VisionSnapshot(
            image_path=str(Path(image_path)),
            width=width,
            height=height,
            summary=summary,
            key_elements=key_elements,
            recommended_action="Use visible primary UI control relevant to the current step.",
        )

    def _build_vlam_prompt(self, current_step: str) -> str:
        return (
            "You are a computer vision assistant.\n\n"
            "Analyze this screen and identify UI elements.\n\n"
            f"Goal: {current_step}\n\n"
            "Instructions:\n"
            "- Describe what is visible\n"
            "- Identify buttons, icons, text\n"
            "- Suggest exact action to take\n\n"
            "Output:\n"
            "{\n"
            '  "screen_state": "...",\n'
            '  "elements": ["button1", "icon2"],\n'
            '  "recommended_action": "..."\n'
            "}\n"
            "Output only valid JSON, no markdown, no extra keys."
        )


def _infer_key_elements(edge_density: float, brightness: float) -> list[str]:
    hints: list[str] = []
    if edge_density > 0.12:
        hints.append("ui-rich-screen")
    else:
        hints.append("low-structure-screen")

    if brightness > 170:
        hints.append("light-theme-likely")
    elif brightness < 80:
        hints.append("dark-theme-likely")
    else:
        hints.append("mixed-lighting")

    return hints


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VisionError("Vision response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise VisionError("Vision response root must be an object.")
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VisionError(f"Field '{key}' must be a non-empty string.")
    return value.strip()


def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise VisionError(f"Field '{key}' must be a non-empty list.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise VisionError(f"Field '{key}' contains invalid list items.")
        normalized.append(item.strip())
    return normalized
