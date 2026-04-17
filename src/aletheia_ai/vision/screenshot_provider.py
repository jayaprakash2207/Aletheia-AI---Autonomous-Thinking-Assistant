"""Screenshot capture provider abstraction."""

from __future__ import annotations

import logging
from pathlib import Path

import pyautogui

from aletheia_ai.core.exceptions import VisionError

logger = logging.getLogger(__name__)


class ScreenshotProvider:
    def capture(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = pyautogui.screenshot()
            image.save(path)
        except Exception as exc:  # noqa: BLE001 - external IO boundary
            raise VisionError(f"Failed to capture screenshot: {exc}") from exc

        logger.info("Screenshot captured", extra={"path": str(path)})
        return str(path)
