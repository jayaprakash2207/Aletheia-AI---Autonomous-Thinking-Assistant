"""Action engine that executes plan steps through desktop and browser automation."""

from __future__ import annotations

import logging
import os
import subprocess
import time

import pyautogui

from aletheia_ai.action.browser_controller import BrowserController
from aletheia_ai.core.exceptions import ActionExecutionError
from aletheia_ai.core.models import ActionResult, ActionType, PlanStep, StepStatus

logger = logging.getLogger(__name__)


class ActionEngine:
    def __init__(self, browser_controller: BrowserController) -> None:
        self._browser = browser_controller
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.2

    def execute(self, step: PlanStep) -> ActionResult:
        logger.info(
            "Executing action step",
            extra={"step_id": step.id, "action_type": step.action_type.value, "desc": step.description},
        )

        try:
            if step.action_type == ActionType.OPEN_APP:
                app_name = str(step.parameters["name"])
                app_norm = app_name.strip().lower()
                launched = False

                if os.name == "nt":
                    known_apps = {
                        "calculator": ["calc.exe"],
                        "calc": ["calc.exe"],
                        "notepad": ["notepad.exe"],
                        "mspaint": ["mspaint.exe"],
                        "paint": ["mspaint.exe"],
                    }
                    command = known_apps.get(app_norm)
                    if command is not None:
                        subprocess.Popen(command, shell=False)
                        launched = True
                        self._focus_window(["Calculator"])

                if not launched:
                    # Fallback for unknown apps: use Start menu search.
                    pyautogui.hotkey("win")
                    pyautogui.write(app_name, interval=0.03)
                    pyautogui.press("enter")

                details = f"Requested app launch: {app_name}"

            elif step.action_type == ActionType.MOUSE_CLICK:
                x = int(step.parameters["x"])
                y = int(step.parameters["y"])
                button = str(step.parameters.get("button", "left"))
                pyautogui.click(x=x, y=y, button=button)
                details = f"Mouse clicked at ({x},{y})"

            elif step.action_type == ActionType.MOUSE_MOVE:
                x = int(step.parameters["x"])
                y = int(step.parameters["y"])
                duration = float(step.parameters.get("duration", 0.2))
                pyautogui.moveTo(x=x, y=y, duration=duration)
                details = f"Mouse moved to ({x},{y})"

            elif step.action_type == ActionType.KEYBOARD_WRITE:
                text = str(step.parameters["text"])
                interval = float(step.parameters.get("interval", 0.02))
                pyautogui.write(text, interval=interval)
                details = "Text typed via keyboard"

            elif step.action_type == ActionType.KEYBOARD_HOTKEY:
                keys = step.parameters.get("keys")
                if not isinstance(keys, list) or not keys:
                    raise ActionExecutionError("keyboard_hotkey requires non-empty list in 'keys'.")
                normalized = [str(k) for k in keys]
                if len(normalized) == 1:
                    pyautogui.press(normalized[0])
                else:
                    pyautogui.hotkey(*normalized)
                details = f"Hotkey sent: {'+'.join(normalized)}"

            elif step.action_type == ActionType.BROWSER_OPEN:
                self._browser.open(url=step.parameters.get("url"))
                details = "Browser opened"

            elif step.action_type == ActionType.BROWSER_CLICK:
                selector = str(step.parameters["css_selector"])
                self._browser.click(css_selector=selector)
                details = f"Browser clicked selector {selector}"

            elif step.action_type == ActionType.BROWSER_TYPE:
                selector = str(step.parameters["css_selector"])
                text = str(step.parameters["text"])
                clear_first = bool(step.parameters.get("clear_first", True))
                self._browser.type_text(css_selector=selector, text=text, clear_first=clear_first)
                details = f"Browser typed into selector {selector}"

            elif step.action_type == ActionType.WAIT:
                seconds = float(step.parameters.get("seconds", 1.0))
                time.sleep(seconds)
                details = f"Waited for {seconds:.2f} seconds"

            else:
                raise ActionExecutionError(f"Unsupported action type: {step.action_type}")

            return ActionResult(
                step_id=step.id,
                status=StepStatus.SUCCEEDED,
                details=details,
                observation={"action_type": step.action_type.value},
            )

        except KeyError as exc:
            raise ActionExecutionError(f"Missing required step parameter: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - boundary for hardware/browser actions
            logger.exception("Action execution failed", extra={"step_id": step.id})
            return ActionResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                details=str(exc),
                observation={"action_type": step.action_type.value},
            )

    def shutdown(self, close_browser: bool = True) -> None:
        self._browser.shutdown(close_browser=close_browser)

    def _focus_window(self, title_candidates: list[str]) -> None:
        # Best-effort foreground focus for desktop apps before typing actions.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                for title in title_candidates:
                    windows = pyautogui.getWindowsWithTitle(title)
                    if windows:
                        window = windows[0]
                        try:
                            if window.isMinimized:
                                window.restore()
                        except Exception:
                            pass
                        window.activate()
                        time.sleep(0.2)
                        return
            except Exception:
                return
            time.sleep(0.2)
