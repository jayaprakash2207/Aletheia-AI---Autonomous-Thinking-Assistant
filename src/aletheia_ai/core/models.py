"""Core domain models shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    OPEN_APP = "open_app"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    KEYBOARD_WRITE = "keyboard_write"
    KEYBOARD_HOTKEY = "keyboard_hotkey"
    BROWSER_OPEN = "browser_open"
    BROWSER_CLICK = "browser_click"
    BROWSER_TYPE = "browser_type"
    WAIT = "wait"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class ReasoningOutput:
    intent: str
    constraints: list[str]
    strategy: str
    success_criteria: list[str]


@dataclass(slots=True)
class PlanStep:
    id: int
    description: str
    action_type: ActionType
    parameters: dict[str, Any] = field(default_factory=dict)
    validation_hint: str = ""


@dataclass(slots=True)
class TaskPlan:
    goal: str
    steps: list[PlanStep]


@dataclass(slots=True)
class VisionSnapshot:
    image_path: str
    width: int
    height: int
    summary: str
    key_elements: list[str]
    recommended_action: str = ""


@dataclass(slots=True)
class ActionResult:
    step_id: int
    status: StepStatus
    details: str
    observation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    confidence: float
    rationale: str
    suggested_correction: str = ""
    next_step: str = "continue"


@dataclass(slots=True)
class ActionDecision:
    action_type: str
    target: str
    input_text: str = ""


@dataclass(slots=True)
class SelfCorrectionOutput:
    new_strategy: str
    updated_action: str
