"""Feedback loop for self-correction and retry orchestration."""

from __future__ import annotations

import copy
import logging
import time

from aletheia_ai.core.models import ActionResult, ActionType, PlanStep, SelfCorrectionOutput, ValidationResult
from aletheia_ai.core.models import StepStatus
from aletheia_ai.core.models import VisionSnapshot

logger = logging.getLogger(__name__)


class FeedbackLoop:
    def __init__(
        self,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    def execute_with_feedback(
        self,
        step: PlanStep,
        action_decider,
        execute_fn,
        capture_snapshot_fn,
        validate_fn,
        self_correct_fn,
        success_criteria: list[str],
    ) -> tuple[ActionResult, ValidationResult, VisionSnapshot]:
        current_step = copy.deepcopy(step)
        previous_snapshot = capture_snapshot_fn(prompt=f"Pre-step capture: {current_step.description}")
        deterministic_action_types = {
            ActionType.OPEN_APP,
            ActionType.WAIT,
            ActionType.KEYBOARD_WRITE,
            ActionType.KEYBOARD_HOTKEY,
            ActionType.BROWSER_OPEN,
            ActionType.BROWSER_CLICK,
            ActionType.BROWSER_TYPE,
        }

        for attempt in range(1, self._max_retries + 2):
            if current_step.action_type in deterministic_action_types:
                executable_step = copy.deepcopy(current_step)
            else:
                decision = action_decider.decide(current_step, previous_snapshot)
                executable_step = action_decider.to_executable_step(current_step, decision)
            action_result = execute_fn(executable_step)
            snapshot = capture_snapshot_fn(prompt=f"Post-step capture: {current_step.description}")
            validation = validate_fn(
                step=current_step,
                action_result=action_result,
                previous_snapshot=previous_snapshot,
                current_snapshot=snapshot,
                success_criteria=success_criteria,
            )

            if validation.passed and validation.next_step == "continue":
                logger.info(
                    "Step succeeded after feedback loop",
                    extra={"step_id": step.id, "attempt": attempt},
                )
                return action_result, validation, snapshot

            logger.warning(
                "Step validation failed",
                extra={
                    "step_id": step.id,
                    "attempt": attempt,
                    "rationale": validation.rationale,
                    "suggested_correction": validation.suggested_correction,
                },
            )

            if attempt > self._max_retries:
                failed_result = ActionResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    details=(
                        "Exceeded retries. Last action result: "
                        f"{action_result.details}; validation: {validation.rationale}"
                    ),
                    observation=action_result.observation,
                )
                return failed_result, validation, snapshot

            correction = self_correct_fn(step=current_step, reason=validation.rationale)
            self._apply_correction(current_step, correction)
            previous_snapshot = snapshot
            time.sleep(self._retry_backoff_seconds * attempt)

        raise RuntimeError("Feedback loop reached an unexpected state")

    def _apply_correction(self, step: PlanStep, correction: SelfCorrectionOutput) -> None:
        strategy = correction.new_strategy.lower().strip()
        updated_action = correction.updated_action.strip()

        if not strategy and not updated_action:
            return

        if updated_action:
            step.description = updated_action
            step.validation_hint = updated_action

        if "wait longer" in strategy:
            current_wait = float(step.parameters.get("seconds", 1.0))
            step.parameters["seconds"] = current_wait + 1.0

        if "incremental" in strategy and "duration" in step.parameters:
            step.parameters["duration"] = float(step.parameters["duration"]) + 0.2

        if "selector" in strategy and "css_selector" in step.parameters:
            step.parameters["css_selector"] = str(step.parameters["css_selector"]).strip()
