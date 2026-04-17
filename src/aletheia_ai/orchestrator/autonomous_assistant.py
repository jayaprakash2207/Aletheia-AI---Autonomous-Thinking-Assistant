"""Primary orchestration workflow for Aletheia AI."""

from __future__ import annotations

import logging

from aletheia_ai.core.models import StepStatus

logger = logging.getLogger(__name__)


class AutonomousAssistant:
    def __init__(
        self,
        reasoner,
        planner,
        vision,
        action_decider,
        action_engine,
        validator,
        self_corrector,
        feedback_loop,
        max_replans: int = 2,
    ) -> None:
        self._reasoner = reasoner
        self._planner = planner
        self._vision = vision
        self._action_decider = action_decider
        self._action_engine = action_engine
        self._validator = validator
        self._self_corrector = self_corrector
        self._feedback_loop = feedback_loop
        self._max_replans = max(0, max_replans)

    def run(self, task: str, context: str | None = None) -> dict:
        logger.info("Assistant run started", extra={"task": task})

        reasoning = self._reasoner.reason(task=task, context=context)
        plan = self._planner.create_plan(task=task, reasoning=reasoning)
        execution_log: list[dict] = []
        replan_count = 0

        while True:
            failed_step = None
            failure_reason = ""

            for step in plan.steps:
                logger.info("Running step", extra={"step_id": step.id, "description": step.description})
                action_result, validation, snapshot = self._feedback_loop.execute_with_feedback(
                    step=step,
                    action_decider=self._action_decider,
                    execute_fn=self._action_engine.execute,
                    capture_snapshot_fn=self._vision.capture_and_analyze,
                    validate_fn=self._validator.validate,
                    self_correct_fn=self._self_corrector.self_correct,
                    success_criteria=reasoning.success_criteria,
                )

                execution_log.append(
                    {
                        "replan_cycle": replan_count,
                        "step_id": step.id,
                        "description": step.description,
                        "action_status": action_result.status.value,
                        "action_details": action_result.details,
                        "validation_passed": validation.passed,
                        "validation_confidence": validation.confidence,
                        "validation_rationale": validation.rationale,
                        "snapshot_summary": snapshot.summary,
                    }
                )

                if action_result.status == StepStatus.FAILED or not validation.passed:
                    failed_step = step
                    failure_reason = f"{action_result.details}; {validation.rationale}"
                    logger.warning(
                        "Step failed, triggering replan",
                        extra={"step_id": step.id, "replan_cycle": replan_count, "reason": failure_reason},
                    )
                    break

            if failed_step is None:
                logger.info("Assistant completed task successfully", extra={"task": task})
                return {
                    "task": task,
                    "status": "succeeded",
                    "reasoning": reasoning,
                    "plan": plan,
                    "execution_log": execution_log,
                    "replans": replan_count,
                }

            if replan_count >= self._max_replans:
                logger.error(
                    "Execution halted after max replan attempts",
                    extra={"max_replans": self._max_replans, "failed_step": failed_step.id},
                )
                return {
                    "task": task,
                    "status": "failed",
                    "reasoning": reasoning,
                    "plan": plan,
                    "execution_log": execution_log,
                    "replans": replan_count,
                    "failure": {
                        "step_id": failed_step.id,
                        "step_description": failed_step.description,
                        "reason": failure_reason,
                    },
                }

            replan_count += 1
            replan_context = self._build_replan_context(
                original_context=context,
                failed_step_id=failed_step.id,
                failed_step_description=failed_step.description,
                failure_reason=failure_reason,
                execution_log=execution_log,
                replan_count=replan_count,
            )
            reasoning = self._reasoner.reason(task=task, context=replan_context)
            plan = self._planner.create_plan(task=task, reasoning=reasoning)

    def _build_replan_context(
        self,
        original_context: str | None,
        failed_step_id: int,
        failed_step_description: str,
        failure_reason: str,
        execution_log: list[dict],
        replan_count: int,
    ) -> str:
        recent_events = execution_log[-5:]
        return (
            f"Original context: {original_context or 'None'}\n"
            f"Replan count: {replan_count}\n"
            f"Failed step: {failed_step_id} - {failed_step_description}\n"
            f"Failure reason: {failure_reason}\n"
            f"Recent execution history: {recent_events}\n"
            "Replan instructions: Think before acting, avoid repeating failed assumptions, "
            "and produce safer, verifiable next actions."
        )

    def shutdown(self, close_browser: bool = True) -> None:
        self._action_engine.shutdown(close_browser=close_browser)
