"""Behavioral contracts for independent module implementations."""

from __future__ import annotations

from typing import Protocol

from aletheia_ai.core.models import (
    ActionDecision,
    ActionResult,
    PlanStep,
    ReasoningOutput,
    SelfCorrectionOutput,
    TaskPlan,
    ValidationResult,
    VisionSnapshot,
)


class Reasoner(Protocol):
    def reason(self, task: str, context: str | None = None) -> ReasoningOutput:
        ...


class Planner(Protocol):
    def create_plan(self, task: str, reasoning: ReasoningOutput) -> TaskPlan:
        ...


class VisionAnalyzer(Protocol):
    def capture_and_analyze(self, prompt: str) -> VisionSnapshot:
        ...


class ActionExecutor(Protocol):
    def execute(self, step: PlanStep) -> ActionResult:
        ...


class ActionDecider(Protocol):
    def decide(self, step: PlanStep, screen_snapshot: VisionSnapshot) -> ActionDecision:
        ...

    def to_executable_step(self, source_step: PlanStep, decision: ActionDecision) -> PlanStep:
        ...


class StepValidator(Protocol):
    def validate(
        self,
        step: PlanStep,
        action_result: ActionResult,
        previous_snapshot: VisionSnapshot,
        current_snapshot: VisionSnapshot,
        success_criteria: list[str],
    ) -> ValidationResult:
        ...


class SelfCorrector(Protocol):
    def self_correct(self, step: PlanStep, reason: str) -> SelfCorrectionOutput:
        ...
