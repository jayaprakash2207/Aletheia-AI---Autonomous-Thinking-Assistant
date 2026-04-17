"""Dependency wiring for Aletheia AI runtime."""

from __future__ import annotations

from aletheia_ai.action.action_engine import ActionEngine
from aletheia_ai.action.action_decision_engine import ActionDecisionEngine
from aletheia_ai.action.browser_controller import BrowserController
from aletheia_ai.action.function_calling_agent import FunctionCallingAgent
from aletheia_ai.config import AppConfig
from aletheia_ai.feedback.feedback_loop import FeedbackLoop
from aletheia_ai.feedback.self_correction_engine import SelfCorrectionEngine
from aletheia_ai.orchestrator.autonomous_assistant import AutonomousAssistant
from aletheia_ai.planning.task_planner import TaskPlanner
from aletheia_ai.reasoning.gemini_reasoner import GeminiReasoner
from aletheia_ai.validation.validator import Validator
from aletheia_ai.vision.screenshot_provider import ScreenshotProvider
from aletheia_ai.vision.vision_analyzer import OpenCVVisionAnalyzer


def build_assistant(config: AppConfig) -> AutonomousAssistant:
    screenshot_provider = ScreenshotProvider()
    vision = OpenCVVisionAnalyzer(config=config, screenshot_provider=screenshot_provider)
    browser = BrowserController(config=config)
    action = ActionEngine(browser_controller=browser)
    function_agent = FunctionCallingAgent(config=config)
    action_decider = ActionDecisionEngine(config=config, function_agent=function_agent)
    reasoner = GeminiReasoner(config=config)
    planner = TaskPlanner(config=config)
    validator = Validator(config=config)
    self_corrector = SelfCorrectionEngine(config=config)
    feedback = FeedbackLoop(
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )

    return AutonomousAssistant(
        reasoner=reasoner,
        planner=planner,
        vision=vision,
        action_decider=action_decider,
        action_engine=action,
        validator=validator,
        self_corrector=self_corrector,
        feedback_loop=feedback,
        max_replans=config.max_replans,
    )
