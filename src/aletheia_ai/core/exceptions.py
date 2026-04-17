"""Domain-specific exceptions for Aletheia AI."""

from __future__ import annotations


class AletheiaError(Exception):
    """Base exception for all assistant errors."""


class ConfigurationError(AletheiaError):
    """Raised when environment or runtime configuration is invalid."""


class ReasoningError(AletheiaError):
    """Raised when LLM reasoning fails or returns malformed output."""


class PlanningError(AletheiaError):
    """Raised when a task plan cannot be generated or parsed."""


class VisionError(AletheiaError):
    """Raised when screenshot capture or visual analysis fails."""


class ActionExecutionError(AletheiaError):
    """Raised when an action cannot be executed."""


class ValidationError(AletheiaError):
    """Raised when validation logic cannot evaluate a result."""
