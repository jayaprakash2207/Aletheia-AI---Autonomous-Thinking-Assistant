"""Reusable retry policies for resilient module calls."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 1.0


def with_retry(operation: Callable[[], T], policy: RetryPolicy, op_name: str) -> T:
    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - explicit centralized retry boundary
            if _is_non_retriable(exc):
                logger.warning(
                    "Operation failed with non-retriable error",
                    extra={"operation": op_name, "error": str(exc)},
                )
                raise

            last_error = exc
            logger.warning(
                "Operation failed; retrying",
                extra={
                    "operation": op_name,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "error": str(exc),
                },
            )
            if attempt < policy.max_attempts:
                time.sleep(policy.backoff_seconds * attempt)

    assert last_error is not None
    raise last_error


def _is_non_retriable(exc: Exception) -> bool:
    message = str(exc).upper()
    non_retriable_markers = {
        "RESOURCE_EXHAUSTED",
        "QUOTA EXCEEDED",
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
    }
    return any(marker in message for marker in non_retriable_markers)
