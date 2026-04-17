"""CLI entrypoint for Aletheia AI."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from aletheia_ai.bootstrap import build_assistant
from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import AletheiaError
from aletheia_ai.logging_config import configure_logging
from aletheia_ai.ui.app import run_gui

logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()

    if args.gui or not args.task:
        config = AppConfig.from_env()
        configure_logging(config.log_level)
        run_gui(config)
        return

    assistant = None

    try:
        config = AppConfig.from_env()
        configure_logging(config.log_level)
        assistant = build_assistant(config)

        result = assistant.run(task=args.task, context=args.context)
        print(json.dumps(_serialize(result), indent=2, ensure_ascii=True))

    except AletheiaError as exc:
        logger.exception("Application error", extra={"error": str(exc)})
        raise SystemExit(2) from exc
    except Exception as exc:  # noqa: BLE001 - top-level fatal guard
        logger.exception("Unhandled fatal error", extra={"error": str(exc)})
        raise SystemExit(1) from exc
    finally:
        if assistant is not None:
            try:
                assistant.shutdown(close_browser=not config.keep_browser_open)
            except Exception:
                logger.exception("Failed to shutdown assistant cleanly")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aletheia AI Autonomous Thinking Assistant")
    parser.add_argument("--task", default=None, help="Natural language goal to execute")
    parser.add_argument("--context", default=None, help="Optional runtime context for reasoning")
    parser.add_argument("--gui", action="store_true", help="Launch the desktop frontend")
    return parser.parse_args()


def _serialize(data: Any) -> Any:
    if is_dataclass(data):
        return {k: _serialize(v) for k, v in asdict(data).items()}
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(item) for item in data]
    return data


if __name__ == "__main__":
    main()
