"""Tkinter desktop frontend for Aletheia AI."""

from __future__ import annotations

import json
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from aletheia_ai.bootstrap import build_assistant
from aletheia_ai.config import AppConfig
from aletheia_ai.core.exceptions import AletheiaError
from aletheia_ai.logging_config import configure_logging

logger = logging.getLogger(__name__)


class _QueueLogHandler(logging.Handler):
    def __init__(self, message_queue: queue.Queue[str]) -> None:
        super().__init__()
        self._queue = message_queue
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(self.format(record))
        except Exception:
            self.handleError(record)


class AletheiaGUI:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._assistant = build_assistant(config)
        self._root = tk.Tk()
        self._root.title("Aletheia AI - Autonomous Thinking Assistant")
        self._root.geometry("1100x780")
        self._root.minsize(980, 720)

        self._message_queue: queue.Queue[str] = queue.Queue()
        self._result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._busy = False

        self.task_var = tk.StringVar(value="Open browser and search for weather in Bangalore")
        self.context_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self._install_log_handler()
        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.after(100, self._poll_queues)

    def run(self) -> None:
        self._root.mainloop()

    def close(self) -> None:
        try:
            self._assistant.shutdown(close_browser=not self._config.keep_browser_open)
        except Exception:
            logger.exception("Failed to shutdown assistant from GUI")
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    def _install_log_handler(self) -> None:
        handler = _QueueLogHandler(self._message_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(self._config.log_level)
        self._gui_log_handler = handler

    def _build_ui(self) -> None:
        self._root.configure(bg="#111827")
        main = ttk.Frame(self._root, padding=18)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Aletheia AI", font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main,
            text="Enter a goal, run the autonomous agent, and watch the result without using the terminal.",
        )
        subtitle.pack(anchor="w", pady=(4, 18))

        form = ttk.LabelFrame(main, text="Task", padding=14)
        form.pack(fill="x")

        ttk.Label(form, text="Goal").grid(row=0, column=0, sticky="w")
        task_entry = ttk.Entry(form, textvariable=self.task_var)
        task_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        ttk.Label(form, text="Context").grid(row=2, column=0, sticky="w")
        context_entry = ttk.Entry(form, textvariable=self.context_var)
        context_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        self.run_button = ttk.Button(form, text="Run Agent", command=self._on_run)
        self.run_button.grid(row=4, column=0, sticky="w")

        ttk.Label(form, textvariable=self.status_var, foreground="#2563eb").grid(row=4, column=1, sticky="e")
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(main, text="Live Output", padding=14)
        output_frame.pack(fill="both", expand=True, pady=(16, 0))

        self.output_text = tk.Text(output_frame, wrap="word", height=24, font=("Consolas", 10))
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self._append_text("Ready. Enter a goal and click Run Agent.\n")

    def _on_run(self) -> None:
        if self._busy:
            return

        task = self.task_var.get().strip()
        context = self.context_var.get().strip() or None
        if not task:
            self._append_text("Task is required.\n")
            return

        self._busy = True
        self.run_button.configure(state="disabled")
        self.status_var.set("Running...")
        self._append_text(f"\n=== Running: {task} ===\n")

        worker = threading.Thread(target=self._run_task, args=(task, context), daemon=True)
        worker.start()

    def _run_task(self, task: str, context: str | None) -> None:
        try:
            result = self._assistant.run(task=task, context=context)
            self._result_queue.put(result)
        except AletheiaError as exc:
            self._result_queue.put({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled GUI execution error")
            self._result_queue.put({"error": str(exc)})

    def _poll_queues(self) -> None:
        try:
            while True:
                message = self._message_queue.get_nowait()
                self._append_text(message + "\n")
        except queue.Empty:
            pass

        try:
            while True:
                result = self._result_queue.get_nowait()
                self._handle_result(result)
        except queue.Empty:
            pass

        self._root.after(100, self._poll_queues)

    def _handle_result(self, result: dict[str, Any]) -> None:
        self._busy = False
        self.run_button.configure(state="normal")
        self.status_var.set("Ready")

        self._append_text("\n=== Result ===\n")
        self._append_text(json.dumps(result, indent=2, ensure_ascii=True))
        self._append_text("\n")

    def _append_text(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")


def run_gui(config: AppConfig | None = None) -> None:
    gui: AletheiaGUI | None = None
    if config is None:
        config = AppConfig.from_env()
        configure_logging(config.log_level)

    try:
        gui = AletheiaGUI(config)
        gui.run()
    finally:
        try:
            if gui is not None:
                gui.close()
        except Exception:
            pass
