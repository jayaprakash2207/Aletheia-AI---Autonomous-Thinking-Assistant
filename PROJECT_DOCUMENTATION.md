# Aletheia AI – Autonomous Thinking Assistant
## Project Documentation

---

| Field | Details |
|---|---|
| **Project Name** | Aletheia AI – Autonomous Thinking Assistant |
| **Version** | 0.1.0 |
| **Language** | Python 3.11+ |
| **License** | MIT |
| **Repository** | https://github.com/jayaprakash2207/Aletheia-AI---Autonomous-Thinking-Assistant |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features & Innovations](#2-key-features--innovations)
3. [System Architecture](#3-system-architecture)
4. [Module Reference](#4-module-reference)
   - 4.1 [Entry Point – `main.py`](#41-entry-point--mainpy)
   - 4.2 [Configuration – `config.py`](#42-configuration--configpy)
   - 4.3 [Bootstrap / Dependency Wiring – `bootstrap.py`](#43-bootstrap--dependency-wiring--bootstrappy)
   - 4.4 [Core Domain – `core/`](#44-core-domain--core)
   - 4.5 [Reasoning Engine – `reasoning/`](#45-reasoning-engine--reasoning)
   - 4.6 [Task Planner – `planning/`](#46-task-planner--planning)
   - 4.7 [Vision Module – `vision/`](#47-vision-module--vision)
   - 4.8 [Action Engine – `action/`](#48-action-engine--action)
   - 4.9 [Validation Module – `validation/`](#49-validation-module--validation)
   - 4.10 [Feedback & Self-Correction – `feedback/`](#410-feedback--self-correction--feedback)
   - 4.11 [Orchestrator – `orchestrator/`](#411-orchestrator--orchestrator)
   - 4.12 [LLM Router – `llm/`](#412-llm-router--llm)
   - 4.13 [Desktop GUI – `ui/`](#413-desktop-gui--ui)
   - 4.14 [Utilities – `utils/`](#414-utilities--utils)
5. [Data Models & Contracts](#5-data-models--contracts)
6. [LLM Provider Strategy](#6-llm-provider-strategy)
7. [Execution Flow (End-to-End)](#7-execution-flow-end-to-end)
8. [Configuration Reference](#8-configuration-reference)
9. [Installation & Setup](#9-installation--setup)
10. [Usage Guide](#10-usage-guide)
11. [Error Handling & Resilience](#11-error-handling--resilience)
12. [Security Considerations](#12-security-considerations)
13. [Dependencies](#13-dependencies)
14. [Project Structure](#14-project-structure)
15. [Glossary](#15-glossary)

---

## 1. Project Overview

**Aletheia AI** is a production-grade, autonomous AI agent that bridges natural language instructions with real-world desktop and browser actions. Given a plain-English goal (e.g., *"Search for the latest AI news"* or *"Add 5 and 3 using the calculator"*), Aletheia AI:

1. **Reasons** about the goal using a Large Language Model (LLM).
2. **Plans** a sequence of concrete, typed, executable steps.
3. **Acts** on the desktop or browser using keyboard, mouse, and Selenium automation.
4. **Sees** the current screen state via screenshots analyzed with Google Gemini vision and OpenCV.
5. **Validates** whether each step succeeded.
6. **Self-corrects** by retrying with an adapted strategy when steps fail.
7. **Re-plans** up to a configurable maximum number of times before reporting failure.

The system is fully modular – each capability is isolated behind a Python Protocol (interface), making it easy to swap LLM backends, add new action types, or integrate alternative vision models.

---

## 2. Key Features & Innovations

| Feature | Description |
|---|---|
| **Autonomous Reasoning Loop** | LLM-driven chain-of-thought analysis before any action is taken |
| **Multi-LLM Backend Support** | Supports Google Gemini (primary), OpenRouter (cloud fallback), and Ollama (local/offline fallback) |
| **Long-Horizon Planning** | Decomposes complex goals into up to 6 atomic, typed, executable steps |
| **Screenshot-Based Vision** | Captures live desktop screenshots; analyzed by Gemini multimodal API + OpenCV fallback |
| **Desktop Automation** | Controls mouse, keyboard, and desktop apps via PyAutoGUI |
| **Browser Automation** | Controls browser via Selenium WebDriver (Edge, Chrome, Firefox) |
| **Step Validation** | LLM-based and heuristic checks to confirm whether each action succeeded |
| **Self-Correction Engine** | On failure, generates an alternative strategy and modifies the step before retry |
| **Adaptive Re-planning** | If all retries fail, the entire task is re-reasoned and re-planned from updated context |
| **Desktop GUI** | Tkinter-based GUI for zero-command-line usage |
| **Structured Logging** | JSON-aware structured logging throughout; ready for log aggregators |
| **Protocol-Based Contracts** | All core components are decoupled through Python Protocols for testability and extensibility |

---

## 3. System Architecture

### High-Level Flow

```
User (CLI / GUI)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                     AutonomousAssistant                     │
│  (orchestrator/autonomous_assistant.py)                     │
│                                                             │
│  1. Reasoner ──────► ReasoningOutput                       │
│  2. Planner  ──────► TaskPlan (list of PlanStep)           │
│                                                             │
│  For each PlanStep:                                         │
│    FeedbackLoop {                                           │
│      Vision.capture_and_analyze()  ◄── ScreenshotProvider  │
│      ActionDecisionEngine.decide() ◄── FunctionCallingAgent│
│      ActionEngine.execute()        ──► Desktop / Browser   │
│      Vision.capture_and_analyze()  ◄── ScreenshotProvider  │
│      Validator.validate()                                   │
│      if failed: SelfCorrectionEngine.self_correct()        │
│               → retry (up to max_retries)                  │
│    }                                                        │
│    if step ultimately failed: replan (up to max_replans)   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
  Result JSON  {task, status, execution_log, replans, …}
```

### Component Dependency Diagram

```
main.py
  └── bootstrap.py (wires all components)
        ├── GeminiReasoner        (uses LLMRouter → Ollama / OpenRouter / Gemini)
        ├── TaskPlanner           (uses LLMRouter → Ollama / OpenRouter / Gemini)
        ├── OpenCVVisionAnalyzer  (uses ScreenshotProvider + Gemini multimodal)
        ├── ActionDecisionEngine  (uses FunctionCallingAgent → LLM)
        ├── ActionEngine          (uses BrowserController → Selenium)
        ├── Validator             (uses LLMRouter → Ollama / OpenRouter / Gemini)
        ├── SelfCorrectionEngine  (uses LLMRouter → Ollama / OpenRouter / Gemini)
        ├── FeedbackLoop          (pure retry orchestrator, no LLM)
        └── AutonomousAssistant   (top-level orchestrator)
```

---

## 4. Module Reference

### 4.1 Entry Point – `main.py`

**File:** `src/aletheia_ai/main.py`

The CLI entry point for the application. Parses command-line arguments using `argparse` and either:
- Starts the **Tkinter GUI** (`--gui` flag or when no `--task` is given).
- Runs the **autonomous agent** for a given task string (`--task "..."`) and prints the result as JSON to stdout.

**CLI Arguments:**

| Argument | Type | Description |
|---|---|---|
| `--task` | `str` | Natural language goal to execute (e.g., `"Search for AI news"`) |
| `--context` | `str` | Optional additional context provided to the reasoning engine |
| `--gui` | flag | Launch the desktop Tkinter GUI instead of CLI mode |

**Exit Codes:**
- `0` – Success
- `1` – Unhandled fatal error
- `2` – Known `AletheiaError` (configuration, reasoning, planning, etc.)

---

### 4.2 Configuration – `config.py`

**File:** `src/aletheia_ai/config.py`

Loads all runtime configuration from environment variables (`.env` file or OS environment) into a typed `AppConfig` dataclass.

**`AppConfig` Fields:**

| Field | Env Variable | Default | Description |
|---|---|---|---|
| `gemini_api_key` | `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `gemini_model` | `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model identifier |
| `openrouter_api_key` | `OPENROUTER_API_KEY` | `None` | OpenRouter API key (optional) |
| `openrouter_model` | `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model |
| `openrouter_base_url` | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `local_llm_provider` | `LOCAL_LLM_PROVIDER` | `ollama` | Local LLM provider type |
| `local_llm_base_url` | `LOCAL_LLM_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `local_llm_model` | `LOCAL_LLM_MODEL` | `llama3.1:8b` | Local model name |
| `local_llm_api_key` | `LOCAL_LLM_API_KEY` | `None` | Local LLM API key (optional) |
| `log_level` | `LOG_LEVEL` | `INFO` | Logging level |
| `max_retries` | `MAX_RETRIES` | `3` | Max per-step retry attempts |
| `max_replans` | `MAX_REPLANS` | `2` | Max whole-task re-plan cycles |
| `retry_backoff_seconds` | `RETRY_BACKOFF_SECONDS` | `1.5` | Backoff multiplier between retries |
| `screenshot_path` | `SCREENSHOT_PATH` | `./runtime/latest_screen.png` | Path to save screenshots |
| `browser_headless` | `BROWSER_HEADLESS` | `false` | Run browser in headless mode |
| `keep_browser_open` | `KEEP_BROWSER_OPEN` | `true` | Keep browser window open after run |
| `selenium_driver` | `SELENIUM_DRIVER` | `edge` | Browser driver: `edge`, `chrome`, or `firefox` |
| `selenium_base_url` | `SELENIUM_BASE_URL` | `https://www.google.com` | Default browser start URL |

---

### 4.3 Bootstrap / Dependency Wiring – `bootstrap.py`

**File:** `src/aletheia_ai/bootstrap.py`

The **composition root** of the application. The `build_assistant(config)` function instantiates and wires all components together, returning a fully configured `AutonomousAssistant` ready to run tasks. This is the only place where concrete classes are instantiated – all other modules depend only on interfaces.

---

### 4.4 Core Domain – `core/`

**Files:** `src/aletheia_ai/core/`

Contains the foundational building blocks shared across all modules.

#### `exceptions.py` – Exception Hierarchy

| Exception | Parent | Raised When |
|---|---|---|
| `AletheiaError` | `Exception` | Base for all project errors |
| `ConfigurationError` | `AletheiaError` | Invalid/missing environment config |
| `ReasoningError` | `AletheiaError` | LLM reasoning returns malformed output |
| `PlanningError` | `AletheiaError` | Task plan cannot be generated/parsed |
| `VisionError` | `AletheiaError` | Screenshot capture or vision analysis fails |
| `ActionExecutionError` | `AletheiaError` | A desktop/browser action fails to execute |
| `ValidationError` | `AletheiaError` | Validation logic cannot evaluate a result |

#### `models.py` – Domain Data Models

| Model | Type | Purpose |
|---|---|---|
| `ActionType` | Enum | All supported action types (9 values) |
| `StepStatus` | Enum | Step lifecycle states: PENDING, RUNNING, SUCCEEDED, FAILED |
| `ReasoningOutput` | Dataclass | LLM reasoning result: intent, constraints, strategy, success_criteria |
| `PlanStep` | Dataclass | A single executable step: id, description, action_type, parameters, validation_hint |
| `TaskPlan` | Dataclass | Ordered list of `PlanStep`s for a goal |
| `VisionSnapshot` | Dataclass | Screenshot analysis result: path, dimensions, summary, key_elements, recommended_action |
| `ActionResult` | Dataclass | Outcome of executing a step: step_id, status, details, observation |
| `ValidationResult` | Dataclass | Validation outcome: passed, confidence, rationale, suggested_correction, next_step |
| `ActionDecision` | Dataclass | LLM-decided action: action_type, target, input_text |
| `SelfCorrectionOutput` | Dataclass | Correction from self-correction engine: new_strategy, updated_action |

**Supported `ActionType` values:**

| Value | Description |
|---|---|
| `open_app` | Launch a desktop application |
| `mouse_click` | Click at screen coordinates |
| `mouse_move` | Move cursor to screen coordinates |
| `keyboard_write` | Type text |
| `keyboard_hotkey` | Send keyboard shortcut (e.g., Enter, Ctrl+C) |
| `browser_open` | Open a URL in the browser |
| `browser_click` | Click a CSS-selected element |
| `browser_type` | Type into a CSS-selected element |
| `wait` | Pause execution for N seconds |

#### `contracts.py` – Behavioral Protocols

Defines Python `Protocol` interfaces that all core components must satisfy:

| Protocol | Key Method | Implemented By |
|---|---|---|
| `Reasoner` | `reason(task, context) → ReasoningOutput` | `GeminiReasoner` |
| `Planner` | `create_plan(task, reasoning) → TaskPlan` | `TaskPlanner` |
| `VisionAnalyzer` | `capture_and_analyze(prompt) → VisionSnapshot` | `OpenCVVisionAnalyzer` |
| `ActionExecutor` | `execute(step) → ActionResult` | `ActionEngine` |
| `ActionDecider` | `decide(step, snapshot) → ActionDecision` | `ActionDecisionEngine` |
| `StepValidator` | `validate(step, result, prev, curr, criteria) → ValidationResult` | `Validator` |
| `SelfCorrector` | `self_correct(step, reason) → SelfCorrectionOutput` | `SelfCorrectionEngine` |

---

### 4.5 Reasoning Engine – `reasoning/`

**File:** `src/aletheia_ai/reasoning/gemini_reasoner.py`

**Class:** `GeminiReasoner`

Converts a natural language task into a structured `ReasoningOutput` by prompting an LLM to identify the goal, break it into logical steps, and predict risks.

**LLM Prompt Schema (output):**
```json
{
  "goal": "...",
  "steps": ["step1", "step2", "step3"],
  "risks": ["possible failure 1", "possible failure 2"]
}
```

**Provider cascade:**
1. **Local LLM (Ollama)** – tried first if configured.
2. **OpenRouter** – tried second if an API key is set.
3. **Google Gemini** – primary cloud backend.
4. **Deterministic fallback** – if all LLMs fail, a safe generic 4-step plan is returned so execution does not halt entirely.

---

### 4.6 Task Planner – `planning/`

**File:** `src/aletheia_ai/planning/task_planner.py`

**Class:** `TaskPlanner`

Converts a `ReasoningOutput` into a `TaskPlan` (a typed list of up to 6 `PlanStep`s).

**LLM Prompt Schema (output):**
```json
[
  {"step": 1, "action": "Open browser"},
  {"step": 2, "action": "Type search query"}
]
```

**Key behaviours:**
- **Deterministic shortcut:** For tasks containing keywords `calculator`, `calc`, or `search`, a hand-coded fallback plan is used immediately (no LLM call needed). This guarantees reliability for common automation patterns.
- **Sanity checking:** If the LLM returns steps that are off-topic or contain dependency-setup keywords (e.g., `chromedriver`, `webdriver`), the plan is discarded and the deterministic fallback is used.
- **Action inference:** The text of each step is heuristically mapped to a typed `ActionType` and parameter set (e.g., `"Open browser"` → `BROWSER_OPEN`).
- **Provider cascade:** Ollama → OpenRouter → Gemini → deterministic fallback.

---

### 4.7 Vision Module – `vision/`

**Files:** `src/aletheia_ai/vision/`

#### `ScreenshotProvider`
Captures the current desktop screen using PyAutoGUI and saves it to the configured path.

#### `OpenCVVisionAnalyzer`
Analyzes the captured screenshot to produce a `VisionSnapshot`.

**Primary path (Gemini Multimodal):**
- Reads the screenshot with OpenCV.
- Sends the image bytes + analysis prompt to Gemini's vision API.
- Parses JSON response describing screen state, UI elements, and recommended action.

**Fallback path (OpenCV only):**
- Runs Canny edge detection and brightness analysis.
- Infers whether the screen is UI-rich or low-structure, and light or dark themed.
- Returns a descriptive summary without LLM dependency.

**Vision Prompt Schema (output):**
```json
{
  "screen_state": "Browser window showing Google homepage",
  "elements": ["search bar", "Google logo", "I'm Feeling Lucky button"],
  "recommended_action": "Click on the search bar"
}
```

---

### 4.8 Action Engine – `action/`

**Files:** `src/aletheia_ai/action/`

#### `ActionEngine`
Executes a `PlanStep` by dispatching to the appropriate hardware/software handler:

| ActionType | Handler | Technology |
|---|---|---|
| `OPEN_APP` | Subprocess / Start menu | `subprocess.Popen`, PyAutoGUI |
| `MOUSE_CLICK` | Direct coordinates | PyAutoGUI |
| `MOUSE_MOVE` | Direct coordinates | PyAutoGUI |
| `KEYBOARD_WRITE` | Text typing | PyAutoGUI |
| `KEYBOARD_HOTKEY` | Key combination | PyAutoGUI |
| `BROWSER_OPEN` | Navigate to URL | Selenium |
| `BROWSER_CLICK` | CSS selector click | Selenium |
| `BROWSER_TYPE` | Type into element | Selenium |
| `WAIT` | Sleep | Python `time.sleep` |

**Known apps (Windows):** calculator, notepad, paint – launched via `subprocess.Popen` with `calc.exe`, `notepad.exe`, `mspaint.exe`.

#### `BrowserController`
Selenium WebDriver adapter. Supports **Edge** (default), **Chrome**, and **Firefox**. Handles:
- Lazy driver initialization on first use.
- Session recovery: automatically resets the WebDriver on `InvalidSessionIdException` or `NoSuchWindowException`.
- Headless mode support.
- Detach mode (keep browser open after script exits).

#### `ActionDecisionEngine`
Uses an LLM (`FunctionCallingAgent`) to decide the concrete action (target element, text) for non-deterministic steps, based on the current vision snapshot.

#### `FunctionCallingAgent`
LLM wrapper that produces an `ActionDecision` (action_type, target, input_text) from a step description and screen context.

---

### 4.9 Validation Module – `validation/`

**File:** `src/aletheia_ai/validation/validator.py`

**Class:** `Validator`

Determines whether a plan step succeeded after execution by comparing before/after `VisionSnapshot`s and the `ActionResult`.

**Validation cascade:**
1. **Immediate failure check** – If `action_result.status != "succeeded"`, return failed immediately.
2. **Deterministic heuristics** – For reliable action types (BROWSER_OPEN, OPEN_APP, BROWSER_CLICK, BROWSER_TYPE), always passes with high confidence (0.9–0.95). For KEYBOARD_WRITE and KEYBOARD_HOTKEY, passes if screen changed.
3. **Local LLM (Ollama)** – LLM-based semantic validation if available.
4. **OpenRouter** – Cloud LLM fallback.
5. **Google Gemini** – Primary cloud LLM validation.
6. **Screen-change fallback** – If all LLMs fail, passes only if the screen summary changed.

**LLM Validation Prompt Schema (output):**
```json
{
  "status": "success",
  "reason": "Search results are now visible on screen",
  "next_step": "continue"
}
```

---

### 4.10 Feedback & Self-Correction – `feedback/`

**Files:** `src/aletheia_ai/feedback/`

#### `FeedbackLoop`
The retry orchestrator for individual plan steps. For each step:

1. Captures a pre-execution `VisionSnapshot`.
2. Optionally calls `ActionDecisionEngine` for non-deterministic steps.
3. Executes the step via `ActionEngine`.
4. Captures a post-execution `VisionSnapshot`.
5. Calls `Validator.validate()`.
6. If validation passes → returns success.
7. If validation fails and retries remain → calls `SelfCorrectionEngine.self_correct()` to update the step, waits `backoff_seconds × attempt`, retries.
8. If max retries exceeded → returns a `FAILED` `ActionResult`.

**Correction application logic (`_apply_correction`):**
- Updates step description and validation hint with the new action text.
- Increases wait duration if the correction strategy says `"wait longer"`.
- Adjusts CSS selector if strategy mentions `"selector"`.

#### `SelfCorrectionEngine`
Calls an LLM to produce a corrected strategy and updated action description when a step fails.

**LLM Prompt Schema (output):**
```json
{
  "new_strategy": "Click on the search input using a more specific selector",
  "updated_action": "Click the element with id='search-input'"
}
```

**Provider cascade:** Ollama → OpenRouter → Gemini → hardcoded heuristic fallback.

---

### 4.11 Orchestrator – `orchestrator/`

**File:** `src/aletheia_ai/orchestrator/autonomous_assistant.py`

**Class:** `AutonomousAssistant`

The top-level coordinator of the entire agent lifecycle.

**`run(task, context)` algorithm:**

```
1. reason(task, context)           → ReasoningOutput
2. create_plan(task, reasoning)    → TaskPlan

3. LOOP:
   for step in plan.steps:
     execute_with_feedback(step)   → (ActionResult, ValidationResult, VisionSnapshot)
     log execution entry
     if step failed:
       break inner loop

   if all steps passed:
     return {status: "succeeded", ...}

   if replan_count >= max_replans:
     return {status: "failed", ...}

   replan_count++
   build_replan_context(failed_step, failure_reason, recent_history)
   reason(task, replan_context) → new ReasoningOutput
   create_plan(task, new_reasoning) → new TaskPlan
   continue LOOP
```

**Re-plan context includes:**
- Original context string.
- Replan cycle number.
- The failed step ID and description.
- The failure reason.
- The last 5 execution log entries.
- Instruction to avoid repeating failed assumptions.

**Return payload:**
```json
{
  "task": "...",
  "status": "succeeded | failed",
  "reasoning": { "intent": "...", "constraints": [...], "strategy": "...", "success_criteria": [...] },
  "plan": { "goal": "...", "steps": [...] },
  "execution_log": [
    {
      "replan_cycle": 0,
      "step_id": 1,
      "description": "...",
      "action_status": "succeeded",
      "action_details": "...",
      "validation_passed": true,
      "validation_confidence": 0.95,
      "validation_rationale": "...",
      "snapshot_summary": "..."
    }
  ],
  "replans": 0
}
```

---

### 4.12 LLM Router – `llm/`

**Files:** `src/aletheia_ai/llm/`

#### `LLMRouter`
A simple dataclass that holds an optional `OllamaClient` and `OpenRouterClient`. Created from `AppConfig` via `LLMRouter.from_config(config)`.

#### `OllamaClient`
HTTP client for locally-hosted Ollama models. Enabled when `LOCAL_LLM_PROVIDER=ollama` and the server is reachable. Provides `generate_text(prompt) → str`.

#### `OpenRouterClient`
HTTP client for the OpenRouter cloud proxy (supports GPT-4o, Claude, Mistral, etc.). Enabled when `OPENROUTER_API_KEY` is set. Provides `generate_text(prompt) → str`.

---

### 4.13 Desktop GUI – `ui/`

**File:** `src/aletheia_ai/ui/app.py`

**Class:** `AletheiaGUI`

A Tkinter-based desktop interface that runs the autonomous assistant without requiring terminal access.

**UI Components:**
- **Goal** text field – enter natural language task.
- **Context** text field – optional additional context.
- **Run Agent** button – starts execution in a background thread.
- **Status label** – shows "Ready" or "Running…".
- **Live Output** scrollable text area – streams real-time log messages via a thread-safe `queue.Queue`.

**Threading model:**  
The agent runs in a daemon worker thread. Log messages and results are relayed to the UI thread via `queue.Queue` and polled every 100 ms using Tkinter's `after()` mechanism (no direct cross-thread widget access).

---

### 4.14 Utilities – `utils/`

**File:** `src/aletheia_ai/utils/retry.py`

#### `RetryPolicy`
Dataclass holding `max_attempts` and `backoff_seconds`.

#### `with_retry(operation, policy, op_name)`
Generic retry wrapper. Retries `operation()` up to `policy.max_attempts` times, sleeping `backoff_seconds * attempt` between each attempt. Logs warnings on each failure and raises the last exception if all attempts are exhausted.

---

## 5. Data Models & Contracts

### Class Diagram (simplified)

```
AppConfig
    │
    ├── GeminiReasoner ─────────────────────► ReasoningOutput
    │       └── LLMRouter (OllamaClient / OpenRouterClient / Gemini)
    │
    ├── TaskPlanner ─────────────────────────► TaskPlan
    │       └── [PlanStep, PlanStep, ...]
    │
    ├── OpenCVVisionAnalyzer ────────────────► VisionSnapshot
    │       └── ScreenshotProvider
    │
    ├── ActionDecisionEngine ────────────────► ActionDecision
    │       └── FunctionCallingAgent
    │
    ├── ActionEngine ────────────────────────► ActionResult
    │       └── BrowserController (Selenium)
    │
    ├── Validator ───────────────────────────► ValidationResult
    │
    ├── SelfCorrectionEngine ────────────────► SelfCorrectionOutput
    │
    ├── FeedbackLoop
    │       └── (orchestrates: Vision → ActionDecider → Action → Vision → Validate → SelfCorrect)
    │
    └── AutonomousAssistant
            └── (orchestrates: Reason → Plan → FeedbackLoop × N → Replan × M)
```

---

## 6. LLM Provider Strategy

Aletheia AI follows a **three-tier provider cascade** to maximize reliability and support offline/low-cost deployments:

```
Tier 1 – Local (Ollama)
  └── Lowest latency, no API cost, fully offline
  └── Enabled by: LOCAL_LLM_PROVIDER=ollama + running Ollama server

Tier 2 – OpenRouter (Cloud Proxy)
  └── Access to many models (GPT-4o, Claude, etc.) via one API key
  └── Enabled by: setting OPENROUTER_API_KEY

Tier 3 – Google Gemini (Primary Cloud)
  └── Required; used when Tiers 1 & 2 are unavailable or disabled
  └── Required: GEMINI_API_KEY

Tier 4 – Deterministic Fallback
  └── Hard-coded safe behavior; no LLM required
  └── Used as last resort to prevent total failure
```

This cascade is applied independently in: Reasoning, Planning, Validation, and Self-Correction.

---

## 7. Execution Flow (End-to-End)

The following is a concrete step-by-step trace for the task `"Search for GitHub Copilot"`:

```
User:  python -m aletheia_ai.main --task "Search for GitHub Copilot"

1. main.py → AppConfig.from_env()
2. main.py → build_assistant(config)
3. AutonomousAssistant.run("Search for GitHub Copilot")

4. GeminiReasoner.reason("Search for GitHub Copilot")
   → ReasoningOutput(
       intent="Search GitHub Copilot on the web",
       constraints=["Browser must be available"],
       strategy="1. Open browser  2. Navigate  3. Type query  4. Submit",
       success_criteria=["Results visible"]
     )

5. TaskPlanner.create_plan(...)
   [Deterministic path: "search" in task]
   → TaskPlan(goal=..., steps=[
       PlanStep(1, "Open browser",       BROWSER_OPEN,  {url: "https://duckduckgo.com/"}),
       PlanStep(2, "Focus search bar",   BROWSER_CLICK, {css_selector: "input[name='q']"}),
       PlanStep(3, "Type search query",  BROWSER_TYPE,  {css_selector: "input[name='q']", text: "GitHub Copilot"}),
       PlanStep(4, "Click search button",BROWSER_CLICK, {css_selector: "button[type='submit']"}),
     ])

6. For each step → FeedbackLoop.execute_with_feedback(step, ...)
   ├── VisionSnapshot (pre) = capture_and_analyze("Pre-step: Open browser")
   ├── ActionEngine.execute(step)   → BrowserController.open("https://duckduckgo.com/")
   ├── VisionSnapshot (post) = capture_and_analyze("Post-step: Open browser")
   ├── Validator.validate(...)
   │   → Deterministic: BROWSER_OPEN always passes (confidence=0.95)
   │   → ValidationResult(passed=True, next_step="continue")
   └── log: {step_id:1, action_status:"succeeded", validation_passed:true, ...}
   ... (steps 2-4 similarly)

7. All steps succeeded → return {task: ..., status: "succeeded", execution_log: [...], replans: 0}
8. main.py → print JSON result
```

---

## 8. Configuration Reference

Copy `.env.example` to `.env` and populate the required fields:

```bash
# ── Required ──────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# ── Optional: OpenRouter (cloud LLM proxy) ─────────────────
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ── Optional: Local LLM via Ollama ────────────────────────
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3.2:1b
LOCAL_LLM_API_KEY=

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR

# ── Retry / Resilience ────────────────────────────────────
MAX_RETRIES=3                # Per-step retry attempts
MAX_REPLANS=2                # Whole-task replan cycles
RETRY_BACKOFF_SECONDS=1.5    # Seconds multiplier between retries

# ── Vision ────────────────────────────────────────────────
SCREENSHOT_PATH=./runtime/latest_screen.png

# ── Browser ───────────────────────────────────────────────
BROWSER_HEADLESS=false
KEEP_BROWSER_OPEN=true
SELENIUM_DRIVER=chrome       # edge | chrome | firefox
SELENIUM_BASE_URL=https://www.google.com
```

---

## 9. Installation & Setup

### Prerequisites

- Python 3.11 or later
- A Google Gemini API key (get one at https://aistudio.google.com/apikey)
- The appropriate WebDriver for your browser:
  - **Chrome**: ChromeDriver (matching your Chrome version)
  - **Edge**: Microsoft Edge WebDriver
  - **Firefox**: GeckoDriver
- *(Optional)* Ollama installed and running locally for offline/local LLM support

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/jayaprakash2207/Aletheia-AI---Autonomous-Thinking-Assistant
cd Aletheia-AI---Autonomous-Thinking-Assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

---

## 10. Usage Guide

### Terminal / CLI Mode

Run a task directly from the command line:

```bash
# Web search
python -m aletheia_ai.main --task "Open browser and search for Python tutorials"

# Desktop automation
python -m aletheia_ai.main --task "Open calculator and add 25 and 37"

# With additional context
python -m aletheia_ai.main \
  --task "Search for quarterly report" \
  --context "Use DuckDuckGo, look for the 2024 Q3 report"
```

**Output:** JSON result printed to stdout:
```json
{
  "task": "Open browser and search for Python tutorials",
  "status": "succeeded",
  "replans": 0,
  "execution_log": [ ... ]
}
```

### GUI Mode

Launch the desktop interface:

```bash
python -m aletheia_ai.main --gui
# or simply:
python -m aletheia_ai.main
```

A Tkinter window opens. Enter your goal in the **Goal** field and click **Run Agent**. Live logs appear in the output panel.

### Installed Command

After `pip install -e .`:

```bash
aletheia-ai --task "Search for the latest AI news"
aletheia-ai --gui
```

---

## 11. Error Handling & Resilience

### Retry Strategy

Every step is retried up to `MAX_RETRIES` times (default: 3) with exponential backoff (`RETRY_BACKOFF_SECONDS × attempt`). Between each retry, the `SelfCorrectionEngine` adapts the step.

### Re-planning Strategy

If a step exhausts all retries, the entire task is re-planned from scratch up to `MAX_REPLANS` times (default: 2). The re-plan context includes the full failure history, guiding the LLM to avoid repeating the same mistakes.

### LLM Fallbacks

Every LLM call follows the cascade: Local → OpenRouter → Gemini → Deterministic fallback. The application never hard-crashes due to LLM unavailability alone.

### Browser Session Recovery

The `BrowserController` automatically detects stale/invalid WebDriver sessions and reinitializes the driver before retrying browser operations.

### Logging

All components use Python's standard `logging` module with structured `extra` fields. Log level is controlled by the `LOG_LEVEL` environment variable. The GUI captures logs via a thread-safe queue handler.

---

## 12. Security Considerations

| Concern | Mitigation |
|---|---|
| **API Key Exposure** | Keys loaded from `.env` only; `.env` is in `.gitignore` |
| **Desktop Automation Scope** | PyAutoGUI `FAILSAFE=True` – move mouse to corner to abort |
| **Browser Isolation** | Test in a VM/sandbox for untrusted tasks |
| **Trusted Domains Only** | Limit `SELENIUM_BASE_URL` and tasks to trusted sites |
| **No Secret Logging** | API keys are never logged; structured `extra` fields only contain safe data |
| **Input Validation** | All LLM JSON responses are strictly validated before use; malformed output raises typed exceptions |

---

## 13. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `google-genai` | ≥1.12.0 | Google Gemini LLM & Vision API client |
| `pydantic` | ≥2.8.0 | Data validation (used by genai client) |
| `python-dotenv` | ≥1.0.1 | Load `.env` files into environment |
| `pyautogui` | ≥0.9.54 | Desktop mouse, keyboard, screenshot automation |
| `selenium` | ≥4.21.0 | Browser automation (WebDriver) |
| `Pillow` | ≥10.3.0 | Image processing (screenshot support) |
| `opencv-python` | ≥4.10.0.84 | Computer vision: edge detection, image analysis |
| `numpy` | ≥1.26.0 | Numerical operations for image analysis |

**Build system:** `setuptools ≥68` with `wheel`.

---

## 14. Project Structure

```
Aletheia-AI---Autonomous-Thinking-Assistant/
│
├── .env.example                    # Template for environment configuration
├── .gitignore
├── LICENSE                         # MIT License
├── README.md                       # Quick-start README
├── PROJECT_DOCUMENTATION.md        # This document
├── pyproject.toml                  # Build system and project metadata
├── requirements.txt                # Runtime dependencies
│
├── docs/
│   └── index.html                  # HTML documentation page
│
└── src/
    └── aletheia_ai/
        ├── __init__.py
        ├── main.py                 # CLI entry point
        ├── bootstrap.py            # Dependency wiring / composition root
        ├── config.py               # AppConfig – env-based configuration
        ├── logging_config.py       # Logging setup
        │
        ├── core/
        │   ├── contracts.py        # Protocol interfaces
        │   ├── exceptions.py       # Domain exception hierarchy
        │   └── models.py           # Shared domain data models
        │
        ├── reasoning/
        │   └── gemini_reasoner.py  # LLM-based task reasoning
        │
        ├── planning/
        │   └── task_planner.py     # LLM-based step planning
        │
        ├── vision/
        │   ├── screenshot_provider.py   # Desktop screenshot capture
        │   └── vision_analyzer.py       # Gemini + OpenCV image analysis
        │
        ├── action/
        │   ├── action_engine.py         # PyAutoGUI + Browser step executor
        │   ├── action_decision_engine.py # LLM-based action decision
        │   ├── browser_controller.py    # Selenium WebDriver adapter
        │   └── function_calling_agent.py # LLM function-calling wrapper
        │
        ├── validation/
        │   └── validator.py        # Step success verification
        │
        ├── feedback/
        │   ├── feedback_loop.py        # Per-step retry orchestration
        │   └── self_correction_engine.py # LLM-based step correction
        │
        ├── orchestrator/
        │   └── autonomous_assistant.py # Top-level task orchestrator
        │
        ├── llm/
        │   ├── llm_router.py       # Local/cloud LLM router
        │   ├── ollama_client.py    # Ollama local LLM client
        │   └── openrouter_client.py # OpenRouter cloud proxy client
        │
        ├── ui/
        │   └── app.py              # Tkinter desktop GUI
        │
        └── utils/
            └── retry.py            # Generic retry policy
```

---

## 15. Glossary

| Term | Definition |
|---|---|
| **Agent** | An autonomous software component that perceives its environment and takes actions to achieve a goal |
| **LLM** | Large Language Model – e.g., Google Gemini, GPT-4o, LLaMA |
| **Reasoning** | The process of analyzing a task and producing an intent, strategy, and constraints before acting |
| **Planning** | Breaking down a reasoning output into a concrete ordered sequence of executable steps |
| **Vision** | The ability of the agent to capture and understand the current state of the screen |
| **Action** | A concrete, typed operation executed on the desktop or browser |
| **Validation** | Checking whether an executed action produced the expected outcome |
| **Self-Correction** | Generating an alternative strategy for a failed step before retrying |
| **Re-planning** | Restarting the reasoning and planning cycle with updated failure context |
| **Feedback Loop** | The per-step cycle of: execute → capture → validate → (correct → retry) |
| **Orchestrator** | The top-level component that drives the full task lifecycle |
| **Protocol** | A Python structural interface (duck typing) used to decouple implementations |
| **Ollama** | An open-source tool for running LLMs locally |
| **OpenRouter** | A cloud proxy providing unified access to multiple LLM providers |
| **Gemini** | Google's family of multimodal large language models |
| **Selenium** | A web automation framework for controlling browsers programmatically |
| **PyAutoGUI** | A Python library for programmatic desktop mouse and keyboard control |
| **OpenCV** | Open Computer Vision library used for image processing and analysis |
| **Headless** | Running a browser without a visible window (background mode) |
| **Backoff** | Increasing wait time between retry attempts to reduce load and improve success chances |

---

*Document prepared for internal company submission.*  
*Project: Aletheia AI – Autonomous Thinking Assistant | Version 0.1.0 | MIT License*
