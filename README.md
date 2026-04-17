# Aletheia AI - Autonomous Thinking Assistant

Aletheia AI is a production-oriented autonomous assistant that can reason, plan, perceive the screen, execute actions, validate outcomes, and self-correct with a feedback loop.

## Core Capabilities

- Think before acting through an explicit reasoning stage backed by Gemini.
- Break complex requests into ordered executable steps.
- Understand visual context from desktop screenshots.
- Execute desktop and browser actions through pluggable engines.
- Validate outcomes and retry with adaptive corrections.

## Architecture

- Reasoning Engine: LLM-backed intent understanding and strategy generation.
- Task Planner: Converts strategy into executable step plans.
- Vision Module: Captures and analyzes screenshots for state awareness.
- Action Engine: Executes mouse, keyboard, and browser actions.
- Validator: Evaluates whether a step succeeded.
- Feedback Loop: Applies retries, diagnostics, and corrective actions.

## Project Structure

```text
Aletheia AI/
  .env.example
  requirements.txt
  README.md
  src/
    aletheia_ai/
      main.py
      bootstrap.py
      config.py
      logging_config.py
      core/
      reasoning/
      planning/
      vision/
      action/
      validation/
      feedback/
      orchestrator/
      utils/
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Configure environment variables:

```bash
copy .env.example .env
```

4. Add your Gemini API key to `.env`.

## Run

```bash
python -m aletheia_ai.main --task "Open browser and search for weather in Bangalore"
```

## GUI Frontend

Launch the desktop app and enter tasks in a window instead of the terminal:

```bash
python -m aletheia_ai.main --gui
```

You can also omit `--task` entirely and the GUI will open by default.

## Safety Notes

- Desktop automation can control keyboard and mouse globally. Run in a controlled environment.
- Use browser automation only on domains you trust.
- Consider policy checks before enabling sensitive actions.

## Production Recommendations

- Add secrets manager integration instead of plain `.env` for production.
- Configure centralized logging sink (for example ELK, Datadog, or OpenTelemetry).
- Add policy guardrails and permission scopes for high-risk actions.
- Add integration tests against a deterministic sandbox environment.
