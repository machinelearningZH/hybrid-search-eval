# AGENTS Guidelines (Python)

Guidelines for agent-assisted development. Python 3.12+, managed with `uv`.

## Process & Security

- **Living Doc**: Update this file with new findings/notes.
- **Scope**: Work only in this repo.
- **Approvals**: **ALWAYS** ask before fetching external resources or installing packages.
- **Secrets**: Store in `.env`, load with `python-dotenv`. Never hardcode or commit secrets.
- **Assumptions**: Surface inconsistencies, unclear intent, or ambiguous requirements—ask for clarification before proceeding. Push back on bad ideas. Present trade-offs when relevant.
- **Simplicity**: Favor simple, explicit, maintainable solutions. No over-engineering. Always prefer the simplest solution that meets requirements.
- **Scope Discipline**: Only modify code directly related to the current task. Never change, move, or remove code, comments, or logic that is orthogonal to your task—even if it seems "wrong" or you don't fully understand it. If you spot issues elsewhere, flag them separately.
- **Cleanup**: Remove dead code, temporary files, and dev artifacts after each step. Release resources (files, connections, temp dirs) using `with` context managers or `atexit` for global cleanup.
- **Self-Check Before Finishing**: Before presenting a solution, verify: (1) Did I make assumptions I should have clarified? (2) Is this the simplest solution? (3) Did I change unrelated code as a side effect? (4) What alternatives/tradeoffs should I mention?

## Environment (`uv`)

Manage env with `uv`. Re-run `uv sync` after changes.

```bash
uv sync                 # Lock/Sync
uv add [--dev] <pkg>    # Add dependency
uv run <cmd>            # Run in env
```

## Python Conventions

- **Types**: Modern syntax (`list[str]`, `X | None`, `Self`). No `typing.List`.
- **Data**: Use `dataclasses` or `TypedDict`.
- **Paths**: `pathlib.Path` only.
- **Errors**: Specific exceptions with messages. No bare `except:`.
- **Formatting**: f-strings. Use debug format `f"{var=}"` → outputs `var=value`.

## Configuration

- **Settings**: Store in `config.yaml`, load with `pyyaml`.
- **Secrets**: Store in `.env`, load with `python-dotenv`. Never commit to git.
- **No magic values**: All configurable parameters belong in config files.

## Logging

Use `logging` module with JSON output for structured logs:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level": record.levelname, "message": record.getMessage(), "module": record.module})

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
```

## Principles

1. **Flat Architecture**: Explicit, linear control flow. No metaclasses, `exec`, or dynamic attribute generation.
2. **Predictable**: Consistent layout, standard patterns, deterministic tests.
3. **Modular**: Decoupled modules, config-driven behavior.
4. **Quality**: Descriptive names, structured logging.

## Documentation

- **Comments/Docstrings**: Explain _why_. Follow PEP 257.
- **README**: Concise usage/examples. No fluff.
- **Files**: Avoid extra docs. Update README/AGENTS.md instead.

## Testing

- **Location**: `tests/` directory, mirroring source structure.
- **Naming**: `test_<module>.py`, functions `test_<behavior>()`.
- **Fixtures**: Use `conftest.py` for shared fixtures.
- **Run**: `uv run pytest -v` (verbose) or `uv run pytest -x` (stop on first failure).

## Git

- **Commits**: Use conventional commits: `type(scope): message`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
  - Example: `feat(auth): add OAuth2 login flow`
- **Branches**: `feature/<name>`, `fix/<name>`, `refactor/<name>`
- **Keep clean**: Commit small, logical changes. No WIP commits on main.

## Code Quality

```bash
uv run ruff format .          # Format
uv run ruff check . [--fix]   # Lint
uv run pytest                 # Test
uv run ruff format . && uv run ruff check . && uv run pytest  # All checks
```

## Core Stack

- **Config**: `pyyaml` for YAML, `python-dotenv` for env vars.
- **CLI**: `typer` (not `argparse`). Type hints, `typer.Argument()`, `typer.Option()`. Use Enum for fixed choices.
- **HTTP**: `httpx` for async requests.
- **Output**: `rich` (`Console`, `Table`) for terminal output.

## Domain-Specific Stack

- **FastAPI**: Pydantic validation, `app/routers/` modules, dependency injection, `async` I/O.
- **Streamlit**: `st.sidebar` for controls, `st.session_state`, `@st.cache_data`.
- **LLM**: OpenRouter via OpenAI-compatible client. Load API key from `.env`. Config in `config.yaml` (model, temp, tokens). Concurrent: `ThreadPoolExecutor`.
- **Embeddings**: `sentence-transformers` (local, e.g., `intfloat/multilingual-e5-small`).
- **Scraping**: `playwright` — async, headless, built-in selectors (`role`, `text`).
- **Data Science**: Jupyter, pandas (vectorized), pyarrow/parquet, scikit-learn, seaborn.
- **Document Parsing**: `docling` — parse docs to MD (`export_to_markdown`).

## Discovery Log

- (Agent: Add project-specific notes/stack decisions here)
