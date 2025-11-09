# Build Mode

Best for multi-file features, refactors, or fixes that need robust validation.

- Start from an approved plan (or produce one quickly if missing).
- Keep the working tree clean; group edits logically and avoid drive-by changes unrelated to the task.
- Document any new public behavior in README, docs, or docstrings.
- Format and lint everything you touched: `uv run ruff format <paths>` then `uv run ruff check --fix`.
- Run the relevant test matrix:
  - Unit: `uv run pytest tests`
  - Type checks (if configured later): `uv run mypy <paths>`
- Record any tests you skipped and why.
- Before wrapping up, scan for TODOs or temporary scaffolding and remove them unless explicitly requested.
- Final response should summarize behavior changes, validation performed, and any risks or follow-ups.
