# Default Mode

Use this for quick, well-scoped edits that can be wrapped up in one sitting.

- Confirm current behavior and the desired change before editing.
- Scope the diff tightly; delete dead code when you touch it.
- Prefer `apply_patch` for single-file updates. If you must touch many files, explain why.
- Formatting: `uv run ruff format <paths>` for the files you changed.
- Lint: `uv run ruff check --fix <paths>` and call out any warnings you cannot auto-fix.
- Tests: run the smallest relevant slice, e.g. `uv run pytest tests/path::TestClass::test_case`.
- If validation is impossible (missing deps, long runtimes), say so and propose how the user can verify.
- Final message must lead with what changed, include file references (`path/to/file.py:12`), and note follow-up steps only when mandatory.
