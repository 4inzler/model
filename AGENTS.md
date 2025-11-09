# Codex Agents

The `.codex/modes` directory defines how this project expects Codex agents to behave. Each mode below is tuned for a different kind of task while sharing the same baseline expectations:

- Assume work happens against Python 3.11 managed with `uv`. Keep dependencies minimal and declare them in `pyproject.toml` once it exists.
- Format with `uv run ruff format` and lint with `uv run ruff check --fix`.
- Run targeted tests with `uv run pytest path/to/tests`. Aim to run the smallest useful slice; call out anything you could not run.
- Prefer `apply_patch` for surgical edits and keep responses concise, noting risks and follow-ups.

| Agent | Mode file | Use when |
| - | - | - |
| `@codex` | `.codex/modes/default.md` | Fast fixes or small changes that can be shipped immediately. |
| `@codex plan` | `.codex/modes/plan.md` | Medium/large tasks that benefit from a written plan before editing. |
| `@codex build` | `.codex/modes/build.md` | New features or refactors that require multiple files or validation steps. |
| `@codex spec` | `.codex/modes/spec.md` | Early design work or when requirements are unclear and need exploration. |
| `@codex review` | `.codex/modes/review.md` | Code review passes when a user asks for feedback instead of changes. |

## default
Stay nimble and decisive. Confirm the goal, outline the minimal change, mention relevant tests, and run or suggest the narrowest verification needed. If you cannot validate something, explain why and propose the next best check. Keep the final message short—lead with the change and list follow-up actions only if they are truly required.

## plan
Use the plan tool for any multi-step effort. Provide at least three concrete steps, update the plan after each major action, and call out risks or unknowns. Plans should mention where code will live, how it will be tested, and any docs that might need updates.

## build
Ideal for feature work. Start from a clear plan, implement changes with well-placed comments when the code is non-obvious, and keep commits logically scoped. Always rerun relevant linters/tests before finishing and summarize remaining risks or chores (docs, follow-up issues) for the user.

## spec
When requirements are fuzzy, produce a short proposal before touching code. Capture user goals, constraints, open questions, and a suggested implementation strategy with estimated effort. This mode should finish with explicit next steps (e.g., “confirm approach”, “start build mode”).

## review
Operate like a focused code reviewer. Lead with findings ordered by severity, reference files with line numbers (e.g., `src/app.py:42`), and explain impact plus the suggested fix. If everything looks good, say so and outline any testing gaps or remaining risks.
