# Repository Guidelines

## Project Structure & Module Organization

- `src/fh5/experiment.py` owns the public record/replay experiment interface, packet decoding, and diagnostics.
- `src/fh5/cli.py` adapts UDP and command-line input; `report.py` and `report.html` produce offline reports.
- `control.py` owns bounded calibration; `live.py` adapts Windows, UDP, and the virtual controller. Read `docs/control.md` before live control or driver changes.
- `tests/` checks observable experiment behavior using synthetic packets and loopback UDP. `configs/` contains versioned configuration examples.
- `runs/` holds ignored local recordings. Keep recordings, credentials, model checkpoints, and machine tooling out of commits.
- Before changing scope, read `docs/PRD.md`. Before changing control or learning behavior, read `docs/driving-learning-design.md`. `CONTEXT.md` holds domain terminology; `docs/adr/` records consequential decisions.
- Before adding or changing resource limits, storage, training continuation, or recovery, read `docs/resource-policy.md`. Use measured capacity, interface requirements, or explicit experiment budgets; handle growing data structurally and preserve completed work when optional diagnostics fail.

## Build, Test, and Development Commands

Run from the repository root with Python 3.12 and uv:

- `uv sync --locked`: create the environment from `uv.lock`.
- `uv run --locked fh5 --help`: inspect recording and replay commands; examples are in `README.md`.
- `uv run --locked pytest`: run the behavioral suite; append `tests/test_experiment.py` for focused checks.
- `uv run --locked mypy`: run strict source type checking.
- `uv run --locked ruff check .` and `uv run --locked ruff format --check .`: check lint and formatting. Use `ruff format .` to format changes.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/files, and `PascalCase` types. Keep the experiment interface independent of transport adapters. Recording and replay use the standard library; Windows control dependencies stay optional and lazily loaded.

Use UTF-8, two-space JSON indentation, and lowercase kebab-case document names. Preserve Chinese product documentation and sequential `NNNN-short-topic.md` ADR names.

## Testing Guidelines

Use pytest files named `test_*.py`; no numeric coverage threshold is imposed. Test through the agreed experiment-run seam with independent expected results. Synthetic packets and loopback tests establish software behavior; record real-game evidence separately. See `docs/recording.md` for protocol assumptions and live checks. Before changing rewards or evaluation, consult `docs/reward-and-validity-design.md`.

## Commit & Pull Request Guidelines

Use concise imperative subjects with a scope prefix, following `docs: add project plan and agent workflow configuration`. Reference relevant issues. PR descriptions explain resulting behavior, validation, and outstanding limitations; driving changes need logs or clips. Keep incomplete live acceptance explicit.

## Agent Skills

- For issue workflows, read `docs/agents/issue-tracker.md`; GitHub is the tracker.
- Before triage, read `docs/agents/triage-labels.md`.
- Before domain exploration, read `docs/agents/domain.md`.
