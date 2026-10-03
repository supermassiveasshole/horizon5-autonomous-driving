# Repository Guidelines

## Project Structure & Module Organization

- `src/fh5/experiment.py` is the public experiment interface; `cli.py` and `commands/` assemble CLI operations.
- `telemetry/`, `observation/`, `capture/`, and `collection/` own packets, model inputs, image capture, and independent recording. `driving/` owns control and realtime execution; `learning/` contains BC, SAC, and the learning loop; `evaluation/` owns validity and candidate selection.
- `artifacts/` owns shared file operations; `reporting/` renders results. Keep low-level types and storage independent of workflow dispatch. See [architecture](docs/architecture.md) for dependency rules.
- `tests/` mirrors these responsibilities; shared fixtures live in `tests/support/`. `configs/` keeps stable versioned examples.
- `runs/` holds ignored local recordings. Keep recordings, credentials, model checkpoints, and machine tooling out of commits.
- Read `docs/design/PRD.md` before scope changes, `docs/design/driving-learning-design.md` before driving/learning changes, and `docs/guides/runtime/control.md` before live control. Terminology lives in `CONTEXT.md`; decisions live in `docs/adr/`.
- Read `docs/design/resource-policy.md` before resource, storage, or recovery changes. Base limits on measured capacity, interface requirements, or explicit budgets; handle growth structurally and preserve completed work when diagnostics fail.

## Build, Test, and Development Commands

Run from the repository root with Python 3.12 and uv:

- `uv sync --locked`: create the environment from `uv.lock`.
- `uv run --locked fh5 --help`: inspect recording and replay commands; examples are in `README.md`.
- `uv run --locked pytest`: run the behavioral suite; append `tests/telemetry/` for focused checks.
- `uv run --locked mypy`: run strict source type checking.
- `uv run --locked ruff check .` and `uv run --locked ruff format --check .`: check lint and formatting. Use `ruff format .` to format changes.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/files, and `PascalCase` types. Keep the experiment interface independent of transport adapters. Recording and replay use the standard library; Windows control dependencies stay optional and lazily loaded.

Use UTF-8, two-space JSON indentation, and lowercase kebab-case document names. Preserve Chinese product documentation and sequential `NNNN-short-topic.md` ADR names.

## Testing Guidelines

Use `test_*.py`; no numeric coverage threshold is imposed. Test observable behavior through `run_experiment` or the CLI. Synthetic packets and loopback tests establish software behavior; record real-game evidence separately. See `docs/guides/capture/recording.md` for protocol checks and `docs/design/reward-and-validity-design.md` before reward/evaluation changes.

## Commit & Pull Request Guidelines

Use concise imperative subjects with a scope prefix, following `docs: add project plan and agent workflow configuration`. Reference relevant issues. PR descriptions explain resulting behavior, validation, and outstanding limitations; driving changes need logs or clips. Keep incomplete live acceptance explicit.

## Agent Skills

- For issue workflows, read `docs/agents/issue-tracker.md`; GitHub is the tracker.
- Before triage, read `docs/agents/triage-labels.md`.
- Before domain exploration, read `docs/agents/domain.md`.
