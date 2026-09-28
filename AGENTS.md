# Repository Guidelines

## Project Structure & Module Organization

This checkout is documentation-first; source, test, and runtime asset directories have not been created.

- `README.md` indexes the project documents.
- `docs/PRD.md` defines current requirements and acceptance criteria. Read it before changing scope or behavior.
- `CONTEXT.md` contains domain terminology; keep it a glossary.
- `docs/driving-learning-design.md` describes the control and learning architecture. Consult it before implementation changes.
- `docs/adr/` records consequential design decisions. Use `docs/scope-decisions.md` to trace scope choices; historical research does not override the PRD.
- `docs/hardware-snapshot.json` records observed hardware, not runtime benchmarks.

## Build, Test, and Development Commands

Run these from the repository root in PowerShell:

- `rg --files --hidden`: inventory files, including future tooling configuration.
- `rg -n "FR-|M2a|M3" docs/PRD.md`: locate requirements and milestones.
- `Get-Content -Raw docs/hardware-snapshot.json | ConvertFrom-Json | Out-Null`: validate JSON syntax.

No build, application launch, or automated test commands are configured. Add reproducible commands to `README.md` when introducing implementation tooling.

## Coding Style & Naming Conventions

Use UTF-8 Markdown with descriptive ATX headings, fenced command examples, and blank lines around lists and tables. Preserve the existing Chinese language in product documents. Indent JSON with two spaces.

Name new documents with lowercase kebab-case, preserving established names such as `PRD.md` and `CONTEXT.md`. Number ADRs sequentially as `NNNN-short-topic.md`. No formatter, linter, or source-language conventions are configured yet; establish them alongside the first implementation.

## Testing Guidelines

No testing framework or coverage threshold exists. For documentation changes, check relative links, JSON syntax, terminology, and consistency with the PRD.

When adding executable code, document its test runner and naming convention. Prioritize behavioral checks for telemetry parsing, action timing, recovery boundaries, and reward exploits. Consult `docs/reward-and-validity-design.md` before changing rewards or evaluation. Report actual checks performed and separate unverified game behavior from observed results.

## Commit & Pull Request Guidelines

The initial Git history establishes no commit convention. Use concise imperative subjects, such as `docs: clarify rewind validation`.

PR descriptions should explain the problem, resulting behavior, relevant PRD requirement IDs or issues, and validation performed. Attach logs or clips for driving changes; identify remaining limitations. Update affected design documents when behavior or acceptance rules change.

## Agent skills

### Issue tracker

For issue workflows, use GitHub; read `docs/agents/issue-tracker.md`.

### Triage labels

Before triage, read the default mappings in `docs/agents/triage-labels.md`.

### Domain docs

Before exploration, read `docs/agents/domain.md` for the single-context layout.
