# Domain Docs

This repository uses a single context: root `CONTEXT.md` and `docs/adr/`.

## Before Exploring

Read [CONTEXT.md](../../CONTEXT.md), then the accepted ADRs relevant to the work in [docs/adr/](../adr/). If a domain document is absent, proceed silently; `domain-modeling` creates it when terminology or decisions are resolved.

Read [PRD.md](../design/PRD.md) before changing scope or acceptance criteria, and [driving-learning-design.md](../design/driving-learning-design.md) before implementation changes. [scope-decisions.md](../archive/scope-decisions.md) records the user's decisions; historical research remains supporting evidence rather than the current product baseline.

## File Layout

```text
CONTEXT.md
docs/
  README.md
  design/
    PRD.md
  archive/
    scope-decisions.md
  adr/
    0001-known-route-first.md
    0002-separate-driving-and-recovery.md
    0003-validity-before-performance.md
```

## Vocabulary and Decisions

Use the glossary's terminology in issues, proposals, experiments, and tests. Keep `CONTEXT.md` a glossary. When a needed concept is missing, distinguish an unnecessary synonym from a real gap and use `domain-modeling` for the latter.

Call out conflicts with accepted ADRs explicitly, naming the ADR and the evidence for reconsideration. Record an agreed change in the relevant documents before treating it as the new baseline. New ADR filenames follow `NNNN-short-topic.md`.
