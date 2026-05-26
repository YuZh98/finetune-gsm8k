# Architecture Decision Records

Hard-to-reverse decisions for `finetune-gsm8k`. One file per decision, numbered sequentially.

## Status values

- **Accepted** — decision is live, code reflects it
- **Superseded** — replaced by a later ADR (link forward)
- **Deprecated** — no longer applies, no replacement

## Template

```markdown
# ADR NNNN — <short title>

**Status:** Accepted | Superseded by ADR-MMMM | Deprecated
**Date:** YYYY-MM-DD

## Context
What forced the decision. Constraints, evidence, prior state.

## Decision
What we are doing. One sentence if possible.

## Consequences
What changes. What breaks. What follow-up work this creates.

## Alternatives considered
Each with one-line rejection reason.
```

## Index

- [0001 — Fallback from Qwen2.5-3B-Instruct to Qwen2.5-3B-base](0001-fallback-to-base-model.md)
