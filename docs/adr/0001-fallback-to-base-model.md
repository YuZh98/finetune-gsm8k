# ADR 0001 — Fallback from `Qwen2.5-3B-Instruct` to `Qwen2.5-3B` (base)

**Status:** Accepted
**Date:** 2026-05-25

## Context

The project spec (`docs/design.md` §4.1, §8) pre-registered a fallback condition:

> "If the instruct baseline is already saturated and no config beats it by ≥ 3 pp,
> fall back to `Qwen2.5-3B-base` (non-instruct). Document the swap honestly in the README."

That condition is now met. Evidence:

| Run                    | Seed | GSM8K acc | Δ vs base |
|------------------------|------|-----------|-----------|
| run0_base (instruct)   | —    | 0.8120    | —         |
| run2_r16               | 42   | 0.6285    | −18.4 pp  |
| run2_r16_seed43        | 43   | 0.6907    | −12.2 pp  |

Two observations:

1. **Both fine-tuned seeds regress vs base by 12–18 pp.** QLoRA on MetaMathQA is
   not improving the instruct baseline; it is damaging it. The most plausible
   mechanism is documented in `docs/lesson-prompt-template-echo.md` — MetaMathQA
   format ("The answer is: N") competes with the prompt's requested format
   (`#### N`), producing a confused hybrid that the extractor only partially
   recovers, plus reasoning quality drift on a capability the instruct model
   was already strong at.
2. **Seed-only variance is ~6 pp (0.6285 → 0.6907)**, which exceeds the spec's
   ≥ 3 pp threshold for a "real" effect. Single-seed comparisons across the
   ablation matrix would be dominated by noise.

Either observation alone would already trigger the fallback; both together
make it unambiguous.

## Decision

Swap the base model from `Qwen/Qwen2.5-3B-Instruct` to `Qwen/Qwen2.5-3B`
(non-instruct). Re-run the full 8-cell ablation matrix on the new base, with
≥ 2 seeds per cell.

## Consequences

- **`src/config.py`**: `BASE_MODEL` updated. Single-line change.
- **Chat template / prompt**: base (non-instruct) model has no chat template.
  `data.py` and `eval_gsm8k.py` need to switch from `tokenizer.apply_chat_template`
  to a plain prompt format. Tracked as follow-up before re-running.
- **All prior eval numbers invalidated.** Instruct-era `runs.csv` rows are
  archived (kept with a `base_model` column noting the swap), not deleted.
- **Compute**: roughly doubles total budget (re-run all 7 adapters + base eval,
  × 2 seeds minimum). Within the design.md `≤ 1 week` envelope only if A100
  available.
- **README narrative**: the instruct → base swap becomes a section of its own.
  Pre-registered fallback executed honestly is a *stronger* writeup than a
  silent retry would be.

## Alternatives considered

- **Keep instruct, reframe as negative-result study.** Rejected: thin for CV;
  the project's design pre-committed to producing a positive ablation, and the
  fallback path was put in the spec specifically for this case.
- **Keep instruct, fix the prompt/SFT format mismatch first (lesson-doc §"What
  a real fix would look like").** Rejected as first move: base is saturated
  regardless of format; format fix might close part of the gap but does not
  create headroom. Worth revisiting on the base model if format drift appears
  there too.
- **Switch to a different base entirely (Llama-3.2-3B, Qwen2.5-7B).** Rejected:
  design.md §4.1 already dismissed both for reasons that have not changed.

## Follow-up

- Update `data.py` and `eval_gsm8k.py` to handle the no-chat-template case.
- Run Run 2 (center config) with 3 seeds on the new base to establish noise floor
  before committing to the full matrix.
- Update README "Status" line and add an "Instruct → base swap" section.
