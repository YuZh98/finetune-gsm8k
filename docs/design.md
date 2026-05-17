# Project Spec: QLoRA Fine-Tuning Qwen2.5-3B-Instruct on GSM8K

**Repo:** [`YuZh98/finetune-gsm8k`](https://github.com/YuZh98/finetune-gsm8k)
**Status:** Spec locked 2026-05-17. Scaffold pushed. Training runs not yet executed.
**Estimated effort:** ~5 days with AI assistance.
**Compute:** Google Colab (T4 minimum, A100 recommended).

Companion documents:
- `FINETUNE_TUTORIAL.md` — conceptual walkthrough for beginners
- `FINETUNE_TIPS_CAVEATS.md` — checklist of known pitfalls

---

## 1. Goal

Run a complete QLoRA supervised fine-tuning study on a public LLM, with a clean ablation across the most important LoRA knobs (rank, target modules, learning rate, data scale), evaluated on a well-known benchmark with a held-out test set.

Outcome the project commits to delivering:

- A public GitHub repo with reproducible training scripts and evaluation harness
- A results table comparing the base model and 7 fine-tuned configurations on GSM8K
- A README that explains what was tried, what worked, and what did not
- Loss curves and an ablation bar chart
- A frozen, archived state — no future maintenance promised

Outcomes the project explicitly does **not** commit to:

- State-of-the-art performance (Qwen2.5-3B-Instruct is already strong on GSM8K)
- Production deployment
- Paper-worthy novelty

---

## 2. Why this project

The "fine-tuning" line on the AI Lab CV currently has no concrete backing. Existing work (VAE from scratch, XGBoost, HMC) is real ML but does not count as fine-tuning, which specifically means *adapting a pretrained checkpoint*. This project closes that gap with verifiable evidence.

Secondary goal: build durable mental model of LoRA mechanics by running an ablation rather than a single script.

---

## 3. Constraints

| Constraint | Value |
|------------|-------|
| Time budget | ≤ 1 week wall-clock |
| Hardware | Colab T4 (free) baseline, A100 (paid) for primary runs |
| Maintenance after completion | None |
| Domain | Generic LLM. No quant flavor required. |
| Audience for the writeup | AI Lab recruiters, peer fine-tuning practitioners |

---

## 4. Design decisions

### 4.1 Base model: `Qwen2.5-3B-Instruct`

**Why:**
- Permissive license (Apache 2.0).
- Strong base math ability — a meaningful target above which to push.
- Fits Colab T4 (16 GB) with 4-bit quantization. Comfortable on A100.
- Modern architecture, standard chat template, well-supported in HF.

**Dismissed alternatives:**
- `Llama-3.2-3B-Instruct` — weaker out-of-box on math, would force the writeup to spend pages on baseline weakness.
- `Qwen2.5-1.5B-Instruct` — too weak a baseline; hard to interpret gains.
- `Qwen2.5-7B-Instruct` — fits with QLoRA but eats the ablation compute budget.
- `Qwen2.5-3B-base` (non-instruct) — backup option if instruct baseline is saturated and gains are too small to claim.

### 4.2 Method: QLoRA SFT

**Stack:**
- Quantization: 4-bit NF4 with double quantization, bf16 compute dtype (A100) or fp16 (T4)
- LoRA via HuggingFace `peft`
- Training loop via `trl.SFTTrainer`
- Optimizer: `paged_adamw_8bit`

**Why QLoRA, not full FT:**
- Fits on consumer/Colab hardware.
- LoRA adapters are ~5 MB, trivial to version and share.
- Ablation across configs becomes cheap.
- This matches what AI Lab teams expect candidates to know in 2026.

### 4.3 Training data: MetaMathQA subsampled to 20k examples

**Why:**
- GSM8K train set is only 7.5k examples — too small for meaningful data-scale ablation.
- MetaMathQA (~395k problems) is the standard SFT corpus for GSM8K-style fine-tuning.
- GSM8K test set stays fully held out.

**Subsample size:**
- 20k chosen as the sweet spot between training time and meaningful gradient signal. One epoch at this size runs in roughly 1-2 hours on A100, 3-4 hours on T4.

### 4.4 Eval: GSM8K test set, exact-match on final numeric answer

**Why GSM8K:**
- Well-defined benchmark, 1319 problems, executable scoring.
- Established baselines exist for every common model size.
- Greedy decoding gives reproducible numbers.

**Scoring:**
- Greedy decode with chain-of-thought prompt.
- Regex extraction of final answer (`#### <number>` format).
- Exact-match against ground truth numeric answer.
- Single sample per problem (greedy, so sampling has no effect).

**Honest reporting:**
- Qwen2.5 pretraining corpus almost certainly contains GSM8K. Base score reported as the comparison baseline, not as proof of generalization. Gains over base are the only honest metric.

---

## 5. Ablation matrix

Eight runs total. Each varies one axis from a center configuration (Run 2).

| Run | Rank `r` | Alpha | Target modules | LR | Data | Notes |
|-----|----------|-------|----------------|------|------|-------|
| 0 | — | — | (base, no training) | — | — | Baseline reference |
| 1 | 8 | 16 | attn-only (q, v) | 2e-4 | 20k | Smallest rank |
| 2 | 16 | 32 | attn-only (q, v) | 2e-4 | 20k | **Center config** |
| 3 | 64 | 128 | attn-only (q, v) | 2e-4 | 20k | Largest rank |
| 4 | 16 | 32 | attn + mlp (all linear) | 2e-4 | 20k | Target-modules ablation |
| 5 | 16 | 32 | attn-only (q, v) | 5e-5 | 20k | LR low |
| 6 | 16 | 32 | attn-only (q, v) | 1e-3 | 20k | LR high (likely diverges) |
| 7 | 16 | 32 | attn-only (q, v) | 2e-4 | 5k | Data scale ablation |

**Total compute budget:**
- A100: ~10-15 GPU-hours (single epoch each at 1-2 hr)
- T4: ~25-30 GPU-hours

`alpha = 2r` convention preserved across rank variations so the effective adapter learning rate stays constant.

**What each comparison answers:**

| Comparison | Question |
|------------|----------|
| Runs 1 vs 2 vs 3 | Does rank matter? Where is the saturation point? |
| Run 2 vs 4 | Does targeting MLP layers help, or is attention enough? |
| Runs 2 vs 5 vs 6 | How sensitive is QLoRA to learning rate? |
| Run 2 vs 7 | Does 4× the data help, or does 5k suffice? |

---

## 6. Repository layout

```
finetune-gsm8k/
├── README.md                  # Headline results, repro instructions, takeaways
├── LICENSE                    # MIT
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_train.ipynb         # Parameterized Colab training notebook
│   └── 02_eval.ipynb          # GSM8K eval harness
├── src/
│   ├── __init__.py
│   ├── config.py              # All hyperparameters and constants in one file
│   ├── train.py               # Single entrypoint, CLI args per ablation knob
│   └── eval_gsm8k.py          # Exact-match scorer + answer extraction
├── results/
│   ├── runs.csv               # One row per run: hparams + eval metrics
│   └── plots/                 # Loss curves, ablation bar chart
└── docs/
    └── design.md              # Snapshot of this spec, frozen at completion
```

Single responsibility per file. No magic numbers outside `src/config.py`.

---

## 7. Data flow

1. Load MetaMathQA from HuggingFace Hub, shuffle with fixed seed, subsample to target size.
2. Format with Qwen's chat template via `tokenizer.apply_chat_template`.
3. Verify by printing one tokenized example end-to-end before training begins.
4. QLoRA-train with TRL `SFTTrainer`, save adapter only (~5 MB).
5. Load adapter on top of frozen 4-bit base for evaluation.
6. Generate on GSM8K test set with greedy decoding and CoT prompt.
7. Regex-extract final numeric answer, compare to ground truth, compute exact-match accuracy.
8. Append a row to `results/runs.csv`.

Loss curves and ablation chart generated from `runs.csv` and per-run training logs.

---

## 8. Success criteria

The project is considered complete when all of the following hold:

- All 8 runs (including Run 0 base) complete without errors and have results logged to `runs.csv`.
- At least one fine-tuned configuration beats the base by ≥ 3 percentage points on GSM8K test accuracy. (Target: +5 pp. Soft floor: +3 pp.)
- The ablation table shows at least two separable, interpretable effects (e.g., "rank 8 and 16 are close, rank 64 overfits"; "MLP targeting helps by X pp over attention-only").
- README is readable in under 5 minutes and contains: headline result, ablation table, two plots, repro command, honest discussion of GSM8K contamination.
- Code reproduces the headline result from a fresh Colab session.

If the instruct baseline is already saturated and no config beats it by ≥ 3 pp, fall back to `Qwen2.5-3B-base` (non-instruct). Document the swap honestly in the README.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Colab T4 too slow for the full matrix | Document A100 as recommended; for T4, drop to 4 highest-signal runs |
| Instruct baseline saturated, gains too small | Swap to `Qwen2.5-3B-base`, larger headroom |
| GSM8K contamination overstates base | Disclose in README. Report gains, not absolute numbers, as the headline |
| Adapter merge issues with 4-bit base | Always evaluate with adapter loaded on top, never merge |
| Eval flakiness | Greedy decoding, fixed seed, single sample per problem |
| Chat template mismatch between train and eval | Print one tokenized example in both pipelines, diff manually |
| OOM on Colab during longer runs | `paged_adamw_8bit`, `gradient_accumulation_steps=4`, `max_seq_length=1024` |
| Run divergence at high LR (Run 6) | Expected. Run still gets logged. Negative result is a result. |

---

## 10. What this project is not

- Not a paper. No novel contribution claimed.
- Not RLHF or DPO. Those are separate projects.
- Not vision. LoRA on ResNet/ViT is a different (and simpler) workflow.
- Not multi-GPU. Single-GPU Colab end-to-end.
- Not actively maintained. Frozen on completion.

---

## 11. Effort breakdown

| Day | Task |
|-----|------|
| 1 | Repo scaffold, environment setup on Colab, smoke-test single short run |
| 2 | Implement training script with CLI args, verify all 8 configs parse |
| 3 | Run all 8 training jobs (overnight on A100 if available) |
| 4 | Run evaluation harness, populate `runs.csv`, generate plots |
| 5 | Write README, polish, sanity-check repro from clean session, archive |

---

## 12. CV statement (target wording, post-completion)

> **Fine-tuning (LoRA/QLoRA)** — Adapted Qwen2.5-3B-Instruct to GSM8K math reasoning via QLoRA SFT. Ran an 8-config ablation across rank, target modules, learning rate, and data scale; documented effects on held-out accuracy. ([repo link](https://github.com/YuZh98/finetune-gsm8k))

Honest, specific, defensible in interview.
