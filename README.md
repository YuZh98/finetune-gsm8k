# finetune-gsm8k

QLoRA fine-tuning of `Qwen2.5-3B-Instruct` on GSM8K math reasoning, with a clean ablation across the LoRA hyperparameters that actually matter (rank, target modules, learning rate, data scale).

**Status:** scaffold pushed, training runs pending.

This is a learning project. The goal is a complete, reproducible, honest study, not state-of-the-art performance.

## What this project answers

By running 8 configurations on a single base model with the same eval harness, the README will eventually answer:

- Does LoRA rank matter? Where is the saturation point?
- Does targeting MLP layers help, or is attention enough?
- How sensitive is QLoRA to learning rate?
- Does more SFT data linearly help, or are there diminishing returns?

## Method

- **Base:** `Qwen2.5-3B-Instruct`
- **Quantization:** 4-bit NF4 with double quantization, bf16 compute (A100) or fp16 (T4)
- **Adapter:** LoRA via HuggingFace `peft`
- **Training:** `trl.SFTTrainer`, `paged_adamw_8bit`
- **Data:** MetaMathQA subsampled to 20k (or 5k for the data-scale ablation)
- **Eval:** GSM8K test set (1319 problems), greedy decoding, exact-match on the final numeric answer

## Ablation matrix

| Run | Rank `r` | Alpha | Target modules | LR | Data |
|-----|----------|-------|----------------|------|------|
| 0 | — | — | (base, no training) | — | — |
| 1 | 8 | 16 | attn-only (q, v) | 2e-4 | 20k |
| 2 | 16 | 32 | attn-only (q, v) | 2e-4 | 20k |
| 3 | 64 | 128 | attn-only (q, v) | 2e-4 | 20k |
| 4 | 16 | 32 | attn + mlp (all linear) | 2e-4 | 20k |
| 5 | 16 | 32 | attn-only (q, v) | 5e-5 | 20k |
| 6 | 16 | 32 | attn-only (q, v) | 1e-3 | 20k |
| 7 | 16 | 32 | attn-only (q, v) | 2e-4 | 5k |

Run 2 is the center configuration. Each other run varies exactly one axis.

## Results

_To be filled in after training runs complete._ Headline number, ablation table, loss curves, and ablation bar chart will live here.

## Honest caveat: GSM8K contamination

Qwen2.5 was pretrained on a corpus that almost certainly contains GSM8K. The base score on the test set is therefore not a measure of zero-shot generalization but of memorization plus reasoning. The headline metric for this project is **gain over base**, not absolute test accuracy.

## Reproduce

The code is built for Google Colab. A100 recommended; T4 viable for a subset of runs.

```bash
# In Colab:
!git clone https://github.com/YuZh98/finetune-gsm8k.git
%cd finetune-gsm8k
!pip install -r requirements.txt

# Train one configuration (e.g., the center config, Run 2 -> run_id "run2_r16")
!python src/train.py --rank 16 --alpha 32 --target attn --lr 2e-4 --data 20000 --output ./runs/run2_r16

# Evaluate the adapter on GSM8K test
!python src/eval_gsm8k.py --adapter ./runs/run2_r16/adapter
```

Notebook front-ends live in `notebooks/`.

## Layout

```
src/
  config.py          # All hyperparameters and constants (incl. ABLATION_MATRIX)
  data.py            # MetaMathQA loading + chat-template formatting
  utils.py           # Shared model/tokenizer/adapter loading + answer extraction
  train.py           # Single entrypoint, CLI args per ablation knob
  eval_gsm8k.py      # Exact-match scorer
notebooks/
  01_train.ipynb     # Parameterized Colab training notebook
  02_eval.ipynb      # Eval harness front-end
results/
  runs.csv           # One row per run: hparams + eval metrics
  plots/             # Loss curves, ablation bar chart
docs/
  design.md          # Frozen spec, snapshot at completion
```

## Companion documents

The conceptual tutorial and pitfall checklist this project was built around are not in this repo — they live in a separate notes repository.

## Status note

This is a frozen learning project. Once the eight runs are logged and the README is filled in, the repo will be archived. No future maintenance, issues, or PRs will be triaged.

## License

MIT. See `LICENSE`.
