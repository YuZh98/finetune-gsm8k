#!/bin/bash
# Full CLI workflow for Google Colab.
# Paste cells into Colab prefixed with ! (or run as %%bash).
# Recommended runtime: A100. Total time: ~10h train + ~4h eval.

set -e

# ─── Setup ───────────────────────────────────────────────────────────────────
git clone https://github.com/YuZh98/finetune-gsm8k.git
cd finetune-gsm8k
pip install -q -r requirements.txt

# ─── Smoke test (optional, ~2 min) ──────────────────────────────────────────
python -m src.train --rank 16 --alpha 32 --target attn --lr 2e-4 --data 100 --output ./runs/smoke_test
python -m src.eval_gsm8k --adapter ./runs/smoke_test/adapter --run_id smoke --limit 16
rm -rf ./runs/smoke_test

# ─── Train all 7 configs ────────────────────────────────────────────────────
python -m src.train --rank 8  --alpha 16  --target attn --lr 2e-4 --data 20000 --output ./runs/run1_r8
python -m src.train --rank 16 --alpha 32  --target attn --lr 2e-4 --data 20000 --output ./runs/run2_r16
python -m src.train --rank 64 --alpha 128 --target attn --lr 2e-4 --data 20000 --output ./runs/run3_r64
python -m src.train --rank 16 --alpha 32  --target all  --lr 2e-4 --data 20000 --output ./runs/run4_mlp
python -m src.train --rank 16 --alpha 32  --target attn --lr 5e-5 --data 20000 --output ./runs/run5_lr_low
python -m src.train --rank 16 --alpha 32  --target attn --lr 1e-3 --data 20000 --output ./runs/run6_lr_high
python -m src.train --rank 16 --alpha 32  --target attn --lr 2e-4 --data 5000  --output ./runs/run7_data5k

# ─── Evaluate base model ────────────────────────────────────────────────────
python -m src.eval_gsm8k --run_id run0_base --output results/runs.csv

# ─── Evaluate each adapter ──────────────────────────────────────────────────
python -m src.eval_gsm8k --adapter ./runs/run1_r8/adapter      --run_id run1_r8      --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run2_r16/adapter     --run_id run2_r16     --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run3_r64/adapter     --run_id run3_r64     --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run4_mlp/adapter     --run_id run4_mlp     --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run5_lr_low/adapter  --run_id run5_lr_low  --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run6_lr_high/adapter --run_id run6_lr_high --output results/runs.csv
python -m src.eval_gsm8k --adapter ./runs/run7_data5k/adapter  --run_id run7_data5k  --output results/runs.csv

# ─── Results ────────────────────────────────────────────────────────────────
cat results/runs.csv
