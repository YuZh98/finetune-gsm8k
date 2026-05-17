"""Single source of truth for hyperparameters and constants.

Edit values here, not scattered across train.py / eval_gsm8k.py.
"""

from dataclasses import dataclass


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_DATASET = "meta-math/MetaMathQA"
EVAL_DATASET = "openai/gsm8k"
EVAL_CONFIG = "main"

SEED = 42

# Default ablation: Run 2 (center config) from FINETUNE_PROJECT_SPEC.md
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_TARGET_MODULES_ATTN = ["q_proj", "v_proj"]
DEFAULT_TARGET_MODULES_ALL = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_LR = 2e-4
DEFAULT_DATA_SIZE = 20_000

# Training loop defaults
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MAX_SEQ_LENGTH = 1024

# Eval defaults
EVAL_MAX_NEW_TOKENS = 512
EVAL_BATCH_SIZE = 8
ANSWER_REGEX = r"####\s*(-?\d+(?:\.\d+)?)"


@dataclass
class RunConfig:
    """One row of the ablation matrix."""

    run_id: str
    rank: int
    alpha: int
    target_modules: list
    lr: float
    data_size: int
    notes: str = ""


ABLATION_MATRIX = [
    RunConfig("run0_base", 0, 0, [], 0.0, 0, "Baseline, no training"),
    RunConfig("run1_r8", 8, 16, DEFAULT_TARGET_MODULES_ATTN, 2e-4, 20_000, "Smallest rank"),
    RunConfig("run2_r16", 16, 32, DEFAULT_TARGET_MODULES_ATTN, 2e-4, 20_000, "Center config"),
    RunConfig("run3_r64", 64, 128, DEFAULT_TARGET_MODULES_ATTN, 2e-4, 20_000, "Largest rank"),
    RunConfig("run4_mlp", 16, 32, DEFAULT_TARGET_MODULES_ALL, 2e-4, 20_000, "All linear"),
    RunConfig("run5_lr_low", 16, 32, DEFAULT_TARGET_MODULES_ATTN, 5e-5, 20_000, "LR low"),
    RunConfig("run6_lr_high", 16, 32, DEFAULT_TARGET_MODULES_ATTN, 1e-3, 20_000, "LR high"),
    RunConfig("run7_data5k", 16, 32, DEFAULT_TARGET_MODULES_ATTN, 2e-4, 5_000, "Data scale"),
]
