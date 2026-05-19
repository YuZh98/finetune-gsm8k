"""QLoRA SFT training entrypoint.

One run = one row of the ablation matrix in src/config.py.

Usage (CLI):
    python src/train.py \
        --rank 16 --alpha 32 --target attn --lr 2e-4 --data 20000 \
        --output ./runs/run2_r16

Saves only the LoRA adapter (~5 MB) under <output>/adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_TARGET_MODULES_ALL,
    DEFAULT_TARGET_MODULES_ATTN,
    DEFAULT_WARMUP_RATIO,
    SEED,
)
from .data import load_metamath
from .utils import load_base_model, load_tokenizer, pick_compute_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT on MetaMathQA, eval on GSM8K")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--target",
        type=str,
        default="attn",
        choices=["attn", "all"],
        help="Target modules: attn-only (q,v) or all linear (attn+mlp)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data", type=int, default=20_000, help="MetaMathQA subsample size")
    parser.add_argument("--output", type=str, required=True, help="Output dir for adapter")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device train batch size"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help="Gradient accumulation steps",
    )
    return parser.parse_args()


def resolve_target_modules(target: str) -> list[str]:
    return DEFAULT_TARGET_MODULES_ATTN if target == "attn" else DEFAULT_TARGET_MODULES_ALL


def train_one_run(
    rank: int,
    alpha: int,
    target: str,
    lr: float,
    data_size: int,
    output_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
) -> None:
    """Run a single QLoRA SFT job and save the adapter to <output_dir>/adapter."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"

    compute_dtype = pick_compute_dtype()
    tokenizer = load_tokenizer(padding_side="right")
    model = load_base_model(compute_dtype)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=resolve_target_modules(target),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_metamath(data_size)

    sft_config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        bf16=(compute_dtype.__str__() == "torch.bfloat16"),
        fp16=(compute_dtype.__str__() == "torch.float16"),
        optim="paged_adamw_8bit",
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        max_length=DEFAULT_MAX_SEQ_LENGTH,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to: {adapter_dir}")


def main() -> None:
    args = parse_args()
    train_one_run(
        rank=args.rank,
        alpha=args.alpha,
        target=args.target,
        lr=args.lr,
        data_size=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()
