"""QLoRA SFT training entrypoint.

CLI args drive one row of the ablation matrix. See src/config.py for defaults.

Usage:
    python src/train.py --rank 16 --alpha 32 --target attn --lr 2e-4 --data 20000

Not yet implemented: this is a scaffold. The training loop body is added in a
follow-on commit once the spec is locked.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT on MetaMathQA, eval on GSM8K")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--target",
        type=str,
        default="attn",
        choices=["attn", "all"],
        help="Target modules: attn-only or all linear",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data", type=int, default=20_000, help="MetaMathQA subsample size")
    parser.add_argument("--output", type=str, required=True, help="Output dir for adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Training loop not yet implemented (called with {args!r}). "
        "See FINETUNE_PROJECT_SPEC.md section 7 for the data flow."
    )


if __name__ == "__main__":
    main()
