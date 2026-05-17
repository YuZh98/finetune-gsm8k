"""GSM8K evaluation: greedy decode, regex-extract final number, exact match.

Usage:
    python src/eval_gsm8k.py --adapter ./runs/run2_r16/adapter

Not yet implemented: this is a scaffold. The eval loop body is added in a
follow-on commit. The scoring rule and answer-extraction regex live in
src/config.py (ANSWER_REGEX).
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adapter on GSM8K test set")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter. Omit to evaluate the base model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/runs.csv",
        help="CSV to append the result row to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Eval loop not yet implemented (called with {args!r}). "
        "See FINETUNE_PROJECT_SPEC.md section 4.4 for the scoring rule."
    )


if __name__ == "__main__":
    main()
