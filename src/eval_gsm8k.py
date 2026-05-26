"""GSM8K evaluation harness.

Loads the base model (in 4-bit) optionally with a LoRA adapter on top,
generates greedy answers on the GSM8K test set, regex-extracts the final
numeric answer, scores exact-match, and appends a row to results/runs.csv.

When ``--save_predictions`` is passed, also writes a per-question JSONL
to ``results/predictions/{run_id}.jsonl`` (or the path given by
``--predictions_path``). Each line is one record; see ``make_record``.

Usage (CLI):
    # Base model
    python src/eval_gsm8k.py --output results/runs.csv

    # Adapter on top of base + per-question dump
    python src/eval_gsm8k.py \
        --adapter ./runs/run2_r16/adapter \
        --run_id run2_r16 \
        --output results/runs.csv \
        --save_predictions
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import torch
from datasets import load_dataset
from tqdm import tqdm

from .config import (
    ANSWER_EXTRACTOR_VERSION,
    EVAL_BATCH_SIZE,
    EVAL_CONFIG,
    EVAL_DATASET,
    EVAL_MAX_NEW_TOKENS,
    PROMPT_PREFIX,
)
from .predictions import make_record, write_predictions_jsonl
from .utils import extract_answer, load_model_for_eval, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K test set")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter. Omit to evaluate the base model.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="base",
        help="Identifier for this run in results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/runs.csv",
        help="CSV to append the result row to",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of test problems (for debugging)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=EVAL_BATCH_SIZE,
        help="Generation batch size",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Write per-question JSONL of predictions for analysis",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Override JSONL path (default: results/predictions/{run_id}.jsonl)",
    )
    return parser.parse_args()


def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    """Apply Qwen's chat template with an instruction + question per problem."""
    prompts: list[str] = []
    for q in questions:
        messages = [{"role": "user", "content": PROMPT_PREFIX + q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    completions: list[str] = []
    for i in range(outputs.shape[0]):
        new_tokens = outputs[i, inputs["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return completions


def evaluate(
    adapter_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
) -> tuple[dict, list[dict]]:
    """Run eval. Returns (aggregate metrics, per-question records)."""
    tokenizer = load_tokenizer(padding_side="left")  # left for generation
    model = load_model_for_eval(adapter_path)

    test = load_dataset(EVAL_DATASET, EVAL_CONFIG, split="test")
    if limit is not None:
        test = test.select(range(limit))

    questions = [ex["question"] for ex in test]
    gold_raw = [ex["answer"] for ex in test]
    gold = [extract_answer(g) for g in gold_raw]

    prompts = build_prompts(tokenizer, questions)

    records: list[dict] = []
    total = len(prompts)
    start = time.time()

    for i in tqdm(range(0, total, batch_size), desc="GSM8K eval"):
        batch_prompts = prompts[i : i + batch_size]
        completions = generate_batch(model, tokenizer, batch_prompts, EVAL_MAX_NEW_TOKENS)
        for j, pred_text in enumerate(completions):
            idx = i + j
            records.append(make_record(idx, questions[idx], gold[idx], pred_text))

    elapsed = time.time() - start
    correct = sum(1 for r in records if r["correct"])
    accuracy = correct / total if total else 0.0
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = {
        "n_problems": total,
        "n_correct": correct,
        "accuracy": accuracy,
        "elapsed_sec": round(elapsed, 1),
    }
    return metrics, records


def append_row(csv_path: str, row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    metrics, records = evaluate(args.adapter, args.limit, args.batch_size)
    row = {
        "run_id": args.run_id,
        "adapter": args.adapter or "",
        "extractor": ANSWER_EXTRACTOR_VERSION,
        **metrics,
    }
    append_row(args.output, row)
    logger.info(
        "[%s] accuracy=%.4f (%d/%d) in %.1fs -> appended to %s",
        args.run_id, metrics["accuracy"], metrics["n_correct"],
        metrics["n_problems"], metrics["elapsed_sec"], args.output,
    )

    if args.save_predictions:
        pred_path = Path(
            args.predictions_path
            or f"results/predictions/{args.run_id}.jsonl"
        )
        write_predictions_jsonl(
            pred_path, args.run_id, records, extractor_version=ANSWER_EXTRACTOR_VERSION
        )
        logger.info("[%s] wrote %d predictions to %s",
                    args.run_id, len(records), pred_path)


if __name__ == "__main__":
    main()
