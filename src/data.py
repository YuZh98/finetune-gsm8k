"""Data loading and formatting for MetaMathQA SFT.

MetaMathQA columns of interest:
- "query"    : the math problem
- "response" : a chain-of-thought solution ending with the final answer

The chat template is applied by SFTTrainer at training time via the
`messages` field. This module just produces that conversational form.
"""

from typing import Any

from datasets import Dataset, load_dataset

from .config import SEED, TRAIN_DATASET


def _to_messages(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def load_metamath(n: int) -> Dataset:
    """Load MetaMathQA, shuffle deterministically, subsample to n rows.

    Returns a Dataset with a single "messages" column ready for SFTTrainer.
    """
    raw = load_dataset(TRAIN_DATASET, split="train")
    raw = raw.shuffle(seed=SEED).select(range(n))
    formatted = raw.map(_to_messages, remove_columns=raw.column_names)
    return formatted
