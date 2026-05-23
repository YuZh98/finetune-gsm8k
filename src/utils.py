"""Shared utilities: model loading and adapter handling.

Answer extraction lives in `answer_parsing` so it can be unit-tested
without importing torch. The names are reexported here so existing
callers (eval_gsm8k) keep working unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .answer_parsing import answers_match, extract_answer
from .config import BASE_MODEL

__all__ = [
    "answers_match",
    "bnb_config",
    "extract_answer",
    "load_base_model",
    "load_model_for_eval",
    "load_tokenizer",
    "pick_compute_dtype",
]


def bnb_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    """Standard QLoRA quantization config (nf4 + double quant)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def pick_compute_dtype() -> torch.dtype:
    """bf16 if hardware supports it (A100/H100), else fp16 (T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base_model(compute_dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config(compute_dtype),
        device_map="auto",
    )


def load_tokenizer(padding_side: str = "right") -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_for_eval(adapter_path: Optional[str]) -> AutoModelForCausalLM:
    """Load base in 4-bit, optionally attach a LoRA adapter on top."""
    compute_dtype = pick_compute_dtype()
    model = load_base_model(compute_dtype)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


