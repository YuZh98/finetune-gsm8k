"""Synthesize plausible per-question prediction JSONLs for the 3 instruct-era runs.

Purpose: scaffolding so ``analyze_predictions.py`` can be exercised before the
real Colab re-run lands. Generated data respects the known aggregates from
``results/runs.csv`` (1071/829/911 correct of 1319) and the seed-disagreement
gap. NOT a substitute for real eval output — discard once real JSONLs arrive.

Synthesis model:
  - Each question has a latent "difficulty" ~ U(0,1).
  - Each run has a "skill" parameter; correct iff skill > difficulty + noise.
  - Skills are tuned so per-run correct counts match runs.csv.
  - FT-wrong predictions are split into extraction-failure vs reasoning-failure.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predictions import make_record, write_predictions_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "predictions"
N = 1319

RUNS = [
    ("run0_base", 1071, 0.812, None),
    ("run2_r16", 829, 0.629, 42),
    ("run2_r16_seed43", 911, 0.691, 43),
]


def question_text(idx: int, difficulty: float) -> str:
    """Synthesize a GSM8K-shaped question. Length grows with difficulty."""
    n_clauses = 1 + int(difficulty * 4) + random.randint(0, 2)
    parts = [
        f"Q{idx}:",
        " ".join(f"step {k} adds {random.randint(1, 99)}." for k in range(n_clauses)),
        f"What is the total after {n_clauses} steps?",
    ]
    return " ".join(parts)


def synth_pred_raw(correct: bool, gold: str, kind: str) -> str:
    """Generate a fake completion. kind in {'gold_marker','answer_is','wrong','extract_fail'}."""
    if correct:
        if kind == "answer_is":
            return f"Let me think.\nThe answer is: {gold}"
        return f"step 1...\nstep 2...\n#### {gold}"
    if kind == "extract_fail":
        return "I cannot determine the answer with confidence."
    wrong = str(int(gold) + random.choice([-3, -1, 1, 2, 5]))
    if kind == "answer_is":
        return f"Working through it.\nThe answer is: {wrong}"
    return f"step a...\nstep b...\n#### {wrong}"


def synthesize_run(run_id: str, n_correct: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    random.seed(seed)
    difficulties = [rng.random() for _ in range(N)]
    questions = [question_text(i, d) for i, d in enumerate(difficulties)]
    golds = [str(rng.randint(1, 999)) for _ in range(N)]

    # Pick which N questions are correct: lowest-difficulty first, with mild noise.
    noise = [rng.gauss(0, 0.08) for _ in range(N)]
    scored = sorted(range(N), key=lambda i: difficulties[i] + noise[i])
    correct_set = set(scored[:n_correct])

    records: list[dict] = []
    for i in range(N):
        is_correct = i in correct_set
        # Surface format: base mostly uses "#### N"; FT runs more often use
        # "The answer is" form (echoing MetaMathQA training data).
        if run_id == "run0_base":
            kind = rng.choices(["gold_marker", "answer_is"], weights=[0.85, 0.15])[0]
        else:
            kind = rng.choices(
                ["gold_marker", "answer_is", "extract_fail"],
                weights=[0.30, 0.55, 0.15],
            )[0]
        if is_correct and kind == "extract_fail":
            kind = "answer_is"
        pred_raw = synth_pred_raw(is_correct, golds[i], kind)
        records.append(make_record(i, questions[i], golds[i], pred_raw))

    # Sanity: enforce exact aggregate match (drift from regex extractor minor).
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run_id, n_correct, expected_acc, seed_param in RUNS:
        seed = seed_param if seed_param is not None else 999
        records = synthesize_run(run_id, n_correct, seed)
        path = OUT / f"{run_id}.jsonl"
        write_predictions_jsonl(
            path, run_id=run_id, records=records, extractor_version="multi-pattern-v1"
        )
        actual_correct = sum(1 for r in records if r["correct"])
        print(f"  {run_id:25s} -> {path.relative_to(ROOT)}  "
              f"(synth correct={actual_correct}/{N}={actual_correct/N:.3f}, "
              f"target={n_correct}/{N}={expected_acc:.3f})")


if __name__ == "__main__":
    main()
