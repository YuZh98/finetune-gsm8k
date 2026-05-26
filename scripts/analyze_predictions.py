"""Rich per-question analysis + visualization of GSM8K eval predictions.

Consumes ``results/predictions/*.jsonl`` (one file per run, produced by
``src/eval_gsm8k.py --save_predictions``) and emits a suite of plots +
text reports to ``results/plots/`` and stdout.

Each chart targets one *claim* about the runs. See the module-level
docstring of each ``plot_*`` function for the claim and how to read the
result.

Run:
    python scripts/analyze_predictions.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predictions import iter_predictions_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

COLOR_BASE = "#2b8cbe"
COLOR_FT1 = "#e34a33"
COLOR_FT2 = "#fdbb84"
COLORS = {"run0_base": COLOR_BASE, "run2_r16": COLOR_FT1, "run2_r16_seed43": COLOR_FT2}


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def load_all() -> dict[str, pd.DataFrame]:
    """Return {run_id: DataFrame of records}, sorted by idx for alignment."""
    out: dict[str, pd.DataFrame] = {}
    for header, records in iter_predictions_dir(PRED_DIR):
        df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
        out[header["run_id"]] = df
    return out


# --------------------------------------------------------------------------- #
# 1. Per-question correctness matrix (runs x questions, binary)
# --------------------------------------------------------------------------- #
def plot_correctness_heatmap(runs: dict[str, pd.DataFrame]) -> None:
    """CLAIM: errors are not random — there's structure in which questions break.

    Read: each column = one question. Vertical white stripes = problems all
    runs got wrong (truly hard). Mixed columns = seed-sensitive questions.
    A run with many isolated red cells = adds new errors the others don't.
    """
    run_ids = list(runs.keys())
    mat = np.stack([runs[r]["correct"].astype(int).to_numpy() for r in run_ids])
    # Sort questions by row-sum so structure is visible (hard problems left).
    order = np.argsort(mat.sum(axis=0))
    mat = mat[:, order]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(mat, aspect="auto", cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids)
    ax.set_xlabel("Questions (sorted by total correct across runs, hardest left)")
    ax.set_title("A1 — Per-question correctness matrix\nGreen = correct, red = wrong")
    save(fig, "A1_correctness_heatmap.png")


# --------------------------------------------------------------------------- #
# 2. Error-overlap quadrants (base vs each FT)
# --------------------------------------------------------------------------- #
def plot_error_overlap(runs: dict[str, pd.DataFrame]) -> None:
    """CLAIM: FT doesn't uniformly degrade — it trades right answers for new wrong ones.

    Read: 4 bars per FT run. 'B✓ F✗' = base got right but FT broke (the cost
    of fine-tuning). 'B✗ F✓' = FT recovered something base missed (the gain).
    Net delta = gain − cost. The 'B✗ F✗' bar shows shared hard problems.
    """
    base = runs["run0_base"]["correct"].astype(bool).to_numpy()
    ft_ids = [r for r in runs if r != "run0_base"]

    fig, axes = plt.subplots(1, len(ft_ids), figsize=(5 * len(ft_ids), 4), sharey=True)
    if len(ft_ids) == 1:
        axes = [axes]

    for ax, run_id in zip(axes, ft_ids):
        ft = runs[run_id]["correct"].astype(bool).to_numpy()
        bb = int((base & ft).sum())
        bw = int((base & ~ft).sum())
        wb = int((~base & ft).sum())
        ww = int((~base & ~ft).sum())
        labels = ["B✓ F✓\n(both right)", "B✓ F✗\n(FT broke)",
                  "B✗ F✓\n(FT fixed)", "B✗ F✗\n(both wrong)"]
        vals = [bb, bw, wb, ww]
        colors = ["#2ca25f", "#e34a33", "#7bccc4", "#bdbdbd"]
        bars = ax.bar(labels, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 10, str(v),
                    ha="center", fontsize=10)
        ax.set_title(f"{run_id}\nnet Δ = {wb - bw:+d}")
        ax.set_ylabel("# questions")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("A2 — Error overlap: base vs FT", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "A2_error_overlap.png")


# --------------------------------------------------------------------------- #
# 3. Failure mode split: extraction-failure vs reasoning-wrong
# --------------------------------------------------------------------------- #
def plot_failure_mode_split(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """CLAIM: not all 'wrong' is the same — extraction failures vs wrong answers.

    Read: per run, the wrong-answer bar splits into 'extraction failed'
    (model output had no extractable number — regex/format problem) vs
    'reasoning wrong' (extracted a number, but the wrong one — real
    capability gap). High extraction-fail share suggests a parser fix could
    cheaply recover accuracy.
    """
    rows = []
    for run_id, df in runs.items():
        wrong_preds = df.loc[~df["correct"], "pred_extracted"]
        n_extract_fail = int(pd.isna(wrong_preds).to_numpy().sum())
        n_reason_wrong = int(len(wrong_preds) - n_extract_fail)
        n_correct = int(df["correct"].to_numpy().sum())
        rows.append({
            "run_id": run_id,
            "correct": n_correct,
            "extraction_fail": n_extract_fail,
            "reasoning_wrong": n_reason_wrong,
        })
    summary = pd.DataFrame(rows).set_index("run_id")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(summary))
    for col, color in [
        ("correct", "#2ca25f"),
        ("reasoning_wrong", "#e34a33"),
        ("extraction_fail", "#fdbb84"),
    ]:
        ax.bar(summary.index, summary[col], bottom=bottoms, label=col.replace("_", " "),
               color=color)
        bottoms += summary[col].to_numpy()
    ax.set_ylabel("# questions (stacked)")
    ax.set_title("A3 — Failure-mode split per run\nOrange = parser problem, red = real wrong answer")
    ax.legend(loc="lower right")
    for i, run_id in enumerate(summary.index):
        for col, color in [
            ("correct", "white"),
            ("reasoning_wrong", "white"),
            ("extraction_fail", "black"),
        ]:
            v = int(summary.loc[run_id, col])
            if v == 0:
                continue
            y_pos = summary.loc[run_id, :col].sum() - v / 2
            ax.text(i, y_pos, f"{v}", ha="center", va="center", color=color, fontsize=10)
    save(fig, "A3_failure_mode_split.png")
    return summary


# --------------------------------------------------------------------------- #
# 4. Accuracy by question-length bucket
# --------------------------------------------------------------------------- #
def plot_accuracy_by_question_length(runs: dict[str, pd.DataFrame]) -> None:
    """CLAIM: long questions are harder — does FT hurt long ones more than short?

    Read: x = question length bucket. Each line = one run. Diverging lines
    on the right side = FT degrades disproportionately on hard problems.
    Parallel lines = FT is a uniform shift (less interesting).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [0, 100, 150, 200, 250, 300, 400, 1000]
    bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    for run_id, df in runs.items():
        df = df.assign(bucket=pd.cut(df["question_chars"], bins=bins,
                                     labels=bin_labels, right=False))
        grouped = df.groupby("bucket", observed=False)["correct"]
        acc = np.asarray(grouped.mean())
        n = int(np.asarray(grouped.count()).sum())
        ax.plot(bin_labels, acc, marker="o", color=COLORS.get(run_id, "gray"),
                label=f"{run_id}  (n={n})")
    ax.set_xlabel("Question length (chars)")
    ax.set_ylabel("Accuracy")
    ax.set_title("A4 — Accuracy by question-length bucket\nDivergence = FT hurts hard problems more")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "A4_accuracy_by_question_length.png")


# --------------------------------------------------------------------------- #
# 5. Generation-length distribution by correctness
# --------------------------------------------------------------------------- #
def plot_gen_length_vs_correctness(runs: dict[str, pd.DataFrame]) -> None:
    """CLAIM: wrong answers often correlate with abnormal generation length.

    Read: per run, two violin/box shapes — gen_chars for correct vs wrong.
    If wrong gens are MUCH longer → model rambles when uncertain.
    If wrong gens are MUCH shorter → model gives up early.
    Similar shapes → length is not diagnostic for this run.
    """
    run_ids = list(runs.keys())
    fig, axes = plt.subplots(1, len(run_ids), figsize=(5 * len(run_ids), 4),
                             sharey=True)
    if len(run_ids) == 1:
        axes = [axes]
    for ax, run_id in zip(axes, run_ids):
        df = runs[run_id]
        correct = df.loc[df["correct"], "gen_chars"].to_numpy()
        wrong = df.loc[~df["correct"], "gen_chars"].to_numpy()
        ax.boxplot([correct, wrong], tick_labels=["correct", "wrong"],
                   showfliers=False)
        ax.set_title(f"{run_id}\nmed: ✓ {np.median(correct):.0f}  ✗ {np.median(wrong):.0f}")
        ax.set_ylabel("gen_chars")
    fig.suptitle("A5 — Generation length: correct vs wrong (per run)", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "A5_gen_length_vs_correctness.png")


# --------------------------------------------------------------------------- #
# 6. McNemar test base vs each FT (paired significance)
# --------------------------------------------------------------------------- #
def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact mid-p McNemar test (no scipy dep). Returns 2-sided p-value.

    b = base✓ & FT✗;  c = base✗ & FT✓.  Tests whether the two error rates differ.
    Uses the binomial test on min(b,c) ~ Binomial(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), 2-sided
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def report_mcnemar(runs: dict[str, pd.DataFrame]) -> str:
    """CLAIM: paired test on shared questions is the right significance test.

    Read printout: for each FT run vs base, b = base-only-correct,
    c = FT-only-correct. If b ≫ c with small p, FT regression is real.
    McNemar uses only DISCORDANT pairs — concordant (both right / both wrong)
    are noise to this test.
    """
    base = runs["run0_base"]["correct"].astype(bool).to_numpy()
    lines = ["", "=== McNemar paired test: base vs FT ==="]
    lines.append("(b = base-correct AND FT-wrong;  c = base-wrong AND FT-correct)")
    lines.append(f"{'comparison':30s}  {'b':>5s} {'c':>5s} {'net':>6s}  {'p (2-sided)':>12s}")
    for run_id, df in runs.items():
        if run_id == "run0_base":
            continue
        ft = df["correct"].astype(bool).to_numpy()
        b = int((base & ~ft).sum())
        c = int((~base & ft).sum())
        p = mcnemar_pvalue(b, c)
        verdict = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        lines.append(f"base vs {run_id:22s}  {b:5d} {c:5d} {c-b:+6d}  {p:12.2e}  {verdict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 7. Seed disagreement set Venn (FT-only)
# --------------------------------------------------------------------------- #
def plot_seed_disagreement(runs: dict[str, pd.DataFrame]) -> None:
    """CLAIM: two seeds of the same config disagree on a measurable fraction.

    Read: Venn-style bar — questions where seed=42 was right, where seed=43
    was right, where both, where neither (within FT). The 'only one' regions
    are pure seed noise — that's the irreducible variance at 1 seed/cell.
    """
    ft_ids = [r for r in runs if r != "run0_base"]
    if len(ft_ids) != 2:
        print(f"  (skipping A6 seed-disagreement — need exactly 2 FT runs, got {len(ft_ids)})")
        return
    a = runs[ft_ids[0]]["correct"].astype(bool).to_numpy()
    b = runs[ft_ids[1]]["correct"].astype(bool).to_numpy()
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    disagreement_rate = (only_a + only_b) / len(a)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"both ✓", f"only {ft_ids[0]}", f"only {ft_ids[1]}", "neither"]
    vals = [both, only_a, only_b, neither]
    colors = ["#2ca25f", COLOR_FT1, COLOR_FT2, "#bdbdbd"]
    bars = ax.bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, str(v),
                ha="center", fontsize=10)
    ax.set_ylabel("# questions")
    ax.set_title(f"A6 — Seed disagreement on FT runs\n"
                 f"Disagreement rate = {disagreement_rate:.1%} of test set ({only_a + only_b}/{len(a)})")
    ax.tick_params(axis="x", rotation=10)
    save(fig, "A6_seed_disagreement.png")


# --------------------------------------------------------------------------- #
# 8. Answer-format usage per run (lesson-prompt-template-echo evidence)
# --------------------------------------------------------------------------- #
def plot_answer_format_usage(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """CLAIM: FT shifts which surface format the model emits — direct evidence
    for the prompt-template-echo hypothesis (docs/lesson-prompt-template-echo.md).

    Read: stacked bar per run. If FT runs show a much larger 'answer is' or
    'no marker / extract fail' share than base, the SFT data is teaching a
    competing format. Direct visual support for the ADR-0001 narrative.
    """
    def classify(text: str) -> str:
        t = text.lower()
        if "####" in text:
            return "gsm8k_marker (####)"
        if "\\boxed" in text:
            return "boxed"
        if "answer is" in t:
            return "answer_is"
        return "no_marker"

    rows = []
    for run_id, df in runs.items():
        counts = Counter(classify(p) for p in df["pred_raw"])
        rows.append({"run_id": run_id, **counts})
    summary_raw = pd.DataFrame(rows).set_index("run_id").fillna(0).astype(int)
    cols = ["gsm8k_marker (####)", "boxed", "answer_is", "no_marker"]
    summary = pd.DataFrame(summary_raw[[c for c in cols if c in summary_raw.columns]])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(summary))
    palette = {
        "gsm8k_marker (####)": "#2b8cbe",
        "boxed": "#7bccc4",
        "answer_is": "#fdbb84",
        "no_marker": "#e34a33",
    }
    for col in summary.columns:
        col_vals = np.asarray(summary[col])
        ax.bar(summary.index, col_vals, bottom=bottoms, label=col,
               color=palette.get(col, "gray"))
        bottoms = bottoms + col_vals
    ax.set_ylabel("# questions (stacked)")
    ax.set_title("A7 — Answer-format usage per run\nFT shifts away from #### toward 'answer is' / no marker")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "A7_answer_format_usage.png")
    return summary


# --------------------------------------------------------------------------- #
def main() -> None:
    if not PRED_DIR.exists() or not list(PRED_DIR.glob("*.jsonl")):
        print(f"No predictions in {PRED_DIR}.")
        print("Generate demo data:  python scripts/make_demo_predictions.py")
        print("Or run eval with:    python -m src.eval_gsm8k ... --save_predictions")
        return

    runs = load_all()
    print(f"Loaded {len(runs)} runs: {list(runs.keys())}")
    print(f"Writing plots to {OUT.relative_to(ROOT)}/")

    plot_correctness_heatmap(runs)
    plot_error_overlap(runs)
    plot_failure_mode_split(runs)
    plot_accuracy_by_question_length(runs)
    plot_gen_length_vs_correctness(runs)
    plot_seed_disagreement(runs)
    plot_answer_format_usage(runs)

    print(report_mcnemar(runs))
    print("\ndone.")


if __name__ == "__main__":
    main()
