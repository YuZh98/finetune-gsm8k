"""
Generate teaching plots from results/runs.csv (instruct-era closeout).

Each function emits one PNG to results/plots/ and demonstrates a different
way to read the same 3-row dataset (base, FT seed=42, FT seed=43).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "runs.csv"
OUT = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

EFFECT_PP = 3.0  # spec-defined "real effect" threshold (docs/design.md §8)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["label"] = df["run_id"].str.replace("run0_base", "base", regex=False)
    return df


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def plot_01_naive_bar(df: pd.DataFrame) -> None:
    """LESSON: plain bars, y-axis from 0. Honest but flat — hides the seed gap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["label"], df["accuracy"], color=["#2b8cbe", "#e34a33", "#fdbb84"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("01 — Naive bar chart (y from 0)\nHonest; visually flat")
    for i, v in enumerate(df["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "01_naive_bar.png")


def plot_02_misleading_truncated(df: pd.DataFrame) -> None:
    """LESSON: truncated y-axis exaggerates differences. ANTI-PATTERN."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["label"], df["accuracy"], color=["#2b8cbe", "#e34a33", "#fdbb84"])
    ax.set_ylim(0.60, 0.85)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("02 — Truncated y-axis (ANTI-PATTERN)\nSame data, looks 3x worse")
    for i, v in enumerate(df["accuracy"]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "02_misleading_truncated.png")


def plot_03_bar_with_baseline(df: pd.DataFrame) -> None:
    """LESSON: reference line + threshold band = comparison-aware bar."""
    base = df.loc[df["run_id"] == "run0_base", "accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["label"], df["accuracy"], color=["#2b8cbe", "#e34a33", "#fdbb84"])
    ax.axhline(base, ls="--", color="#2b8cbe", label=f"base = {base:.3f}")
    ax.axhspan(base - EFFECT_PP / 100, base + EFFECT_PP / 100,
               alpha=0.15, color="#2b8cbe", label=f"±{EFFECT_PP}pp (effect threshold)")
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("03 — Bar + baseline + threshold band\nFT runs sit well below the band")
    ax.legend(loc="lower right")
    for i, v in enumerate(df["accuracy"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "03_bar_with_baseline.png")


def plot_04_dot_seed_range(df: pd.DataFrame) -> None:
    """LESSON: dot plot makes the seed variance visible as a vertical line."""
    base = df.loc[df["run_id"] == "run0_base", "accuracy"].iloc[0]
    ft = df[df["run_id"] != "run0_base"]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.axhline(base, ls="--", color="#2b8cbe", lw=2, label=f"base = {base:.3f}")
    ax.axhspan(base - EFFECT_PP / 100, base + EFFECT_PP / 100,
               alpha=0.15, color="#2b8cbe")

    xs = [1] * len(ft)
    ax.scatter(xs, ft["accuracy"], s=140, color="#e34a33", zorder=3)
    ax.plot([1, 1], [ft["accuracy"].min(), ft["accuracy"].max()],
            color="#e34a33", lw=2, alpha=0.5, zorder=2)
    for _, r in ft.iterrows():
        seed_val = int(r["seed"])  # type: ignore[arg-type]
        acc_val = float(r["accuracy"])  # type: ignore[arg-type]
        ax.annotate(f"seed={seed_val}: {acc_val:.3f}",
                    (1.0, acc_val), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9)

    gap = ft["accuracy"].max() - ft["accuracy"].min()
    ax.text(1, (ft["accuracy"].max() + ft["accuracy"].min()) / 2,
            f"  seed gap = {gap*100:.1f}pp", color="#e34a33",
            ha="left", va="center", fontsize=10, style="italic")

    ax.set_xticks([1]); ax.set_xticklabels(["r=16 QLoRA"])
    ax.set_xlim(0.5, 2.0)
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("04 — Dot plot w/ seed variance\nSpread itself is the finding")
    ax.legend(loc="upper right")
    save(fig, "04_dot_seed_range.png")


def plot_05_effect_size_forest(df: pd.DataFrame) -> None:
    """LESSON: forest plot — Δ vs base in pp. Standard for ablations."""
    base = df.loc[df["run_id"] == "run0_base", "accuracy"].iloc[0]
    ft = df[df["run_id"] != "run0_base"].copy()
    ft["delta_pp"] = (ft["accuracy"] - base) * 100

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(-EFFECT_PP, EFFECT_PP, color="gray", alpha=0.15,
               label=f"±{EFFECT_PP}pp noise band")

    ys = np.arange(len(ft))
    ax.scatter(ft["delta_pp"], ys, s=140, color="#e34a33", zorder=3)
    for y, (_, r) in zip(ys, ft.iterrows()):
        d = float(r["delta_pp"])  # type: ignore[arg-type]
        ax.annotate(f"{d:+.1f}pp",
                    (d, float(y)), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"r=16, seed={int(s)}" for s in ft["seed"]])
    ax.set_xlabel(f"Δ accuracy vs base (pp). Base = {base:.3f}")
    ax.set_xlim(-22, 6)
    ax.set_title("05 — Forest plot of effect sizes\nBoth FT runs land far left of zero")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    save(fig, "05_effect_size_forest.png")


def plot_06_noise_vs_effect(df: pd.DataFrame) -> None:
    """LESSON: pedagogical — when seed noise > target effect, experiment is under-powered."""
    ft = df[df["run_id"] != "run0_base"]
    seed_gap = (ft["accuracy"].max() - ft["accuracy"].min()) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(["Target effect\n(spec §8)", "Observed seed gap\n(r=16, 2 seeds)"],
                   [EFFECT_PP, seed_gap],
                   color=["#2b8cbe", "#e34a33"])
    for bar, v in zip(bars, [EFFECT_PP, seed_gap]):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}pp", va="center", fontsize=11)
    ax.set_xlabel("Magnitude (percentage points)")
    ax.set_title("06 — Noise floor vs target effect\nSeed gap exceeds the effect we want to detect")
    ax.set_xlim(0, max(seed_gap, EFFECT_PP) * 1.3)
    save(fig, "06_noise_vs_effect.png")


def plot_07_panel_summary(df: pd.DataFrame) -> None:
    """LESSON: dashboard combines absolute level + effect size + noise context."""
    base = df.loc[df["run_id"] == "run0_base", "accuracy"].iloc[0]
    ft = df[df["run_id"] != "run0_base"].copy()
    ft["delta_pp"] = (ft["accuracy"] - base) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: absolute accuracy
    ax = axes[0]
    ax.bar(df["label"], df["accuracy"], color=["#2b8cbe", "#e34a33", "#fdbb84"])
    ax.axhline(base, ls="--", color="#2b8cbe")
    ax.axhspan(base - EFFECT_PP / 100, base + EFFECT_PP / 100,
               alpha=0.15, color="#2b8cbe")
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("A. Absolute accuracy")
    ax.tick_params(axis="x", rotation=20)

    # Panel B: effect sizes
    ax = axes[1]
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(-EFFECT_PP, EFFECT_PP, color="gray", alpha=0.15)
    ys = np.arange(len(ft))
    ax.scatter(ft["delta_pp"], ys, s=140, color="#e34a33")
    ax.set_yticks(ys); ax.set_yticklabels([f"seed={int(s)}" for s in ft["seed"]])
    ax.set_xlabel("Δ vs base (pp)")
    ax.set_xlim(-22, 6)
    ax.set_title("B. Effect size")
    ax.invert_yaxis()

    # Panel C: noise vs effect
    ax = axes[2]
    seed_gap = (ft["accuracy"].max() - ft["accuracy"].min()) * 100
    ax.barh(["target", "observed gap"], [EFFECT_PP, seed_gap],
            color=["#2b8cbe", "#e34a33"])
    ax.set_xlabel("pp")
    ax.set_title("C. Noise vs target effect")

    fig.suptitle("07 — Instruct-era closeout dashboard", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "07_panel_summary.png")


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} runs from {CSV.relative_to(ROOT)}")
    print(df[["run_id", "seed", "accuracy"]].to_string(index=False))
    print(f"\nWriting plots to {OUT.relative_to(ROOT)}/")
    plot_01_naive_bar(df)
    plot_02_misleading_truncated(df)
    plot_03_bar_with_baseline(df)
    plot_04_dot_seed_range(df)
    plot_05_effect_size_forest(df)
    plot_06_noise_vs_effect(df)
    plot_07_panel_summary(df)
    print("done.")


if __name__ == "__main__":
    main()
