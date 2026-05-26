"""Full instruct-era closeout: 11 plots from the first-round Drive bundle.

Inputs (extracted from results/first-round/finetune-gsm8k-runs/):
  - extracted/runs.csv: all 14 eval rows (8 cells × 1 seed + 1 extra seed + 3 stale extractor)
  - extracted/trainer_state_run2_r16_seed43.json: 63 logged training steps

Outputs (results/plots/first-round/):
  B1–B7: aggregate ablation plots
  C1–C4: training dynamics (n=1, run2_r16_seed43 only)
  D1–D2: cross-cutting smoking-gun charts

Reads RUN_CONFIGS from src.config for hparam metadata so charts can label
each cell by its actual ablation knob (rank / target_modules / lr / data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ABLATION_MATRIX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "first-round" / "extracted"
RUNS_CSV = SRC / "runs.csv"
TRAINER_STATE = SRC / "trainer_state_run2_r16_seed43.json"
OUT = ROOT / "results" / "plots" / "first-round"
OUT.mkdir(parents=True, exist_ok=True)

EFFECT_PP = 3.0
EXTRACTOR_LATEST = "multi-pattern-v1"

CONFIG_BY_RUN = {rc.run_id: rc for rc in ABLATION_MATRIX}


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (latest_extractor_df, stale_extractor_df)."""
    df = pd.read_csv(RUNS_CSV)
    df["acc_pct"] = df["accuracy"] * 100
    latest = df[df["answer_regex"] == EXTRACTOR_LATEST].copy()
    stale = df[df["answer_regex"] != EXTRACTOR_LATEST].copy()
    return latest.reset_index(drop=True), stale.reset_index(drop=True)


def with_hparams(df: pd.DataFrame) -> pd.DataFrame:
    """Attach r/alpha/target_modules/lr/data_size columns from RUN_CONFIGS."""
    rows = []
    for _, row in df.iterrows():
        rid = row["run_id"]
        cfg = CONFIG_BY_RUN.get(rid)
        if cfg is None:
            # Multi-seed runs (e.g. run2_r16_seed43) reuse base run config
            base_rid = rid.rsplit("_seed", 1)[0]
            cfg = CONFIG_BY_RUN.get(base_rid)
        if cfg is None:
            continue
        rows.append({
            **row.to_dict(),
            "r": cfg.rank,
            "alpha": cfg.alpha,
            "target_modules": ",".join(cfg.target_modules) if cfg.target_modules else "",
            "n_target_modules": len(cfg.target_modules),
            "lr": cfg.lr,
            "data_size": cfg.data_size,
            "label": cfg.notes,
        })
    return pd.DataFrame(rows)


# =========================================================================== #
# B1 — Full ablation bar + baseline + threshold band
# =========================================================================== #
def plot_B1_full_ablation(latest: pd.DataFrame) -> None:
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]
    df = latest.sort_values("accuracy", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2b8cbe" if r == "run0_base" else "#e34a33" for r in df["run_id"]]
    bars = ax.bar(df["run_id"], df["accuracy"], color=colors)
    ax.axhline(base_acc, ls="--", color="#2b8cbe", lw=2, label=f"base = {base_acc:.3f}")
    ax.axhspan(base_acc - EFFECT_PP / 100, base_acc + EFFECT_PP / 100,
               alpha=0.15, color="#2b8cbe", label=f"±{EFFECT_PP}pp effect band")
    for bar, v in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("B1 — Full ablation matrix (sorted by accuracy)\nALL 7 FT runs fall outside the effect band below base")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "B1_full_ablation_bar.png")


# =========================================================================== #
# B2 — Effect-size forest plot
# =========================================================================== #
def plot_B2_effect_forest(latest: pd.DataFrame) -> None:
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]
    ft = latest[latest["run_id"] != "run0_base"].copy()
    ft["delta_pp"] = (ft["accuracy"] - base_acc) * 100
    ft = ft.sort_values("delta_pp").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(-EFFECT_PP, EFFECT_PP, color="gray", alpha=0.15,
               label=f"±{EFFECT_PP}pp noise band")
    ys = np.arange(len(ft))
    ax.scatter(ft["delta_pp"], ys, s=160, color="#e34a33", zorder=3)
    for y, (_, r) in zip(ys, ft.iterrows()):
        d = float(r["delta_pp"])
        ax.annotate(f"{d:+.1f}pp", (d, float(y)), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels(ft["run_id"])
    ax.set_xlabel(f"Δ accuracy vs base (pp). Base = {base_acc:.3f}")
    ax.set_xlim(-25, 5)
    ax.set_title("B2 — Effect sizes vs base (forest plot)\nEvery FT run lands far left of zero — uniformly negative")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    save(fig, "B2_effect_forest.png")


# =========================================================================== #
# B3 — Ranked accuracy with config annotations
# =========================================================================== #
def plot_B3_ranked_with_labels(latest: pd.DataFrame) -> None:
    df = with_hparams(latest).sort_values("accuracy", ascending=False).reset_index(drop=True)

    def descr(r) -> str:
        if r["run_id"] == "run0_base":
            return "(no training)"
        bits = [f"r={int(r['r'])}", f"lr={r['lr']:.0e}"]
        if r["n_target_modules"] > 2:
            bits.append("ALL modules")
        if r["data_size"] != 20_000:
            bits.append(f"n={int(r['data_size'])//1000}k")
        if "seed" in r["run_id"]:
            bits.append("seed=43")
        return ", ".join(bits)

    df["descr"] = df.apply(descr, axis=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(df["run_id"], df["accuracy"],
                   color=["#2b8cbe" if r == "run0_base" else "#e34a33" for r in df["run_id"]])
    for bar, v, d in zip(bars, df["accuracy"], df["descr"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"  {v:.3f}   [{d}]", va="center", fontsize=10)
    ax.set_xlim(0.55, 0.92)
    ax.set_xlabel("GSM8K accuracy")
    ax.set_title("B3 — Ranked accuracy with config annotations\nReads top-down: best to worst, each labeled by its ablation knob")
    ax.invert_yaxis()
    save(fig, "B3_ranked_with_labels.png")


# =========================================================================== #
# B4 — Per-hparam sweep panels (4)
# =========================================================================== #
def plot_B4_hparam_sweeps(latest: pd.DataFrame) -> None:
    df = with_hparams(latest)
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Rank: r=8, r=16 (default seed=42), r=64
    ax = axes[0, 0]
    rank_runs = df[df["run_id"].isin(["run1_r8", "run2_r16", "run3_r64"])]
    rank_runs = rank_runs.sort_values("r")
    ax.plot(rank_runs["r"], rank_runs["accuracy"], "o-", color="#e34a33", markersize=10)
    for _, r in rank_runs.iterrows():
        ax.annotate(f" {r['accuracy']:.3f}", (r["r"], r["accuracy"]), fontsize=10)
    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 64])
    ax.set_xticklabels([8, 16, 64])
    ax.set_xlabel("LoRA rank (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Rank sweep (q,v only; lr=2e-4; n=20k)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Target modules: q,v only (run2_r16) vs all-7 (run4_mlp)
    ax = axes[0, 1]
    tm_runs = df[df["run_id"].isin(["run2_r16", "run4_mlp"])]
    labels = ["q,v only\n(run2_r16)", "all 7 modules\n(run4_mlp)"]
    ax.bar(labels, tm_runs["accuracy"], color="#e34a33")
    for i, v in enumerate(tm_runs["accuracy"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_title("Target modules (r=16; lr=2e-4; n=20k)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # LR: low / default / high
    ax = axes[1, 0]
    lr_runs = df[df["run_id"].isin(["run5_lr_low", "run2_r16", "run6_lr_high"])]
    lr_runs = lr_runs.sort_values("lr")
    ax.plot(lr_runs["lr"], lr_runs["accuracy"], "o-", color="#e34a33", markersize=10)
    for _, r in lr_runs.iterrows():
        ax.annotate(f" {r['accuracy']:.3f}", (r["lr"], r["accuracy"]), fontsize=10)
    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LR sweep (r=16; q,v; n=20k)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Data size
    ax = axes[1, 1]
    data_runs = df[df["run_id"].isin(["run7_data5k", "run2_r16"])]
    data_runs = data_runs.sort_values("data_size")
    ax.plot(data_runs["data_size"], data_runs["accuracy"], "o-",
            color="#e34a33", markersize=10)
    for _, r in data_runs.iterrows():
        ax.annotate(f" {r['accuracy']:.3f}", (r["data_size"], r["accuracy"]), fontsize=10)
    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_xlabel("Training data size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Data scale (r=16; lr=2e-4; q,v)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("B4 — Per-hyperparameter sweeps\nNone of the knobs recover the base", fontsize=13, y=1.00)
    fig.tight_layout()
    save(fig, "B4_hparam_sweeps.png")


# =========================================================================== #
# B5 — Extractor evolution (single-pattern vs multi-pattern)
# =========================================================================== #
def plot_B5_extractor_evolution(latest: pd.DataFrame, stale: pd.DataFrame) -> None:
    """Compare same runs under old extractor (#### only) vs new (multi-pattern)."""
    stale_clean = stale.drop_duplicates(subset=["run_id", "answer_regex"], keep="first")

    paired = []
    for run_id in stale_clean["run_id"].unique():
        if run_id not in latest["run_id"].values:
            continue
        old_acc = stale_clean.loc[stale_clean["run_id"] == run_id, "accuracy"].iloc[0]
        new_acc = latest.loc[latest["run_id"] == run_id, "accuracy"].iloc[0]
        paired.append({"run_id": run_id, "single_pattern": old_acc, "multi_pattern": new_acc})
    pdf = pd.DataFrame(paired)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pdf))
    w = 0.35
    ax.bar(x - w/2, pdf["single_pattern"], w, color="#fdbb84",
           label="single-pattern (#### only)")
    ax.bar(x + w/2, pdf["multi_pattern"], w, color="#2b8cbe",
           label="multi-pattern-v1 (####, boxed, 'answer is', fallback)")
    for i, row in pdf.iterrows():
        diff = (row["multi_pattern"] - row["single_pattern"]) * 100
        ax.text(int(i), max(row["single_pattern"], row["multi_pattern"]) + 0.02,
                f"+{diff:.1f}pp", ha="center", fontsize=10, color="green",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pdf["run_id"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("B5 — Extractor evolution: same model, different regex\n"
                 "Extractor fix shifts accuracy by 30+ pp — bigger than any hparam knob")
    ax.legend()
    save(fig, "B5_extractor_evolution.png")


# =========================================================================== #
# B6 — Trainable params vs accuracy scatter
# =========================================================================== #
def trainable_params(r: int, target_modules: list[str], hidden: int = 2048,
                     intermediate: int = 11008, n_layers: int = 36) -> int:
    """Rough LoRA trainable param count for Qwen2.5-3B-Instruct."""
    per_module = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, hidden // 4),  # GQA
        "v_proj": (hidden, hidden // 4),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    total = 0
    for m in target_modules:
        d_in, d_out = per_module.get(m, (hidden, hidden))
        total += r * (d_in + d_out)
    return total * n_layers


def plot_B6_params_vs_accuracy(latest: pd.DataFrame) -> None:
    df = with_hparams(latest)
    df = df[df["run_id"] != "run0_base"].copy()
    df["params"] = df.apply(
        lambda r: trainable_params(int(r["r"]), r["target_modules"].split(",") if r["target_modules"] else []),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["params"], df["accuracy"], s=150, color="#e34a33", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"  {r['run_id']}",
                    (r["params"], r["accuracy"]), fontsize=9, va="center")
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]
    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("# trainable LoRA params (log scale, rough)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("B6 — Trainable params vs accuracy\n"
                 "More capacity does NOT help — r=64 has 4× r=16's params but lower accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "B6_params_vs_accuracy.png")


# =========================================================================== #
# B7 — Seed variance inside full ablation distribution
# =========================================================================== #
def plot_B7_seed_in_distribution(latest: pd.DataFrame) -> None:
    df = latest.copy()
    base_acc = df.loc[df["run_id"] == "run0_base", "accuracy"].iloc[0]
    ft = df[df["run_id"] != "run0_base"].sort_values("accuracy").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, row) in enumerate(ft.iterrows()):
        is_seed_pair = row["run_id"].startswith("run2_r16")
        color = "#e34a33" if is_seed_pair else "#bdbdbd"
        ax.scatter(i, row["accuracy"], s=150, color=color, zorder=3,
                   edgecolors="black" if is_seed_pair else "none", linewidths=1.5)
        ax.annotate(f"{row['accuracy']:.3f}", (i, row["accuracy"]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    seed_pair = ft[ft["run_id"].str.startswith("run2_r16")]
    if len(seed_pair) == 2:
        idxs = [ft.index[ft["run_id"] == rid][0] for rid in seed_pair["run_id"]]
        ax.plot(idxs, seed_pair["accuracy"], color="#e34a33", lw=2, alpha=0.5)
        gap = (seed_pair["accuracy"].max() - seed_pair["accuracy"].min()) * 100
        mid = sum(idxs) / 2
        ax.text(mid, seed_pair["accuracy"].mean(),
                f" seed gap = {gap:.1f}pp", color="#e34a33",
                fontsize=11, fontweight="bold", va="center")

    ax.axhline(base_acc, ls="--", color="#2b8cbe", label=f"base = {base_acc:.3f}")
    ax.set_xticks(range(len(ft)))
    ax.set_xticklabels(ft["run_id"], rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("B7 — Single-seed pair in context of full ablation\n"
                 "Seed gap (red, 6pp) is comparable to between-cell range — ablation noise-dominated")
    ax.legend()
    save(fig, "B7_seed_in_distribution.png")


# =========================================================================== #
# C1 — Training loss curve
# =========================================================================== #
def load_trainer_state() -> list[dict]:
    if not TRAINER_STATE.exists():
        return []
    return json.load(open(TRAINER_STATE))["log_history"]


def plot_C1_loss_curve(log: list[dict]) -> None:
    steps = [r["step"] for r in log if "loss" in r]
    losses = [r["loss"] for r in log if "loss" in r]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, color="#e34a33", lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("C1 — Training loss curve (run2_r16_seed43, n=1 only)\n"
                 "Smooth descent — training is healthy, the problem is eval, not train")
    ax.grid(alpha=0.3)
    save(fig, "C1_training_loss.png")


# =========================================================================== #
# C2 — Loss + grad_norm + lr 3-panel
# =========================================================================== #
def plot_C2_training_diagnostics(log: list[dict]) -> None:
    rows = [r for r in log if "loss" in r and "grad_norm" in r]
    steps = [r["step"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, [r["loss"] for r in rows], color="#e34a33")
    axes[0].set_ylabel("loss")
    axes[0].set_title("C2 — Training diagnostics: loss / grad_norm / lr (run2_r16_seed43)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [r["grad_norm"] for r in rows], color="#fdbb84")
    axes[1].set_ylabel("grad_norm")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, [r["learning_rate"] for r in rows], color="#2b8cbe")
    axes[2].set_ylabel("learning_rate")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "C2_training_diagnostics.png")


# =========================================================================== #
# C3 — mean_token_accuracy curve
# =========================================================================== #
def plot_C3_token_accuracy(log: list[dict]) -> None:
    rows = [r for r in log if "mean_token_accuracy" in r]
    steps = [r["step"] for r in rows]
    accs = [r["mean_token_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, accs, color="#e34a33", lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("mean_token_accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.axhline(accs[-1], ls=":", color="#e34a33", alpha=0.5,
               label=f"final = {accs[-1]:.3f}")
    ax.set_title("C3 — Mean token accuracy over training (run2_r16_seed43)\n"
                 "Climbs to 0.93 — model is mastering the TRAINING distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "C3_token_accuracy.png")


# =========================================================================== #
# C4 — Entropy decay
# =========================================================================== #
def plot_C4_entropy(log: list[dict]) -> None:
    rows = [r for r in log if "entropy" in r]
    steps = [r["step"] for r in rows]
    ents = [r["entropy"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, ents, color="#e34a33", lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("C4 — Output entropy over training (run2_r16_seed43)\n"
                 f"{ents[0]:.3f} → {ents[-1]:.3f}: model getting confident "
                 "(diagnostic for over-confidence collapse)")
    ax.grid(alpha=0.3)
    save(fig, "C4_entropy.png")


# =========================================================================== #
# D1 — Train-vs-eval gap (smoking gun)
# =========================================================================== #
def plot_D1_train_vs_eval_gap(log: list[dict], latest: pd.DataFrame) -> None:
    accs = [r["mean_token_accuracy"] for r in log if "mean_token_accuracy" in r]
    train_token_acc = accs[-1]
    eval_acc = latest.loc[latest["run_id"] == "run2_r16_seed43", "accuracy"].iloc[0]
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        ["TRAIN\nmean token accuracy\n(MetaMathQA format)",
         "EVAL\nGSM8K answer accuracy\n(after fine-tuning)",
         "EVAL\nbase model\n(no fine-tuning)"],
        [train_token_acc, eval_acc, base_acc],
        color=["#2ca25f", "#e34a33", "#2b8cbe"],
    )
    for bar, v in zip(bars, [train_token_acc, eval_acc, base_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    gap = (train_token_acc - eval_acc) * 100
    ax.set_title(f"D1 — Train vs eval gap (run2_r16_seed43)\n"
                 f"{gap:.0f}pp gap = model mastered the MetaMathQA surface format, "
                 "not GSM8K reasoning")
    ax.annotate("", xy=(0.5, train_token_acc), xytext=(0.5, eval_acc),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(0.6, (train_token_acc + eval_acc) / 2, f"  {gap:.0f}pp gap",
            fontsize=12, fontweight="bold", va="center")
    save(fig, "D1_train_vs_eval_gap.png")


# =========================================================================== #
# D2 — Annotated ablation heatmap
# =========================================================================== #
def plot_D2_design_space_heatmap(latest: pd.DataFrame) -> None:
    df = with_hparams(latest)
    df = df[df["run_id"] != "run0_base"].copy()
    base_acc = latest.loc[latest["run_id"] == "run0_base", "accuracy"].iloc[0]
    df["delta_pp"] = (df["accuracy"] - base_acc) * 100

    rows = []
    for _, r in df.iterrows():
        if r["run_id"] == "run4_mlp":
            rows.append(("target_modules", "all_7", r["delta_pp"]))
        elif r["run_id"] in ["run1_r8", "run2_r16", "run3_r64"]:
            rows.append(("rank", f"r={int(r['r'])}", r["delta_pp"]))
        elif r["run_id"] == "run5_lr_low":
            rows.append(("lr", "5e-5", r["delta_pp"]))
        elif r["run_id"] == "run6_lr_high":
            rows.append(("lr", "1e-3", r["delta_pp"]))
        elif r["run_id"] == "run7_data5k":
            rows.append(("data_size", "5k", r["delta_pp"]))
        elif r["run_id"] == "run2_r16_seed43":
            rows.append(("seed", "43", r["delta_pp"]))

    # Add reference cells (default knob values, same baseline run = run2_r16)
    default_delta = df.loc[df["run_id"] == "run2_r16", "delta_pp"].iloc[0]
    rows.extend([
        ("lr", "2e-4 (default)", default_delta),
        ("data_size", "20k (default)", default_delta),
        ("target_modules", "q,v (default)", default_delta),
        ("seed", "42 (default)", default_delta),
    ])
    grid = pd.DataFrame(rows, columns=["knob", "value", "delta_pp"])
    pivot = grid.pivot_table(index="knob", columns="value", values="delta_pp",
                             aggfunc="first")

    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=10, color="black")
    plt.colorbar(im, ax=ax, label="Δ accuracy vs base (pp)")
    ax.set_title("D2 — Design-space heatmap (Δpp vs base; NaN = not tried)\n"
                 "Every cell is negative; less-negative knobs: r=8, lr-shift, data=5k")
    save(fig, "D2_design_space_heatmap.png")


# =========================================================================== #
def main() -> None:
    if not RUNS_CSV.exists():
        sys.exit(f"missing {RUNS_CSV}")
    latest, stale = load_runs()
    print(f"Loaded {len(latest)} latest-extractor rows, {len(stale)} stale rows")
    print(f"Writing to {OUT.relative_to(ROOT)}/")

    plot_B1_full_ablation(latest)
    plot_B2_effect_forest(latest)
    plot_B3_ranked_with_labels(latest)
    plot_B4_hparam_sweeps(latest)
    plot_B5_extractor_evolution(latest, stale)
    plot_B6_params_vs_accuracy(latest)
    plot_B7_seed_in_distribution(latest)

    log = load_trainer_state()
    if log:
        plot_C1_loss_curve(log)
        plot_C2_training_diagnostics(log)
        plot_C3_token_accuracy(log)
        plot_C4_entropy(log)
        plot_D1_train_vs_eval_gap(log, latest)
    else:
        print("(no trainer_state.json — skipping C1–C4, D1)")

    plot_D2_design_space_heatmap(latest)
    print("done.")


if __name__ == "__main__":
    main()
