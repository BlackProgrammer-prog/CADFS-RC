"""Figures for CADFS-Next benchmarks; outputs never replace legacy figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
LABELS = {
    "astar": "A*", "wastar": "WA*",
    "learn_focal_W": "L-Focal W",
    "learn_focal_wstar": "L-Focal tuned",
    "cadfs": "Legacy CADFS-RC", "cadfs_next": "CADFS Next",
    "cadfs_next_geometry": "geometry only",
    "cadfs_next_uniform": "uniform experts",
    "cadfs_next_nointra": "without intra-U",
    "cadfs_next_nointer": "without inter-U",
    "cadfs_next_noconf": "without confidence",
    "cadfs_next_linear": "linear controller",
    "cadfs_next_threshold": "threshold controller",
    "cadfs_next_fixed": "fixed controller",
    "cadfs_next_mlp": "MLP controller",
}
SPLITS = ["test", "shift_density", "shift_size", "shift_family"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/logs/bench_next.csv")
    parser.add_argument("--tag", default="next")
    parser.add_argument("--bound", type=float, default=2.0)
    return parser.parse_args()


def save(fig, tag: str, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{tag}_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = ROOT / path
    data = pd.read_csv(path)
    data["label"] = data.method.map(LABELS).fillna(data.method)
    data["split"] = pd.Categorical(data.split, SPLITS, ordered=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    main_methods = [name for name in (
        "astar", "wastar", "learn_focal_wstar", "cadfs", "cadfs_next")
        if name in set(data.method)]
    selected = data[data.method.isin(main_methods)]
    order = [LABELS[name] for name in main_methods]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    sns.barplot(selected, x="split", y="expansions", hue="label",
                hue_order=order, errorbar=("ci", 95), ax=ax)
    ax.set_yscale("log")
    ax.set(xlabel="", ylabel="mean expansions (log scale)",
           title="CADFS Next: search effort under distribution shift")
    save(fig, args.tag, "main_expansions")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    sns.barplot(selected, x="split", y="runtime_ms", hue="label",
                hue_order=order, errorbar=("ci", 95), ax=ax)
    ax.set_yscale("log")
    ax.set(xlabel="", ylabel="mean runtime in ms (log scale)",
           title="CADFS Next: end-to-end runtime")
    save(fig, args.tag, "runtime")

    adaptive = data[data.method.isin(
        [name for name in ("cadfs", "cadfs_next") if name in set(data.method)])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    sns.boxplot(adaptive, x="split", y="mean_w", hue="label", ax=axes[0])
    sns.boxplot(adaptive, x="split", y="fallback_rate", hue="label", ax=axes[1])
    axes[0].set(xlabel="", ylabel="mean focal width", title="Width adaptation")
    axes[1].set(xlabel="", ylabel="fallback activation rate",
                title="Fallback activation")
    save(fig, args.tag, "width_fallback")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    sns.stripplot(selected, x="label", y="ratio", hue="split",
                  size=2.5, alpha=0.55, ax=ax)
    ax.axhline(args.bound, color="red", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=20)
    ax.set(xlabel="", ylabel="cost / C*", title="Empirical bounded suboptimality")
    save(fig, args.tag, "suboptimality")

    ablation_methods = [name for name in LABELS
                        if name.startswith("cadfs_next") and name in set(data.method)]
    ablation = data[data.method.isin(ablation_methods)]
    if not ablation.empty:
        fig, ax = plt.subplots(figsize=(10.0, 4.0))
        sns.barplot(ablation, x="label", y="expansions", hue="split",
                    errorbar=("ci", 95), ax=ax)
        ax.tick_params(axis="x", rotation=28)
        ax.set(xlabel="", ylabel="mean expansions",
               title="CADFS Next component/controller ablations")
        save(fig, args.tag, "ablations")


if __name__ == "__main__":
    main()
