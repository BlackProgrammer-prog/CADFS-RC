"""Create reproducible Legacy/CADFS-Next tables from a benchmark CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from analysis.stats import (  # noqa: E402
    bootstrap_ci,
    bootstrap_ci_mean_diff,
    holm_adjust,
    paired_rank_biserial,
    paired_wilcoxon,
    spearman,
)

TAB = ROOT / "results/tables"

ORDER = [
    "dijkstra", "astar", "wastar", "focal_plain", "learn_focal_W",
    "learn_focal_wstar", "cadfs", "cadfs_next",
    "cadfs_next_geometry", "cadfs_next_uniform",
    "cadfs_next_nointra", "cadfs_next_nointer",
    "cadfs_next_noconf", "cadfs_next_linear",
    "cadfs_next_threshold", "cadfs_next_fixed", "cadfs_next_mlp",
    "cadfs_next_metric_r1", "cadfs_next_metric_r3",
    "cadfs_next_metric_r7", "cadfs_next_metric_tuned",
    "cadfs_linear", "cadfs_norisk", "cadfs_randomrisk",
    "cadfs_permutedrisk", "cadfs_nofallback", "cadfs_noconf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/logs/bench_next.csv")
    parser.add_argument("--tag", default="next")
    return parser.parse_args()


def percentile95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def output_path(tag: str, name: str) -> Path:
    prefix = f"{tag}_" if tag else ""
    return TAB / f"{prefix}{name}.csv"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    df = pd.read_csv(input_path)
    TAB.mkdir(parents=True, exist_ok=True)
    present = [method for method in ORDER if method in set(df.method)]

    summary = (df.groupby(["split", "method"], observed=True)
                 .agg(
                     expansions_mean=("expansions", "mean"),
                     expansions_median=("expansions", "median"),
                     expansions_p95=("expansions", percentile95),
                     runtime_mean_ms=("runtime_ms", "mean"),
                     runtime_median_ms=("runtime_ms", "median"),
                     runtime_p95_ms=("runtime_ms", percentile95),
                     ratio_mean=("ratio", "mean"),
                     ratio_max=("ratio", "max"),
                     success_rate=("found", "mean"),
                     fallback_rate=("fallback_rate", "mean"),
                     mean_width=("mean_w", "mean"),
                     width_oscillation=("mean_abs_dw", "mean"),
                 ).reset_index())
    ci_rows = []
    for ci_seed, ((split, method), group) in enumerate(
            df.groupby(["split", "method"], observed=True)):
        exp_mean = bootstrap_ci(
            group["expansions"].to_numpy(float), "mean", seed=ci_seed)
        runtime_mean = bootstrap_ci(
            group["runtime_ms"].to_numpy(float), "mean", seed=1000 + ci_seed)
        runtime_median = bootstrap_ci(
            group["runtime_ms"].to_numpy(float), "median",
            seed=2000 + ci_seed)
        ratio_mean = bootstrap_ci(
            group["ratio"].to_numpy(float), "mean", seed=3000 + ci_seed)
        ci_rows.append({
            "split": split, "method": method,
            "expansions_mean_ci95_low": exp_mean[0],
            "expansions_mean_ci95_high": exp_mean[1],
            "runtime_mean_ci95_low": runtime_mean[0],
            "runtime_mean_ci95_high": runtime_mean[1],
            "runtime_median_ci95_low": runtime_median[0],
            "runtime_median_ci95_high": runtime_median[1],
            "ratio_mean_ci95_low": ratio_mean[0],
            "ratio_mean_ci95_high": ratio_mean[1],
        })
    summary = summary.merge(
        pd.DataFrame(ci_rows), on=["split", "method"], how="left")
    summary["method"] = pd.Categorical(
        summary["method"], categories=present, ordered=True)
    summary = summary.sort_values(["split", "method"])
    summary.round(4).to_csv(
        output_path(args.tag, "main_results"), index=False)

    comparisons = [
        ("cadfs_next", "astar"),
        ("cadfs_next", "wastar"),
        ("cadfs_next", "focal_plain"),
        ("cadfs_next", "cadfs"),
        ("cadfs_next", "learn_focal_wstar"),
        ("cadfs_next", "learn_focal_W"),
    ]
    for method in present:
        if method.startswith("cadfs_next_metric_"):
            comparisons.extend([
                (method, "cadfs_next"),
                (method, "astar"),
                (method, "wastar"),
            ])
    paired_rows = []
    pair_key = "problem_id" if "problem_id" in df.columns else "instance"
    for split, group in df.groupby("split", observed=True):
        for metric in ("expansions", "runtime_ms", "ratio"):
            pivot = group.pivot_table(
                index=pair_key, columns="method", values=metric,
                aggfunc="first")
            family_start = len(paired_rows)
            for candidate, baseline in comparisons:
                if candidate not in pivot or baseline not in pivot:
                    continue
                pairs = pivot[[candidate, baseline]].dropna()
                a = pairs[candidate].to_numpy(float)
                b = pairs[baseline].to_numpy(float)
                differences = a - b
                _, p_value = paired_wilcoxon(a, b)
                low, high = bootstrap_ci_mean_diff(a, b)
                paired_rows.append({
                    "split": split,
                    "metric": metric,
                    "comparison": f"{candidate} - {baseline}",
                    "n": len(pairs),
                    "zero_differences": int(np.isclose(differences, 0.0).sum()),
                    "mean_difference": float(differences.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilcoxon_p": p_value,
                    "rank_biserial": paired_rank_biserial(a, b),
                    "percent_change": 100.0 * (a.mean() / b.mean() - 1.0),
                })
            family = paired_rows[family_start:]
            adjusted = holm_adjust(
                np.array([row["wilcoxon_p"] for row in family]))
            for row, value in zip(family, adjusted):
                row["holm_p"] = float(value)
                row["significant_0_05"] = bool(value < 0.05)
    pd.DataFrame(paired_rows).round(6).to_csv(
        output_path(args.tag, "paired_tests"), index=False)

    correlation_rows = []
    for method in ("cadfs", "cadfs_next"):
        selected = df[df.method == method]
        for split, group in selected.groupby("split", observed=True):
            if len(group) < 3:
                continue
            for metric in ("expansions", "runtime_ms"):
                rho, p_value, (low, high) = spearman(
                    group["mean_R"].to_numpy(), group[metric].to_numpy())
                correlation_rows.append({
                    "method": method, "split": split,
                    "x": "mean_R", "y": metric, "n": len(group),
                    "spearman_rho": rho, "p": p_value,
                    "ci95_low": low, "ci95_high": high,
                })
    pd.DataFrame(correlation_rows).round(6).to_csv(
        output_path(args.tag, "risk_correlation"), index=False)

    ablations = [name for name in present if name.startswith("cadfs_next_")]
    if "cadfs_next" in present:
        ablations.insert(0, "cadfs_next")
    if "cadfs" in present:
        ablations.insert(0, "cadfs")
    ablation = (df[df.method.isin(ablations)]
                .groupby("method", observed=True)
                .agg(expansions=("expansions", "mean"),
                     runtime_ms=("runtime_ms", "mean"),
                     ratio=("ratio", "mean"), ratio_max=("ratio", "max"),
                     success=("found", "mean"),
                     fallback=("fallback_rate", "mean"),
                     mean_w=("mean_w", "mean"))
                .reindex(ablations))
    ablation.round(4).to_csv(output_path(args.tag, "ablation"))

    for path in sorted(TAB.glob(f"{args.tag}_*.csv")):
        print(path)


if __name__ == "__main__":
    main()
