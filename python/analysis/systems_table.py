"""Create the systems/latency table required for CADFS-SH claims."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "problem_id", "method", "found", "ratio", "expansions", "runtime_ms",
    "model_eval_count", "model_member_evals", "model_cache_hit_rate",
    "model_eval_time_ms",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/logs/bench_next.csv")
    parser.add_argument("--tag", default="next")
    parser.add_argument("--baseline", default="astar")
    parser.add_argument("--out")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} predates systems telemetry; missing columns: {missing}")
    duplicated = frame.duplicated(["problem_id", "method"])
    if duplicated.any():
        raise ValueError(
            "systems table requires one row per paired problem/method")

    baseline = frame[frame.method == args.baseline][
        ["problem_id", "runtime_ms", "expansions"]
    ].rename(columns={
        "runtime_ms": "baseline_runtime_ms",
        "expansions": "baseline_expansions",
    })
    if baseline.empty:
        raise ValueError(f"baseline method {args.baseline!r} is absent")
    paired = frame.merge(
        baseline, on="problem_id", how="left", validate="many_to_one")
    if paired["baseline_runtime_ms"].isna().any():
        raise ValueError("some methods do not share the baseline instance set")

    paired["speedup_vs_baseline"] = (
        paired["baseline_runtime_ms"] /
        paired["runtime_ms"].clip(lower=np.finfo(float).tiny))
    paired["expansion_reduction"] = (
        1.0 - paired["expansions"] /
        paired["baseline_expansions"].clip(lower=1))
    paired["model_time_share"] = np.where(
        paired["runtime_ms"] > 0,
        paired["model_eval_time_ms"] / paired["runtime_ms"],
        0.0)
    paired["members_per_model_eval"] = np.where(
        paired["model_eval_count"] > 0,
        paired["model_member_evals"] / paired["model_eval_count"],
        0.0)

    summary = paired.groupby("method", sort=False).agg(
        instances=("problem_id", "nunique"),
        success_rate=("found", "mean"),
        mean_ratio=("ratio", "mean"),
        max_ratio=("ratio", "max"),
        mean_expansions=("expansions", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
        mean_runtime_ms=("runtime_ms", "mean"),
        median_speedup_vs_baseline=("speedup_vs_baseline", "median"),
        median_expansion_reduction=("expansion_reduction", "median"),
        mean_model_eval_time_ms=("model_eval_time_ms", "mean"),
        mean_model_time_share=("model_time_share", "mean"),
        mean_model_eval_count=("model_eval_count", "mean"),
        mean_members_per_eval=("members_per_model_eval", "mean"),
        mean_cache_hit_rate=("model_cache_hit_rate", "mean"),
    ).reset_index()

    output = (
        Path(args.out) if args.out
        else ROOT / "results/tables" / f"systems_{args.tag}.csv")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"systems table -> {output}")


if __name__ == "__main__":
    main()
