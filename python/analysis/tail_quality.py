"""Audit empirical tail quality and the predeclared observed-maximum gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--candidate", default="cadfs_next_metric_tuned")
    parser.add_argument("--baseline", default="wastar")
    parser.add_argument("--bound", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=731)
    return parser.parse_args()


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(np.mean(tail))


def statistic(values: np.ndarray, name: str) -> float:
    if name == "p95":
        return float(np.quantile(values, .95))
    if name == "p99":
        return float(np.quantile(values, .99))
    if name == "max":
        return float(np.max(values))
    if name == "cvar95":
        return cvar(values, .95)
    raise ValueError(name)


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, name: str,
                                count: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(count, dtype=float)
    # Chunking avoids a large count-by-n allocation on the final benchmark.
    for index in range(count):
        selected = rng.integers(0, len(a), size=len(a))
        estimates[index] = statistic(a[selected], name) - statistic(b[selected], name)
    return (float(np.quantile(estimates, .025)),
            float(np.quantile(estimates, .975)))


def safe_ratio_values(group: pd.DataFrame, bound: float) -> np.ndarray:
    values = group["ratio"].to_numpy(float, copy=True)
    found = group["found"].astype(bool).to_numpy()
    # A finite penalty keeps bootstrap/sign tests defined while still making
    # every failure a bound violation and strictly worse than any valid run.
    values[~found] = bound + 1.0
    return values


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    df = pd.read_csv(input_path)
    required = {"split", "method", "ratio", "found", "problem_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    methods = set(df["method"])
    if args.candidate not in methods or args.baseline not in methods:
        raise ValueError(
            f"input must contain {args.candidate!r} and {args.baseline!r}")
    if args.bound <= 1.0 or args.bootstrap < 100:
        raise ValueError("bound must exceed 1 and bootstrap must be >= 100")

    summary_rows = []
    for (split, method), group in df.groupby(["split", "method"], observed=True):
        values = safe_ratio_values(group, args.bound)
        worst_index = int(np.argmax(values))
        worst = group.iloc[worst_index]
        summary_rows.append({
            "split": split,
            "method": method,
            "n": len(group),
            "success_rate": float(group["found"].astype(bool).mean()),
            "bound_violations": int(np.sum(values > args.bound + 1e-9)),
            "ratio_mean": float(np.mean(values)),
            "ratio_median": float(np.median(values)),
            "ratio_p95": statistic(values, "p95"),
            "ratio_p99": statistic(values, "p99"),
            "ratio_cvar95": cvar(values),
            "ratio_max": float(np.max(values)),
            "worst_problem_id": worst["problem_id"],
            "worst_map_id": worst.get("map_id", ""),
            "mean_expansions": float(group["expansions"].mean()),
            "median_runtime_ms": float(group["runtime_ms"].median()),
        })
    summary = pd.DataFrame(summary_rows)

    comparisons = []
    gates = []
    for split, group in df.groupby("split", observed=True):
        pivot = group.pivot_table(
            index="problem_id", columns="method", values=["ratio", "found"],
            aggfunc="first")
        a = pivot[("ratio", args.candidate)].to_numpy(float)
        b = pivot[("ratio", args.baseline)].to_numpy(float)
        a_found = pivot[("found", args.candidate)].astype(bool).to_numpy()
        b_found = pivot[("found", args.baseline)].astype(bool).to_numpy()
        a[~a_found] = args.bound + 1.0
        b[~b_found] = args.bound + 1.0
        if len(a) == 0:
            continue
        differences = a - b
        nonzero = differences[~np.isclose(differences, 0.0)]
        lower = int(np.sum(differences < 0.0))
        equal = int(np.sum(np.isclose(differences, 0.0)))
        higher = int(np.sum(differences > 0.0))
        sign_p = (
            float(st.binomtest(lower, lower + higher, .5).pvalue)
            if lower + higher else 1.0)
        wilcoxon_p = (
            float(st.wilcoxon(a, b, zero_method="wilcox").pvalue)
            if len(nonzero) else 1.0)

        row = {
            "split": split, "candidate": args.candidate,
            "baseline": args.baseline, "n": len(a),
            "candidate_better": lower, "equal": equal,
            "candidate_worse": higher, "sign_test_p": sign_p,
            "paired_wilcoxon_p": wilcoxon_p,
        }
        for offset, name in enumerate(("p95", "p99", "cvar95", "max")):
            av = statistic(a, name)
            bv = statistic(b, name)
            low, high = paired_bootstrap_difference(
                a, b, name, args.bootstrap, args.seed + 100 * offset)
            row[f"candidate_{name}"] = av
            row[f"baseline_{name}"] = bv
            row[f"difference_{name}"] = av - bv
            row[f"difference_{name}_ci95_low"] = low
            row[f"difference_{name}_ci95_high"] = high
        comparisons.append(row)

        success_ok = bool(np.all(a_found) and np.all(b_found))
        bound_violations = int(np.sum(a > args.bound + 1e-9))
        max_win = bool(np.max(a) < np.max(b))
        gates.append({
            "split": split,
            "n": len(a),
            "success_ok": success_ok,
            "candidate_bound_violations": bound_violations,
            "observed_max_strictly_better": max_win,
            "p95_nonworse": bool(statistic(a, "p95") <= statistic(b, "p95")),
            "p99_nonworse": bool(statistic(a, "p99") <= statistic(b, "p99")),
            "cvar95_nonworse": bool(cvar(a) <= cvar(b)),
            "primary_split_pass": bool(success_ok and not bound_violations and max_win),
        })

    TAB.mkdir(parents=True, exist_ok=True)
    summary_path = TAB / f"{args.tag}_tail_summary.csv"
    comparison_path = TAB / f"{args.tag}_tail_comparison.csv"
    gate_path = TAB / f"{args.tag}_claim_gate.json"
    summary.round(8).to_csv(summary_path, index=False)
    pd.DataFrame(comparisons).round(8).to_csv(comparison_path, index=False)
    gate_payload = {
        "claim_type": "empirical observed maximum; not theoretical worst-case",
        "candidate": args.candidate,
        "baseline": args.baseline,
        "declared_bound": args.bound,
        "splits": gates,
        "primary_claim_pass_all_splits": bool(
            gates and all(item["primary_split_pass"] for item in gates)),
        "secondary_tail_pass_all_splits": bool(
            gates and all(
                item["p95_nonworse"] and item["p99_nonworse"]
                and item["cvar95_nonworse"] for item in gates)),
        "bootstrap_note": (
            "Paired nonparametric percentile intervals for tail-statistic "
            "differences; observed maxima remain sample-size dependent."
        ),
    }
    gate_path.write_text(json.dumps(gate_payload, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(comparison_path)
    print(gate_path)
    print("PRIMARY CLAIM:",
          "PASS" if gate_payload["primary_claim_pass_all_splits"] else "FAIL")


if __name__ == "__main__":
    main()
