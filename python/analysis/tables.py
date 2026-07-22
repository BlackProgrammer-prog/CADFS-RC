"""Build paper tables from results/logs/bench.csv -> results/tables/*.csv|md."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from analysis.stats import paired_wilcoxon, bootstrap_ci_mean_diff, spearman  # noqa

TAB = ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)

ORDER = ["astar", "wastar", "focal_plain", "learn_focal_W", "learn_focal_wstar",
         "cadfs", "cadfs_linear", "cadfs_norisk", "cadfs_randomrisk",
         "cadfs_permutedrisk", "cadfs_nofallback", "cadfs_noconf", "cadfs_riskctrl"]


def main() -> None:
    df = pd.read_csv(ROOT / "results/logs/bench.csv")

    # ---- Table 2/3: main comparison per split ----
    g = (df.groupby(["split", "method"])
           .agg(expansions=("expansions", "mean"),
                expansions_med=("expansions", "median"),
                runtime_ms=("runtime_ms", "mean"),
                ratio=("ratio", "mean"), ratio_max=("ratio", "max"),
                success=("found", "mean"),
                fallback=("fallback_rate", "mean"),
                mean_w=("mean_w", "mean"), osc=("mean_abs_dw", "mean"))
           .reset_index())
    g["method"] = pd.Categorical(g["method"], ORDER, ordered=True)
    g = g.sort_values(["split", "method"])
    g.round(3).to_csv(TAB / "main_results.csv", index=False)

    # ---- paired tests: CADFS vs the two key baselines, per split ----
    rows = []
    for split, d in df.groupby("split"):
        piv = d.pivot_table(index="instance", columns="method",
                            values="expansions")
        for base in ("learn_focal_W", "learn_focal_wstar"):
            a = piv["cadfs"].to_numpy(float)
            b = piv[base].to_numpy(float)
            _, p = paired_wilcoxon(a, b)
            lo, hi = bootstrap_ci_mean_diff(a, b)
            rows.append(dict(split=split, comparison=f"cadfs - {base}",
                             mean_diff=round(float((a - b).mean()), 1),
                             ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                             wilcoxon_p=f"{p:.2e}",
                             pct_change=round(100 * (a.mean() / b.mean() - 1), 1)))
    pd.DataFrame(rows).to_csv(TAB / "paired_tests.csv", index=False)

    # ---- Table 5: risk-difficulty Spearman correlation (CADFS rows) ----
    rows = []
    dc = df[df.method == "cadfs"]
    for split, d in dc.groupby("split"):
        for metric in ("expansions", "runtime_ms"):
            rho, p, (lo, hi) = spearman(d["mean_R"].to_numpy(),
                                        d[metric].to_numpy())
            rows.append(dict(split=split, x="mean_R_search", y=metric,
                             spearman_rho=round(rho, 3),
                             ci=f"[{lo:.2f}, {hi:.2f}]", p=f"{p:.2e}",
                             n=len(d)))
    rho, p, (lo, hi) = spearman(dc["mean_R"].to_numpy(),
                                dc["expansions"].to_numpy())
    rows.append(dict(split="ALL", x="mean_R_search", y="expansions",
                     spearman_rho=round(rho, 3),
                     ci=f"[{lo:.2f}, {hi:.2f}]", p=f"{p:.2e}", n=len(dc)))
    pd.DataFrame(rows).to_csv(TAB / "risk_correlation.csv", index=False)

    # ---- Table 4: ablation summary (all splits pooled + shift-only) ----
    abl = [m for m in ORDER if m.startswith("cadfs") or m == "learn_focal_wstar"]
    for name, dd in (("ablation_all", df),
                     ("ablation_shift",
                      df[df.split.isin(["shift_density", "shift_size",
                                        "shift_family"])])):
        t = (dd[dd.method.isin(abl)]
             .groupby("method")
             .agg(expansions=("expansions", "mean"), ratio=("ratio", "mean"),
                  success=("found", "mean"), fallback=("fallback_rate", "mean"),
                  mean_w=("mean_w", "mean"), osc=("mean_abs_dw", "mean"))
             .reindex(abl).round(3))
        t.to_csv(TAB / f"{name}.csv")

    print(*(str(p) for p in sorted(TAB.glob("*.csv"))), sep="\n")


if __name__ == "__main__":
    main()
