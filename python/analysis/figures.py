"""Paper figures from bench.csv -> results/figures/*.png (matplotlib+seaborn).

NOTE ON TERMINOLOGY (reviewer-safety): we report "fallback ACTIVATION rate",
i.e. P(fallback), never "fallback CATCH rate". Catch rate would require a
ground-truth "unsafe / high-error" label per instance and computing
Precision = P(high error | fallback) and Recall = P(fallback | high error);
we have not computed those and therefore do not claim them. The calibration
figure below reports three purely descriptive, paired statistics (activation
rate, worst-case suboptimality, mean expansions) and lets the reader draw
the (appropriately hedged) conclusion in the paper text.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

# Method -> display label. "cadfs" is CADFS-RC (alpha=1, beta=0: risk enters
# the controller only, never the node ranking -- see Sec. "risk-controller-
# only ranking"). cadfs_riskctrl is a LEGACY key from an earlier ad-hoc
# diagnostic run (before CADFS-RC's default became risk-controller-only) and
# is now numerically redundant with "cadfs". We keep the label for backward
# compatibility with old logs but exclude it from all figures below by
# default -- see `_present()` filtering, which also prints an explicit
# warning for any requested method missing from bench.csv, rather than
# silently drawing an empty bar.
LABEL = {"astar": "A*", "wastar": "WA*", "focal_plain": "Focal (plain)",
         "learn_focal_W": "L-Focal ($W$)", "learn_focal_wstar": "L-Focal ($w^*$)",
         "cadfs": "CADFS-RC", "cadfs_linear": "CADFS-RC (linear ctrl.)",
         "cadfs_norisk": "no risk", "cadfs_randomrisk": "random risk",
         "cadfs_permutedrisk": "permuted risk",
         "cadfs_nofallback": "no fallback", "cadfs_noconf": "no confidence",
         "cadfs_riskctrl": "CADFS-RC (legacy duplicate, excluded by default)"}
SPLITS = ["test", "shift_density", "shift_size", "shift_family"]
SPLIT_LBL = {"test": "in-dist", "shift_density": "density shift",
             "shift_size": "size shift", "shift_family": "family shift"}

# ---------------------------------------------------------------------------
# Old-vs-GeoCal threshold comparison (Fig: calibration).
#
# "Old" = pre-calibration thresholds (theta_r=0.80, theta_dev=0.50), the
#   defaults inherited from the in-distribution-only validation sweep.
# "GeoCal" = thresholds re-tuned on the val_shift calibration proxy using
#   the purely-geometric R_obs/R_mob signal (theta_r=0.35, theta_dev=0.15);
#   see tune_validation.py's theta_r/theta_dev sweep.
#
# These numbers come from a paired, same-instance, same-seed comparison on
# the REAL shift_family TEST split (n=40, seed=3, same ensemble), run
# directly with cadfs_engine.run_cadfs under each threshold setting -- NOT
# from bench.csv (which only stores the currently-tuned/GeoCal run). This
# dict is the recorded result of that comparison; regenerate it by rerunning
# the same paired script if the dataset, ensemble, or seed changes.
CALIBRATION_COMPARISON = {
    "Old":    dict(fallback_rate=0.003, worst_case_ratio=1.44, mean_expansions=761.1),
    "GeoCal": dict(fallback_rate=0.447, worst_case_ratio=1.28, mean_expansions=801.2),
}
CALIB_N = 40          # paired instances
CALIB_SPLIT = "shift_family (test)"
CALIB_SEED = 3


def _present(df: pd.DataFrame, methods: list[str], context: str) -> list[str]:
    """Return only methods that actually exist in df; warn (don't silently
    draw an empty bar) for any that don't."""
    have = set(df.method.unique())
    missing = [m for m in methods if m not in have]
    if missing:
        print(f"[figures.py] WARNING ({context}): methods not found in "
              f"bench.csv, excluded rather than drawn as empty bars: "
              f"{missing}")
    return [m for m in methods if m in have]


def main() -> None:
    df = pd.read_csv(ROOT / "results/logs/bench.csv")
    df["Method"] = df.method.map(LABEL)
    df["Split"] = df.split.map(SPLIT_LBL)

    # ======================================================================
    # Fig A: main comparison, expansions per split (log scale)
    # ======================================================================
    main_m = _present(
        df, ["astar", "wastar", "learn_focal_W", "learn_focal_wstar", "cadfs"],
        "main comparison")
    d = df[df.method.isin(main_m)]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    sns.barplot(d, x="Split", y="expansions", hue="Method",
                order=[SPLIT_LBL[s] for s in SPLITS],
                hue_order=[LABEL[m] for m in main_m],
                errorbar=("ci", 95), ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Mean node expansions (log scale)")
    ax.set_xlabel("")
    ax.legend(ncols=3, fontsize=8, loc="upper left")
    fig.text(0.5, -0.02,
             "Error bars: 95% CI (bootstrap over paired instances). "
             "Success rate and suboptimality ratio per method/split: see Table 2/3.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    fig.savefig(FIG / "fig_main_expansions.png", dpi=200, bbox_inches="tight")

    # ======================================================================
    # Fig B: degradation vs obstacle density (the "killer" curve)
    # ======================================================================
    dens_m = _present(df, ["learn_focal_wstar", "cadfs", "learn_focal_W"],
                      "density degradation curve")
    dd = df[df.split.isin(["test", "shift_density"]) & df.method.isin(dens_m)].copy()
    dd["dens_bin"] = pd.cut(dd.density, [0, .15, .25, .35, .45, .55])
    g = (dd.groupby(["dens_bin", "Method"], observed=True)["expansions"]
           .mean().reset_index())
    g["density"] = [iv.mid for iv in g.dens_bin]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    sns.lineplot(g, x="density", y="expansions", hue="Method", marker="o", ax=ax)
    ax.axvspan(0.05, 0.32, color="green", alpha=0.07)
    ax.text(0.17, ax.get_ylim()[1] * 0.95, "train range", ha="center",
            fontsize=8, color="green")
    ax.set(xlabel="obstacle density", ylabel="mean node expansions",
           title="Degradation with shift severity")
    fig.tight_layout()
    fig.savefig(FIG / "fig_degradation_density.png", dpi=200)

    # ======================================================================
    # Fig C: adaptive width and fallback ACTIVATION rate across splits
    # ======================================================================
    dc = df[df.method == "cadfs"]
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
    sns.boxplot(dc, x="Split", y="mean_w",
                order=[SPLIT_LBL[s] for s in SPLITS], ax=ax[0], color="#4c72b0")
    ax[0].set(ylabel="mean $w_t$", xlabel="",
             title="CADFS-RC: adaptive width")
    sns.boxplot(dc, x="Split", y="fallback_rate",
                order=[SPLIT_LBL[s] for s in SPLITS], ax=ax[1], color="#dd8452")
    ax[1].set(ylabel="fallback activation rate  $P(\\mathrm{fallback})$", xlabel="",
             title="CADFS-RC: fallback activation")
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_width_fallback.png", dpi=200)

    # ======================================================================
    # Fig D: risk vs difficulty scatter (Spearman diagnostic)
    # ======================================================================
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    sns.scatterplot(dc, x="mean_R", y="expansions", hue="Split",
                    hue_order=[SPLIT_LBL[s] for s in SPLITS], s=22,
                    alpha=0.75, ax=ax)
    ax.set_yscale("log")
    ax.set(xlabel="mean structural risk $\\bar R$ during search",
           ylabel="node expansions (log scale)",
           title="Structural risk vs empirical difficulty "
                 "(diagnostic; Table 5 reports Spearman $\\rho$)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_risk_vs_difficulty.png", dpi=200)

    # ======================================================================
    # Fig E (APPENDIX): ablation, family shift ONLY (not pooled), showing
    # expansions alongside the fallback activation rate side by side so the
    # safety-relevant signal isn't hidden by an expansions-only summary.
    # ======================================================================
    abl_requested = ["cadfs", "cadfs_linear", "cadfs_norisk", "cadfs_randomrisk",
                     "cadfs_permutedrisk", "cadfs_nofallback", "cadfs_noconf",
                     "learn_focal_wstar"]
    # cadfs_riskctrl is intentionally NOT included: since CADFS-RC's default is
    # already risk-controller-only (alpha=1, beta=0), that legacy diagnostic
    # method is numerically redundant with "cadfs" in the current pipeline.
    abl = _present(df, abl_requested, "ablation (appendix)")
    da = df[(df.split == "shift_family") & df.method.isin(abl)]

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    sns.barplot(da, x="Method", y="expansions", order=[LABEL[m] for m in abl],
                errorbar=("ci", 95), ax=ax[0], color="#55a868")
    ax[0].set(xlabel="", ylabel="mean node expansions",
             title="Family shift: node expansions")
    sns.barplot(da, x="Method", y="fallback_rate", order=[LABEL[m] for m in abl],
                errorbar=("ci", 95), ax=ax[1], color="#c44e52")
    ax[1].set(xlabel="", ylabel="fallback activation rate",
             title="Family shift: fallback activation")
    for a in ax:
        a.tick_params(axis="x", rotation=30, labelsize=7.5)
    fig.suptitle("Appendix: component ablation under family shift "
                 "(pooled-split version moved out of the main text)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation_appendix.png", dpi=200)

    # ======================================================================
    # Fig F: suboptimality ratio vs the bound W
    # ======================================================================
    sub_m = _present(df, ["cadfs", "learn_focal_W", "wastar"], "suboptimality plot")
    dm = df[df.method.isin(sub_m)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    sns.stripplot(dm, x="Method", y="ratio", hue="Split", dodge=False,
                  size=2.5, alpha=0.5, ax=ax, legend=False)
    ax.axhline(2.0, color="red", ls="--", lw=1, label="bound $W=2$")
    ax.set(ylabel="cost / $C^*$", xlabel="", title="Suboptimality vs bound")
    ax.set_xticklabels([LABEL[m] for m in sub_m])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_suboptimality.png", dpi=200)

    # ======================================================================
    # Fig G (NEW): Old vs GeoCal threshold comparison, 3-panel, shift_family
    # ======================================================================
    order = ["Old", "GeoCal"]
    colors = {"Old": "#8c8c8c", "GeoCal": "#4c72b0"}
    panels = [
        ("fallback_rate", "Fallback activation rate\n$P(\\mathrm{fallback})$", "{:.1%}"),
        ("worst_case_ratio", "Worst-case suboptimality\n$\\max(\\mathrm{cost}/C^*)$", "{:.2f}"),
        ("mean_expansions", "Mean node expansions", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for a, (key, ylabel, fmt) in zip(axes, panels):
        vals = [CALIBRATION_COMPARISON[o][key] for o in order]
        bars = a.bar(order, vals, color=[colors[o] for o in order], width=0.55)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                  ha="center", va="bottom", fontsize=9)
        a.set_ylabel(ylabel, fontsize=8.5)
        a.set_ylim(0, max(vals) * 1.25)
    axes[1].axhline(2.0, color="red", ls="--", lw=1)
    axes[1].text(0.02, 2.02, "$W=2$", color="red", fontsize=7,
                transform=axes[1].get_yaxis_transform())
    fig.suptitle(
        f"Threshold calibration on family shift (test split, n={CALIB_N}, "
        f"seed={CALIB_SEED}, paired same instances)\n"
        "Old: $\\theta_r{=}0.80,\\ \\theta_{dev}{=}0.50$   |   "
        "GeoCal: $\\theta_r{=}0.35,\\ \\theta_{dev}{=}0.15$ "
        "(tuned on the val_shift geometric-shift proxy)",
        fontsize=8.5)
    fig.text(0.5, -0.03,
             "\"Fallback activation rate\" reports $P(\\mathrm{fallback})$, not a "
             "catch rate; we do not claim Precision/Recall against a ground-truth "
             "error label here.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration_comparison.png", dpi=200, bbox_inches="tight")

    print(*(str(p) for p in sorted(FIG.glob("fig_*.png"))), sep="\n")


if __name__ == "__main__":
    main()
