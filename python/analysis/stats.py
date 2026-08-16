"""Paired statistics helpers (paper Sec. 11.13)."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def bootstrap_ci(values: np.ndarray, statistic: str = "mean",
                 n_boot: int = 5000, seed: int = 0):
    """Deterministic percentile bootstrap CI for a one-sample statistic."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx]
    estimates = (samples.mean(axis=1) if statistic == "mean"
                 else np.median(samples, axis=1))
    return (float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Wilcoxon signed-rank on paired samples; returns (stat, p)."""
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0
    return st.wilcoxon(a, b, zero_method="wilcox")


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0):
    """95% CI for mean(a - b) via paired bootstrap."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size in [-1, 1]."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    differences = differences[~np.isclose(differences, 0.0)]
    if len(differences) == 0:
        return 0.0
    ranks = st.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def spearman(x: np.ndarray, y: np.ndarray, n_boot=5000, seed=0):
    """Spearman rho with bootstrap 95% CI and p-value."""
    rho, p = st.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([st.spearmanr(x[i], y[i])[0] for i in idx])
    return float(rho), float(p), (float(np.percentile(boots, 2.5)),
                                  float(np.percentile(boots, 97.5)))
