"""Paired statistics helpers (paper Sec. 11.13)."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


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


def spearman(x: np.ndarray, y: np.ndarray, n_boot=5000, seed=0):
    """Spearman rho with bootstrap 95% CI and p-value."""
    rho, p = st.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([st.spearmanr(x[i], y[i])[0] for i in idx])
    return float(rho), float(p), (float(np.percentile(boots, 2.5)),
                                  float(np.percentile(boots, 97.5)))
