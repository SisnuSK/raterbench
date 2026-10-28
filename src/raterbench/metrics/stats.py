"""Resampling confidence intervals and paired significance tests."""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    """Percentile bootstrap over items.

    `stat` receives an index array (into the item list) and returns the statistic, so any
    metric over paired arrays can be bootstrapped without copying data. Returns
    (point estimate on all items, lower, upper). Resamples yielding NaN are dropped.
    """
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_items))
    vals = np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(n_resamples)])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on paired correctness (two-sided binomial test on discordant pairs)."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else binomtest(only_a, n, 0.5, alternative="two-sided").pvalue
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
