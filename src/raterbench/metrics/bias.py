"""Subgroup analysis (RQ4): per-subgroup performance, gaps, and a test of equal error rates."""

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2_contingency

from raterbench.metrics import fast
from raterbench.metrics.stats import bootstrap_ci


def subgroup_report(
    gold: Sequence[str],
    pred: Sequence[str | None],
    groups: Sequence[str],
    labels: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 13,
) -> dict:
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    g_arr = np.array([index[x] for x in gold])
    p_arr = np.array([index.get(x, k) if x is not None else k for x in pred])
    groups = list(groups)
    rows = {}
    for g in sorted(set(groups)):
        idx = np.array([i for i, x in enumerate(groups) if x == g])
        gg, pp = g_arr[idx], p_arr[idx]

        def f1(sample: np.ndarray, gg=gg, pp=pp) -> float:
            return fast.macro_f1(fast.confusion(gg[sample], pp[sample], k))

        point, lo, hi = bootstrap_ci(f1, len(idx), n_resamples, seed=seed)
        n_err = int(np.sum(pp != gg))
        rows[g] = {
            "n": len(idx),
            "macro_f1": point,
            "macro_f1_ci": [lo, hi],
            "error_rate": n_err / len(idx),
            "n_errors": n_err,
        }

    f1s = [r["macro_f1"] for r in rows.values()]
    errs = [r["error_rate"] for r in rows.values()]
    # Chi-square test of independence between subgroup and correct/incorrect.
    table = np.array([[r["n_errors"], r["n"] - r["n_errors"]] for r in rows.values()])
    if len(rows) > 1 and (table.sum(axis=0) > 0).all():
        chi2, p, dof, _ = chi2_contingency(table)
    else:
        chi2, p, dof = float("nan"), float("nan"), 0
    return {
        "subgroups": rows,
        "max_macro_f1_gap": float(max(f1s) - min(f1s)),
        "max_error_rate_gap": float(max(errs) - min(errs)),
        "error_rate_chi2": float(chi2),
        "error_rate_dof": int(dof),
        "error_rate_p_value": float(p),
    }
