"""Vectorized agreement metrics from integer-coded labels, for fast bootstrapping.

Labels are coded 0..K-1; an invalid prediction (parse failure) is coded K, which never
matches a gold label. Results match `agreement.agreement_metrics` (tested).
"""

import numpy as np


def confusion(gold: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """(K) x (K+1) matrix: rows = gold label, columns = predicted label (+ invalid)."""
    return np.bincount(gold * (k + 1) + pred, minlength=k * (k + 1)).reshape(k, k + 1)


def macro_f1(c: np.ndarray) -> float:
    k = c.shape[0]
    tp = np.diag(c[:, :k]).astype(float)
    pred_tot = c[:, :k].sum(axis=0)
    gold_tot = c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(pred_tot + gold_tot > 0, 2 * tp / (pred_tot + gold_tot), 0.0)
    return float(f1.mean())


def accuracy(c: np.ndarray) -> float:
    k = c.shape[0]
    return float(np.trace(c[:, :k]) / c.sum())


def cohen_kappa(c: np.ndarray) -> float:
    k = c.shape[0]
    n = c.sum()
    sq = np.zeros((k + 1, k + 1))
    sq[:k, :] = c
    po = np.trace(sq) / n
    pe = float(sq.sum(axis=1) @ sq.sum(axis=0)) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def krippendorff_alpha(c: np.ndarray) -> float:
    """Nominal alpha for two complete raters, via the coincidence matrix."""
    k = c.shape[0]
    sq = np.zeros((k + 1, k + 1))
    sq[:k, :] = c
    o = sq + sq.T
    n = o.sum()
    nc = o.sum(axis=1)
    denom = n**2 - float(nc @ nc)
    if denom == 0:
        return float("nan")
    return float(1 - (n - 1) * (n - np.trace(o)) / denom)


METRICS = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "cohen_kappa": cohen_kappa,
    "krippendorff_alpha": krippendorff_alpha,
}
