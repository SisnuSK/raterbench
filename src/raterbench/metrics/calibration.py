"""Calibration of a confidence signal for predicting correctness (RQ2).

All functions take `conf` in [0, 1] (the signal's claimed probability that the prediction
is correct) and `correct` as 0/1.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width bins on [0, 1]; the last bin includes 1.0."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": int(m.sum()),
                "mean_conf": float(conf[m].mean()) if m.any() else float("nan"),
                "accuracy": float(correct[m].mean()) if m.any() else float("nan"),
            }
        )
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: sum over bins of (n_b / N) * |accuracy_b - mean_conf_b|."""
    n = len(conf)
    if n == 0:
        return float("nan")
    return float(
        sum(
            b["n"] / n * abs(b["accuracy"] - b["mean_conf"])
            for b in reliability_bins(conf, correct, n_bins)
            if b["n"]
        )
    )


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    """Brier score of the confidence as a forecast of correctness."""
    conf = np.asarray(conf, dtype=float)
    return (
        float(np.mean((conf - np.asarray(correct, dtype=float)) ** 2))
        if len(conf)
        else float("nan")
    )


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of the confidence for separating correct from incorrect predictions."""
    correct = np.asarray(correct, dtype=int)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective-prediction curve: auto-label the most confident fraction first.

    Returns (coverage, risk, AURC). Ties in confidence are broken by input order, which is
    fixed (sorted item IDs), so the curve is deterministic.
    """
    conf = np.asarray(conf, dtype=float)
    err = 1 - np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="stable")
    cum_err = np.cumsum(err[order])
    k = np.arange(1, len(conf) + 1)
    coverage = k / len(conf)
    risk = cum_err / k
    return coverage, risk, float(np.mean(risk))
