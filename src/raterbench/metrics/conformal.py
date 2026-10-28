"""Split conformal prediction and the human-in-the-loop triage policy (RQ3).

Inputs are per-item probability vectors over the label set (rows sum to 1). With
exchangeable calibration and test items, the prediction sets cover the true label with
probability >= 1 - alpha (marginally).
"""

import math

import numpy as np


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return math.inf  # too few calibration points for this alpha: always return full sets
    return float(np.quantile(scores, level, method="higher"))


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-ambiguous set-valued classifier score: 1 - p(true label)."""
    return 1 - probs[np.arange(len(y)), y]


def lac_sets(
    cal_probs: np.ndarray, cal_y: np.ndarray, test_probs: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    q = _quantile(lac_scores(cal_probs, cal_y), alpha)
    return (1 - test_probs) <= q + 1e-12, q


def _aps_scores(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Randomized APS score for every label (Romano et al., 2020).

    score(y) = mass of labels ranked strictly above y + u * p(y), with u ~ U(0, 1) per item.
    The randomization breaks ties (the non-randomized version gives the lowest-ranked label
    a score of exactly 1, which makes sets very conservative with few classes).
    """
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum_sorted = np.cumsum(sorted_p, axis=1)
    cum = np.empty_like(probs)
    np.put_along_axis(cum, order, cum_sorted, axis=1)
    return cum - (1 - u[:, None]) * probs


def aps_sets(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
    seed: int = 13,
) -> tuple[np.ndarray, float]:
    """Randomized APS sets (seeded). Sets may be empty; triage routes those to a human."""
    rng = np.random.default_rng(seed)
    u_cal = rng.random(len(cal_probs))
    scores = _aps_scores(cal_probs, u_cal)[np.arange(len(cal_y)), cal_y]
    q = _quantile(scores, alpha)
    u_test = rng.random(len(test_probs))
    return _aps_scores(test_probs, u_test) <= q + 1e-12, q


def triage_report(sets: np.ndarray, test_y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Auto-label items whose set is a single label; route the rest to a human.

    Error rate on auto-labeled items compares the (single) set label to gold. Human effort
    saved is the fraction of items that need no human label.
    """
    n = len(test_y)
    covered = sets[np.arange(n), test_y]
    size = sets.sum(axis=1)
    auto = size == 1
    auto_label = sets.argmax(axis=1)
    auto_err = float(np.mean(auto_label[auto] != test_y[auto])) if auto.any() else float("nan")
    return {
        "n": n,
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "frac_routed_to_human": float(1 - auto.mean()),
        "auto_label_error_rate": auto_err,
        "human_effort_saved": float(auto.mean()),
        "point_prediction_error_rate": float(np.mean(pred != test_y)),
    }
