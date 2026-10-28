"""Agreement metrics between predicted labels and gold labels (RQ1).

Missing predictions (parse failures, refusals) are kept and scored as errors: they are
mapped to an extra INVALID category that never matches a gold label.
"""

from collections.abc import Sequence
from typing import Any

import krippendorff
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

INVALID = "__invalid__"


def krippendorff_alpha_nominal(a: Sequence[str], b: Sequence[str]) -> float:
    """Krippendorff's alpha (nominal) for two complete raters."""
    domain = sorted(set(a) | set(b))
    if len(domain) < 2:
        return float("nan")  # undefined when only one value occurs
    index = {v: i for i, v in enumerate(domain)}
    data = np.array([[index[x] for x in a], [index[x] for x in b]], dtype=float)
    return float(
        krippendorff.alpha(
            reliability_data=data,
            level_of_measurement="nominal",
            value_domain=list(range(len(domain))),
        )
    )


def agreement_metrics(
    gold: Sequence[str], pred: Sequence[str | None], labels: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    pred_ = [p if p is not None else INVALID for p in pred]
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(
        gold, pred_, labels=labels, average=None, zero_division=0
    )
    return {
        "n": len(gold),
        "n_invalid": sum(x == INVALID for x in pred_),
        "accuracy": float(accuracy_score(gold, pred_)),
        "macro_f1": float(np.mean(f)),
        "cohen_kappa": float(cohen_kappa_score(gold, pred_)),
        "krippendorff_alpha": krippendorff_alpha_nominal(gold, pred_),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
    }
