import numpy as np
import pytest

from raterbench.metrics import fast
from raterbench.metrics.agreement import agreement_metrics


@pytest.mark.parametrize(
    "k,seed,invalid_rate", [(2, 0, 0.0), (3, 1, 0.05), (3, 2, 0.0), (5, 3, 0.1)]
)
def test_fast_metrics_match_reference(k: int, seed: int, invalid_rate: float) -> None:
    rng = np.random.default_rng(seed)
    labels = tuple(f"l{i}" for i in range(k))
    gold = rng.integers(0, k, 300)
    pred = np.where(rng.random(300) < 0.6, gold, rng.integers(0, k, 300))
    pred = np.where(rng.random(300) < invalid_rate, k, pred)
    ref = agreement_metrics(
        [labels[g] for g in gold], [labels[p] if p < k else None for p in pred], labels
    )
    c = fast.confusion(gold, pred, k)
    for name, fn in fast.METRICS.items():
        assert fn(c) == pytest.approx(ref[name], abs=1e-9), name
