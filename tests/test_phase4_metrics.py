import numpy as np
import pytest

from raterbench.metrics.bias import subgroup_report
from raterbench.metrics.calibration import auroc, brier, ece, reliability_bins, risk_coverage
from raterbench.metrics.conformal import aps_sets, lac_sets, triage_report
from raterbench.metrics.stats import bootstrap_ci, holm, mcnemar_exact


def test_holm_hand_computed() -> None:
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_mcnemar_exact_hand_computed() -> None:
    # 10 vs 2 discordant pairs: p = 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12 = 158/4096
    a = [True] * 10 + [False] * 2 + [True] * 5
    b = [False] * 10 + [True] * 2 + [True] * 5
    r = mcnemar_exact(a, b)
    assert (r["only_a_correct"], r["only_b_correct"]) == (10, 2)
    assert r["p_value"] == pytest.approx(158 / 4096)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    x = np.random.default_rng(0).normal(5, 1, 400)
    point, lo, hi = bootstrap_ci(lambda idx: float(x[idx].mean()), len(x), 1000, seed=1)
    assert lo < point < hi
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(400), rel=0.2)
    assert bootstrap_ci(lambda idx: float(x[idx].mean()), len(x), 200, seed=1) == bootstrap_ci(
        lambda idx: float(x[idx].mean()), len(x), 200, seed=1
    )


def test_ece_brier_auroc_hand_computed() -> None:
    conf = np.array([0.9, 0.9, 0.1, 0.1])
    correct = np.array([1, 0, 0, 0])
    # bin 9: |0.5 - 0.9| * 2/4 = 0.2 ; bin 1: |0 - 0.1| * 2/4 = 0.05
    assert ece(conf, correct) == pytest.approx(0.25)
    assert brier(conf, correct) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert auroc(conf, correct) == pytest.approx(0.8333, abs=1e-4)  # 2.5 of 3 pairs ordered
    assert sum(b["n"] for b in reliability_bins(np.array([0.0, 1.0, 0.55]), np.ones(3))) == 3


def test_perfect_calibration_has_zero_ece() -> None:
    conf = np.array([1.0] * 5 + [0.0] * 5)
    assert ece(conf, np.array([1] * 5 + [0] * 5)) == 0.0


def test_risk_coverage() -> None:
    cov, risk, aurc = risk_coverage(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 0]))
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.5])
    assert aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def _synthetic(n: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    probs = rng.dirichlet(np.ones(k) * 0.5, n)
    y = np.array([rng.choice(k, p=p) for p in probs])
    return probs, y


@pytest.mark.parametrize("method", [lac_sets, aps_sets])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_conformal_coverage_on_synthetic_data(method, alpha: float) -> None:
    rng = np.random.default_rng(42)
    n_cal, covs = 300, []
    for _ in range(200):
        cal_p, cal_y = _synthetic(n_cal, 3, rng)
        te_p, te_y = _synthetic(500, 3, rng)
        sets, _ = method(cal_p, cal_y, te_p, alpha)
        covs.append(sets[np.arange(500), te_y].mean())
    mean_cov = float(np.mean(covs))
    # Theory: 1 - alpha <= E[coverage] <= 1 - alpha + 1/(n_cal + 1); allow MC slack.
    assert 1 - alpha - 0.01 <= mean_cov <= 1 - alpha + 1 / (n_cal + 1) + 0.01


def test_conformal_too_few_points_returns_full_sets() -> None:
    p = np.array([[0.7, 0.3]] * 5)
    sets, q = lac_sets(p, np.zeros(5, int), p, alpha=0.05)  # needs n >= 19
    assert np.isinf(q) and sets.all()


def test_triage_report() -> None:
    sets = np.array([[True, False], [False, True], [True, True], [True, False]])
    y = np.array([0, 0, 1, 1])
    r = triage_report(sets, y, pred=np.array([0, 1, 0, 0]))
    assert r["frac_routed_to_human"] == pytest.approx(0.25)
    assert r["human_effort_saved"] == pytest.approx(0.75)
    assert r["auto_label_error_rate"] == pytest.approx(2 / 3)
    assert r["empirical_coverage"] == pytest.approx(0.5)
    assert r["mean_set_size"] == pytest.approx(1.25)


def test_subgroup_report_gap_and_test() -> None:
    gold = ["a", "b"] * 50 + ["a", "b"] * 50
    pred = ["a", "b"] * 50 + ["b", "a"] * 40 + ["a", "b"] * 10
    groups = ["g1"] * 100 + ["g2"] * 100
    r = subgroup_report(gold, pred, groups, ("a", "b"), n_resamples=100)
    assert r["subgroups"]["g1"]["error_rate"] == 0.0
    assert r["subgroups"]["g2"]["error_rate"] == pytest.approx(0.8)
    assert r["max_error_rate_gap"] == pytest.approx(0.8)
    assert r["error_rate_p_value"] < 1e-10
