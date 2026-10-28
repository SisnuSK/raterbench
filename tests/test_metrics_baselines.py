import math

import pytest

from raterbench.baselines import majority
from raterbench.data.schema import Item
from raterbench.metrics.agreement import agreement_metrics, krippendorff_alpha_nominal


def _two_raters(a: int, b: int, c: int, d: int) -> tuple[list[str], list[str]]:
    """2x2 table: a = both yes, b = r1 yes / r2 no, c = r1 no / r2 yes, d = both no."""
    r1 = ["yes"] * (a + b) + ["no"] * (c + d)
    r2 = ["yes"] * a + ["no"] * b + ["yes"] * c + ["no"] * d
    return r1, r2


def test_kappa_and_alpha_hand_computed() -> None:
    # a=45, b=15, c=25, d=15 (N=100):
    #   p_o = 0.60; p_e = 0.6*0.7 + 0.4*0.3 = 0.54; kappa = 0.06/0.46 = 0.130435
    #   alpha: n = 200 pairable values, n_yes = 130, n_no = 70, off-diagonal o = 80
    #          alpha = 1 - (n-1) * 80 / (2 * 130 * 70) = 1 - 15920/18200 = 0.125275
    r1, r2 = _two_raters(45, 15, 25, 15)
    m = agreement_metrics(r1, r2, ("yes", "no"))
    assert m["accuracy"] == pytest.approx(0.60)
    assert m["cohen_kappa"] == pytest.approx(0.06 / 0.46, abs=1e-6)
    assert m["krippendorff_alpha"] == pytest.approx(1 - 15920 / 18200, abs=1e-6)


def test_perfect_agreement() -> None:
    gold = ["a", "b", "c", "a"]
    m = agreement_metrics(gold, gold, ("a", "b", "c"))
    assert m["accuracy"] == m["macro_f1"] == m["cohen_kappa"] == m["krippendorff_alpha"] == 1.0


def test_per_class_and_macro_f1() -> None:
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    m = agreement_metrics(gold, pred, ("a", "b"))
    # a: P=1, R=0.5, F1=2/3; b: P=2/3, R=1, F1=0.8
    assert m["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["b"]["f1"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_missing_predictions_count_as_errors() -> None:
    gold = ["a", "b", "a", "b"]
    m = agreement_metrics(gold, ["a", None, "a", "b"], ("a", "b"))
    assert m["n_invalid"] == 1
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["b"]["recall"] == pytest.approx(0.5)


def test_constant_predictor_has_zero_kappa() -> None:
    m = agreement_metrics(["a", "b", "a", "a"], ["a"] * 4, ("a", "b"))
    assert m["cohen_kappa"] == pytest.approx(0.0)


def test_alpha_undefined_with_single_value() -> None:
    assert math.isnan(krippendorff_alpha_nominal(["a", "a"], ["a", "a"]))


def test_majority_label_and_tiebreak() -> None:
    items = [
        Item(id=str(i), text="t", label=lab, subgroup="s", split="train")
        for i, lab in enumerate(["b", "b", "a", "c"])
    ]
    label, probs = majority.majority_label(items, ("a", "b", "c"))
    assert label == "b" and probs["b"] == pytest.approx(0.5)
    tie = items[:1] + items[2:3]  # one "b", one "a": tie -> first in label order
    assert majority.majority_label(tie, ("a", "b", "c"))[0] == "a"
