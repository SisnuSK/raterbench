"""End-to-end `analyze` on fabricated predictions (no model calls, no raw data)."""

import json
import random
from pathlib import Path

import pytest

from raterbench.analysis.analyze import Analysis, AnalyzeConfig

LABELS = ("none", "against", "favor")
TARGETS = ("abortion", "atheism", "climate")


def _items(role: str, n: int, rng: random.Random) -> list[dict]:
    return [
        {
            "item_id": f"{role}-{i:04d}",
            "subgroup": TARGETS[i % 3],
            "gold": rng.choice(LABELS),
            "role": role,
        }
        for i in range(n)
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def fake_results(tmp_path: Path) -> Path:
    rng = random.Random(0)
    items = _items("eval", 150, rng) + _items("calibration", 120, rng)
    run_rows = []
    for model, skill in (("llm-a", 0.7), ("llm-b", 0.55)):
        for cond in ("zero_shot", "codebook"):
            for it in items:
                right = rng.random() < skill
                label = it["gold"] if right else rng.choice([x for x in LABELS if x != it["gold"]])
                conf = rng.randint(70, 95) if right else rng.randint(50, 90)
                base = {
                    "dataset": "tweet_eval_stance",
                    "model": model,
                    "condition": cond,
                    "gold": it["gold"],
                    "subgroup": it["subgroup"],
                    "role": it["role"],
                    "item_id": it["item_id"],
                }
                bad = rng.random() < 0.02
                run_rows.append(
                    {
                        **base,
                        "sample_idx": 0,
                        "outcome": "parse_failure" if bad else "ok",
                        "label": None if bad else label,
                        "confidence": conf,
                        "label_prob": conf / 100 + 0.04,
                        "reasoning": None,
                        "raw_text": "{}",
                    }
                )
                if cond == "codebook":
                    for s in range(1, 6):
                        lab = (
                            label if rng.random() < (0.9 if right else 0.6) else rng.choice(LABELS)
                        )
                        run_rows.append(
                            {
                                **base,
                                "sample_idx": s,
                                "outcome": "ok",
                                "label": lab,
                                "confidence": conf,
                            }
                        )
    run_dir = tmp_path / "run"
    _write(run_dir / "predictions.jsonl", run_rows)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": {"run": {"conditions": ["zero_shot", "codebook"]}},
                "models": {"llm-a": {}, "llm-b": {}},
            }
        )
    )

    bdir = tmp_path / "baselines" / "predictions" / "tweet_eval_stance"
    maj, enc = [], []
    for it in items:
        base = {
            "dataset": "tweet_eval_stance",
            "item_id": it["item_id"],
            "role": it["role"],
            "gold": it["gold"],
            "subgroup": it["subgroup"],
            "sample_idx": 0,
            "outcome": "ok",
        }
        maj.append(
            {
                **base,
                "model": "majority",
                "condition": "baseline",
                "label": "against",
                "confidence": 48.0,
                "probs": {"none": 0.26, "against": 0.48, "favor": 0.26},
            }
        )
        p = [rng.random() for _ in LABELS]
        p[LABELS.index(it["gold"])] += 1.5
        p = [x / sum(p) for x in p]
        enc.append(
            {
                **base,
                "model": "roberta-base-ft",
                "condition": "supervised",
                "seed": 13,
                "label": LABELS[p.index(max(p))],
                "confidence": 100 * max(p),
                "probs": dict(zip(LABELS, p, strict=True)),
            }
        )
    _write(bdir / "majority.jsonl", maj)
    _write(bdir / "roberta-base-ft_seed13.jsonl", enc)
    return tmp_path


def test_coverage_lower_bound_matches_beta_binomial() -> None:
    from raterbench.analysis.analyze import coverage_lower_bound

    # n_cal=305, alpha=0.05: l=15, Beta(291, 15): mean 0.95098; total SD 0.01481 with n=715.
    assert coverage_lower_bound(305, 715, 0.05) == pytest.approx(0.9214, abs=2e-4)
    assert coverage_lower_bound(5, 100, 0.05) == pytest.approx(0.95)


def test_analyze_end_to_end(fake_results: Path) -> None:
    cfg = AnalyzeConfig(
        run_dir=fake_results / "run",
        baselines_dir=fake_results / "baselines",
        perturbation_dir=fake_results / "none",
        output_dir=fake_results / "analysis",
        error_samples_dir=fake_results / "errors",
        n_bootstrap=50,
    )
    out = Analysis(cfg).run()
    for name in (
        "table1_main_results",
        "table4_calibration",
        "table6_conformal_triage",
        "table7_subgroups",
        "table_significance",
    ):
        assert (out / f"{name}.csv").exists() and (out / f"{name}.md").exists(), name
    for fig in (
        "fig2_prompting_effect",
        "fig3_reliability",
        "fig5_risk_coverage",
        "fig7_subgroups",
    ):
        assert (out / "figures" / f"{fig}.png").exists(), fig
    res = json.loads((out / "analysis.json").read_text())
    signals = {r["signal"] for r in res["calibration"] if r["condition"] == "codebook"}
    assert signals == {"verbalized", "logprob", "self_consistency"}
    assert res["reliability_model"] == "llm-a"
    conf = [r for r in res["conformal"] if r["model"] == "roberta-base-ft"]
    assert conf and all(0 <= r["pct_routed_to_human"] <= 100 for r in conf)
    assert res["perturbation"] == "not run"
    assert any((fake_results / "errors" / "tweet_eval_stance").glob("*.jsonl"))
    # Majority baseline: kappa exactly 0.
    k = res["agreement"]["tweet_eval_stance/majority/baseline"]["cohen_kappa"][0]
    assert k == pytest.approx(0.0)
