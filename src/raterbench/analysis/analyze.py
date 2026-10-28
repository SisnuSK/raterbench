"""`raterbench analyze`: every table and figure in spec Section 9 from saved predictions."""

import csv
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from raterbench.analysis import figures
from raterbench.analysis.load import Key, PredSet, load_baselines, load_llm_run, read_jsonl
from raterbench.metrics import fast
from raterbench.metrics.bias import subgroup_report
from raterbench.metrics.calibration import auroc, brier, ece, reliability_bins, risk_coverage
from raterbench.metrics.conformal import aps_sets, lac_sets, triage_report
from raterbench.metrics.stats import bootstrap_ci, holm, mcnemar_exact

ENCODER = "roberta-base-ft"
BASELINE_ORDER = ["majority", ENCODER]


class AnalyzeConfig(BaseModel):
    run_dir: Path = Path("results/runs/full_local")
    baselines_dir: Path = Path("results/baselines")
    perturbation_dir: Path = Path("results/perturbation")
    output_dir: Path = Path("results/analysis")
    error_samples_dir: Path = Path("results/error_samples")
    encoder_seed: int = 13  # seed whose predictions get CIs / significance tests
    n_bootstrap: int = 1000
    seed: int = 13
    conformal_alphas: list[float] = Field(default_factory=lambda: [0.10, 0.05])
    focus_condition: str = "codebook"  # the condition with all three confidence signals
    error_samples_per_cell: int = 25


# ---- helpers ---------------------------------------------------------------------------


def coverage_lower_bound(n_cal: int, n_eval: int, alpha: float, k_sd: float = 2.0) -> float:
    """Mean - k SD of empirical split-conformal coverage on n_eval test points.

    With l = floor((n_cal + 1) * alpha), true coverage ~ Beta(n_cal + 1 - l, l) (Vovk 2012;
    Angelopoulos & Bates 2021); test sampling adds binomial variance on top.
    """
    lo = math.floor((n_cal + 1) * alpha)
    if lo == 0:
        return 1.0 - alpha  # too few calibration points: sets are full, coverage is 1
    a, b = n_cal + 1 - lo, lo
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1)) + mean * (1 - mean) / n_eval
    return mean - k_sd * math.sqrt(var)


def _fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _ci(p: float, lo: float, hi: float) -> str:
    return f"{_fmt(p)} [{_fmt(lo)}, {_fmt(hi)}]"


def _write_table(
    out: Path, name: str, rows: list[dict[str, Any]], title: str, note: str = ""
) -> None:
    if not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def cell(v: Any) -> str:
        return _fmt(v) if isinstance(v, float) else str(v)

    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r[c]) for c in cols) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Analysis:
    def __init__(self, cfg: AnalyzeConfig):
        self.cfg = cfg
        self.preds: dict[Key, PredSet] = {
            **load_baselines(cfg.baselines_dir / "predictions", cfg.encoder_seed),
            **load_llm_run(cfg.run_dir / "predictions.jsonl"),
        }
        manifest = json.loads((cfg.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.conditions = manifest["config"]["run"]["conditions"]
        self.llms = list(manifest["models"])
        self.datasets = sorted({k[0] for k in self.preds})
        self.results: dict[str, Any] = {}
        self._check_alignment()

    def _check_alignment(self) -> None:
        by_role = defaultdict(set)
        for (ds, _, _, role), ps in self.preds.items():
            by_role[(ds, role)].add(tuple(ps.item_ids))
        bad = {k: len(v) for k, v in by_role.items() if len(v) > 1}
        if bad:
            raise ValueError(f"prediction sets cover different items: {bad}")

    def get(self, ds: str, model: str, cond: str, role: str = "eval") -> PredSet | None:
        return self.preds.get((ds, model, cond, role))

    def rows_for(self, ds: str) -> list[tuple[str, str]]:
        """(model, condition) in display order: baselines first, then LLM x condition."""
        rows = [("majority", "baseline"), (ENCODER, "supervised")]
        rows += [(m, c) for m in self.llms for c in self.conditions]
        return [(m, c) for m, c in rows if self.get(ds, m, c)]

    def boot(self, ps: PredSet, metric: str) -> tuple[float, float, float]:
        fn = fast.METRICS[metric]
        return bootstrap_ci(
            lambda i: fn(fast.confusion(ps.gold[i], ps.pred[i], ps.k)),
            len(ps.gold),
            self.cfg.n_bootstrap,
            seed=self.cfg.seed,
        )

    # ---- RQ1 -----------------------------------------------------------------------------

    def agreement(self) -> None:
        out = self.cfg.output_dir
        seed_stats = self._encoder_seed_stats()
        rows, store = [], {}
        for ds in self.datasets:
            for model, cond in self.rows_for(ds):
                ps = self.get(ds, model, cond)
                m = {
                    k: self.boot(ps, k)
                    for k in ("macro_f1", "cohen_kappa", "krippendorff_alpha", "accuracy")
                }
                store[(ds, model, cond)] = m
                row = {
                    "dataset": ds,
                    "model": model,
                    "condition": cond,
                    "n": len(ps.gold),
                    "invalid": int(np.sum(ps.pred == ps.k)),
                    "macro_f1": _ci(*m["macro_f1"]),
                    "cohen_kappa": _ci(*m["cohen_kappa"]),
                    "krippendorff_alpha": _ci(*m["krippendorff_alpha"]),
                    "accuracy": _fmt(m["accuracy"][0]),
                }
                if model == ENCODER and ds in seed_stats:
                    row["model"] = (
                        f"{ENCODER} (seed {self.cfg.encoder_seed}; 3-seed kappa {seed_stats[ds]})"
                    )
                rows.append(row)
        _write_table(
            out,
            "table1_main_results",
            rows,
            "Main results (eval role)",
            f"Point estimate [95% percentile bootstrap CI, {self.cfg.n_bootstrap} "
            "resamples over items]. Invalid outputs count as errors.",
        )
        self.results["agreement"] = {f"{d}/{m}/{c}": v for (d, m, c), v in store.items()}
        figures.prompting_effect(
            store, self.datasets, self.llms, self.conditions, ENCODER, out / "figures"
        )

    def _encoder_seed_stats(self) -> dict[str, str]:
        path = self.cfg.baselines_dir / "metrics.csv"
        if not path.exists():
            return {}
        vals = defaultdict(list)
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] == ENCODER and r["role"] == "eval":
                    vals[r["dataset"]].append(float(r["cohen_kappa"]))
        return {
            ds: f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}" for ds, v in vals.items() if len(v) > 1
        }

    def significance(self) -> None:
        rows = []
        for ds in self.datasets:
            pairs = []
            for m in self.llms:
                pairs += [((m, a), (m, b)) for a, b in combinations(self.conditions, 2)]
            for c in self.conditions:
                pairs += [((a, c), (b, c)) for a, b in combinations(self.llms, 2)]
            enc = self.get(ds, ENCODER, "supervised")
            if enc is not None:
                pairs += [
                    ((m, c), (ENCODER, "supervised")) for m in self.llms for c in self.conditions
                ]
            tests = []
            for (ma, ca), (mb, cb) in pairs:
                a, b = self.get(ds, ma, ca), self.get(ds, mb, cb)
                if a is None or b is None:
                    continue
                r = mcnemar_exact(a.correct, b.correct)
                tests.append((f"{ma}/{ca}", f"{mb}/{cb}", a.correct.mean(), b.correct.mean(), r))
            adj = holm([t[4]["p_value"] for t in tests])
            for (a, b, acc_a, acc_b, r), p_adj in zip(tests, adj, strict=True):
                rows.append(
                    {
                        "dataset": ds,
                        "a": a,
                        "b": b,
                        "acc_a": float(acc_a),
                        "acc_b": float(acc_b),
                        "only_a_correct": r["only_a_correct"],
                        "only_b_correct": r["only_b_correct"],
                        "p": r["p_value"],
                        "p_holm": p_adj,
                        "significant_0.05": p_adj < 0.05,
                    }
                )
        _write_table(
            self.cfg.output_dir,
            "table_significance",
            rows,
            "Paired significance tests (exact McNemar, eval role)",
            "Holm-Bonferroni correction within each dataset's family of tests.",
        )
        self.results["significance"] = rows

    # ---- RQ2 -----------------------------------------------------------------------------

    def calibration(self) -> None:
        rows, curves, reliab = [], defaultdict(list), {}
        for ds in self.datasets:
            cells = [(m, c) for m in self.llms for c in self.conditions]
            cells += [(ENCODER, "supervised")]
            for model, cond in cells:
                ps = self.get(ds, model, cond)
                if ps is None:
                    continue
                for sig, conf in ps.signals.items():
                    ok = ~np.isnan(conf)
                    c, y = conf[ok], ps.correct[ok].astype(float)
                    e = bootstrap_ci(
                        lambda i, c=c, y=y: ece(c[i], y[i]),
                        len(c),
                        self.cfg.n_bootstrap,
                        seed=self.cfg.seed,
                    )
                    a = bootstrap_ci(
                        lambda i, c=c, y=y: auroc(c[i], y[i]),
                        len(c),
                        self.cfg.n_bootstrap,
                        seed=self.cfg.seed,
                    )
                    _, _, aurc = risk_coverage(np.where(ok, conf, -1.0), ps.correct)
                    rows.append(
                        {
                            "dataset": ds,
                            "model": model,
                            "condition": cond,
                            "signal": sig,
                            "n_with_signal": int(ok.sum()),
                            "n_distinct_values": int(len(np.unique(c))),
                            "ece": _ci(*e),
                            "brier": brier(c, y),
                            "auroc": _ci(*a),
                            "aurc": aurc,
                        }
                    )
                    reliab[(ds, model, cond, sig)] = reliability_bins(c, y)
                    if cond in (self.cfg.focus_condition, "supervised"):
                        cov, risk, _ = risk_coverage(np.where(ok, conf, -1.0), ps.correct)
                        curves[ds].append((model, sig, cov, risk))
        _write_table(
            self.cfg.output_dir,
            "table4_calibration",
            rows,
            "Calibration of confidence signals (eval role)",
            "ECE (10 equal-width bins) and AUROC with 95% bootstrap CIs; Brier score "
            "of confidence as a forecast of correctness; AURC = area under the "
            "risk-coverage curve (lower is better). Self-consistency = share of the "
            "5 T=0.7 samples that agree with the T=0 label.",
        )
        self.results["calibration"] = rows
        best = self._best_llm()
        if best:
            figures.reliability(
                reliab,
                self.datasets,
                best,
                self.cfg.focus_condition,
                self.cfg.output_dir / "figures",
            )
        figures.risk_coverage_plot(
            curves, self.datasets, self.llms, ENCODER, self.cfg.output_dir / "figures"
        )
        self.results["reliability_model"] = best

    def _best_llm(self) -> str | None:
        scores = {}
        for m in self.llms:
            ks = [
                fast.cohen_kappa(fast.confusion(p.gold, p.pred, p.k))
                for ds in self.datasets
                if (p := self.get(ds, m, self.cfg.focus_condition))
            ]
            if ks:
                scores[m] = float(np.mean(ks))
        return max(scores, key=scores.get) if scores else None

    # ---- RQ3 -----------------------------------------------------------------------------

    def conformal(self) -> None:
        rows = []
        cells = [(m, self.cfg.focus_condition) for m in self.llms] + [(ENCODER, "supervised")]
        for ds in self.datasets:
            for model, cond in cells:
                cal, ev = self.get(ds, model, cond, "calibration"), self.get(ds, model, cond)
                if cal is None or ev is None:
                    continue
                for sig in ev.probs:
                    for method, fn in (("LAC", lac_sets), ("APS", aps_sets)):
                        for alpha in self.cfg.conformal_alphas:
                            sets, q = fn(cal.probs[sig], cal.gold, ev.probs[sig], alpha)
                            r = triage_report(sets, ev.gold, ev.pred)
                            target = 1 - alpha
                            lower = coverage_lower_bound(len(cal.gold), r["n"], alpha)
                            rows.append(
                                {
                                    "dataset": ds,
                                    "model": model,
                                    "condition": cond,
                                    "signal": sig,
                                    "method": method,
                                    "target_coverage": target,
                                    "n_cal": len(cal.gold),
                                    "n_eval": r["n"],
                                    "threshold": q,
                                    "empirical_coverage": r["empirical_coverage"],
                                    "coverage_lower_2sd": lower,
                                    "coverage_ok": r["empirical_coverage"] >= lower,
                                    "mean_set_size": r["mean_set_size"],
                                    "pct_routed_to_human": 100 * r["frac_routed_to_human"],
                                    "auto_label_error_rate": r["auto_label_error_rate"],
                                    "human_effort_saved_pct": 100 * r["human_effort_saved"],
                                    "error_rate_if_all_auto": r["point_prediction_error_rate"],
                                }
                            )
        _write_table(
            self.cfg.output_dir,
            "table6_conformal_triage",
            rows,
            "Split conformal triage (calibration role -> eval role)",
            "Items whose prediction set has exactly one label are auto-labeled; all "
            "others go to a human. coverage_ok: empirical coverage >= coverage_lower_2sd, "
            "the mean minus 2 SD of split-conformal coverage given n_cal (Beta) and n_eval "
            "(binomial). The guarantee is a lower bound on expected coverage; coarse "
            "confidence values can make sets over-cover. It covers the prediction set, not "
            "the error rate on auto-labeled items. Dreaddit's calibration set is held out "
            "from train, not test (see data/DATASETS.md), so exchangeability is approximate "
            "there.",
        )
        self.results["conformal"] = rows

    # ---- RQ4 -----------------------------------------------------------------------------

    def subgroups(self) -> None:
        rows, plot = [], {}
        cells = [(m, self.cfg.focus_condition) for m in self.llms] + [(ENCODER, "supervised")]
        for ds in self.datasets:
            views = (
                [("domain", True), ("subreddit", False)]
                if ds == "dreaddit"
                else [("target", False)]
            )
            for view, use_domain in views:
                for model, cond in cells:
                    ps = self.get(ds, model, cond)
                    if ps is None:
                        continue
                    groups = ps.domain() if use_domain else ps.subgroup
                    labels = list(ps.labels)
                    rep = subgroup_report(
                        [labels[i] for i in ps.gold],
                        [labels[i] if i < ps.k else None for i in ps.pred],
                        groups,
                        labels,
                        self.cfg.n_bootstrap,
                        self.cfg.seed,
                    )
                    if view != "subreddit":
                        plot[(ds, model)] = rep
                    for g, s in rep["subgroups"].items():
                        rows.append(
                            {
                                "dataset": ds,
                                "view": view,
                                "model": model,
                                "condition": cond,
                                "subgroup": g,
                                "n": s["n"],
                                "macro_f1": _ci(s["macro_f1"], *s["macro_f1_ci"]),
                                "error_rate": s["error_rate"],
                                "max_f1_gap": rep["max_macro_f1_gap"],
                                "chi2_p_error_rates": rep["error_rate_p_value"],
                            }
                        )
        _write_table(
            self.cfg.output_dir,
            "table7_subgroups",
            rows,
            "Subgroup analysis (eval role)",
            "Per-subgroup macro-F1 [95% bootstrap CI]. chi2_p tests whether error "
            "rates differ across subgroups. Subgroups differ in size and label mix, "
            "so gaps are not purely model bias.",
        )
        self.results["subgroups"] = rows
        figures.subgroup_plot(
            plot, self.datasets, self.llms, ENCODER, self.cfg.output_dir / "figures"
        )

    def perturbation(self) -> None:
        path = self.cfg.perturbation_dir / "predictions.jsonl"
        if not path.exists():
            self.results["perturbation"] = "not run"
            return
        recs = read_jsonl(path)
        orig = {
            (r["dataset"], r["model"], r["condition"], r["item_id"]): r
            for r in recs
            if r["perturbation"] == "original"
        }
        cells: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
        for r in recs:
            if r["perturbation"] == "original":
                continue
            o = orig[(r["dataset"], r["model"], r["condition"], r["original_id"])]
            cells[(r["dataset"], r["model"], r["condition"], r["perturbation"])].append((o, r))
        rows = []
        for (ds, model, cond, ptype), pairs in sorted(cells.items()):
            flips = np.array([o.get("label") != p.get("label") for o, p in pairs], dtype=float)
            fr = bootstrap_ci(
                lambda i, f=flips: float(f[i].mean()),
                len(flips),
                self.cfg.n_bootstrap,
                seed=self.cfg.seed,
            )
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "condition": cond,
                    "perturbation": ptype,
                    "n": len(pairs),
                    "flip_rate": _ci(*fr),
                    "acc_original": float(np.mean([o.get("label") == o["gold"] for o, _ in pairs])),
                    "acc_perturbed": float(
                        np.mean([p.get("label") == p["gold"] for _, p in pairs])
                    ),
                    "invalid_perturbed": sum(p.get("label") is None for _, p in pairs),
                }
            )
        _write_table(
            self.cfg.output_dir,
            "table8_perturbation_flips",
            rows,
            "Label flip rates under meaning-preserving perturbations (T=0)",
            "flip_rate = share of items whose label changes after the edit "
            "[95% bootstrap CI]. Read alongside the manual spot check "
            "(data/perturbations/spot_check.md).",
        )
        self.results["perturbation"] = rows

    # ---- error samples -------------------------------------------------------------------

    def error_samples(self) -> None:
        out = self.cfg.error_samples_dir
        rng = random.Random(self.cfg.seed)
        counts = {}
        for ds in self.datasets:
            for m in self.llms:
                for c in self.conditions:
                    ps = self.get(ds, m, c)
                    if ps is None:
                        continue
                    errs = [r for r in ps.records if r.get("label") != r["gold"]]
                    sample = rng.sample(errs, min(self.cfg.error_samples_per_cell, len(errs)))
                    path = out / ds / f"{m}__{c}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    keep = (
                        "item_id",
                        "subgroup",
                        "gold",
                        "label",
                        "confidence",
                        "reasoning",
                        "outcome",
                        "parse_error",
                    )  # no raw_text: model output can quote the (non-redistributable) text
                    with path.open("w", encoding="utf-8") as f:
                        for r in sorted(sample, key=lambda x: x["item_id"]):
                            f.write(
                                json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n"
                            )
                    counts[f"{ds}/{m}/{c}"] = {"errors": len(errs), "sampled": len(sample)}
        self.results["error_samples"] = counts

    def run(self) -> Path:
        started = datetime.now(UTC).isoformat()
        for step in (
            self.agreement,
            self.significance,
            self.calibration,
            self.conformal,
            self.subgroups,
            self.perturbation,
            self.error_samples,
        ):
            print(f"analyze: {step.__name__}", flush=True)
            step()
        self.results["meta"] = {
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "config": self.cfg.model_dump(mode="json"),
        }
        out = self.cfg.output_dir / "analysis.json"
        out.write_text(json.dumps(self.results, indent=2, default=str), encoding="utf-8")
        return self.cfg.output_dir
