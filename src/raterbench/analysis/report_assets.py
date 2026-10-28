"""Generate the report's LaTeX tables, in-text number macros, and the README results table.

Everything is read from files that `raterbench analyze` (and the run/baseline steps) wrote,
so no number in the report or README is typed by hand.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

DATASETS = [("dreaddit", "Dreaddit"), ("tweet_eval_stance", "TweetEval")]
LLMS = ["llama3.1-8b", "qwen2.5-7b"]
ENCODER = "roberta-base-ft"
CONDITIONS = ["zero_shot", "few_shot", "codebook", "codebook_reasoning"]
COND_NAME = {
    "zero_shot": "zero-shot",
    "few_shot": "few-shot",
    "codebook": "codebook",
    "codebook_reasoning": "codebook+reason",
    "baseline": "--",
    "supervised": "--",
}
MODEL_NAME = {
    "majority": "Majority class",
    ENCODER: "RoBERTa-base (fine-tuned)",
    "llama3.1-8b": "Llama 3.1 8B",
    "qwen2.5-7b": "Qwen 2.5 7B",
}
SIGNAL_NAME = {
    "verbalized": "verbalized",
    "logprob": "token prob.",
    "self_consistency": "self-consistency",
    "softmax": "softmax",
}
_CI = re.compile(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]")


def _ci(s: str) -> tuple[float, float, float]:
    m = _CI.match(s)
    if not m:
        raise ValueError(f"not a CI string: {s!r}")
    return tuple(float(x) for x in m.groups())  # type: ignore[return-value]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Assets:
    def __init__(self, analysis_dir: Path, run_dir: Path, baselines_dir: Path):
        self.res = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.analysis_dir = analysis_dir
        self.baseline_metrics = _read_csv(baselines_dir / "metrics.csv")
        self.numbers: dict[str, str] = {}

    def agree(self, ds: str, model: str, cond: str, metric: str) -> tuple[float, float, float]:
        return tuple(self.res["agreement"][f"{ds}/{model}/{cond}"][metric])  # type: ignore

    def rows(self) -> list[tuple[str, str]]:
        return [("majority", "baseline"), (ENCODER, "supervised")] + [
            (m, c) for m in LLMS for c in CONDITIONS
        ]

    # ---- tables --------------------------------------------------------------------------

    def main_table(self) -> str:
        best = {
            ds: max(
                ((m, c) for m in LLMS for c in CONDITIONS),
                key=lambda mc, ds=ds: self.agree(ds, *mc, "cohen_kappa")[0],
            )
            for ds, _ in DATASETS
        }
        lines = [
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r" & & \multicolumn{2}{c}{Dreaddit (stress)}"
            r" & \multicolumn{2}{c}{TweetEval (stance)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Model & Prompt & $\kappa$ [95\% CI] & F1 & $\kappa$ [95\% CI] & F1 \\",
            r"\midrule",
        ]
        prev = None
        for model, cond in self.rows():
            if prev and prev != model and model in LLMS:
                lines.append(r"\midrule")
            cells = []
            for ds, _ in DATASETS:
                k, lo, hi = self.agree(ds, model, cond, "cohen_kappa")
                f1 = self.agree(ds, model, cond, "macro_f1")[0]
                kc = f"{k:.2f} [{lo:.2f}, {hi:.2f}]"
                if best[ds] == (model, cond):
                    kc = rf"\textbf{{{k:.2f}}} [{lo:.2f}, {hi:.2f}]"
                cells += [kc, f"{f1:.2f}"]
            name = MODEL_NAME[model] if model != prev else ""
            lines.append(f"{name} & {COND_NAME[cond]} & " + " & ".join(cells) + r" \\")
            prev = model
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def calibration_table(self) -> str:
        cal = {
            (r["dataset"], r["model"], r["signal"]): r
            for r in self.res["calibration"]
            if r["condition"] in ("codebook", "supervised")
        }
        cells_order = [(m, s) for m in LLMS for s in ("verbalized", "logprob", "self_consistency")]
        cells_order.append((ENCODER, "softmax"))
        lines = [
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r" & & \multicolumn{2}{c}{Dreaddit} & \multicolumn{2}{c}{TweetEval} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Model & Signal & ECE$\downarrow$ & AUROC$\uparrow$"
            r" & ECE$\downarrow$ & AUROC$\uparrow$ \\",
            r"\midrule",
        ]
        prev = None
        for model, sig in cells_order:
            cells = []
            for ds, _ in DATASETS:
                r = cal[(ds, model, sig)]
                cells += [f"{_ci(r['ece'])[0]:.2f}", f"{_ci(r['auroc'])[0]:.2f}"]
            name = MODEL_NAME[model].replace(" (fine-tuned)", "") if model != prev else ""
            lines.append(f"{name} & {SIGNAL_NAME[sig]} & " + " & ".join(cells) + r" \\")
            prev = model
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def conformal_table(self, target: float = 0.9) -> str:
        conf = {
            (r["dataset"], r["model"], r["signal"]): r
            for r in self.res["conformal"]
            if r["method"] == "LAC" and abs(r["target_coverage"] - target) < 1e-9
        }
        cells_order = [(m, s) for m in LLMS for s in ("logprob", "verbalized", "self_consistency")]
        cells_order.append((ENCODER, "softmax"))
        lines = [
            r"\begin{tabular}{llcccccc}",
            r"\toprule",
            r" & & \multicolumn{3}{c}{Dreaddit} & \multicolumn{3}{c}{TweetEval} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"Model & Signal & Cov. & Human\% & Err.\% & Cov. & Human\% & Err.\% \\",
            r"\midrule",
        ]
        prev = None
        for model, sig in cells_order:
            cells = []
            for ds, _ in DATASETS:
                r = conf[(ds, model, sig)]
                err = r["auto_label_error_rate"]
                err_s = "--" if err is None or err != err else f"{100 * err:.1f}"
                cells += [
                    f"{r['empirical_coverage']:.3f}",
                    f"{r['pct_routed_to_human']:.0f}",
                    err_s,
                ]
            name = MODEL_NAME[model].replace(" (fine-tuned)", "") if model != prev else ""
            lines.append(f"{name} & {SIGNAL_NAME[sig]} & " + " & ".join(cells) + r" \\")
            prev = model
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def perturbation_table(self, condition: str = "codebook") -> str:
        rows = self.res.get("perturbation")
        if not isinstance(rows, list):
            return r"\emph{Perturbation test not run.}"
        pert = {
            (r["dataset"], r["model"], r["perturbation"]): r
            for r in rows
            if r["condition"] == condition
        }
        lines = [
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Model & Edit & Dreaddit & TweetEval \\",
            r"\midrule",
        ]
        for model in LLMS:
            for i, ptype in enumerate(("gender_swap", "typo", "paraphrase")):
                cells = []
                for ds, _ in DATASETS:
                    r = pert.get((ds, model, ptype))
                    if r is None:
                        cells.append("n/a")
                    else:
                        p, lo, hi = _ci(r["flip_rate"])
                        cells.append(f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
                name = MODEL_NAME[model] if i == 0 else ""
                lines.append(f"{name} & {ptype.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
            if model != LLMS[-1]:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def error_table(self) -> str | None:
        rows = _read_csv(self.analysis_dir / "table9_error_taxonomy.csv")
        if not rows or not sum(int(r["total_primary"]) for r in rows):
            return None
        keys = [k for k in rows[0] if k.endswith("_primary") and k != "total_primary"]
        lines = [
            r"\begin{tabular}{l" + "c" * (len(keys) + 1) + "}",
            r"\toprule",
            "Error type & "
            + " & ".join(dict(DATASETS).get(k.removesuffix("_primary"), k) for k in keys)
            + r" & Total \\",
            r"\midrule",
        ]
        for r in sorted(rows, key=lambda x: -int(x["total_primary"])):
            if int(r["total_primary"]) == 0:
                continue
            lines.append(
                r["error_type"].replace("_", " ")
                + " & "
                + " & ".join(r[k] for k in keys)
                + f" & {r['total_primary']}"
                + r" \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def _error_numbers(self) -> None:
        """Counts per error type and dataset, the most frequent type, and who coded them."""
        rows = _read_csv(self.analysis_dir / "table9_error_taxonomy.csv")
        status_path = self.analysis_dir / "table9_status.json"
        if not rows or not status_path.exists():
            return
        status = json.loads(status_path.read_text(encoding="utf-8"))
        n = self.numbers
        n["errStatus"] = status["text"]
        n["errNCoded"] = str(status["n_coded"])
        for ds, short in (("dreaddit", "Dr"), ("tweet_eval_stance", "Te")):
            col = f"{ds}_primary"
            if col not in rows[0]:
                continue
            n[f"errN{short}"] = str(status["per_dataset"].get(ds, 0))
            for r in rows:
                camel = "".join(w.capitalize() for w in r["error_type"].split("_"))
                n[f"err{camel}{short}"] = r[col]
            top = max(rows, key=lambda r, col=col: int(r[col]))
            n[f"errTopName{short}"] = top["error_type"].replace("_", " ")
            n[f"errTopN{short}"] = top[col]
        ambig = next(r for r in rows if r["error_type"] == "ambiguous_gold")
        n["errAmbiguousGoldTotal"] = ambig["total_primary"]

    # ---- in-text numbers -----------------------------------------------------------------

    def compute_numbers(self) -> None:
        n = self.numbers
        self._error_numbers()
        summary = self.manifest["summary"]
        n["nCalls"] = f"{sum(r['n'] for r in summary):,}"
        n["nParseFail"] = str(sum(r["parse_failure"] + r["refusal"] for r in summary))
        t0, t1 = self.manifest["started_utc"], self.manifest["finished_utc"]
        from datetime import datetime

        hours = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 3600
        n["runHours"] = f"{hours:.1f}"
        items = self.manifest["items"]
        for ds, short in (("dreaddit", "Dr"), ("tweet_eval_stance", "Te")):
            n[f"nEval{short}"] = f"{items[ds]['eval']:,}"
            n[f"nCal{short}"] = f"{items[ds]['calibration']:,}"
            enc = self.agree(ds, ENCODER, "supervised", "cohen_kappa")
            n[f"kappaFt{short}"] = f"{enc[0]:.2f}"
            seeds = [
                float(r["cohen_kappa"])
                for r in self.baseline_metrics
                if r["dataset"] == ds and r["model"] == ENCODER and r["role"] == "eval"
            ]
            if len(seeds) > 1:
                mean = sum(seeds) / len(seeds)
                sd = (sum((x - mean) ** 2 for x in seeds) / (len(seeds) - 1)) ** 0.5
                n[f"kappaFtSeeds{short}"] = f"{mean:.2f} $\\pm$ {sd:.2f}"
            best = max(
                ((m, c) for m in LLMS for c in CONDITIONS),
                key=lambda mc, ds=ds: self.agree(ds, *mc, "cohen_kappa")[0],
            )
            k = self.agree(ds, *best, "cohen_kappa")
            n[f"kappaBest{short}"] = f"{k[0]:.2f}"
            n[f"kappaBestCI{short}"] = f"[{k[1]:.2f}, {k[2]:.2f}]"
            n[f"bestModel{short}"] = MODEL_NAME[best[0]]
            n[f"bestCond{short}"] = COND_NAME[best[1]]
            zs = [self.agree(ds, m, "zero_shot", "cohen_kappa")[0] for m in LLMS]
            lo_s, hi_s = f"{min(zs):.2f}", f"{max(zs):.2f}"
            n[f"kappaZsRange{short}"] = lo_s if lo_s == hi_s else f"{lo_s}--{hi_s}"
            # Conformal: best (fewest routed) LLM probability-based cell at 90%.
            conf = [
                r
                for r in self.res["conformal"]
                if r["dataset"] == ds
                and r["method"] == "LAC"
                and abs(r["target_coverage"] - 0.9) < 1e-9
                and r["model"] in LLMS
                and r["signal"] == "logprob"
            ]
            c = min(conf, key=lambda r: r["pct_routed_to_human"])
            n[f"cpModel{short}"] = MODEL_NAME[c["model"]]
            n[f"cpRouted{short}"] = f"{c['pct_routed_to_human']:.0f}"
            n[f"cpAutoErr{short}"] = f"{100 * c['auto_label_error_rate']:.1f}"
            n[f"cpAllErr{short}"] = f"{100 * c['error_rate_if_all_auto']:.1f}"
            n[f"cpCov{short}"] = f"{c['empirical_coverage']:.3f}"
        sc_routed = [
            r["pct_routed_to_human"]
            for r in self.res["conformal"]
            if r["signal"] == "self_consistency" and r["method"] == "LAC"
        ]
        n["scRoutedMin"] = f"{min(sc_routed):.0f}"
        verb_te = [
            r["pct_routed_to_human"]
            for r in self.res["conformal"]
            if r["signal"] == "verbalized"
            and r["dataset"] == "tweet_eval_stance"
            and r["model"] in LLMS
        ]
        n["verbRoutedTeMin"] = f"{min(verb_te):.0f}"
        for ds, short in (("dreaddit", "Dr"), ("tweet_eval_stance", "Te")):
            ft = next(
                r
                for r in self.res["conformal"]
                if r["dataset"] == ds
                and r["model"] == ENCODER
                and r["method"] == "LAC"
                and abs(r["target_coverage"] - 0.9) < 1e-9
            )
            n[f"cpFtRouted{short}"] = f"{ft['pct_routed_to_human']:.0f}"
            n[f"cpFtAutoErr{short}"] = f"{100 * ft['auto_label_error_rate']:.1f}"
        self._significance_numbers()
        self._subgroup_numbers()
        cal = [r for r in self.res["calibration"] if r["condition"] == "codebook"]
        for sig, short in (("verbalized", "Verb"), ("logprob", "Lp"), ("self_consistency", "Sc")):
            vals = [_ci(r["auroc"])[0] for r in cal if r["signal"] == sig]
            n[f"auroc{short}Min"] = f"{min(vals):.2f}"
            n[f"auroc{short}Max"] = f"{max(vals):.2f}"
        nd = [r["n_distinct_values"] for r in cal if r["signal"] == "verbalized"]
        n["verbDistinctMax"] = str(max(nd))
        conf_ok = [r["coverage_ok"] for r in self.res["conformal"]]
        n["nConfCells"] = str(len(conf_ok))
        n["nConfOk"] = str(sum(conf_ok))
        pert = self.res.get("perturbation")
        if isinstance(pert, list):
            for ptype, short in (
                ("gender_swap", "Gender"),
                ("typo", "Typo"),
                ("paraphrase", "Para"),
            ):
                vals = [_ci(r["flip_rate"])[0] for r in pert if r["perturbation"] == ptype]
                n[f"flip{short}Min"] = f"{100 * min(vals):.1f}"
                n[f"flip{short}Max"] = f"{100 * max(vals):.1f}"

    @staticmethod
    def _p(p: float) -> str:
        """Relation and value for use as `$p\\macro$`, e.g. "<0.001" or "=0.046"."""
        if p < 0.001:
            return "<0.001"
        return f"={p:.3f}" if p < 0.1 else f"={p:.2f}"

    def _significance_numbers(self) -> None:
        n = self.numbers
        sig = {(r["dataset"], r["a"], r["b"]): r for r in self.res["significance"]}

        def get(ds: str, a: str, b: str) -> dict:
            return sig.get((ds, a, b)) or sig[(ds, b, a)]

        ft = f"{ENCODER}/supervised"
        zs_vs_ft = []
        for ds, short in (("dreaddit", "Dr"), ("tweet_eval_stance", "Te")):
            best = max(
                ((m, c) for m in LLMS for c in CONDITIONS),
                key=lambda mc, ds=ds: self.agree(ds, *mc, "cohen_kappa")[0],
            )
            r = get(ds, f"{best[0]}/{best[1]}", ft)
            n[f"pBestVsFt{short}"] = self._p(r["p_holm"])
            n[f"accBest{short}"] = f"{100 * r['acc_a']:.1f}"
            n[f"accFt{short}"] = f"{100 * r['acc_b']:.1f}"
            zs_vs_ft += [get(ds, f"{m}/zero_shot", ft)["p_holm"] for m in LLMS]
            r = get(ds, f"{LLMS[0]}/codebook", f"{LLMS[1]}/codebook")
            n[f"pLlamaQwenCb{short}"] = self._p(r["p_holm"])
        n["pZsVsFtMax"] = self._p(max(zs_vs_ft))
        zs_cb = [
            get("tweet_eval_stance", f"{m}/zero_shot", f"{m}/codebook")["p_holm"] for m in LLMS
        ]
        n["pZsCbTeMax"] = self._p(max(zs_cb))
        n["nSigTests"] = str(len(self.res["significance"]))

    def _subgroup_numbers(self) -> None:
        n = self.numbers
        seen = set()
        for r in self.res["subgroups"]:
            key = (r["dataset"], r["view"], r["model"])
            if key in seen or r["view"] == "subreddit":
                continue
            seen.add(key)
            ds = "Dr" if r["dataset"] == "dreaddit" else "Te"
            m = {"llama3.1-8b": "Llama", "qwen2.5-7b": "Qwen", ENCODER: "Ft"}[r["model"]]
            n[f"sgGap{m}{ds}"] = f"{r['max_f1_gap']:.2f}"
            n[f"sgP{m}{ds}"] = self._p(r["chi2_p_error_rates"])

    def numbers_tex(self) -> str:
        lines = ["% Generated by `raterbench report-assets`; do not edit."]
        for k, v in sorted(self.numbers.items()):
            lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
        return "\n".join(lines) + "\n"

    def readme_table(self) -> str:
        lines = [
            "| Model | Prompt | Dreaddit κ [95% CI] | TweetEval κ [95% CI] |",
            "|---|---|---|---|",
        ]
        for model, cond in self.rows():
            cells = []
            for ds, _ in DATASETS:
                k, lo, hi = self.agree(ds, model, cond, "cohen_kappa")
                cells.append(f"{k:.2f} [{lo:.2f}, {hi:.2f}]")
            lines.append(f"| {MODEL_NAME[model]} | {COND_NAME[cond]} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def write_assets(
    analysis_dir: Path, run_dir: Path, baselines_dir: Path, out_dir: Path
) -> dict[str, Any]:
    a = Assets(analysis_dir, run_dir, baselines_dir)
    a.compute_numbers()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "tab_main.tex": a.main_table(),
        "tab_calibration.tex": a.calibration_table(),
        "tab_conformal.tex": a.conformal_table(),
        "tab_perturbation.tex": a.perturbation_table(),
        "numbers.tex": a.numbers_tex(),
    }
    err = a.error_table()
    files["tab_errors.tex"] = err or "% error coding not done yet"
    files["error_flags.tex"] = (
        "% error coding done" if err else r"\newcommand{\errorcodingpending}{}"
    )
    for name, body in files.items():
        (out_dir / name).write_text(body + "\n", encoding="utf-8")
    (analysis_dir / "readme_main_table.md").write_text(a.readme_table(), encoding="utf-8")
    return {"files": sorted(files), "numbers": a.numbers, "error_coding_done": bool(err)}
