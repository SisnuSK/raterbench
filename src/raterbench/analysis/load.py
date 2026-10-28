"""Assemble predictions from LLM runs and baselines into aligned numeric arrays."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from raterbench.data import registry
from raterbench.data.dreaddit import DOMAINS

Key = tuple[str, str, str, str]  # (dataset, model, condition, role)


@dataclass
class PredSet:
    dataset: str
    model: str
    condition: str
    role: str
    labels: tuple[str, ...]
    item_ids: list[str]
    gold: np.ndarray  # int codes
    pred: np.ndarray  # int codes; len(labels) = invalid (parse failure / refusal / error)
    subgroup: list[str]
    records: list[dict]
    # Confidence that the prediction is correct, per signal, in [0, 1] (NaN if unavailable).
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-item probability vectors over labels, per signal (rows sum to 1), for conformal.
    probs: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.labels)

    @property
    def correct(self) -> np.ndarray:
        return self.pred == self.gold

    def domain(self) -> list[str]:
        return [DOMAINS.get(s, s) for s in self.subgroup]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _spread(chosen: np.ndarray, conf: np.ndarray, k: int) -> np.ndarray:
    """Probability vector from a single confidence: p(chosen) = c, the rest split evenly.

    Used for signals that only give the chosen label's probability (verbalized confidence,
    label log-probability). Invalid predictions or missing confidence get a uniform vector.
    """
    n = len(chosen)
    out = np.full((n, k), 1.0 / k)
    ok = (chosen < k) & ~np.isnan(conf)
    c = np.clip(conf[ok], 0.0, 1.0)
    rows = np.where(ok)[0]
    out[rows] = ((1 - c) / (k - 1))[:, None]
    out[rows, chosen[ok]] = c
    return out


def _build(recs: list[dict], labels: tuple[str, ...]) -> PredSet:
    recs = sorted(recs, key=lambda r: r["item_id"])
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    r0 = recs[0]
    return PredSet(
        dataset=r0["dataset"],
        model=r0["model"],
        condition=r0["condition"],
        role=r0["role"],
        labels=labels,
        item_ids=[r["item_id"] for r in recs],
        gold=np.array([idx[r["gold"]] for r in recs]),
        pred=np.array([idx.get(r.get("label"), k) if r.get("label") else k for r in recs]),
        subgroup=[r["subgroup"] for r in recs],
        records=recs,
    )


def load_llm_run(pred_path: Path) -> dict[Key, PredSet]:
    rows = read_jsonl(pred_path)
    main: dict[Key, list[dict]] = defaultdict(list)
    sc: dict[Key, dict[str, list[str | None]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["dataset"], r["model"], r["condition"], r["role"])
        if r["sample_idx"] == 0:
            main[key].append(r)
        else:
            sc[key][r["item_id"]].append(r.get("label"))
    out = {}
    for key, recs in main.items():
        labels = registry.get(key[0]).labels
        ps = _build(recs, labels)
        k = ps.k
        verbal = np.array(
            [r["confidence"] / 100 if r.get("outcome") == "ok" else np.nan for r in ps.records],
            dtype=float,
        )
        ps.signals["verbalized"] = verbal
        ps.probs["verbalized"] = _spread(ps.pred, verbal, k)
        lp = np.array(
            [
                r.get("label_prob") if r.get("label_prob") is not None else np.nan
                for r in ps.records
            ],
            dtype=float,
        )
        if not np.all(np.isnan(lp)):
            ps.signals["logprob"] = lp
            ps.probs["logprob"] = _spread(ps.pred, lp, k)
        if key in sc:
            samples = sc[key]
            agree, freq = [], np.zeros((len(ps.item_ids), k))
            for i, iid in enumerate(ps.item_ids):
                s = samples.get(iid, [])
                main_label = ps.records[i].get("label")
                agree.append(np.mean([x == main_label for x in s]) if s and main_label else np.nan)
                valid = [labels.index(x) for x in s if x in labels]
                if valid:
                    freq[i] = np.bincount(valid, minlength=k) / len(valid)
                else:
                    freq[i] = 1.0 / k
            ps.signals["self_consistency"] = np.array(agree, dtype=float)
            ps.probs["self_consistency"] = freq
        out[key] = ps
    return out


def load_baselines(pred_dir: Path, encoder_seed: int) -> dict[Key, PredSet]:
    out = {}
    for ds_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        labels = registry.get(ds_dir.name).labels
        files = [ds_dir / "majority.jsonl", *sorted(ds_dir.glob(f"*_seed{encoder_seed}.jsonl"))]
        for f in files:
            by_role: dict[str, list[dict]] = defaultdict(list)
            for r in read_jsonl(f):
                by_role[r["role"]].append(r)
            for role, recs in by_role.items():
                ps = _build(recs, labels)
                probs = np.array([[r["probs"][lab] for lab in labels] for r in ps.records])
                ps.probs["softmax"] = probs
                ps.signals["softmax"] = probs[np.arange(len(probs)), np.minimum(ps.pred, ps.k - 1)]
                out[(ps.dataset, ps.model, ps.condition, role)] = ps
    return out
