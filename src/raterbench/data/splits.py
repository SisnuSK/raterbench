"""Build the fixed evaluation / calibration / training roles and persist them as sample IDs.

Roles:
- eval: items every model is scored on.
- calibration: held-out labeled items for split conformal prediction (never in eval).
- train: pool for few-shot examples and supervised baselines (never in eval or calibration).
- val: the published validation split, if any (for baseline model selection).
"""

import json
from pathlib import Path

from raterbench.config import DatasetConfig
from raterbench.data.sampling import stratified_sample
from raterbench.data.schema import Item

ROLES = ("eval", "calibration", "train", "val")


def _stratum(it: Item) -> tuple[str, str]:
    return (it.subgroup, it.label)


def _without(items: list[Item], drop: list[Item]) -> list[Item]:
    ids = {it.id for it in drop}
    return [it for it in items if it.id not in ids]


def build_splits(items: list[Item], cfg: DatasetConfig) -> dict[str, list[Item]]:
    by_split = {s: [it for it in items if it.split == s] for s in ("train", "val", "test")}
    test, train = by_split["test"], by_split["train"]
    if cfg.eval_size > len(test):
        raise ValueError(
            f"{cfg.name}: eval_size={cfg.eval_size} exceeds the {len(test)} test items available"
        )

    eval_items = stratified_sample(test, cfg.eval_size, _stratum, seed=cfg.seed)
    cal = cfg.calibration
    if cal.source == "test_remainder":
        rest = _without(test, eval_items)
        size = len(rest) if cal.size is None else cal.size
        calibration = stratified_sample(rest, size, _stratum, seed=cfg.seed + 1)
    else:
        if cal.size is None:
            raise ValueError(f"{cfg.name}: calibration.size is required for train_holdout")
        calibration = stratified_sample(train, cal.size, _stratum, seed=cfg.seed + 1)
        train = _without(train, calibration)

    return {"eval": eval_items, "calibration": calibration, "train": train, "val": by_split["val"]}


def write_samples(splits: dict[str, list[Item]], path: Path) -> None:
    """Save item IDs per role (plus a text hash for integrity; no raw text or labels)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for role in ROLES:
            for it in sorted(splits[role], key=lambda x: x.id):
                rec = {
                    "id": it.id,
                    "role": role,
                    "source_split": it.split,
                    "subgroup": it.subgroup,
                    "text_sha256": it.text_sha256,
                }
                f.write(json.dumps(rec) + "\n")


def read_samples(items: list[Item], path: Path) -> dict[str, list[Item]]:
    """Resolve a saved sample file against loaded items, verifying text hashes."""
    by_id = {it.id: it for it in items}
    splits: dict[str, list[Item]] = {r: [] for r in ROLES}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            it = by_id.get(rec["id"])
            if it is None:
                raise ValueError(f"{path}: id {rec['id']} not found in loaded data")
            if it.text_sha256 != rec["text_sha256"]:
                raise ValueError(f"{path}: text hash mismatch for id {rec['id']}")
            splits[rec["role"]].append(it)
    return splits
