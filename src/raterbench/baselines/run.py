"""Run the majority and fine-tuned encoder baselines and write predictions + metrics."""

import csv
import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from raterbench.baselines import majority
from raterbench.config import MainConfig, load_yaml
from raterbench.data import registry, splits
from raterbench.data.sampling import stratified_sample
from raterbench.data.schema import Item
from raterbench.metrics.agreement import agreement_metrics

METRIC_KEYS = ("accuracy", "macro_f1", "cohen_kappa", "krippendorff_alpha")


class BaselineConfig(BaseModel):
    main: Path = Path("configs/main.yaml")
    output_dir: Path = Path("results/baselines")
    run_encoder: bool = True
    seeds: list[int] = Field(default_factory=lambda: [13, 14, 15])
    dev_seed: int = 13
    dev_fraction: float = 0.1  # grouped hold-out from train when no validation split exists
    encoder: dict[str, Any] = Field(default_factory=dict)  # EncoderSettings fields
    max_length: dict[str, int] = Field(default_factory=dict)  # per-dataset override


def _records(
    ds: str,
    role: str,
    items: list[Item],
    preds: list[dict],
    model: str,
    condition: str,
    seed: int | None,
) -> list[dict]:
    by_id = {it.id: it for it in items}
    out = []
    for p in preds:
        it = by_id[p["item_id"]]
        out.append(
            {
                "dataset": ds,
                "role": role,
                "item_id": it.id,
                "subgroup": it.subgroup,
                "gold": it.label,
                "model": model,
                "condition": condition,
                "seed": seed,
                "sample_idx": 0,
                "outcome": "ok",
                **p,
            }
        )
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _score(rows: list[dict], labels: tuple[str, ...]) -> dict[str, Any]:
    return agreement_metrics([r["gold"] for r in rows], [r["label"] for r in rows], labels)


def dev_split(roles: dict[str, list[Item]], cfg: BaselineConfig) -> tuple[list[Item], list[Item]]:
    """Return (train, dev). Uses the published val split if present."""
    if roles["val"]:
        return roles["train"], roles["val"]
    n_dev = round(cfg.dev_fraction * len(roles["train"]))
    dev = stratified_sample(roles["train"], n_dev, lambda it: (it.subgroup, it.label), cfg.dev_seed)
    dev_ids = {it.id for it in dev}
    return [it for it in roles["train"] if it.id not in dev_ids], dev


def run_baselines(cfg: BaselineConfig, log=print) -> Path:
    main = load_yaml(cfg.main, MainConfig)
    out = cfg.output_dir
    started = datetime.now(UTC)
    metric_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    device = None
    if cfg.run_encoder:
        import torch

        from raterbench.baselines.encoder import EncoderSettings, train_encoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"encoder device: {device}")

    for ds in main.dataset_configs():
        info = registry.get(ds.name)
        labels = info.labels
        roles = splits.read_samples(info.load(main.raw_dir), main.samples_dir / f"{ds.name}.jsonl")
        targets = {"eval": roles["eval"], "calibration": roles["calibration"]}
        details[ds.name] = {}

        # Majority class (fit on the train role only).
        label, probs = majority.majority_label(roles["train"], labels)
        rows = []
        for role, items in targets.items():
            role_rows = _records(
                ds.name,
                role,
                items,
                majority.predict(items, label, probs),
                "majority",
                "baseline",
                None,
            )
            rows += role_rows
            m = _score(role_rows, labels)
            metric_rows.append(
                {
                    "dataset": ds.name,
                    "model": "majority",
                    "seed": "",
                    "role": role,
                    **{k: m[k] for k in ("n", *METRIC_KEYS)},
                }
            )
            details[ds.name][f"majority/{role}"] = m
        _write_jsonl(out / "predictions" / ds.name / "majority.jsonl", rows)
        log(f"{ds.name}: majority label = {label!r} (train share {probs[label]:.3f})")

        if not cfg.run_encoder:
            continue
        settings = EncoderSettings(**cfg.encoder)
        settings.max_length = cfg.max_length.get(ds.name, settings.max_length)
        train, dev = dev_split(roles, cfg)
        model_name = settings.model_name.split("/")[-1] + "-ft"
        details[ds.name]["encoder_setup"] = {
            "settings": asdict(settings),
            "n_train": len(train),
            "n_dev": len(dev),
            "dev_source": "val split"
            if roles["val"]
            else f"grouped {cfg.dev_fraction:.0%} of train",
        }
        for seed in cfg.seeds:
            log(
                f"{ds.name}: training {settings.model_name} seed={seed} "
                f"(train={len(train)}, dev={len(dev)})"
            )
            res = train_encoder(train, dev, targets, labels, settings, seed, device, log=log)
            rows = []
            for role, preds in res.predictions.items():
                items = dev if role == "dev" else targets[role]
                role_rows = _records(ds.name, role, items, preds, model_name, "supervised", seed)
                rows += role_rows
                m = _score(role_rows, labels)
                metric_rows.append(
                    {
                        "dataset": ds.name,
                        "model": model_name,
                        "seed": seed,
                        "role": role,
                        **{k: m[k] for k in ("n", *METRIC_KEYS)},
                    }
                )
                details[ds.name][f"{model_name}/seed{seed}/{role}"] = m
            details[ds.name][f"{model_name}/seed{seed}/training"] = {
                "history": res.history,
                "best_epoch": res.best_epoch,
                "seconds": round(res.seconds, 1),
                "model_revision": res.model_revision,
            }
            _write_jsonl(out / "predictions" / ds.name / f"{model_name}_seed{seed}.jsonl", rows)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        w.writeheader()
        w.writerows(metric_rows)
    (out / "metrics.md").write_text(_markdown(metric_rows), encoding="utf-8")
    (out / "metrics_detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    manifest = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "config": cfg.model_dump(mode="json"),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _versions(),
    }
    (out / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def _versions() -> dict[str, Any]:
    v: dict[str, Any] = {}
    try:
        import torch
        import transformers

        v.update(
            torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__
        )
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    import sklearn

    v["scikit_learn"] = sklearn.__version__
    return v


def _markdown(rows: list[dict[str, Any]]) -> str:
    """Eval-role table: majority row, encoder mean ± sd over seeds, then per-seed rows."""
    lines = [
        "Baselines on the eval role (point estimates; bootstrap CIs are added in Phase 4).",
        "",
        "| dataset | model | seeds | n | accuracy | macro-F1 | Cohen's kappa "
        "| Krippendorff's alpha |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ev = [r for r in rows if r["role"] == "eval"]
    for ds in dict.fromkeys(r["dataset"] for r in ev):
        for model in dict.fromkeys(r["model"] for r in ev if r["dataset"] == ds):
            rs = [r for r in ev if r["dataset"] == ds and r["model"] == model]
            cells = []
            for k in METRIC_KEYS:
                vals = [r[k] for r in rs]
                if len(vals) > 1:
                    cells.append(f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}")
                else:
                    cells.append(f"{vals[0]:.3f}")
            seeds = ",".join(str(r["seed"]) for r in rs if r["seed"] != "") or "-"
            lines.append(f"| {ds} | {model} | {seeds} | {rs[0]['n']} | " + " | ".join(cells) + " |")
    lines += ["", "Encoder rows show mean ± sample SD across seeds.", ""]
    return "\n".join(lines)
