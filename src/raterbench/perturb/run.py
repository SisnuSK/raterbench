"""Generate perturbations for a fixed subset of eval items and measure label flip rates.

The original-text predictions come from the main run (identical prompts hit the response
cache). Perturbed texts are written under data/perturbations/ (git-ignored, since they are
derived from the raw datasets); only IDs, edit counts, and predictions go to results/.
"""

import asyncio
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from raterbench.config import ModelConfig, load_yaml
from raterbench.data.sampling import stratified_sample
from raterbench.data.schema import Item
from raterbench.llm.cache import ResponseCache
from raterbench.llm.cost import CostTracker
from raterbench.llm.types import LLMRequest, Message
from raterbench.perturb.edits import gender_swap, typo_noise
from raterbench.runner import Job, Runner, load_run_config

PARAPHRASE_SYSTEM = (
    "You rewrite text for a research study. Paraphrase the text so it keeps exactly the same "
    "meaning, tone, stance, and level of emotion, and the same point of view (first person "
    "stays first person). Keep hashtags, @mentions, and URLs unchanged. Do not add or remove "
    "information. Output only the rewritten text."
)


class PerturbConfig(BaseModel):
    run_config: Path = Path("configs/full_local.yaml")  # models, prompts, cache, seed
    items_per_dataset: int = 200
    seed: int = 13
    conditions: list[str] = Field(default_factory=lambda: ["zero_shot", "codebook"])
    types: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "dreaddit": ["gender_swap", "typo", "paraphrase"],
            "tweet_eval_stance": ["typo", "paraphrase"],
        }
    )
    typo_rate: float = 0.1
    paraphrase_model: Path = Path("configs/models/qwen2.5-7b.yaml")
    text_dir: Path = Path("data/perturbations")
    output_dir: Path = Path("results/perturbation")
    spot_check_per_type: int = 15


def _clean_paraphrase(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(here is|here's)[^:\n]*:\s*", "", text, flags=re.I)
    text = re.sub(r"^(paraphrase|rewritten text)\s*:\s*", "", text, flags=re.I)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


async def _paraphrase(
    items: list[Item], mc: ModelConfig, cache: ResponseCache, tracker: CostTracker, runner: Runner
) -> dict[str, str]:
    client = runner._client(mc, cache, tracker)

    async def one(it: Item) -> tuple[str, str]:
        req = LLMRequest(
            system=PARAPHRASE_SYSTEM,
            messages=[Message(role="user", content=it.text)],
            temperature=0.0 if client.supports_temperature else None,
            seed=13,
        )
        resp, _, _ = await client.complete(req)
        return it.id, _clean_paraphrase(resp.text)

    return dict(await asyncio.gather(*(one(it) for it in items)))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def run_perturbation(cfg: PerturbConfig) -> Path:
    run_cfg = load_run_config(cfg.run_config)
    runner = Runner(run_cfg)
    cache = ResponseCache(run_cfg.cache_path)
    tracker = CostTracker(budget_usd=run_cfg.budget_usd)
    para_mc = load_yaml(cfg.paraphrase_model, ModelConfig)

    perturbed: list[dict[str, Any]] = []  # one per (item, type) that changed the text
    subsets: dict[str, list[Item]] = {}
    try:
        for ds, roles in runner.items.items():
            subset = stratified_sample(
                roles["eval"], cfg.items_per_dataset, lambda it: (it.subgroup, it.label), cfg.seed
            )
            subset = sorted(subset, key=lambda it: it.id)
            subsets[ds] = subset
            types = cfg.types.get(ds, [])
            paras = (
                await _paraphrase(subset, para_mc, cache, tracker, runner)
                if "paraphrase" in types
                else {}
            )
            for it in subset:
                for t in types:
                    if t == "gender_swap":
                        new, n = gender_swap(it.text)
                    elif t == "typo":
                        new, n = typo_noise(it.text, seed=f"{cfg.seed}:{it.id}", rate=cfg.typo_rate)
                    elif t == "paraphrase":
                        new = paras.get(it.id, "")
                        n = int(bool(new) and new != it.text)
                    else:
                        raise ValueError(f"unknown perturbation {t!r}")
                    if n == 0 or not new.strip():
                        continue
                    perturbed.append(
                        {"dataset": ds, "type": t, "original": it, "text": new, "n_edits": n}
                    )

        cfg.text_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.text_dir / "perturbations.jsonl").open("w", encoding="utf-8") as f:
            for p in perturbed:
                f.write(
                    json.dumps(
                        {
                            "dataset": p["dataset"],
                            "type": p["type"],
                            "item_id": p["original"].id,
                            "n_edits": p["n_edits"],
                            "original": p["original"].text,
                            "perturbed": p["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _write_spot_check(perturbed, cfg)

        # Label every original and perturbed text with each model x condition (T=0).
        records = []
        for mc in runner.model_cfgs:
            client = runner._client(mc, cache, tracker)
            jobs = []
            for ds, subset in subsets.items():
                for cond in cfg.conditions:
                    jobs += [(Job(ds, "eval", it, cond, 0), None) for it in subset]
            for p in perturbed:
                pit = Item(
                    id=f"{p['original'].id}::{p['type']}",
                    text=p["text"],
                    label=p["original"].label,
                    subgroup=p["original"].subgroup,
                    split=p["original"].split,
                )
                for cond in cfg.conditions:
                    jobs.append((Job(p["dataset"], "eval", pit, cond, 0), p))
            print(f"perturbation: {mc.name}: {len(jobs)} jobs", flush=True)
            results = await asyncio.gather(*(runner._run_job(j, mc, client) for j, _ in jobs))
            for (job, p), rec in zip(jobs, results, strict=True):
                rec.pop("raw_text", None)
                rec["perturbation"] = p["type"] if p else "original"
                rec["original_id"] = p["original"].id if p else job.item.id
                rec["n_edits"] = p["n_edits"] if p else 0
                records.append(rec)
    finally:
        cache.close()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "config": cfg.model_dump(mode="json"),
        "subset_ids": {ds: [it.id for it in s] for ds, s in subsets.items()},
        "n_perturbed": {f"{p['dataset']}/{p['type']}": 0 for p in perturbed},
        "cost": tracker.summary(),
    }
    for p in perturbed:
        meta["n_perturbed"][f"{p['dataset']}/{p['type']}"] += 1
    (cfg.output_dir / "perturbation_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return cfg.output_dir


def _write_spot_check(perturbed: list[dict[str, Any]], cfg: PerturbConfig) -> None:
    """A random sample per (dataset, type) for manual review of meaning preservation."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in perturbed:
        by[(p["dataset"], p["type"])].append(p)
    lines = [
        "# Perturbation spot check",
        "",
        "For each pair, check that the edit preserves the meaning and the gold label.",
        "Mark problems inline; flip rates should be read in light of this review.",
        "",
    ]
    rng = random.Random(cfg.seed)
    for (ds, t), ps in sorted(by.items()):
        lines.append(f"## {ds} / {t} ({len(ps)} perturbed)")
        for p in rng.sample(ps, min(cfg.spot_check_per_type, len(ps))):
            o = p["original"]
            lines += [
                "",
                f"**{o.id}** (gold: {o.label}, edits: {p['n_edits']})",
                "",
                f"- original: {o.text.strip()}",
                f"- perturbed: {p['text'].strip()}",
                "- OK? [ ]",
            ]
        lines.append("")
    (cfg.text_dir / "spot_check.md").write_text("\n".join(lines), encoding="utf-8")
