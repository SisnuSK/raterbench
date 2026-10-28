import asyncio
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from fakes import FakeBackend, no_sleep
from raterbench.config import CalibrationConfig, DatasetConfig, RunConfig
from raterbench.data import registry
from raterbench.data.schema import Item
from raterbench.data.splits import build_splits, write_samples
from raterbench.prompting.render import CONDITIONS, PromptBuilder, select_fewshot
from raterbench.runner import Runner

REPO = Path(__file__).resolve().parents[1]
TOY_LABELS = ("yes", "no")


def toy_items() -> list[Item]:
    items = []
    for split in ("train", "test"):
        for sg in ("a", "b"):
            for lab in TOY_LABELS:
                for i in range(10):
                    items.append(
                        Item(
                            id=f"{split}-{sg}-{lab}-{i}",
                            text=f"{lab} text {i} in {sg}",
                            label=lab,
                            subgroup=sg,
                            split=split,
                        )
                    )
    return items


@pytest.fixture
def toy_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    items = toy_items()
    monkeypatch.setitem(
        registry.DATASETS,
        "toy",
        registry.DatasetInfo(
            name="toy",
            labels=TOY_LABELS,
            subgroup_name="sg",
            load=lambda _: items,
            files=[],
            documented={},
            documented_source="-",
        ),
    )
    ds_cfg = DatasetConfig(
        name="toy", eval_size=20, calibration=CalibrationConfig(source="test_remainder")
    )
    write_samples(build_splits(items, ds_cfg), tmp_path / "samples" / "toy.jsonl")
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump(ds_cfg.model_dump()))
    (tmp_path / "main.yaml").write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tmp_path / "raw"),
                "samples_dir": str(tmp_path / "samples"),
                "datasets": [str(tmp_path / "toy.yaml")],
            }
        )
    )
    prompts = tmp_path / "prompts"
    shutil.copytree(REPO / "prompts" / "templates", prompts / "templates")
    (prompts / "tasks").mkdir()
    (prompts / "codebooks").mkdir()
    (prompts / "tasks" / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": "toy",
                "description": "Is it yes?",
                "labels": [{"name": "yes", "definition": "y"}, {"name": "no", "definition": "n"}],
            }
        )
    )
    (prompts / "codebooks" / "toy.md").write_text("Rule: label yes if the text says yes.")
    (tmp_path / "model.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "fake",
                "backend": "fake",
                "model": "fake-1",
                "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 10.0},
                "concurrency": 3,
            }
        )
    )
    return tmp_path


def run_cfg(env: Path, **kw) -> RunConfig:
    base = dict(
        name="t",
        main=env / "main.yaml",
        models=[env / "model.yaml"],
        conditions=["zero_shot", "few_shot", "codebook", "codebook_reasoning"],
        max_items_per_role=8,
        budget_usd=10.0,
        cache_path=env / "cache.sqlite",
        output_dir=env / "runs",
        prompts_dir=env / "prompts",
        self_consistency={"n": 3, "temperature": 0.7, "conditions": ["codebook"]},
    )
    return RunConfig(**{**base, **kw})


def smart_reply(req) -> str:
    """Answers correctly; returns broken output on the first try for texts ending in 3."""
    content = req.messages[0].content
    text = content.split("<text>")[-1]
    label = "yes" if "yes text" in text else "no"
    if " 3 in" in text and len(req.messages) == 1:
        return "I am not sure."
    return json.dumps({"reasoning": "r", "label": label, "confidence": 77})


def test_runner_end_to_end_and_resume_from_cache(toy_env: Path) -> None:
    backend = FakeBackend(reply=smart_reply)
    cfg = run_cfg(toy_env)
    run_dir = asyncio.run(Runner(cfg, backends={"fake": backend}, sleep=no_sleep).run())
    recs = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    # 8 items x (3 single-sample conditions + codebook with 1 + 3 samples)
    assert len(recs) == 8 * (3 + 4)
    assert Counter(r["outcome"] for r in recs) == {"ok": len(recs)}
    assert all(r["label"] == r["gold"] for r in recs)
    retried = [r for r in recs if r["retried"]]
    assert retried and all(" 3 in" in r["item_id"].replace("-3", " 3 in") for r in retried)
    assert {r["temperature"] for r in recs if r["sample_idx"] == 0} == {0.0}
    assert {r["temperature"] for r in recs if r["sample_idx"] > 0} == {0.7}
    first_calls = len(backend.calls)
    assert first_calls == len(recs) + len(retried)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["cost"]["spent_usd"] > 0
    assert set(manifest["prompt_hashes"]["toy"]) == set(cfg.conditions)

    # Second run: everything comes from the cache, no new calls, zero cost.
    run_dir2 = asyncio.run(Runner(cfg, backends={"fake": backend}, sleep=no_sleep).run())
    assert len(backend.calls) == first_calls
    m2 = json.loads((run_dir2 / "run_manifest.json").read_text())
    assert m2["cost"]["spent_usd"] == 0
    recs2 = [json.loads(line) for line in (run_dir2 / "predictions.jsonl").read_text().splitlines()]
    assert all(r["cached"] for r in recs2)
    assert [r["label"] for r in recs2] == [r["label"] for r in recs]


def test_runner_records_parse_failures_after_one_retry(toy_env: Path) -> None:
    backend = FakeBackend(reply=lambda req: "no json here")
    run_dir = asyncio.run(
        Runner(
            run_cfg(toy_env, conditions=["zero_shot"], self_consistency={"n": 0}),
            backends={"fake": backend},
            sleep=no_sleep,
        ).run()
    )
    recs = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert {r["outcome"] for r in recs} == {"parse_failure"}
    assert all(r["retried"] for r in recs)
    assert len(backend.calls) == 2 * len(recs)
    summary = json.loads((run_dir / "run_manifest.json").read_text())["summary"]
    assert summary[0]["parse_failure_rate"] == 1.0


def test_runner_halts_at_budget(toy_env: Path) -> None:
    backend = FakeBackend(reply=smart_reply)
    cfg = run_cfg(toy_env, conditions=["zero_shot"], self_consistency={"n": 0}, budget_usd=0.006)
    run_dir = asyncio.run(Runner(cfg, backends={"fake": backend}, sleep=no_sleep).run())
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["cost"]["halted_by_budget"]
    assert m["cost"]["spent_usd"] <= 0.006
    outcomes = Counter(
        json.loads(x)["outcome"] for x in (run_dir / "predictions.jsonl").read_text().splitlines()
    )
    assert outcomes["skipped_budget"] > 0


def test_fewshot_balanced_deterministic_and_train_only() -> None:
    pool = [it for it in toy_items() if it.split == "train"]
    a = select_fewshot(pool, TOY_LABELS, 5, seed=1, by_subgroup=True)
    b = select_fewshot(pool, TOY_LABELS, 5, seed=1, by_subgroup=True)
    assert {k: [i.id for i in v] for k, v in a.items()} == {
        k: [i.id for i in v] for k, v in b.items()
    }
    for sg, exs in a.items():
        assert len(exs) == 5
        assert all(e.subgroup == sg and e.split == "train" for e in exs)
        assert sorted(Counter(e.label for e in exs).values()) == [2, 3]


@pytest.mark.parametrize("dataset", ["dreaddit", "tweet_eval_stance"])
def test_real_prompts_render_all_conditions(dataset: str) -> None:
    info = registry.get(dataset)
    pool = [
        Item(
            id=f"tr-{lab}-{i}",
            text=f"FEWSHOTEXAMPLE {lab} {i}",
            label=lab,
            subgroup="abortion" if dataset != "dreaddit" else "ptsd",
            split="train",
        )
        for lab in info.labels
        for i in range(3)
    ]
    b = PromptBuilder(dataset, REPO / "prompts", pool, 5, 13)
    item = Item(
        id="x", text="THE ITEM TEXT", label=info.labels[0], subgroup=pool[0].subgroup, split="test"
    )
    hashes = set()
    for cond in CONDITIONS:
        p = b.render(cond, item)
        assert "THE ITEM TEXT" in p
        assert all(f'"{lab}"' in p for lab in info.labels)
        assert ("FEWSHOTEXAMPLE" in p) == (cond == "few_shot")
        assert ("<codebook>" in p) == cond.startswith("codebook")
        assert ('"reasoning"' in p) == (cond == "codebook_reasoning")
        hashes.add(b.prompt_hash(cond))
    assert len(hashes) == len(CONDITIONS)
    if dataset == "tweet_eval_stance":
        assert "Legalization of Abortion" in b.render("zero_shot", item)
