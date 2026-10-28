from collections import Counter
from pathlib import Path

import pytest

from raterbench.config import CalibrationConfig, DatasetConfig
from raterbench.data import registry
from raterbench.data.sampling import allocate, stratified_sample
from raterbench.data.schema import Item
from raterbench.data.splits import build_splits, read_samples, write_samples

RAW_DIR = Path("data/raw")


def make_items(n_per_stratum: int = 20, group_size: int = 1) -> list[Item]:
    items = []
    for split in ("train", "test"):
        for sg in ("a", "b"):
            for label in ("x", "y"):
                for i in range(n_per_stratum):
                    items.append(
                        Item(
                            id=f"{split}-{sg}-{label}-{i}",
                            text=f"text {split} {sg} {label} {i}",
                            label=label,
                            subgroup=sg,
                            split=split,
                            group=f"{split}-{sg}-{label}-g{i // group_size}"
                            if group_size > 1
                            else None,
                        )
                    )
    return items


def test_allocate_largest_remainder() -> None:
    assert allocate({"a": 50, "b": 30, "c": 20}, 10) == {"a": 5, "b": 3, "c": 2}
    # exact 3.5 / 2.1 / 1.4 -> floors 3/2/1, the leftover draw goes to the largest remainder
    assert allocate({"a": 50, "b": 30, "c": 20}, 7) == {"a": 4, "b": 2, "c": 1}
    assert sum(allocate({"a": 3, "b": 3, "c": 3}, 4).values()) == 4


def test_allocate_rejects_oversize() -> None:
    with pytest.raises(ValueError):
        allocate({"a": 2}, 3)


def test_stratified_sample_exact_and_deterministic() -> None:
    items = [it for it in make_items() if it.split == "test"]

    def key(it: Item) -> tuple[str, str]:
        return (it.subgroup, it.label)

    s1 = stratified_sample(items, 40, key, seed=1)
    s2 = stratified_sample(items, 40, key, seed=1)
    s3 = stratified_sample(items, 40, key, seed=2)
    assert [it.id for it in s1] == [it.id for it in s2]
    assert {it.id for it in s1} != {it.id for it in s3}
    assert len(s1) == 40
    assert set(Counter(key(it) for it in s1).values()) == {10}


def test_stratified_sample_keeps_groups_together() -> None:
    items = [it for it in make_items(group_size=4) if it.split == "train"]
    sample = stratified_sample(items, 30, lambda it: (it.subgroup, it.label), seed=0)
    chosen_groups = {it.group for it in sample}
    for it in items:
        assert (it.group in chosen_groups) == (it in sample)


@pytest.mark.parametrize("source", ["test_remainder", "train_holdout"])
def test_build_splits_roles_are_disjoint(source: str) -> None:
    items = make_items()
    cfg = DatasetConfig(
        name="toy",
        eval_size=60,
        calibration=CalibrationConfig(
            source=source, size=None if source == "test_remainder" else 20
        ),
    )
    roles = build_splits(items, cfg)
    ids = {r: {it.id for it in v} for r, v in roles.items()}
    assert not ids["eval"] & ids["calibration"]
    assert not ids["eval"] & ids["train"]
    assert not ids["calibration"] & ids["train"]
    assert all(it.split == "test" for it in roles["eval"])
    assert len(roles["eval"]) == 60
    if source == "test_remainder":
        assert len(roles["calibration"]) == 20
        assert len(roles["train"]) == 80
    else:
        assert len(roles["calibration"]) == 20
        assert len(roles["train"]) == 60


def test_build_splits_rejects_eval_larger_than_test() -> None:
    cfg = DatasetConfig(
        name="toy", eval_size=81, calibration=CalibrationConfig(source="test_remainder")
    )
    with pytest.raises(ValueError, match="exceeds"):
        build_splits(make_items(), cfg)


def test_samples_round_trip_and_hash_check(tmp_path: Path) -> None:
    items = make_items()
    cfg = DatasetConfig(
        name="toy", eval_size=40, calibration=CalibrationConfig(source="test_remainder", size=20)
    )
    roles = build_splits(items, cfg)
    path = tmp_path / "toy.jsonl"
    write_samples(roles, path)
    assert "text" not in path.read_text().split("\n")[0].replace("text_sha256", "")
    back = read_samples(items, path)
    for role, v in roles.items():
        assert sorted(it.id for it in back[role]) == sorted(it.id for it in v)

    items[0].text = "tampered"
    tampered_id = items[0].id
    if any(it.id == tampered_id for v in roles.values() for it in v):
        with pytest.raises(ValueError, match="hash mismatch"):
            read_samples(items, path)


def test_check_documented_reports_mismatch() -> None:
    info = registry.DatasetInfo(
        name="toy",
        labels=("x", "y"),
        subgroup_name="sg",
        load=lambda _: [],
        files=[],
        documented={("test", "*"): 80, ("all", "a"): 81},
        documented_source="toy",
    )
    problems = registry.check_documented(info, make_items())
    assert problems == ["all/a: documented 81, loaded 80"]


@pytest.mark.parametrize("name", sorted(registry.DATASETS))
def test_real_data_matches_documented(name: str) -> None:
    info = registry.get(name)
    if not (RAW_DIR / name).exists():
        pytest.skip(f"raw data for {name} not downloaded")
    items = info.load(RAW_DIR)
    assert registry.check_documented(info, items) == []
    assert len({it.id for it in items}) == len(items)
    assert {it.label for it in items} == set(info.labels)


@pytest.mark.parametrize("name", sorted(registry.DATASETS))
def test_committed_samples_resolve(name: str) -> None:
    path = Path("data/samples") / f"{name}.jsonl"
    if not (RAW_DIR / name).exists() or not path.exists():
        pytest.skip("raw data or sample file missing")
    roles = read_samples(registry.get(name).load(RAW_DIR), path)
    ids = [it.id for v in roles.values() for it in v]
    assert len(ids) == len(set(ids))
