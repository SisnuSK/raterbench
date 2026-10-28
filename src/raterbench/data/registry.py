"""Dataset registry, including the counts documented by each source for verification."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from raterbench.data import download, dreaddit, tweet_eval
from raterbench.data.schema import Item


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    labels: tuple[str, ...]
    subgroup_name: str
    load: Callable[[Path], list[Item]]
    files: list[download.RemoteFile]
    # Documented counts: {(split, subgroup): n}, with subgroup "*" meaning all subgroups.
    documented: dict[tuple[str, str], int]
    documented_source: str


DATASETS: dict[str, DatasetInfo] = {
    "dreaddit": DatasetInfo(
        name="dreaddit",
        labels=dreaddit.LABELS,
        subgroup_name="subreddit",
        load=dreaddit.load,
        files=download.DREADDIT_FILES,
        # Turcan & McKeown (2019): Sec. 5 gives the split sizes; Table 1 gives labeled
        # segments per subreddit (train + test combined, recorded here under split "all").
        documented={
            ("train", "*"): 2838,
            ("test", "*"): 715,
            ("all", "domesticviolence"): 388,
            ("all", "survivorsofabuse"): 315,
            ("all", "anxiety"): 650,
            ("all", "stress"): 78,
            ("all", "almosthomeless"): 99,
            ("all", "assistance"): 355,
            ("all", "food_pantry"): 43,
            ("all", "homeless"): 220,
            ("all", "ptsd"): 711,
            ("all", "relationships"): 694,
        },
        documented_source="Turcan & McKeown (2019), Table 1 and Section 5",
    ),
    "tweet_eval_stance": DatasetInfo(
        name="tweet_eval_stance",
        labels=tweet_eval.LABELS,
        subgroup_name="stance target",
        load=tweet_eval.load,
        files=download.TWEETEVAL_FILES,
        documented={
            ("train", "abortion"): 587,
            ("val", "abortion"): 66,
            ("test", "abortion"): 280,
            ("train", "atheism"): 461,
            ("val", "atheism"): 52,
            ("test", "atheism"): 220,
            ("train", "climate"): 355,
            ("val", "climate"): 40,
            ("test", "climate"): 169,
            ("train", "feminist"): 597,
            ("val", "feminist"): 67,
            ("test", "feminist"): 285,
            ("train", "hillary"): 620,
            ("val", "hillary"): 69,
            ("test", "hillary"): 295,
        },
        documented_source="Hugging Face dataset card for cardiffnlp/tweet_eval",
    ),
}


def get(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]


def check_documented(info: DatasetInfo, items: list[Item]) -> list[str]:
    """Return a list of mismatches between loaded data and documented counts."""
    problems = []
    for (split, subgroup), expected in info.documented.items():
        actual = sum(
            1
            for it in items
            if (split == "all" or it.split == split)
            and (subgroup == "*" or it.subgroup == subgroup)
        )
        if actual != expected:
            problems.append(f"{split}/{subgroup}: documented {expected}, loaded {actual}")
    return problems
