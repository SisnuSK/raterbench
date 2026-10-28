"""Dreaddit (Turcan & McKeown, 2019): Reddit segments labeled for stress."""

import csv
import io
import zipfile
from pathlib import Path

from raterbench.data.download import DREADDIT_FILES, verify
from raterbench.data.schema import Item

LABELS = ("not_stress", "stress")

# Subreddit -> domain, from Table 1 of the paper.
DOMAINS = {
    "domesticviolence": "abuse",
    "survivorsofabuse": "abuse",
    "anxiety": "anxiety",
    "stress": "anxiety",
    "almosthomeless": "financial",
    "assistance": "financial",
    "food_pantry": "financial",
    "homeless": "financial",
    "ptsd": "ptsd",
    "relationships": "social",
}


def load(raw_dir: Path) -> list[Item]:
    root = raw_dir / "dreaddit"
    verify(DREADDIT_FILES, root)
    items = []
    with zipfile.ZipFile(root / "dreaddit.zip") as zf:
        for split in ("train", "test"):
            with zf.open(f"dreaddit-{split}.csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
                for row in reader:
                    items.append(
                        Item(
                            id=row["id"],
                            text=row["text"],
                            label=LABELS[int(row["label"])],
                            subgroup=row["subreddit"],
                            split=split,
                            group=row["post_id"],
                            meta={
                                "domain": DOMAINS[row["subreddit"]],
                                "annotator_agreement": row["confidence"],
                            },
                        )
                    )
    return items
