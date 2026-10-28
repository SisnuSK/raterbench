"""TweetEval stance subsets (Barbieri et al., 2020; from SemEval-2016 Task 6)."""

from pathlib import Path

from raterbench.data.download import TWEETEVAL_FILES, verify
from raterbench.data.schema import Item

TARGETS = ("abortion", "atheism", "climate", "feminist", "hillary")
LABELS = ("none", "against", "favor")
SPLITS = ("train", "val", "test")


def _lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def load(raw_dir: Path) -> list[Item]:
    root = raw_dir / "tweet_eval_stance"
    verify(TWEETEVAL_FILES, root)
    mapping = dict(line.split("\t") for line in _lines(root / "mapping.txt"))
    if tuple(mapping[str(i)] for i in range(len(LABELS))) != LABELS:
        raise ValueError(f"Unexpected label mapping: {mapping}")
    items = []
    for target in TARGETS:
        for split in SPLITS:
            texts = _lines(root / target / f"{split}_text.txt")
            labels = _lines(root / target / f"{split}_labels.txt")
            if len(texts) != len(labels):
                raise ValueError(f"{target}/{split}: {len(texts)} texts vs {len(labels)} labels")
            for i, (text, label) in enumerate(zip(texts, labels, strict=True)):
                items.append(
                    Item(
                        id=f"{target}-{split}-{i:04d}",
                        text=text,
                        label=mapping[label.strip()],
                        subgroup=target,
                        split=split,
                    )
                )
    return items
