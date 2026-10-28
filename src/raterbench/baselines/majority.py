"""Majority-class baseline: always predict the most frequent training label."""

from collections import Counter

from raterbench.data.schema import Item


def majority_label(train: list[Item], labels: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    """Return the majority label (ties broken by label order) and the train label frequencies."""
    counts = Counter(it.label for it in train)
    total = sum(counts.values())
    best = max(labels, key=lambda lab: (counts[lab], -labels.index(lab)))
    return best, {lab: counts[lab] / total for lab in labels}


def predict(items: list[Item], label: str, probs: dict[str, float]) -> list[dict]:
    return [
        {"item_id": it.id, "label": label, "confidence": 100 * probs[label], "probs": probs}
        for it in items
    ]
