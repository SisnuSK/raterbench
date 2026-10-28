"""Deterministic stratified sampling over groups of items."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence

from raterbench.data.schema import Item


def allocate(counts: dict[Hashable, int], n: int) -> dict[Hashable, int]:
    """Proportional allocation of n draws across strata (largest-remainder rounding).

    Ties in the remainder are broken by stratum key order so the result is deterministic.
    """
    total = sum(counts.values())
    if not 0 <= n <= total:
        raise ValueError(f"Cannot draw {n} from {total} items")
    keys = sorted(counts)
    exact = {k: n * counts[k] / total for k in keys}
    alloc = {k: math.floor(exact[k]) for k in keys}
    leftover = n - sum(alloc.values())
    by_remainder = sorted(keys, key=lambda k: -(exact[k] - alloc[k]))
    for k in by_remainder[:leftover]:
        alloc[k] += 1
    return alloc


def stratified_sample(
    items: Sequence[Item],
    n: int,
    stratum: Callable[[Item], Hashable],
    seed: int,
) -> list[Item]:
    """Draw about n items, stratified by `stratum`, keeping each item's group intact.

    Each group (e.g. all segments of one Reddit post) is assigned to the most common
    stratum among its items and drawn as a whole. With singleton groups the sample size is
    exactly n; with larger groups it can overshoot a stratum's quota by less than one group.
    """
    if n >= len(items):
        return list(items)
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.unit].append(it)

    group_stratum = {}
    for unit, members in groups.items():
        votes = Counter(stratum(m) for m in members)
        group_stratum[unit] = min(votes, key=lambda s: (-votes[s], s))

    item_counts = Counter(group_stratum[it.unit] for it in items)
    quotas = allocate(dict(item_counts), n)

    by_stratum: dict[Hashable, list[str]] = defaultdict(list)
    for unit in sorted(groups):
        by_stratum[group_stratum[unit]].append(unit)

    rng = random.Random(seed)
    chosen: list[Item] = []
    for s in sorted(by_stratum):
        units = by_stratum[s]
        rng.shuffle(units)
        taken = 0
        for unit in units:
            if taken >= quotas[s]:
                break
            chosen.extend(groups[unit])
            taken += len(groups[unit])
    return chosen
