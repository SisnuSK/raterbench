"""Human-readable dataset statistics (sizes, label distributions, subgroup counts)."""

from collections import Counter

from raterbench.data.registry import DatasetInfo, check_documented
from raterbench.data.schema import Item


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _breakdown(info: DatasetInfo, groups: dict[str, list[Item]]) -> str:
    """One row per subgroup, one column per (group, label), plus totals."""
    subgroups = sorted({it.subgroup for its in groups.values() for it in its})
    header = [info.subgroup_name]
    for g in groups:
        header += [f"{g}:{lab}" for lab in info.labels] + [f"{g}:n"]
    rows = [header]
    for sg in [*subgroups, "TOTAL"]:
        row = [sg]
        for its in groups.values():
            sel = [it for it in its if sg == "TOTAL" or it.subgroup == sg]
            c = Counter(it.label for it in sel)
            row += [str(c[lab]) for lab in info.labels] + [str(len(sel))]
        rows.append(row)
    return _table(rows)


def report(
    info: DatasetInfo, items: list[Item], splits: dict[str, list[Item]] | None = None
) -> str:
    out = [f"== {info.name} =="]
    by_split = {s: [it for it in items if it.split == s] for s in ("train", "val", "test")}
    by_split = {s: v for s, v in by_split.items() if v}
    out.append("Published splits (label counts by " + info.subgroup_name + "):")
    out.append(_breakdown(info, by_split))
    n_groups = len({it.unit for it in items})
    if n_groups != len(items):
        out.append(f"Items: {len(items)} in {n_groups} groups (e.g. posts)")

    problems = check_documented(info, items)
    if problems:
        out.append(f"MISMATCH vs {info.documented_source}:")
        out += [f"  - {p}" for p in problems]
    else:
        out.append(
            f"All {len(info.documented)} documented counts match ({info.documented_source})."
        )

    if splits is not None:
        out.append("Benchmark roles (from fixed sample file):")
        out.append(_breakdown(info, {r: v for r, v in splits.items() if v}))
    return "\n".join(out)
