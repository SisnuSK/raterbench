"""Manual error coding: export a spreadsheet of model errors, then count the coded types.

The spreadsheet contains dataset text, so it lives under data/ (git-ignored). Only the
aggregated counts go to results/.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from raterbench.analysis.load import load_llm_run
from raterbench.data import registry

TAXONOMY = {
    "sarcasm_irony": "Sarcasm or irony; the literal reading differs from the intended one.",
    "negation": "Negation or contrast flips the meaning (not, never, but).",
    "implicit_stance": "Stance or stress is implied rather than stated; the model missed it.",
    "over_inference": "The model read stance/stress into text that does not express it.",
    "ambiguous_gold": "The gold label is debatable or wrong; a careful annotator could disagree.",
    "missing_context": "The excerpt lacks context needed to decide (thread, link, image).",
    "label_boundary": "Borderline case at the edge of the label definitions.",
    "keyword_reliance": "Decision driven by a hashtag or keyword rather than the message.",
    "target_confusion": "Stance toward a related entity confused with stance toward the target.",
    "other": "None of the above; explain in notes.",
}
COLUMNS = [
    "code_id",
    "dataset",
    "item_id",
    "subgroup",
    "text",
    "gold",
    "model",
    "condition",
    "model_label",
    "model_confidence",
    "model_reasoning",
    "error_type",
    "second_error_type",
    "notes",
]


def export(
    run_dir: Path,
    raw_dir: Path,
    out: Path,
    condition: str = "codebook_reasoning",
    per_cell: int = 25,
    seed: int = 13,
) -> int:
    """Sample `per_cell` eval-set errors per (dataset, model) for `condition`."""
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite coding work")
    preds = load_llm_run(run_dir / "predictions.jsonl")
    rng = random.Random(seed)
    rows = []
    for (ds, model, cond, role), ps in sorted(preds.items()):
        if cond != condition or role != "eval":
            continue
        texts = {it.id: it.text for it in registry.get(ds).load(raw_dir)}
        errors = [r for r in ps.records if r.get("label") != r["gold"]]
        for r in sorted(rng.sample(errors, min(per_cell, len(errors))), key=lambda x: x["item_id"]):
            rows.append(
                {
                    "dataset": ds,
                    "item_id": r["item_id"],
                    "subgroup": r["subgroup"],
                    "text": texts[r["item_id"]].strip(),
                    "gold": r["gold"],
                    "model": model,
                    "condition": cond,
                    "model_label": r.get("label") or "(invalid)",
                    "model_confidence": r.get("confidence"),
                    "model_reasoning": r.get("reasoning") or "",
                    "error_type": "",
                    "second_error_type": "",
                    "notes": "",
                }
            )
    rng.shuffle(rows)  # mix datasets and models so coding order does not track them
    for i, r in enumerate(rows, 1):
        r["code_id"] = f"E{i:03d}"
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens the accents and emoji correctly.
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out.parent / "TAXONOMY.md").write_text(
        "# Error types\n\nPut one code in `error_type` (and optionally `second_error_type`).\n"
        "When you have checked a row (whether or not you changed it), put `yes` in\n"
        "`reviewed_by_author`.\n\n"
        + "\n".join(f"- `{k}`: {v}" for k, v in TAXONOMY.items())
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def summarize(coded: Path, out_dir: Path) -> dict:
    """Count coded error types; write table9 (CSV + Markdown). Validates the codes."""
    with coded.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = sorted(
        {
            c
            for r in rows
            for c in (r["error_type"], r["second_error_type"])
            if c and c not in TAXONOMY
        }
    )
    if bad:
        raise ValueError(f"unknown error codes {bad}; allowed: {sorted(TAXONOMY)}")
    coded_rows = [r for r in rows if r["error_type"]]
    primary = Counter((r["dataset"], r["error_type"]) for r in coded_rows)
    any_mention = Counter(
        (r["dataset"], c)
        for r in coded_rows
        for c in {r["error_type"], r["second_error_type"]}
        if c
    )
    per_ds = Counter(r["dataset"] for r in coded_rows)
    table = []
    for code in TAXONOMY:
        row = {"error_type": code}
        for ds in sorted(per_ds):
            row[f"{ds}_primary"] = primary[(ds, code)]
            row[f"{ds}_any"] = any_mention[(ds, code)]
        row["total_primary"] = sum(primary[(ds, code)] for ds in per_ds)
        table.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "table9_error_taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    status = coding_status(coded_rows)
    cols = list(table[0])
    md = [
        "# Error taxonomy (coded sample of model errors)",
        "",
        f"{len(coded_rows)} of {len(rows)} sampled errors coded; {status['text']}. "
        f"Per dataset: {dict(per_ds)}. `_any` counts primary or secondary codes.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
        *("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table),
    ]
    (out_dir / "table9_error_taxonomy.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    summary = {
        "n_rows": len(rows),
        "n_coded": len(coded_rows),
        "per_dataset": dict(per_ds),
        **status,
    }
    (out_dir / "table9_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def coding_status(rows: list[dict]) -> dict:
    """Who coded the sample and how much the author has reviewed, as a sentence for reports.

    `reviewed_by_author` counts as reviewed when non-empty (e.g. "yes" or "changed").
    Rows without a `coder` column were coded by the author directly.
    """
    ai = [r for r in rows if r.get("coder", "").lower().startswith(("ai", "llm"))]
    reviewed = [r for r in ai if r.get("reviewed_by_author", "").strip()]
    if not ai:
        text = "coded manually by the author"
    elif len(reviewed) == len(ai):
        text = "drafted with a language model and reviewed by the author"
    elif reviewed:
        text = (
            f"drafted with a language model; the author has reviewed {len(reviewed)} "
            f"of {len(ai)} so far"
        )
    else:
        text = "drafted with a language model and not yet reviewed by the author"
    return {"n_ai_drafted": len(ai), "n_reviewed": len(reviewed), "text": text}
