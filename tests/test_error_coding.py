import csv
from pathlib import Path

import pytest

from raterbench.analysis.error_coding import COLUMNS, TAXONOMY, summarize


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def test_summarize_counts_primary_and_any(tmp_path: Path) -> None:
    rows = [
        {"code_id": "E1", "dataset": "a", "error_type": "negation"},
        {
            "code_id": "E2",
            "dataset": "a",
            "error_type": "negation",
            "second_error_type": "ambiguous_gold",
        },
        {"code_id": "E3", "dataset": "b", "error_type": "ambiguous_gold"},
        {"code_id": "E4", "dataset": "b", "error_type": ""},  # not coded yet
    ]
    _write(tmp_path / "coded.csv", rows)
    s = summarize(tmp_path / "coded.csv", tmp_path / "out")
    assert {k: s[k] for k in ("n_rows", "n_coded", "per_dataset")} == {
        "n_rows": 4,
        "n_coded": 3,
        "per_dataset": {"a": 2, "b": 1},
    }
    assert s["text"] == "coded manually by the author"
    with (tmp_path / "out" / "table9_error_taxonomy.csv").open(encoding="utf-8") as f:
        table = {r["error_type"]: r for r in csv.DictReader(f)}
    assert set(table) == set(TAXONOMY)
    assert table["negation"]["a_primary"] == "2" and table["negation"]["total_primary"] == "2"
    assert table["ambiguous_gold"]["a_any"] == "1" and table["ambiguous_gold"]["b_primary"] == "1"


def test_summarize_rejects_unknown_codes(tmp_path: Path) -> None:
    _write(tmp_path / "coded.csv", [{"code_id": "E1", "dataset": "a", "error_type": "typo"}])
    with pytest.raises(ValueError, match="unknown error codes"):
        summarize(tmp_path / "coded.csv", tmp_path / "out")


def test_coding_status_tracks_ai_draft_and_review() -> None:
    from raterbench.analysis.error_coding import coding_status

    ai = {"coder": "LLM draft"}
    assert coding_status([{**ai, "reviewed_by_author": ""}] * 2)["text"].endswith(
        "not yet reviewed by the author"
    )
    partial = coding_status([{**ai, "reviewed_by_author": "yes"}, {**ai, "reviewed_by_author": ""}])
    assert partial["n_reviewed"] == 1 and "reviewed 1 of 2" in partial["text"]
    done = coding_status([{**ai, "reviewed_by_author": "changed"}] * 3)
    assert done["text"].endswith("and reviewed by the author")
