"""Pin the per-question predictions JSONL schema.

These tests cover only the pure helpers (no model load, no GPU), so the
schema can be guarded in CI without the heavy eval stack.

If you change the schema, bump ``PREDICTIONS_SCHEMA_VERSION`` and update
``scripts/analyze_predictions.py`` consumer code in the same commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.predictions import (
    PREDICTIONS_SCHEMA_VERSION,
    make_record,
    read_predictions_jsonl,
    write_predictions_jsonl,
)

EXPECTED_FIELDS = {
    "idx",
    "question",
    "gold",
    "pred_raw",
    "pred_extracted",
    "correct",
    "gen_chars",
    "question_chars",
}


def test_record_has_all_expected_fields():
    rec = make_record(
        idx=0,
        question="Janet has 16 eggs. How many remain after eating 3?",
        gold="13",
        pred_raw="16 - 3 = 13.\n#### 13",
    )
    assert set(rec.keys()) == EXPECTED_FIELDS


def test_record_correct_when_answers_match():
    rec = make_record(0, "q?", "13", "step. \n#### 13")
    assert rec["correct"] is True
    assert rec["pred_extracted"] == "13"


def test_record_incorrect_when_answers_differ():
    rec = make_record(0, "q?", "13", "step.\n#### 14")
    assert rec["correct"] is False
    assert rec["pred_extracted"] == "14"


def test_record_handles_extractor_failure():
    """When extractor finds no number, pred_extracted is None and correct=False."""
    rec = make_record(0, "q?", "13", "I don't know.")
    assert rec["pred_extracted"] is None
    assert rec["correct"] is False


def test_record_handles_none_gold():
    """If gold itself failed extraction upstream, record still well-formed."""
    rec = make_record(0, "q?", None, "answer is 13")
    assert rec["gold"] is None
    assert rec["pred_extracted"] == "13"
    assert rec["correct"] is False  # cannot match against None gold


def test_record_lengths_are_integers():
    rec = make_record(0, "abc", "1", "xyz!")
    assert rec["question_chars"] == 3
    assert rec["gen_chars"] == 4
    assert isinstance(rec["idx"], int)


def test_jsonl_roundtrip_with_metadata_header(tmp_path: Path):
    """First line is metadata; remaining lines are records. Order preserved."""
    records = [
        make_record(0, "q0", "1", "#### 1"),
        make_record(1, "q1", "2", "#### 3"),
    ]
    path = tmp_path / "predictions" / "test_run.jsonl"
    write_predictions_jsonl(
        path, run_id="test_run", records=records, extractor_version="vtest"
    )

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1 + len(records)

    header = json.loads(lines[0])
    assert header["_meta"] is True
    assert header["schema_version"] == PREDICTIONS_SCHEMA_VERSION
    assert header["run_id"] == "test_run"
    assert header["n_records"] == 2

    parsed = [json.loads(line) for line in lines[1:]]
    assert [r["idx"] for r in parsed] == [0, 1]
    assert parsed[0]["correct"] is True
    assert parsed[1]["correct"] is False


def test_jsonl_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "deeply" / "nested" / "preds.jsonl"
    write_predictions_jsonl(path, run_id="r", records=[], extractor_version="test")
    assert path.exists()


def test_schema_version_constant_is_set():
    """Schema version is the load-bearing pinning signal for downstream consumers."""
    assert PREDICTIONS_SCHEMA_VERSION == "v1"


def test_read_predictions_jsonl_roundtrip(tmp_path: Path):
    records = [
        make_record(0, "q0", "1", "#### 1"),
        make_record(1, "q1", "2", "#### 2"),
    ]
    path = tmp_path / "r.jsonl"
    write_predictions_jsonl(path, run_id="r", records=records, extractor_version="vx")

    header, parsed = read_predictions_jsonl(path)
    assert header["run_id"] == "r"
    assert header["extractor"] == "vx"
    assert header["n_records"] == 2
    assert [r["idx"] for r in parsed] == [0, 1]


def test_read_rejects_wrong_schema_version(tmp_path: Path):
    import json as _json
    path = tmp_path / "r.jsonl"
    bad_header = {
        "_meta": True,
        "schema_version": "v999",
        "run_id": "r",
        "extractor": "x",
        "n_records": 0,
    }
    path.write_text(_json.dumps(bad_header) + "\n")
    try:
        read_predictions_jsonl(path)
    except ValueError as e:
        assert "schema version mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on schema mismatch")
