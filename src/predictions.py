"""Per-question prediction records: schema, builder, JSONL writer/reader.

Pure helpers. No torch / transformers / datasets imports — so the schema can
be tested in CI without the heavy eval stack, and downstream analysis scripts
can import this module to read JSONLs without needing a GPU env.

Schema version pinning: bumping ``PREDICTIONS_SCHEMA_VERSION`` is the load-
bearing signal that downstream consumers (``scripts/analyze_predictions.py``)
need an update. Update both in the same commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .answer_parsing import answers_match, extract_answer

PREDICTIONS_SCHEMA_VERSION = "v1"

EXPECTED_FIELDS = frozenset({
    "idx",
    "question",
    "gold",
    "pred_raw",
    "pred_extracted",
    "correct",
    "gen_chars",
    "question_chars",
})


def make_record(
    idx: int,
    question: str,
    gold: Optional[str],
    pred_raw: str,
) -> dict:
    """Build one per-question prediction record. Pure function.

    Fields (schema v1):
      idx: 0-based test-set index
      question: original GSM8K question text
      gold: gold numeric answer (string), or None if extractor failed on gold
      pred_raw: full model completion (post-prompt, decoded)
      pred_extracted: numeric answer extracted from pred_raw, or None
      correct: bool, answers_match(pred_extracted, gold)
      gen_chars: len(pred_raw); cheap proxy for generation length
      question_chars: len(question); for length-bucket analysis
    """
    pred = extract_answer(pred_raw)
    return {
        "idx": idx,
        "question": question,
        "gold": gold,
        "pred_raw": pred_raw,
        "pred_extracted": pred,
        "correct": answers_match(pred, gold),
        "gen_chars": len(pred_raw),
        "question_chars": len(question),
    }


def write_predictions_jsonl(
    path: Path,
    run_id: str,
    records: list[dict],
    extractor_version: str,
) -> None:
    """Write records as JSONL. First line = metadata header, then one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "_meta": True,
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "run_id": run_id,
        "extractor": extractor_version,
        "n_records": len(records),
    }
    with path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_predictions_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """Read JSONL written by ``write_predictions_jsonl``. Returns (header, records).

    Raises ValueError if the schema version is unknown.
    """
    lines = path.read_text().strip().split("\n")
    if not lines:
        raise ValueError(f"empty predictions file: {path}")
    header = json.loads(lines[0])
    if not header.get("_meta"):
        raise ValueError(f"missing metadata header in {path}")
    if header["schema_version"] != PREDICTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"schema version mismatch in {path}: "
            f"file={header['schema_version']!r} expected={PREDICTIONS_SCHEMA_VERSION!r}"
        )
    records = [json.loads(line) for line in lines[1:]]
    return header, records


def iter_predictions_dir(directory: Path) -> Iterator[tuple[dict, list[dict]]]:
    """Yield (header, records) for every *.jsonl in directory, sorted by filename."""
    for path in sorted(directory.glob("*.jsonl")):
        yield read_predictions_jsonl(path)
