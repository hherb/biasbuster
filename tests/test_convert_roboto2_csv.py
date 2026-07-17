"""Tests for the ROBoto2 CSV → ingestion JSON converter.

Pure, no-network conversion, so it is exercised fully offline against small
inline CSV fixtures mirroring the real dataset's shape (a ``manual_assessment``
cell embedding one assessment dict with ``domain_risk_levels``,
``overall_risk_level`` and a flat ``result`` list of per-question signalling
entries; a ``paper_parse`` cell embedding an S2ORC parse).
"""
import csv
import json
from pathlib import Path

from studies.oa_rob_benchmark.convert_roboto2_csv import (
    convert_csv,
    parse_manual_row,
    write_json,
)


def _manual_cell(domain_levels, overall, result):
    return json.dumps([
        {
            "domain_risk_levels": domain_levels,
            "overall_risk_level": overall,
            "paper_id": "x",
            "result": result,
        }
    ])


def _paper_parse_cell(title, abstract, authors):
    return json.dumps({
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "identifiers": {},
    })


def test_parse_manual_row_uses_recorded_labels_and_builds_signalling():
    result = [
        {"domain_number": 1, "question_number": 1, "expertPrediction": "NI"},
        {"domain_number": 1, "question_number": 2, "expertPrediction": "N/PN"},
        {"domain_number": 4, "question_number": 3, "expertPrediction": "NI"},
    ]
    row = {
        "paper_id": "29054",
        "paper_parse": _paper_parse_cell(
            "A trial title",
            "An abstract.",
            [{"first": "Maria", "last": "Balle"}, {"first": "Jon", "last": "Doe"}],
        ),
        "manual_assessment": _manual_cell(
            ["Some concerns", "Some concerns", "Low", "Low", "Some concerns"],
            "Some concerns",
            result,
        ),
        "roboto2_assessment": "[]",
    }

    record = parse_manual_row(row)
    assert record is not None
    # Ground truth is taken verbatim from the recorded expert labels (D4 stays
    # "low" even though re-deriving from signalling would yield some concerns).
    assert record["rob2"] == {
        "overall": "some concerns",
        "randomization": "some concerns",
        "deviations": "some concerns",
        "missing_outcome": "low",
        "measurement": "low",
        "reporting": "some concerns",
    }
    # Raw signalling answers are preserved verbatim, including combined values.
    assert record["signalling"]["randomization"] == {"1.1": "NI", "1.2": "N/PN"}
    assert record["signalling"]["measurement"] == {"4.3": "NI"}
    # Identity fields come from the parse, not from paper_id.
    assert record["paper_id"] == "29054"
    assert record["title"] == "A trial title"
    assert record["authors"] == ["Maria Balle", "Jon Doe"]
    assert record["has_fulltext"] is True
    assert record["source"] == "roboto2_manual"


def test_none_domain_level_is_preserved_not_coerced():
    row = {
        "paper_id": "300",
        "paper_parse": "null",
        "manual_assessment": _manual_cell(
            ["Low", "None", "High", "Low", "Low"], "High", [],
        ),
        "roboto2_assessment": "[]",
    }
    record = parse_manual_row(row)
    assert record is not None
    assert record["rob2"]["deviations"] == "none"
    assert record["rob2"]["overall"] == "high"
    # No paper_parse → no full text, empty identity, still emitted.
    assert record["has_fulltext"] is False
    assert record["title"] == ""
    assert record["authors"] == []


def test_row_without_manual_assessment_is_skipped():
    assert parse_manual_row({"paper_id": "1", "manual_assessment": "null"}) is None
    assert parse_manual_row({"paper_id": "2", "manual_assessment": "[]"}) is None
    assert parse_manual_row({"paper_id": "3", "manual_assessment": ""}) is None


def test_unparseable_manual_assessment_is_dropped_not_raised():
    """A corrupt manual_assessment cell is logged and dropped, never raised —
    one bad cell must not abort the whole conversion."""
    row = {"paper_id": "42", "manual_assessment": "{not valid json"}
    assert parse_manual_row(row) is None


def test_unparseable_paper_parse_keeps_labels_without_identity():
    """A corrupt paper_parse cell costs the row its identity but not its
    already-parsed expert labels: emitted with empty title, has_fulltext False."""
    row = {
        "paper_id": "43",
        "paper_parse": "{not valid json",
        "manual_assessment": _manual_cell(
            ["Low", "Low", "Low", "Low", "Low"], "Low", [],
        ),
    }
    record = parse_manual_row(row)
    assert record is not None
    assert record["title"] == ""
    assert record["has_fulltext"] is False
    assert record["rob2"]["overall"] == "low"


def test_convert_csv_keeps_only_manual_rows(tmp_path):
    csv_path = tmp_path / "roboto2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "paper_parse", "manual_assessment", "roboto2_assessment"])
        writer.writerow([
            "100", "null",
            _manual_cell(["Low", "Low", "Low", "Low", "Low"], "Low", []),
            "[]",
        ])
        # LLM-assisted-only row: no manual_assessment → excluded.
        writer.writerow(["101", "null", "[]", json.dumps([{"domain": "x"}])])

    records = convert_csv(csv_path)
    assert len(records) == 1
    assert records[0]["paper_id"] == "100"

    out_path = tmp_path / "out" / "roboto2.json"
    write_json(records, out_path)
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded == records
