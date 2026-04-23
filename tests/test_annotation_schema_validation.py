"""DB-level annotation JSON Schema validation.

Covers the contract from ``docs/ANNOTATION_JSON_SPEC.md``:

  - Methodologies with a registered schema (cochrane_rob2, quadas_2)
    must validate every annotation before it lands in the DB.
  - Validation failure raises ``jsonschema.ValidationError`` and the
    INSERT does NOT occur (no half-written rows).
  - Methodologies without a schema (currently: ``biasbuster``) are
    untouched — back-compat for the legacy single-call assessor.
  - Real annotations from the recovered DB validate cleanly (smoke
    test via the schemas-on-disk path that production uses).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from biasbuster.database import Database


# ---- Helpers -----------------------------------------------------------

def _fresh_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "fresh.db")
    db.initialize()
    db.insert_paper({"pmid": "P1", "title": "T", "abstract": "A", "source": "test"})
    return db


def _valid_rob2(severity: str = "low") -> dict:
    return {
        "outcomes": [
            {
                "overall_judgement": severity,
                "domains": {
                    "randomization": {
                        "domain": "randomization",
                        "signalling_answers": {"1.1": "Y"},
                        "judgement": severity,
                    },
                },
            },
        ],
        "worst_across_outcomes": severity,
        "overall_severity": severity,
    }


def _valid_quadas2(rating: str = "low") -> dict:
    return {
        "domains": {
            "patient_selection": {
                "domain": "patient_selection",
                "signalling_answers": {"1.1": "yes"},
                "bias_rating": rating,
                "applicability": rating,
            },
            "index_test": {
                "domain": "index_test",
                "signalling_answers": {},
                "bias_rating": rating,
                "applicability": rating,
            },
            "reference_standard": {
                "domain": "reference_standard",
                "signalling_answers": {},
                "bias_rating": rating,
                "applicability": rating,
            },
            "flow_and_timing": {
                "domain": "flow_and_timing",
                "signalling_answers": {},
                "bias_rating": rating,
                "applicability": None,
            },
        },
        "worst_bias": rating,
        "worst_applicability": rating,
        "overall_severity": rating,
        "overall_applicability": rating,
    }


# ---- Schema files themselves ------------------------------------------

class TestSchemaFiles:
    """The on-disk schemas must be valid Draft 2020-12 documents."""

    def test_rob2_schema_loads_and_is_valid(self) -> None:
        path = Path(__file__).resolve().parent.parent / "schemas" / "rob2_annotation.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    def test_quadas2_schema_loads_and_is_valid(self) -> None:
        path = Path(__file__).resolve().parent.parent / "schemas" / "quadas2_annotation.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


# ---- Happy path: valid annotations land in the DB ---------------------

class TestValidAnnotationsAccepted:
    def test_rob2_minimal_valid_inserts(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        ok = db.insert_annotation(
            "P1", "anthropic", _valid_rob2("low"),
            methodology="cochrane_rob2",
            methodology_version="rob2-2019",
        )
        assert ok is True

    def test_quadas2_minimal_valid_inserts(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        ok = db.insert_annotation(
            "P1", "anthropic", _valid_quadas2("low"),
            methodology="quadas_2",
            methodology_version="quadas2-2011",
        )
        assert ok is True

    def test_upsert_rob2_valid_replaces_in_place(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        db.upsert_annotation(
            "P1", "anthropic", _valid_rob2("low"),
            methodology="cochrane_rob2", methodology_version="rob2-2019",
        )
        db.upsert_annotation(
            "P1", "anthropic", _valid_rob2("high"),
            methodology="cochrane_rob2", methodology_version="rob2-2019",
        )
        rows = db.conn.execute(
            "SELECT COUNT(*) AS n, MAX(overall_severity) AS sev "
            "FROM annotations WHERE pmid = 'P1'",
        ).fetchone()
        assert rows["n"] == 1
        assert rows["sev"] == "high"


# ---- Failure path: invalid annotations are rejected -------------------

class TestInvalidAnnotationsRejected:
    def test_rob2_missing_outcomes_rejected(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        bad = {"worst_across_outcomes": "low"}  # no "outcomes" array
        with pytest.raises(ValidationError):
            db.insert_annotation(
                "P1", "anthropic", bad, methodology="cochrane_rob2",
            )
        # And nothing landed.
        n = db.conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE pmid = 'P1'"
        ).fetchone()["n"]
        assert n == 0

    def test_rob2_invalid_judgement_value_rejected(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        bad = _valid_rob2("low")
        bad["outcomes"][0]["overall_judgement"] = "maybe"  # not in enum
        with pytest.raises(ValidationError):
            db.insert_annotation(
                "P1", "anthropic", bad, methodology="cochrane_rob2",
            )

    def test_rob2_invalid_signalling_code_rejected(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        bad = _valid_rob2("low")
        # Z is not in {Y, PY, PN, N, NI}
        bad["outcomes"][0]["domains"]["randomization"]["signalling_answers"]["1.1"] = "Z"
        with pytest.raises(ValidationError):
            db.insert_annotation(
                "P1", "anthropic", bad, methodology="cochrane_rob2",
            )

    def test_quadas2_flow_and_timing_must_be_null_applicability(
        self, tmp_path: Path,
    ) -> None:
        """flow_and_timing carries no applicability dimension per Whiting 2011.
        A non-null applicability on that domain must be rejected."""
        db = _fresh_db(tmp_path)
        bad = _valid_quadas2("low")
        bad["domains"]["flow_and_timing"]["applicability"] = "low"  # should be null
        with pytest.raises(ValidationError):
            db.insert_annotation(
                "P1", "anthropic", bad, methodology="quadas_2",
            )

    def test_quadas2_other_domains_must_have_applicability(
        self, tmp_path: Path,
    ) -> None:
        db = _fresh_db(tmp_path)
        bad = _valid_quadas2("low")
        # patient_selection requires applicability per Whiting 2011
        del bad["domains"]["patient_selection"]["applicability"]
        with pytest.raises(ValidationError):
            db.insert_annotation(
                "P1", "anthropic", bad, methodology="quadas_2",
            )


# ---- Methodologies without a schema are untouched ---------------------

class TestUnregisteredMethodologyBypasses:
    def test_biasbuster_methodology_skips_validation(
        self, tmp_path: Path,
    ) -> None:
        """The legacy biasbuster single-call assessor has no schema; its
        annotations land regardless of shape so existing pipelines keep
        working."""
        db = _fresh_db(tmp_path)
        ok = db.insert_annotation(
            "P1", "anthropic", {"any": "shape", "literally": ["anything"]},
            methodology="biasbuster",
        )
        assert ok is True

    def test_unknown_methodology_skips_validation(self, tmp_path: Path) -> None:
        db = _fresh_db(tmp_path)
        ok = db.insert_annotation(
            "P1", "anthropic", {"shape": "unspecified"},
            methodology="amstar_2",  # registered methodology, no schema (yet)
        )
        assert ok is True
