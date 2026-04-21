"""Tests for the QUADAS-2 ground-truth extractor.

Three concerns:

1. **JATS parsing** — pulls the QUADAS-2 risk-of-bias table out of the
   review XML, decodes domain ratings from cell text, resolves each
   study-label's bibliography cross-reference to PMID/DOI.
2. **Title-similarity heuristic** — flags legacy-DB entries whose
   stored title is a bad match for the expected JATS-bibliography title
   (a common sign the v1 ingest pipeline tied a DOI to the wrong paper).
3. **CLI end-to-end** — subprocess run producing a JSON sidecar with
   the expected structure.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.extract_quadas2_ground_truth import (
    StudyRow,
    _parse_study_label,
    _title_similarity,
    cross_reference_legacy_db,
    parse_quadas2_table,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
JATS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cochrane_reviews" / "jcm-15-01829.xml"


# ---- Helpers -----------------------------------------------------------

def _write_legacy_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Create a minimal legacy-schema papers table with the given rows.

    Each row is (pmid, doi, title). Used to exercise the cross-reference
    logic without needing the real legacy biasbuster DB.
    """
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE papers (
                pmid TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT,
                source TEXT
            );
        """)
        conn.executemany(
            "INSERT INTO papers (pmid, doi, title, source) VALUES (?, ?, ?, 'cochrane_rob')",
            rows,
        )
        conn.commit()


# ---- Unit tests --------------------------------------------------------

class TestParseStudyLabel:
    def test_simple_first_author_year(self) -> None:
        author, year, _ = _parse_study_label("Choudhry (2022)")
        assert author == "Choudhry"
        assert year == 2022

    def test_em_dash_substratum_stripped(self) -> None:
        author, year, _ = _parse_study_label("Egboh (2022)—Male")
        assert author == "Egboh"
        assert year == 2022

    def test_hyphen_substratum_stripped(self) -> None:
        author, year, _ = _parse_study_label(
            "Kadashetti (2015)-BG > 200 mg/dL"
        )
        assert author == "Kadashetti"
        assert year == 2015

    def test_no_year_returns_none(self) -> None:
        author, year, _ = _parse_study_label("Mystery Paper")
        assert author == "Mystery Paper"
        assert year is None


class TestTitleSimilarity:
    def test_identical_titles_return_one(self) -> None:
        t = "Salivary Glucose as a Diagnostic Tool in Type II Diabetes Mellitus"
        assert _title_similarity(t, t) == 1.0

    def test_completely_unrelated_titles_return_zero(self) -> None:
        expected = "Correlation of Salivary Glucose and Blood Glucose"
        actual = "Unusual presentation of dermatofibroma on the face"
        assert _title_similarity(expected, actual) == 0.0

    def test_empty_title_returns_zero(self) -> None:
        assert _title_similarity("", "anything") == 0.0
        assert _title_similarity("anything", "") == 0.0

    def test_case_insensitive(self) -> None:
        expected = "SALIVARY GLUCOSE ASSAY"
        actual = "salivary glucose assay"
        assert _title_similarity(expected, actual) == 1.0

    def test_short_words_ignored(self) -> None:
        """Function words (<4 letters) don't dominate the similarity score."""
        expected = "A study of diabetes"
        actual = "The diabetes study"
        # Both share "diabetes" and "study" as their only content words.
        sim = _title_similarity(expected, actual)
        assert sim == 1.0


# ---- Integration tests over the real JATS fixture ----------------------

class TestParseQuadas2TableFixture:
    """Tests against the real tests/fixtures/cochrane_reviews/jcm-15-01829.xml."""

    def test_extracts_expected_row_count(self) -> None:
        """The salivary-glucose review's Table 2 has 31 rows (25 unique studies
        with some sub-strata). If the fixture is ever replaced, this test
        needs updating; we pin the number so silent regressions in the
        table parser are caught."""
        studies, _ = parse_quadas2_table(JATS_FIXTURE)
        assert len(studies) == 31

    def test_every_row_has_domain_ratings(self) -> None:
        studies, _ = parse_quadas2_table(JATS_FIXTURE)
        for s in studies:
            # Every row should have all 4 QUADAS-2 domain ratings.
            assert set(s.bias_ratings) == {
                "patient_selection", "index_test",
                "reference_standard", "flow_and_timing",
            }
            for rating in s.bias_ratings.values():
                assert rating in ("low", "high", "unclear"), (
                    f"unexpected rating in {s.label}: {rating!r}"
                )

    def test_overall_rating_parsed_for_every_row(self) -> None:
        studies, _ = parse_quadas2_table(JATS_FIXTURE)
        for s in studies:
            assert s.overall in ("low", "high", "unclear"), (
                f"overall rating missing for {s.label}: {s.overall!r}"
            )

    def test_bibliography_resolves_for_rows_with_ref_id(self) -> None:
        studies, bibliography = parse_quadas2_table(JATS_FIXTURE)
        for s in studies:
            if s.ref_id is None:
                continue
            assert s.ref_id in bibliography
            # Every ref that makes it into the table should have a title.
            assert bibliography[s.ref_id]["title"]

    def test_known_corrupt_paper_detected(
        self, tmp_path: Path,
    ) -> None:
        """AlQusayer 2019 (DOI 10.5281/zenodo.2542068) is known to be corrupt
        in the legacy DB — the DOI is associated with a dermatofibroma case
        report instead of a salivary-glucose study. Confirm the heuristic
        flags this.
        """
        studies, _ = parse_quadas2_table(JATS_FIXTURE)
        # Build a tiny fixture DB with the known-corrupt row.
        mini_db = tmp_path / "mini_legacy.db"
        _write_legacy_db(mini_db, [
            (
                "99999999",
                "10.5281/zenodo.2542068",
                "Unusual presentation of dermatofibroma on the face: Case report.",
            ),
        ])
        cross_reference_legacy_db(studies, mini_db)
        alqusayer_rows = [s for s in studies if s.first_author_surname == "AlQusayer"]
        assert len(alqusayer_rows) == 1
        row = alqusayer_rows[0]
        assert row.legacy_db["present"] is True
        assert row.legacy_db["title_plausible"] is False
        assert row.legacy_db["title_jaccard_similarity"] < 0.2

    def test_plausible_match_flagged_true(self, tmp_path: Path) -> None:
        """A DB row whose title matches the JATS bibliography scores well."""
        studies, _ = parse_quadas2_table(JATS_FIXTURE)
        # Find a known-good study: Wang 2017 / PMID 28251153 in the real DB.
        mini_db = tmp_path / "mini_legacy.db"
        _write_legacy_db(mini_db, [
            (
                "28251153",
                "10.1155/2017/2569707",
                "Evaluation of Parotid Salivary Glucose Level for Clinical "
                "Diagnosis and Monitoring Type 2 Diabetes Mellitus Patients.",
            ),
        ])
        cross_reference_legacy_db(studies, mini_db)
        wang_rows = [s for s in studies if s.first_author_surname == "Wang"]
        assert len(wang_rows) == 1
        row = wang_rows[0]
        assert row.legacy_db["present"] is True
        assert row.legacy_db["title_plausible"] is True


# ---- summarise() ------------------------------------------------------

class TestSummarise:
    def test_counts_include_plausible_and_corrupt_buckets(self) -> None:
        # Two studies, one plausible, one corrupt, one absent.
        plausible = StudyRow(
            label="A", first_author_surname="A", year=2020,
            ref_id="B1", pmid="1", doi="x",
            expected_title="cats",
            bias_ratings={}, overall="low",
            legacy_db={"present": True, "title_plausible": True},
        )
        corrupt = StudyRow(
            label="B", first_author_surname="B", year=2021,
            ref_id="B2", pmid="2", doi="y",
            expected_title="dogs",
            bias_ratings={}, overall="low",
            legacy_db={"present": True, "title_plausible": False},
        )
        absent = StudyRow(
            label="C", first_author_surname="C", year=2022,
            ref_id="B3", pmid=None, doi="z",
            expected_title="birds",
            bias_ratings={}, overall="low",
            legacy_db={"present": False},
        )
        result = summarise([plausible, corrupt, absent])
        assert result["total_rows"] == 3
        assert result["with_pmid"] == 2
        assert result["with_doi"] == 3
        assert result["present_in_legacy_db"] == 2
        assert result["legacy_title_plausible"] == 1
        assert result["legacy_likely_corrupt"] == 1


# ---- CLI end-to-end ---------------------------------------------------

class TestCliEndToEnd:
    def test_cli_writes_expected_json_shape(self, tmp_path: Path) -> None:
        """Subprocess-level smoke test: the script runs and produces a
        JSON file with ``source``, ``studies``, ``summary`` top-level keys.
        """
        out_path = tmp_path / "gt.json"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "extract_quadas2_ground_truth.py"),
                "--jats", str(JATS_FIXTURE),
                "--output", str(out_path),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(data) == {"source", "studies", "summary"}
        assert data["source"]["jats_file"] == str(JATS_FIXTURE)
        assert len(data["studies"]) == 31
        # Without --legacy-db the legacy_db dict on each study is empty.
        for s in data["studies"]:
            assert s.get("legacy_db") == {}
