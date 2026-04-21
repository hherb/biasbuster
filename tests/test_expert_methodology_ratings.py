"""Schema + method coverage for ``expert_methodology_ratings``.

Three concerns:

1. **Schema** — fresh DB exposes the table with the expected PK; re-running
   ``initialize()`` is idempotent.
2. **Upsert semantics** — re-ingesting a machine row preserves curator
   fields (``verified``, ``notes``, ``added_by``) unless the caller opts
   into a full overwrite. The generic column set updates as expected.
3. **Read filters** — ``get_expert_ratings`` scopes by methodology,
   source, and pmid; the ``verified_only`` filter excludes unverified rows.

Ingest script end-to-end coverage is kept tight: we build a tiny legacy
DB that the extractor can match against, run the ingest script as a
subprocess, and assert the resulting row count and key fields.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from biasbuster.database import Database

REPO_ROOT = Path(__file__).resolve().parent.parent
JATS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cochrane_reviews" / "jcm-15-01829.xml"


@pytest.fixture
def fresh_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "fresh.db")
    db.initialize()
    return db


# ---- Schema ------------------------------------------------------------

class TestSchema:
    def test_table_created_with_expected_pk(self, fresh_db: Database) -> None:
        # Identity columns form the PK — the same paper can hold multiple
        # rows under different methodologies or from different reviews.
        rows = fresh_db.conn.execute(
            "PRAGMA table_info(expert_methodology_ratings)"
        ).fetchall()
        pk_cols = [r["name"] for r in rows if r["pk"] > 0]
        # SQLite returns pk in the order they were declared in PRIMARY KEY
        assert pk_cols == ["methodology", "rating_source", "study_label"]

    def test_initialize_is_idempotent(self, fresh_db: Database) -> None:
        # Re-initialising must not drop or recreate the table; any rows
        # should survive (we'd lose curator work otherwise).
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-x", study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
        )
        fresh_db.initialize()
        rows = fresh_db.get_expert_ratings()
        assert len(rows) == 1


# ---- Upsert semantics --------------------------------------------------

class TestUpsertSemantics:
    def test_new_row_returns_inserted_status(
        self, fresh_db: Database,
    ) -> None:
        status = fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-1",
            study_label="AlQusayer (2019)",
            domain_ratings={"patient_selection": {"bias": "unclear"}},
            overall_rating="unclear",
            pmid="12345", doi="10.1/x",
            source_reference="Review title",
            added_by="curator",
        )
        assert status == "inserted"
        [row] = fresh_db.get_expert_ratings()
        assert row["study_label"] == "AlQusayer (2019)"
        assert row["pmid"] == "12345"
        assert row["domain_ratings"] == {
            "patient_selection": {"bias": "unclear"},
        }

    def test_existing_row_returns_updated_status(
        self, fresh_db: Database,
    ) -> None:
        """Second call with the same identity returns ``"updated"``.

        Drives the ingest-script summary count so re-runs report
        ``inserted=0`` / ``updated=N`` instead of misleading the user
        into thinking N fresh rows were created.
        """
        first = fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-1a",
            study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
        )
        second = fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-1a",
            study_label="A",
            domain_ratings={"patient_selection": {"bias": "high"}},
            overall_rating="high",
        )
        assert first == "inserted"
        assert second == "updated"

    def test_updated_at_bumped_on_upsert(
        self, fresh_db: Database,
    ) -> None:
        """``updated_at`` must move forward when an existing row changes.

        Audit trail: without this the schema can't answer "when did this
        machine-produced rating last change?", which matters when a
        curator is deciding whether their old ``verified=1`` tag still
        applies to the current row values.
        """
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-ts",
            study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
        )
        first_ts = fresh_db.conn.execute(
            "SELECT added_at, updated_at FROM expert_methodology_ratings "
            "WHERE study_label = 'A'"
        ).fetchone()
        # Bypass SQLite's 1-second clock resolution by forcing the
        # initial timestamp into the past before the update.
        fresh_db.conn.execute(
            "UPDATE expert_methodology_ratings "
            "SET updated_at = '2000-01-01 00:00:00' WHERE study_label = 'A'"
        )
        fresh_db.conn.commit()
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-ts",
            study_label="A",
            domain_ratings={"patient_selection": {"bias": "high"}},
            overall_rating="high",
        )
        second_ts = fresh_db.conn.execute(
            "SELECT added_at, updated_at FROM expert_methodology_ratings "
            "WHERE study_label = 'A'"
        ).fetchone()
        # added_at is frozen at insert time; updated_at has advanced past
        # the forced-stale value.
        assert second_ts["added_at"] == first_ts["added_at"]
        assert second_ts["updated_at"] > "2000-01-01 00:00:00"

    def test_reingest_preserves_curation(
        self, fresh_db: Database,
    ) -> None:
        """Re-running the machine ingest must not clobber ``verified=1``."""
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-2", study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
        )
        # Curator promotes the row.
        fresh_db.conn.execute(
            "UPDATE expert_methodology_ratings "
            "SET verified = 1, notes = 'checked' "
            "WHERE study_label = 'A'"
        )
        fresh_db.conn.commit()
        # Machine ingest runs again with new ratings and no curator info.
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-2", study_label="A",
            domain_ratings={"patient_selection": {"bias": "high"}},
            overall_rating="high",
        )
        [row] = fresh_db.get_expert_ratings()
        assert row["domain_ratings"] == {
            "patient_selection": {"bias": "high"},
        }
        assert row["overall_rating"] == "high"
        # Curation fields preserved
        assert row["verified"] == 1
        assert row["notes"] == "checked"

    def test_force_overwrite_replaces_curation(
        self, fresh_db: Database,
    ) -> None:
        """``preserve_curation=False`` is the explicit escape hatch."""
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-3", study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
            verified=True, notes="human-checked",
        )
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-3", study_label="A",
            domain_ratings={"patient_selection": {"bias": "high"}},
            overall_rating="high",
            verified=False, notes=None,
            preserve_curation=False,
        )
        [row] = fresh_db.get_expert_ratings()
        assert row["verified"] == 0
        assert row["notes"] is None
        assert row["overall_rating"] == "high"

    def test_same_label_different_methodology_coexist(
        self, fresh_db: Database,
    ) -> None:
        """PK includes methodology, so a paper can carry RoB 2 + QUADAS-2."""
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="src-4", study_label="A",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
        )
        fresh_db.upsert_expert_rating(
            methodology="cochrane_rob2", rating_source="src-4",
            study_label="A",
            domain_ratings={"randomization": "low"},
            overall_rating="low",
        )
        assert len(fresh_db.get_expert_ratings()) == 2


# ---- Read filters ------------------------------------------------------

class TestReadFilters:
    @pytest.fixture
    def seeded_db(self, fresh_db: Database) -> Database:
        # Three rows across two methodologies, two sources, two pmids.
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="review-A",
            study_label="Study-1",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low", pmid="111",
            verified=True,
        )
        fresh_db.upsert_expert_rating(
            methodology="quadas_2", rating_source="review-B",
            study_label="Study-2",
            domain_ratings={"patient_selection": {"bias": "high"}},
            overall_rating="high", pmid="222",
        )
        fresh_db.upsert_expert_rating(
            methodology="cochrane_rob2", rating_source="review-A",
            study_label="Study-1",
            domain_ratings={"randomization": "low"},
            overall_rating="low", pmid="111",
        )
        return fresh_db

    def test_methodology_filter(self, seeded_db: Database) -> None:
        rows = seeded_db.get_expert_ratings(methodology="quadas_2")
        assert {r["study_label"] for r in rows} == {"Study-1", "Study-2"}

    def test_source_filter(self, seeded_db: Database) -> None:
        rows = seeded_db.get_expert_ratings(rating_source="review-A")
        assert len(rows) == 2

    def test_pmid_filter(self, seeded_db: Database) -> None:
        rows = seeded_db.get_expert_ratings(pmid="111")
        # Both the QUADAS-2 and RoB 2 entries for PMID 111.
        assert len(rows) == 2
        assert {r["methodology"] for r in rows} == {
            "quadas_2", "cochrane_rob2",
        }

    def test_verified_only_filter(self, seeded_db: Database) -> None:
        rows = seeded_db.get_expert_ratings(verified_only=True)
        assert len(rows) == 1
        assert rows[0]["study_label"] == "Study-1"
        assert rows[0]["verified"] == 1


# ---- Ingest script end-to-end -----------------------------------------

class TestIngestScript:
    """Subprocess-level coverage of scripts/ingest_expert_quadas2_ratings.py.

    Uses a synthetic legacy DB seeded with one matching title for the
    known-plausible Wang 2017 row in the JATS fixture (see
    ``test_extract_quadas2_ground_truth.py`` for how this row was
    identified). One plausible row should land in the target DB; the
    other rows should be skipped because they're absent from our mini
    legacy DB.
    """

    def _seed_mini_legacy_db(self, path: Path) -> None:
        """Minimal legacy schema that the extractor's cross-reference can hit."""
        with sqlite3.connect(str(path)) as conn:
            conn.executescript("""
                CREATE TABLE papers (
                    pmid TEXT PRIMARY KEY,
                    doi TEXT,
                    title TEXT,
                    source TEXT
                );
            """)
            conn.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?)",
                (
                    "28251153",
                    "10.1155/2017/2569707",
                    "Evaluation of Parotid Salivary Glucose Level for "
                    "Clinical Diagnosis and Monitoring Type 2 Diabetes "
                    "Mellitus Patients.",
                    "cochrane_rob",
                ),
            )
            conn.commit()

    def test_ingest_writes_plausible_row(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.db"
        target = tmp_path / "target.db"
        self._seed_mini_legacy_db(legacy)

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "ingest_expert_quadas2_ratings.py"),
                "--jats", str(JATS_FIXTURE),
                "--legacy-db", str(legacy),
                "--target-db", str(target),
                "--rating-source", "jcm-15-01829",
                "--added-by", "test-curator",
            ],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr

        db = Database(target)
        db.initialize()
        try:
            rows = db.get_expert_ratings(methodology="quadas_2")
        finally:
            db.close()

        # Exactly one row because only one PMID was seeded in the mini
        # legacy DB — every other row of the JATS table is classified as
        # "not_in_legacy" and skipped.
        assert len(rows) == 1
        row = rows[0]
        assert row["pmid"] == "28251153"
        assert row["methodology_version"] == "quadas2-2011"
        assert row["rating_source"] == "jcm-15-01829"
        assert row["added_by"] == "test-curator"
        assert row["verified"] == 0
        # Ratings structure is the per-domain {"bias": level} shape
        assert "patient_selection" in row["domain_ratings"]
        assert "bias" in row["domain_ratings"]["patient_selection"]
        assert row["overall_rating"] in ("low", "high", "unclear")

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        """--dry-run parses and classifies but the target DB stays empty."""
        legacy = tmp_path / "legacy.db"
        target = tmp_path / "target.db"
        self._seed_mini_legacy_db(legacy)

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "ingest_expert_quadas2_ratings.py"),
                "--jats", str(JATS_FIXTURE),
                "--legacy-db", str(legacy),
                "--target-db", str(target),
                "--rating-source", "jcm-15-01829",
                "--dry-run",
            ],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "Would ingest:" in result.stdout

        # The ingest script still calls Database.initialize() so the
        # table exists, but no rows were written.
        db = Database(target)
        db.initialize()
        try:
            rows = db.get_expert_ratings()
        finally:
            db.close()
        assert rows == []

    def test_include_unmatched_ingests_jats_only_rows(
        self, tmp_path: Path,
    ) -> None:
        """With --include-unmatched, rows absent from legacy DB are still written."""
        legacy = tmp_path / "legacy_empty.db"
        target = tmp_path / "target.db"
        # Empty legacy DB: every row is classified 'not_in_legacy'.
        with sqlite3.connect(str(legacy)) as conn:
            conn.executescript(
                "CREATE TABLE papers (pmid TEXT PRIMARY KEY, doi TEXT, "
                "title TEXT, source TEXT);"
            )

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "ingest_expert_quadas2_ratings.py"),
                "--jats", str(JATS_FIXTURE),
                "--legacy-db", str(legacy),
                "--target-db", str(target),
                "--rating-source", "jcm-15-01829",
                "--include-unmatched",
            ],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr

        db = Database(target)
        db.initialize()
        try:
            rows = db.get_expert_ratings(methodology="quadas_2")
        finally:
            db.close()
        # Every row of the JATS table (31 of them) should land.
        assert len(rows) == 31


# ---- process_studies classification ------------------------------------

class TestProcessStudies:
    """Unit-test the ingest loop body against synthetic StudyRow fixtures.

    Subprocess tests cover the happy paths against the real JATS fixture
    (which has no incomplete or corrupt rows). These tests pin the
    classification + skip logic by constructing studies directly, which
    is the only way to exercise the ``skipped_incomplete`` counter.
    """

    def _plausible_row(self) -> "StudyRow":
        from scripts.extract_quadas2_ground_truth import StudyRow
        return StudyRow(
            label="Good (2020)",
            first_author_surname="Good",
            year=2020,
            ref_id="B1",
            pmid="111",
            doi="10.1/good",
            expected_title="A plausible study.",
            bias_ratings={
                "patient_selection": "low", "index_test": "low",
                "reference_standard": "low", "flow_and_timing": "low",
            },
            overall="low",
            legacy_db={
                "present": True, "title_plausible": True,
                "pmid_in_db": "111", "doi_in_db": "10.1/good",
                "title_in_db": "A plausible study.",
                "title_jaccard_similarity": 0.9,
            },
        )

    def _corrupt_row(self) -> "StudyRow":
        from scripts.extract_quadas2_ground_truth import StudyRow
        return StudyRow(
            label="Bad (2019)",
            first_author_surname="Bad",
            year=2019,
            ref_id="B2",
            pmid="222",
            doi="10.1/bad",
            expected_title="Expected title.",
            bias_ratings={
                "patient_selection": "low", "index_test": "low",
                "reference_standard": "low", "flow_and_timing": "low",
            },
            overall="low",
            legacy_db={
                "present": True, "title_plausible": False,
                "pmid_in_db": "222", "doi_in_db": "10.1/bad",
                "title_in_db": "Wrong paper entirely.",
                "title_jaccard_similarity": 0.0,
            },
        )

    def _incomplete_row(self) -> "StudyRow":
        from scripts.extract_quadas2_ground_truth import StudyRow
        return StudyRow(
            label="Partial (2021)",
            first_author_surname="Partial",
            year=2021,
            ref_id="B3",
            pmid="333",
            doi="10.1/partial",
            expected_title="Partial study.",
            bias_ratings={},  # empty — caller couldn't parse the row
            overall=None,
            legacy_db={
                "present": True, "title_plausible": True,
                "pmid_in_db": "333", "doi_in_db": "10.1/partial",
                "title_in_db": "Partial study.",
                "title_jaccard_similarity": 0.9,
            },
        )

    def test_plausible_row_inserted(self, tmp_path: Path) -> None:
        from scripts.ingest_expert_quadas2_ratings import process_studies

        db = Database(tmp_path / "t.db")
        db.initialize()
        try:
            counts = process_studies(
                [self._plausible_row()], db,
                rating_source="src", review_meta={},
            )
            db.commit()
        finally:
            db.close()
        assert counts["inserted"] == 1
        assert counts["skipped_corrupt"] == 0

    def test_corrupt_row_skipped(self, tmp_path: Path) -> None:
        """Legacy DB has the paper but the title doesn't match → skip."""
        from scripts.ingest_expert_quadas2_ratings import process_studies

        db = Database(tmp_path / "t.db")
        db.initialize()
        try:
            counts = process_studies(
                [self._corrupt_row()], db,
                rating_source="src", review_meta={},
            )
            db.commit()
            assert db.get_expert_ratings() == []
        finally:
            db.close()
        assert counts["skipped_corrupt"] == 1
        assert counts["inserted"] == 0

    def test_incomplete_row_skipped(self, tmp_path: Path) -> None:
        """Empty bias_ratings or None overall → skip with a warning."""
        from scripts.ingest_expert_quadas2_ratings import process_studies

        db = Database(tmp_path / "t.db")
        db.initialize()
        try:
            counts = process_studies(
                [self._incomplete_row()], db,
                rating_source="src", review_meta={},
            )
            db.commit()
            assert db.get_expert_ratings() == []
        finally:
            db.close()
        assert counts["skipped_incomplete"] == 1
        assert counts["inserted"] == 0

    def test_counts_sum_matches_total(self, tmp_path: Path) -> None:
        """Every row lands in exactly one bucket — no double-count, no drops."""
        from scripts.ingest_expert_quadas2_ratings import process_studies

        db = Database(tmp_path / "t.db")
        db.initialize()
        try:
            counts = process_studies(
                [
                    self._plausible_row(), self._corrupt_row(),
                    self._incomplete_row(),
                ],
                db, rating_source="src", review_meta={},
            )
        finally:
            db.close()
        bucket_sum = (
            counts["inserted"] + counts["updated"]
            + counts["skipped_corrupt"]
            + counts["skipped_not_in_legacy"]
            + counts["skipped_incomplete"]
        )
        assert bucket_sum == counts["total"] == 3


# ---- parse_review_metadata --------------------------------------------

class TestParseReviewMetadata:
    def test_extracts_review_pmid_doi_title(self) -> None:
        from scripts.extract_quadas2_ground_truth import parse_review_metadata
        meta = parse_review_metadata(JATS_FIXTURE)
        # The fixture is a real JCM article; at minimum a DOI and a title
        # should be present. PMID may or may not be embedded depending on
        # how the JATS was generated.
        assert meta["doi"] is not None
        assert meta["title"] is not None
