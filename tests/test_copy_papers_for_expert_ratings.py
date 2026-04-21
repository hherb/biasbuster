"""Coverage for scripts/copy_papers_for_expert_ratings.py.

Focus on the targeted behaviour that matters for the faithfulness
workflow:

- Only PMIDs actually listed in ``expert_methodology_ratings`` are
  copied — never an indiscriminate ``SELECT * FROM papers``.
- Existing target rows are preserved (curator edits don't get clobbered).
- JSON columns survive the round-trip unchanged.
- A ``--methodology`` filter narrows the PMID set.
- Missing-in-source PMIDs are reported, not skipped silently.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from biasbuster.database import Database
from scripts.copy_papers_for_expert_ratings import copy_papers


def _seed_source_db(path: Path, papers: list[dict]) -> None:
    """Create a minimal legacy-shape source DB.

    Only the columns ``copy_papers`` actually reads/writes are
    included; the production legacy DB has more, but they're
    irrelevant to the copy logic.
    """
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE papers (
                pmid TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT NOT NULL DEFAULT '',
                abstract TEXT NOT NULL DEFAULT '',
                journal TEXT,
                year INTEGER,
                authors TEXT,
                grants TEXT,
                mesh_terms TEXT,
                subjects TEXT,
                source TEXT NOT NULL
            );
        """)
        for p in papers:
            conn.execute(
                "INSERT INTO papers "
                "(pmid, doi, title, abstract, journal, year, "
                " authors, grants, mesh_terms, subjects, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    p["pmid"], p.get("doi"), p.get("title", ""),
                    p.get("abstract", ""), p.get("journal"),
                    p.get("year"),
                    json.dumps(p.get("authors")) if p.get("authors") else None,
                    json.dumps(p.get("grants")) if p.get("grants") else None,
                    json.dumps(p.get("mesh_terms"))
                    if p.get("mesh_terms") else None,
                    json.dumps(p.get("subjects"))
                    if p.get("subjects") else None,
                    p.get("source", "legacy"),
                ),
            )
        conn.commit()


def _seed_target_with_ratings(
    db: Database, pmids: list[str], methodology: str = "quadas_2",
) -> None:
    """Populate expert_methodology_ratings so copy_papers has work to do."""
    for i, pmid in enumerate(pmids):
        db.upsert_expert_rating(
            methodology=methodology,
            rating_source="test",
            study_label=f"Study-{i}",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low",
            pmid=pmid,
        )


@pytest.fixture
def source_db(tmp_path: Path) -> Path:
    """Legacy DB pre-populated with three papers, only two of which match."""
    path = tmp_path / "legacy.db"
    _seed_source_db(path, [
        {
            "pmid": "1001", "doi": "10.1/a", "title": "Paper A",
            "abstract": "abs A", "authors": ["Smith J"],
            "mesh_terms": ["Diabetes"], "source": "cochrane_rob",
        },
        {
            "pmid": "1002", "doi": "10.1/b", "title": "Paper B",
            "abstract": "abs B", "authors": ["Jones K"],
            "source": "cochrane_rob",
        },
        # Not referenced by ratings — must NOT be copied.
        {
            "pmid": "9999", "title": "Unrelated",
            "source": "retraction_watch",
        },
    ])
    return path


@pytest.fixture
def target_db_path(tmp_path: Path) -> Path:
    return tmp_path / "target.db"


# ---- Scope: only rated PMIDs are copied --------------------------------

class TestScope:
    def test_only_rated_pmids_are_copied(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        """A bare ``SELECT * FROM papers`` on the source would copy too much."""
        db = Database(target_db_path)
        db.initialize()
        _seed_target_with_ratings(db, ["1001", "1002"])
        db.close()

        counts = copy_papers(source_db, target_db_path)
        assert counts["newly_copied"] == 2
        assert counts["rated"] == 2
        assert counts["missing_in_source"] == 0

        db = Database(target_db_path)
        db.initialize()
        try:
            pmids_in_target = db.get_paper_pmids()
        finally:
            db.close()
        # The unrelated "9999" row must not have leaked in.
        assert pmids_in_target == {"1001", "1002"}

    def test_methodology_filter_narrows_copy_set(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        db = Database(target_db_path)
        db.initialize()
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="t",
            study_label="S1",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low", pmid="1001",
        )
        db.upsert_expert_rating(
            methodology="cochrane_rob2", rating_source="t",
            study_label="S2",
            domain_ratings={"randomization": "low"},
            overall_rating="low", pmid="1002",
        )
        db.close()

        counts = copy_papers(
            source_db, target_db_path, methodology="quadas_2",
        )
        assert counts["rated"] == 1
        assert counts["newly_copied"] == 1

        db = Database(target_db_path)
        db.initialize()
        try:
            assert db.get_paper_pmids() == {"1001"}
        finally:
            db.close()

    def test_ratings_without_pmid_are_dropped(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        """NULL-PMID rows can't be joined against papers; nothing to copy."""
        db = Database(target_db_path)
        db.initialize()
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="t",
            study_label="S-null",
            domain_ratings={"patient_selection": {"bias": "low"}},
            overall_rating="low", pmid=None,
        )
        db.close()

        counts = copy_papers(source_db, target_db_path)
        assert counts["rated"] == 0


# ---- Idempotency: existing rows are preserved --------------------------

class TestIdempotency:
    def test_already_present_rows_are_not_overwritten(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        """A re-run of copy must not clobber edits in the target."""
        db = Database(target_db_path)
        db.initialize()
        _seed_target_with_ratings(db, ["1001"])
        # Pre-populate the target with a modified row: the test asserts
        # that this modified title survives a copy pass.
        db.insert_paper({
            "pmid": "1001", "title": "EDITED IN TARGET",
            "abstract": "curator touched this", "source": "manual",
        })
        db.close()

        counts = copy_papers(source_db, target_db_path)
        assert counts["already_present"] == 1
        assert counts["newly_copied"] == 0

        db = Database(target_db_path)
        try:
            paper = db.get_paper("1001")
        finally:
            db.close()
        assert paper is not None
        assert paper["title"] == "EDITED IN TARGET"


# ---- JSON columns survive ---------------------------------------------

class TestJSONRoundTrip:
    def test_authors_and_mesh_are_preserved(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        """Re-serialisation must not turn a list into a string-of-a-list."""
        db = Database(target_db_path)
        db.initialize()
        _seed_target_with_ratings(db, ["1001"])
        db.close()

        copy_papers(source_db, target_db_path)

        db = Database(target_db_path)
        try:
            paper = db.get_paper("1001")
        finally:
            db.close()
        assert paper is not None
        assert paper["authors"] == ["Smith J"]
        assert paper["mesh_terms"] == ["Diabetes"]


# ---- Missing-from-source reporting -------------------------------------

class TestMissingReporting:
    def test_missing_pmid_counts_into_summary(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        """A rated PMID that doesn't exist in the source must be surfaced."""
        db = Database(target_db_path)
        db.initialize()
        _seed_target_with_ratings(db, ["1001", "NOT_IN_SOURCE"])
        db.close()

        counts = copy_papers(source_db, target_db_path)
        assert counts["rated"] == 2
        assert counts["found_in_source"] == 1
        assert counts["missing_in_source"] == 1
        assert counts["newly_copied"] == 1


# ---- Dry-run writes nothing -------------------------------------------

class TestDryRun:
    def test_dry_run_leaves_target_empty(
        self, source_db: Path, target_db_path: Path,
    ) -> None:
        db = Database(target_db_path)
        db.initialize()
        _seed_target_with_ratings(db, ["1001"])
        db.close()

        counts = copy_papers(source_db, target_db_path, dry_run=True)
        # found_in_source is still populated even though nothing is written,
        # so the user can see what WOULD have been copied.
        assert counts["found_in_source"] == 1
        assert counts["newly_copied"] == 0

        db = Database(target_db_path)
        db.initialize()
        try:
            pmids = db.get_paper_pmids()
        finally:
            db.close()
        assert pmids == set()
