"""Schema + method coverage for the methodology-aware annotations table.

The fresh-DB schema keys annotations by (pmid, model_name, methodology).
These tests check: (a) a clean DB gets the new PK and indices; (b) the
same (pmid, model_name) can hold annotations under different
methodologies; (c) read/write methods filter correctly by methodology;
(d) opening a pre-methodology database raises LegacySchemaError instead
of silently auto-migrating.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from biasbuster.database import Database, LegacySchemaError


@pytest.fixture
def fresh_db(tmp_path: Path) -> Database:
    """A brand-new methodology-aware database with one paper inserted."""
    db = Database(tmp_path / "fresh.db")
    db.initialize()
    db.insert_paper({"pmid": "P1", "title": "T", "abstract": "A", "source": "test"})
    return db


@pytest.fixture
def legacy_db_path(tmp_path: Path) -> Path:
    """Create a DB with the pre-methodology schema (PK (pmid, model_name))."""
    path = tmp_path / "legacy.db"
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE papers (pmid TEXT PRIMARY KEY, title TEXT, source TEXT);
            CREATE TABLE annotations (
                pmid TEXT NOT NULL,
                model_name TEXT NOT NULL,
                annotation JSON NOT NULL,
                overall_severity TEXT,
                overall_bias_probability REAL,
                confidence TEXT,
                annotated_at TEXT,
                PRIMARY KEY (pmid, model_name)
            );
            CREATE TABLE human_reviews (
                pmid TEXT, model_name TEXT, validated INTEGER,
                override_severity TEXT, annotation JSON, flagged INTEGER,
                notes TEXT, reviewed_at TEXT,
                PRIMARY KEY (pmid, model_name)
            );
        """)
        conn.execute("INSERT INTO papers VALUES ('P9', 'legacy', 'test')")
        conn.execute(
            "INSERT INTO annotations VALUES "
            "('P9', 'anthropic', '{\"x\": 1}', 'high', 0.9, 'medium', "
            "'2026-01-01')"
        )
        conn.commit()
    return path


class TestFreshSchema:
    """Verify the fresh DB gets the new three-part PK."""

    def test_annotations_primary_key_includes_methodology(
        self, fresh_db: Database
    ) -> None:
        cols = {
            c[1]: c
            for c in fresh_db.conn.execute(
                "PRAGMA table_info(annotations)"
            ).fetchall()
        }
        assert "methodology" in cols
        # Primary key columns: pk > 0 ordinal.
        pk_cols = sorted(
            (c[1] for c in cols.values() if c[5] > 0),
            key=lambda n: cols[n][5],
        )
        assert pk_cols == ["pmid", "model_name", "methodology"]

    def test_methodology_indices_exist(self, fresh_db: Database) -> None:
        rows = fresh_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {r["name"] for r in rows}
        assert "idx_annotations_methodology" in index_names
        assert "idx_annotations_meth_model" in index_names

    def test_human_reviews_fk_includes_methodology(
        self, fresh_db: Database
    ) -> None:
        fk_rows = fresh_db.conn.execute(
            "PRAGMA foreign_key_list(human_reviews)"
        ).fetchall()
        fk_cols = {r["from"] for r in fk_rows}
        assert {"pmid", "model_name", "methodology"} <= fk_cols


class TestPerMethodologyIsolation:
    """Same (pmid, model) can hold annotations under multiple methodologies."""

    def test_two_methodologies_coexist(self, fresh_db: Database) -> None:
        fresh_db.insert_annotation(
            "P1", "anthropic", {"overall_severity": "moderate"},
            methodology="biasbuster",
        )
        fresh_db.insert_annotation(
            "P1", "anthropic", {"overall_severity": "some_concerns"},
            methodology="cochrane_rob2",
        )
        rows = fresh_db.conn.execute(
            "SELECT methodology, overall_severity FROM annotations "
            "WHERE pmid = ? ORDER BY methodology",
            ("P1",),
        ).fetchall()
        assert [r["methodology"] for r in rows] == [
            "biasbuster", "cochrane_rob2",
        ]
        assert [r["overall_severity"] for r in rows] == [
            "moderate", "some_concerns",
        ]

    def test_has_annotation_filters_by_methodology(
        self, fresh_db: Database
    ) -> None:
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster",
        )
        assert fresh_db.has_annotation(
            "P1", "anthropic", methodology="biasbuster"
        )
        assert not fresh_db.has_annotation(
            "P1", "anthropic", methodology="cochrane_rob2"
        )

    def test_delete_annotation_only_removes_target_methodology(
        self, fresh_db: Database
    ) -> None:
        fresh_db.insert_annotation("P1", "anthropic", {}, methodology="biasbuster")
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="cochrane_rob2"
        )
        removed = fresh_db.delete_annotation(
            "P1", "anthropic", methodology="biasbuster"
        )
        assert removed is True
        remaining = fresh_db.get_annotations(pmid="P1")
        assert len(remaining) == 1
        assert remaining[0]["methodology"] == "cochrane_rob2"

    def test_upsert_preserves_methodology_identity(
        self, fresh_db: Database
    ) -> None:
        fresh_db.upsert_annotation(
            "P1", "anthropic", {"overall_severity": "low"},
            methodology="cochrane_rob2",
            methodology_version="rob2-2019",
        )
        fresh_db.upsert_annotation(
            "P1", "anthropic", {"overall_severity": "high"},
            methodology="cochrane_rob2",
            methodology_version="rob2-2019",
        )
        rows = fresh_db.conn.execute(
            "SELECT COUNT(*) as n FROM annotations WHERE pmid = ?",
            ("P1",),
        ).fetchone()
        assert rows["n"] == 1
        single = fresh_db.get_annotations(pmid="P1")[0]
        assert single["overall_severity"] == "high"
        assert single["methodology"] == "cochrane_rob2"
        assert single["methodology_version"] == "rob2-2019"

    def test_get_annotations_filter(self, fresh_db: Database) -> None:
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster",
        )
        fresh_db.insert_annotation(
            "P1", "deepseek", {}, methodology="biasbuster",
        )
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="cochrane_rob2",
        )
        biasbuster_rows = fresh_db.get_annotations(methodology="biasbuster")
        rob2_rows = fresh_db.get_annotations(methodology="cochrane_rob2")
        assert len(biasbuster_rows) == 2
        assert len(rob2_rows) == 1
        assert rob2_rows[0]["model_name"] == "anthropic"

    def test_get_annotated_pmids_filter(self, fresh_db: Database) -> None:
        fresh_db.insert_paper({"pmid": "P2", "source": "test"})
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster"
        )
        fresh_db.insert_annotation(
            "P2", "anthropic", {}, methodology="cochrane_rob2"
        )
        bb_pmids = fresh_db.get_annotated_pmids("anthropic", methodology="biasbuster")
        rob2_pmids = fresh_db.get_annotated_pmids(
            "anthropic", methodology="cochrane_rob2"
        )
        assert bb_pmids == {"P1"}
        assert rob2_pmids == {"P2"}
        # Without filter, both should appear.
        assert fresh_db.get_annotated_pmids("anthropic") == {"P1", "P2"}

    def test_default_methodology_is_biasbuster(
        self, fresh_db: Database
    ) -> None:
        """Callers that don't pass methodology get 'biasbuster' for back-compat."""
        fresh_db.insert_annotation("P1", "anthropic", {})
        row = fresh_db.conn.execute(
            "SELECT methodology FROM annotations WHERE pmid = ?",
            ("P1",),
        ).fetchone()
        assert row["methodology"] == "biasbuster"

    def test_delete_annotations_for_pmids_respects_methodology(
        self, fresh_db: Database
    ) -> None:
        """Bulk-delete with methodology= leaves other methodologies untouched."""
        fresh_db.insert_paper({"pmid": "P2", "source": "test"})
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster",
        )
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="cochrane_rob2",
        )
        fresh_db.insert_annotation(
            "P2", "anthropic", {}, methodology="biasbuster",
        )
        removed = fresh_db.delete_annotations_for_pmids(
            ["P1", "P2"], methodology="biasbuster"
        )
        assert removed == 2
        remaining = fresh_db.get_annotations()
        assert len(remaining) == 1
        assert remaining[0]["pmid"] == "P1"
        assert remaining[0]["methodology"] == "cochrane_rob2"

    def test_empty_string_methodology_is_not_treated_as_any(
        self, fresh_db: Database
    ) -> None:
        """Empty-string methodology is a distinct (if nonsensical) value, not 'any'.

        Writers default to 'biasbuster'; readers use ``is not None`` so ``""``
        would match zero rows (nothing is tagged ""). This documents the
        behaviour and prevents a future refactor from silently treating
        ``""`` as "no filter".
        """
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster"
        )
        # Reader with empty string filter: must not match the biasbuster row.
        rows = fresh_db.get_annotations(methodology="")
        assert rows == []
        # Reader with None: returns the biasbuster row as "any methodology".
        rows = fresh_db.get_annotations(methodology=None)
        assert len(rows) == 1


class TestLegacySchemaGuard:
    """Opening a legacy DB must refuse to auto-upgrade."""

    def test_initialize_raises_on_legacy_schema(
        self, legacy_db_path: Path
    ) -> None:
        db = Database(legacy_db_path)
        with pytest.raises(LegacySchemaError) as exc:
            db.initialize()
        assert "pre-methodology" in str(exc.value)
        assert "add_methodology_support.py" in str(exc.value)
        db.close()


class TestHumanReviewMethodology:
    """Reviews are keyed by (pmid, model_name, methodology) too."""

    def test_review_scoped_to_methodology(self, fresh_db: Database) -> None:
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="biasbuster"
        )
        fresh_db.insert_annotation(
            "P1", "anthropic", {}, methodology="cochrane_rob2"
        )
        fresh_db.upsert_review(
            "P1", "anthropic", validated=True,
            notes="biasbuster review",
            methodology="biasbuster",
        )
        fresh_db.upsert_review(
            "P1", "anthropic", validated=False,
            notes="rob2 review",
            methodology="cochrane_rob2",
        )
        bb_reviews = fresh_db.get_reviews(
            model_name="anthropic", methodology="biasbuster"
        )
        rob2_reviews = fresh_db.get_reviews(
            model_name="anthropic", methodology="cochrane_rob2"
        )
        assert len(bb_reviews) == 1
        assert bb_reviews[0]["notes"] == "biasbuster review"
        assert bb_reviews[0]["validated"] == 1
        assert len(rob2_reviews) == 1
        assert rob2_reviews[0]["notes"] == "rob2 review"
        assert rob2_reviews[0]["validated"] == 0


class TestMigrationScript:
    """End-to-end test of the legacy-quarantine migration."""

    def test_migrate_creates_fresh_db_and_copies_papers(
        self, legacy_db_path: Path, tmp_path: Path
    ) -> None:
        from scripts.migrations.add_methodology_support import migrate

        new_db = tmp_path / "fresh.db"
        summary = migrate(
            legacy_db_path,
            new_db,
            archive_legacy=False,
            copy_papers=True,
            copy_enrichments=False,
            copy_annotations=False,
            copy_reviews=False,
            force=False,
        )
        assert summary["papers"] == 1
        # New DB should have the new schema.
        db = Database(new_db)
        db.initialize()  # should not raise — already migrated
        papers = db.get_papers()
        assert {p["pmid"] for p in papers} == {"P9"}
        assert db.get_annotations() == []
        db.close()

    def test_migrate_copies_annotations_as_biasbuster_legacy(
        self, legacy_db_path: Path, tmp_path: Path
    ) -> None:
        from scripts.migrations.add_methodology_support import migrate

        new_db = tmp_path / "fresh.db"
        summary = migrate(
            legacy_db_path,
            new_db,
            archive_legacy=False,
            copy_papers=True,
            copy_enrichments=False,
            copy_annotations=True,
            copy_reviews=False,
            force=False,
        )
        assert summary["annotations"] == 1
        db = Database(new_db)
        db.initialize()
        ann = db.get_annotations()[0]
        assert ann["methodology"] == "biasbuster"
        assert ann["methodology_version"] == "legacy"
        assert ann["overall_severity"] == "high"
        db.close()

    def test_migrate_refuses_if_target_exists_without_force(
        self, legacy_db_path: Path, tmp_path: Path
    ) -> None:
        from scripts.migrations.add_methodology_support import migrate

        new_db = tmp_path / "fresh.db"
        new_db.write_text("")  # occupy the path
        with pytest.raises(SystemExit, match="already exists"):
            migrate(
                legacy_db_path, new_db,
                archive_legacy=False, copy_papers=False, copy_enrichments=False,
                copy_annotations=False, copy_reviews=False, force=False,
            )

    def test_migrate_rejects_already_migrated_source(
        self, tmp_path: Path
    ) -> None:
        """Running migrate on an already-migrated DB should error out."""
        from scripts.migrations.add_methodology_support import migrate

        already = tmp_path / "already.db"
        Database(already).initialize()
        with pytest.raises(SystemExit, match="already migrated"):
            migrate(
                already, tmp_path / "newer.db",
                archive_legacy=False, copy_papers=False, copy_enrichments=False,
                copy_annotations=False, copy_reviews=False, force=False,
            )

    def test_migrate_rejects_annotations_without_papers_before_creating_db(
        self, legacy_db_path: Path, tmp_path: Path
    ) -> None:
        """--copy-annotations without --copy-papers fails BEFORE creating the new DB.

        SystemExit does not inherit from Exception and so cannot be caught
        by the migration's in-flight rollback block. Instead, the FK
        dependency is validated up-front so the fresh DB is never created.
        """
        from scripts.migrations.add_methodology_support import migrate

        new_db = tmp_path / "partial.db"
        with pytest.raises(SystemExit, match="requires --copy-papers"):
            migrate(
                legacy_db_path, new_db,
                archive_legacy=False,
                copy_papers=False, copy_enrichments=False,
                copy_annotations=True, copy_reviews=False, force=False,
            )
        assert not new_db.exists(), (
            "fresh DB was created despite invalid --copy-annotations plan"
        )

    def test_archive_legacy_copies_without_modifying_original(
        self, legacy_db_path: Path, tmp_path: Path
    ) -> None:
        """--archive-legacy writes a timestamped copy; original DB is untouched."""
        from scripts.migrations.add_methodology_support import migrate

        new_db = tmp_path / "fresh.db"
        legacy_before_size = legacy_db_path.stat().st_size
        migrate(
            legacy_db_path, new_db,
            archive_legacy=True,
            copy_papers=True, copy_enrichments=False,
            copy_annotations=False, copy_reviews=False, force=False,
        )
        assert legacy_db_path.exists()
        assert legacy_db_path.stat().st_size == legacy_before_size
        siblings = list(legacy_db_path.parent.glob("legacy.legacy_*.db"))
        assert len(siblings) == 1
