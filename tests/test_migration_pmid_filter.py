"""Step-8 helper: migration PMID-filter + batch annotator argparse tests.

These tests cover the two Step-8 preparation tools:

1. ``scripts/migrations/add_methodology_support.py --only-pmids`` —
   restrict the legacy→fresh copy to a hand-picked PMID subset (e.g.
   "the 15 papers that scored kappa=1 in the previous comparison run").
2. ``scripts/batch_annotate_pmids.py`` — argparse smoke tests and the
   PMID-list loader. The actual annotation loop is exercised by
   existing single-paper tests; here we only confirm the argument
   validator catches invalid combinations before spinning up any DB.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.migrations.add_methodology_support import (
    load_pmid_filter,
    migrate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- Fixtures ----------------------------------------------------------

def _write_legacy_db(path: Path, pmids_with_ratings: list[str]) -> None:
    """Create a pre-methodology DB with papers for *pmids*."""
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE papers (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                source TEXT,
                overall_rob TEXT,
                randomization_bias TEXT
            );
            CREATE TABLE enrichments (
                pmid TEXT PRIMARY KEY,
                suspicion_level TEXT
            );
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
        for pmid in pmids_with_ratings:
            conn.execute(
                "INSERT INTO papers VALUES (?, 'Trial ' || ?, 'Abs', "
                "'cochrane_rob', 'low', 'low')",
                (pmid, pmid),
            )
            conn.execute(
                "INSERT INTO enrichments VALUES (?, 'low')", (pmid,),
            )
            conn.execute(
                "INSERT INTO annotations VALUES "
                "(?, 'anthropic', '{\"x\":1}', 'low', 0.2, 'high', '2026-01-01')",
                (pmid,),
            )
        conn.commit()


# ---- load_pmid_filter --------------------------------------------------

class TestLoadPmidFilter:
    def test_reads_one_pmid_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "pmids.txt"
        path.write_text("12345\n67890\n", encoding="utf-8")
        assert load_pmid_filter(path) == {"12345", "67890"}

    def test_strips_comments_and_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "pmids.txt"
        path.write_text(
            "# Header comment\n"
            "12345\n"
            "\n"
            "67890  # trailing comment\n"
            "   # indented comment\n",
            encoding="utf-8",
        )
        assert load_pmid_filter(path) == {"12345", "67890"}

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="not found"):
            load_pmid_filter(tmp_path / "nope.txt")

    def test_empty_file_exits(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("# only a comment\n\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no PMIDs"):
            load_pmid_filter(path)


# ---- Migration with --only-pmids --------------------------------------

class TestMigrationPmidFilter:
    def test_only_listed_pmids_copied(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.db"
        _write_legacy_db(legacy, ["P1", "P2", "P3", "P4"])
        new = tmp_path / "fresh.db"
        summary = migrate(
            legacy, new,
            archive_legacy=False,
            copy_papers=True, copy_enrichments=True,
            copy_annotations=False, copy_reviews=False,
            force=False,
            only_pmids={"P1", "P3"},
        )
        assert summary["papers"] == 2
        assert summary["enrichments"] == 2
        with sqlite3.connect(str(new)) as dst:
            pmids = {r[0] for r in dst.execute("SELECT pmid FROM papers")}
            assert pmids == {"P1", "P3"}
            enrich_pmids = {r[0] for r in dst.execute("SELECT pmid FROM enrichments")}
            assert enrich_pmids == {"P1", "P3"}

    def test_annotation_copy_honours_filter(self, tmp_path: Path) -> None:
        """Annotations for filtered-out papers are not copied (FK stays satisfied)."""
        legacy = tmp_path / "legacy.db"
        _write_legacy_db(legacy, ["P1", "P2", "P3"])
        new = tmp_path / "fresh.db"
        summary = migrate(
            legacy, new,
            archive_legacy=False,
            copy_papers=True, copy_enrichments=False,
            copy_annotations=True, copy_reviews=False,
            force=False,
            only_pmids={"P2"},
        )
        assert summary["papers"] == 1
        assert summary["annotations"] == 1
        with sqlite3.connect(str(new)) as dst:
            ann_pmids = {r[0] for r in dst.execute("SELECT pmid FROM annotations")}
            assert ann_pmids == {"P2"}

    def test_none_filter_copies_everything(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.db"
        _write_legacy_db(legacy, ["P1", "P2", "P3"])
        new = tmp_path / "fresh.db"
        summary = migrate(
            legacy, new,
            archive_legacy=False,
            copy_papers=True, copy_enrichments=False,
            copy_annotations=False, copy_reviews=False,
            force=False,
            only_pmids=None,
        )
        assert summary["papers"] == 3

    def test_cli_e2e_with_only_pmids_flag(self, tmp_path: Path) -> None:
        """End-to-end: CLI invocation reads a PMID file and filters correctly."""
        legacy = tmp_path / "legacy.db"
        _write_legacy_db(legacy, ["P1", "P2", "P3"])
        new = tmp_path / "fresh.db"
        pmid_file = tmp_path / "picks.txt"
        pmid_file.write_text("# 2 of 3\nP1\nP3\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "migrations" / "add_methodology_support.py"),
                "--from", str(legacy),
                "--to", str(new),
                "--copy-papers",
                "--only-pmids", str(pmid_file),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "Filtered to 2 PMID(s)" in result.stdout
        with sqlite3.connect(str(new)) as dst:
            pmids = {r[0] for r in dst.execute("SELECT pmid FROM papers")}
            assert pmids == {"P1", "P3"}


# ---- Batch annotator argparse ------------------------------------------

class TestBatchAnnotatorArgparse:
    """Argparse validator rejects bad flag combinations before DB access."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "batch_annotate_pmids.py"),
                *args,
            ],
            capture_output=True, text=True,
        )

    def test_help_mentions_methodology_flag(self) -> None:
        result = self._run(["--help"])
        assert result.returncode == 0, result.stderr
        assert "--methodology" in result.stdout
        assert "--pmids-file" in result.stdout

    def test_cochrane_rob2_requires_decomposed(self, tmp_path: Path) -> None:
        pmid_file = tmp_path / "x.txt"
        pmid_file.write_text("12345\n", encoding="utf-8")
        result = self._run([
            "--pmids-file", str(pmid_file),
            "--methodology", "cochrane_rob2",
        ])
        assert result.returncode != 0
        assert "requires --decomposed" in (result.stderr + result.stdout)

    def test_cochrane_rob2_rejects_single_call(self, tmp_path: Path) -> None:
        pmid_file = tmp_path / "x.txt"
        pmid_file.write_text("12345\n", encoding="utf-8")
        result = self._run([
            "--pmids-file", str(pmid_file),
            "--methodology", "cochrane_rob2",
            "--single-call",
        ])
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        # Either the mutually-exclusive-group handler or our validator
        # catches this. Both give an actionable message.
        assert "--single-call" in combined or "decomposed" in combined
