"""Tests for the manually_verified FK table and helper."""

import sqlite3
from pathlib import Path

import pytest

from biasbuster.database import Database


@pytest.fixture
def db_with_paper(tmp_path: Path) -> Database:
    """Fresh DB with one paper inserted."""
    db = Database(tmp_path / "t.db")
    db.initialize()
    db.insert_paper(
        {"pmid": "P1", "title": "T", "abstract": "A", "source": "cochrane_rob"}
    )
    return db


def test_manually_verified_table_exists(db_with_paper: Database) -> None:
    row = db_with_paper.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='manually_verified'"
    ).fetchone()
    assert row is not None


def test_manually_verified_composite_pk(db_with_paper: Database) -> None:
    cols = db_with_paper.conn.execute(
        "PRAGMA table_info(manually_verified)"
    ).fetchall()
    pk_cols = sorted(c[1] for c in cols if c[5] > 0)  # pk column index is 5
    assert pk_cols == ["pmid", "verification_set"]


def test_manually_verified_foreign_key(db_with_paper: Database) -> None:
    # FK is IMMEDIATE (no DEFERRABLE clause), so IntegrityError fires at
    # execute time, not at commit time.
    with pytest.raises(sqlite3.IntegrityError):
        db_with_paper.conn.execute(
            "INSERT INTO manually_verified (pmid, verification_set) "
            "VALUES (?, ?)",
            ("UNKNOWN_PMID", "rob2_manual_verify_20260421"),
        )
    db_with_paper.conn.rollback()


def test_manually_verified_insert_valid_row(db_with_paper: Database) -> None:
    db_with_paper.conn.execute(
        "INSERT INTO manually_verified (pmid, verification_set) VALUES (?, ?)",
        ("P1", "rob2_manual_verify_20260421"),
    )
    db_with_paper.conn.commit()
    row = db_with_paper.conn.execute(
        "SELECT fulltext_ok, added_at FROM manually_verified WHERE pmid='P1'"
    ).fetchone()
    assert row is not None
    assert row["fulltext_ok"] == 0
    assert row["added_at"] is not None
