"""Tests for the wrong-paper (RCT030) exclusion in the κ analysis path.

Regression context (2026-07-17, issue #29): RCT030 is the wrong-paper
acquisition (Phase 1 resolved the parent Cochrane review instead of the
primary trial). Its 179 valid judgement rows were counted in every phase-6
κ table because ``compute_phase6_kappa`` had no RCT-level exclusion. The fix
centralises ``WRONG_PAPER_RCTS`` in ``exclusions.py`` and filters it in the
model-vs-Cochrane loaders. These tests pin both the SQL-fragment helper and
the loader behaviour.
"""
from __future__ import annotations

import sqlite3
import sys

import pytest

from tests.conftest import load_study_module

exclusions = load_study_module("exclusions")
# compute_phase6_kappa adds the study dir to sys.path at import time, so its
# own ``from exclusions import ...`` / ``from sanity_check_kappa import ...``
# resolve when exec'd here.
cpk = load_study_module("compute_phase6_kappa")


# --- wrong_paper_filter helper ----------------------------------------

class TestWrongPaperFilter:
    def test_unaliased_fragment_and_params(self) -> None:
        # Expectations derive from the actual set so the test stays correct as
        # the wrong-paper class grows (it went from {RCT030} to 13 in #29).
        expected = sorted(exclusions.WRONG_PAPER_RCTS)
        placeholders = ",".join("?" * len(expected))
        sql, params = exclusions.wrong_paper_filter("")
        assert sql == f" AND rct_id NOT IN ({placeholders})"
        assert params == expected

    def test_aliased_fragment_repeats_per_alias(self) -> None:
        expected = sorted(exclusions.WRONG_PAPER_RCTS)
        placeholders = ",".join("?" * len(expected))
        sql, params = exclusions.wrong_paper_filter("a", "b")
        assert sql == (f" AND a.rct_id NOT IN ({placeholders}) "
                       f"AND b.rct_id NOT IN ({placeholders})")
        # One full copy of the ids per alias, so positional binding stays aligned.
        assert params == expected + expected

    def test_empty_set_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exclusions, "WRONG_PAPER_RCTS", frozenset())
        assert exclusions.wrong_paper_filter("a") == ("", [])

    def test_unrecoverable_is_subset_of_wrong_paper_set(self) -> None:
        # The sensitivity-analysis excludes must all be genuine wrong papers.
        assert exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS <= exclusions.WRONG_PAPER_RCTS
        # RCT030 is both a wrong paper and unrecoverable (not PubMed-indexed).
        assert "RCT030" in exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS

    def test_explicit_exclusion_set_overrides_default(self) -> None:
        # The sensitivity κ passes UNRECOVERABLE_WRONG_PAPER_RCTS so only the
        # two unindexed wrong papers are filtered, not all 13.
        expected = sorted(exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS)
        placeholders = ",".join("?" * len(expected))
        sql, params = exclusions.wrong_paper_filter(
            "", exclusion_set=exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS)
        assert sql == f" AND rct_id NOT IN ({placeholders})"
        assert params == expected

    def test_exclusion_set_none_uses_module_default(self) -> None:
        assert exclusions.wrong_paper_filter("a") == exclusions.wrong_paper_filter(
            "a", exclusion_set=exclusions.WRONG_PAPER_RCTS)

    def test_empty_explicit_set_is_a_noop(self) -> None:
        assert exclusions.wrong_paper_filter(
            "a", exclusion_set=frozenset()) == ("", [])


# --- recoverable / unrecoverable partition ----------------------------

class TestRecoverablePartition:
    def test_recoverable_is_wrong_minus_unrecoverable(self) -> None:
        assert (exclusions.RECOVERABLE_WRONG_PAPER_RCTS
                == exclusions.WRONG_PAPER_RCTS
                - exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS)

    def test_recoverable_and_unrecoverable_partition_the_wrong_set(self) -> None:
        # The two subsets are disjoint and together cover every wrong paper —
        # the sensitivity κ excludes UNRECOVERABLE and re-includes RECOVERABLE.
        assert not (exclusions.RECOVERABLE_WRONG_PAPER_RCTS
                    & exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS)
        assert (exclusions.RECOVERABLE_WRONG_PAPER_RCTS
                | exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS
                == exclusions.WRONG_PAPER_RCTS)


# --- sensitivity mode: active exclusion set ---------------------------

class TestActiveWrongPaperSet:
    def test_default_mode_uses_full_wrong_paper_set(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpk, "SENSITIVITY_MODE", False)
        assert cpk._active_wrong_paper_set() == exclusions.WRONG_PAPER_RCTS

    def test_sensitivity_mode_uses_unrecoverable_only(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpk, "SENSITIVITY_MODE", True)
        assert (cpk._active_wrong_paper_set()
                == exclusions.UNRECOVERABLE_WRONG_PAPER_RCTS)


@pytest.fixture
def sens_conn() -> sqlite3.Connection:
    """benchmark_judgment with unrecoverable + recoverable + normal RCTs.

    RCT030 is unrecoverable (dropped in both modes); RCT008 is a recoverable
    wrong paper (dropped in the primary, re-included in the sensitivity);
    RCT999 is a normal RCT (always kept). Distinct judgments per RCT so the
    surviving pairs identify exactly which RCTs contributed.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE benchmark_judgment (
            rct_id TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            judgment TEXT,
            rationale TEXT,
            valid INTEGER NOT NULL DEFAULT 1,
            raw_label TEXT,
            PRIMARY KEY (rct_id, source, domain)
        );
        """
    )
    rows = [
        ("RCT030", "cochrane", "overall", "high"),            # unrecoverable
        ("RCT030", "sonnet_4_6_fulltext_pass2", "overall", "some_concerns"),
        ("RCT008", "cochrane", "overall", "some_concerns"),   # recoverable
        ("RCT008", "sonnet_4_6_fulltext_pass2", "overall", "high"),
        ("RCT999", "cochrane", "overall", "low"),             # normal
        ("RCT999", "sonnet_4_6_fulltext_pass2", "overall", "low"),
    ]
    conn.executemany(
        "INSERT INTO benchmark_judgment "
        "(rct_id, source, domain, judgment, valid) VALUES (?, ?, ?, ?, 1)",
        rows,
    )
    conn.commit()
    yield conn
    conn.close()


class TestLoaderSensitivityMode:
    def test_primary_drops_both_wrong_papers(
            self, sens_conn: sqlite3.Connection,
            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpk, "SENSITIVITY_MODE", False)
        pairs = cpk.load_pairs(
            sens_conn, "cochrane", "sonnet_4_6_fulltext_pass2", "overall")
        assert pairs == [("low", "low")]  # only RCT999

    def test_sensitivity_keeps_recoverable_drops_unrecoverable(
            self, sens_conn: sqlite3.Connection,
            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpk, "SENSITIVITY_MODE", True)
        pairs = cpk.load_pairs(
            sens_conn, "cochrane", "sonnet_4_6_fulltext_pass2", "overall")
        # RCT008 (recoverable) now contributes; RCT030 (unrecoverable) never does.
        assert ("high", "some_concerns") not in pairs           # RCT030 gone
        assert sorted(pairs) == [("low", "low"), ("some_concerns", "high")]


# --- sensitivity precondition guard -----------------------------------

@pytest.fixture
def rct_conn() -> sqlite3.Connection:
    """Minimal benchmark_rct table (rct_id + notes) for recovery detection."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE benchmark_rct (rct_id TEXT PRIMARY KEY, notes TEXT);")
    yield conn
    conn.close()


def _mark_recovered(conn: sqlite3.Connection, *rct_ids: str) -> None:
    """Insert benchmark_rct rows carrying the shared recovery-note marker."""
    for rct_id in rct_ids:
        note = (f"Recovered {rct_id}: wrong pmid 1 -> correct pmid 2 "
                f"({exclusions.RECOVERY_NOTE_MARKER}).")
        conn.execute(
            "INSERT INTO benchmark_rct (rct_id, notes) VALUES (?, ?)",
            (rct_id, note))
    conn.commit()


class TestRecoveryDetection:
    def test_detects_only_marked_rcts(
            self, rct_conn: sqlite3.Connection) -> None:
        _mark_recovered(rct_conn, "RCT008")
        rct_conn.execute(
            "INSERT INTO benchmark_rct (rct_id, notes) VALUES ('RCT999', NULL)")
        rct_conn.commit()
        assert cpk.recovered_wrong_paper_rcts(rct_conn) == frozenset({"RCT008"})

    def test_writer_and_detector_share_one_marker(
            self, rct_conn: sqlite3.Connection) -> None:
        # The detector must key on the same constant recover_wrong_papers writes.
        assert exclusions.RECOVERY_NOTE_MARKER in (
            "Recovered RCT040: wrong pmid 1 -> correct pmid 2 "
            f"({exclusions.RECOVERY_NOTE_MARKER}).")
        _mark_recovered(rct_conn, "RCT040")
        assert "RCT040" in cpk.recovered_wrong_paper_rcts(rct_conn)


class TestSensitivityGuard:
    def test_no_failures_when_all_recoverable_recovered(
            self, rct_conn: sqlite3.Connection) -> None:
        _mark_recovered(rct_conn, *exclusions.RECOVERABLE_WRONG_PAPER_RCTS)
        assert cpk.sensitivity_precondition_failures(rct_conn) == frozenset()

    def test_unrecovered_recoverables_are_reported(
            self, rct_conn: sqlite3.Connection) -> None:
        # Empty DB: nothing recovered, so every recoverable RCT is a failure.
        assert (cpk.sensitivity_precondition_failures(rct_conn)
                == exclusions.RECOVERABLE_WRONG_PAPER_RCTS)

    def test_partial_recovery_reports_the_remainder(
            self, rct_conn: sqlite3.Connection) -> None:
        recoverable = sorted(exclusions.RECOVERABLE_WRONG_PAPER_RCTS)
        _mark_recovered(rct_conn, *recoverable[:-1])  # all but one
        assert (cpk.sensitivity_precondition_failures(rct_conn)
                == frozenset({recoverable[-1]}))


class TestSensitivityMainGuard:
    def test_main_refuses_sensitivity_on_unrecovered_db(
            self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        # main() flips these module globals via `global`; pin them through
        # monkeypatch so its finaliser restores the defaults and the mutation
        # cannot leak into later tests regardless of collection order.
        monkeypatch.setattr(cpk, "SENSITIVITY_MODE", False)
        monkeypatch.setattr(cpk, "EXCLUDE_FALLBACK", False)
        db = tmp_path / "bench.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE benchmark_rct (rct_id TEXT PRIMARY KEY, notes TEXT);"
            "CREATE TABLE benchmark_judgment ("
            "  rct_id TEXT, source TEXT, domain TEXT, judgment TEXT,"
            "  valid INTEGER DEFAULT 1, raw_label TEXT);")
        conn.commit()
        conn.close()
        monkeypatch.setattr(
            sys, "argv",
            ["compute_phase6_kappa", "--sensitivity", "--db-path", str(db)])
        assert cpk.main() == 2


# --- loader integration -----------------------------------------------

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory benchmark_judgment with a wrong-paper and a normal RCT."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE benchmark_judgment (
            rct_id TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            judgment TEXT,
            rationale TEXT,
            valid INTEGER NOT NULL DEFAULT 1,
            raw_label TEXT,
            PRIMARY KEY (rct_id, source, domain)
        );
        """
    )
    # RCT030 (wrong-paper) and RCT999 (normal), each with a cochrane
    # ground-truth row and one model row for the 'overall' domain.
    rows = [
        ("RCT030", "cochrane", "overall", "high"),
        ("RCT030", "sonnet_4_6_fulltext_pass2", "overall", "some_concerns"),
        ("RCT999", "cochrane", "overall", "low"),
        ("RCT999", "sonnet_4_6_fulltext_pass2", "overall", "low"),
    ]
    conn.executemany(
        "INSERT INTO benchmark_judgment "
        "(rct_id, source, domain, judgment, valid) VALUES (?, ?, ?, ?, 1)",
        rows,
    )
    conn.commit()
    yield conn
    conn.close()


class TestLoaderExcludesWrongPaper:
    def test_load_pairs_drops_wrong_paper_rct(
            self, conn: sqlite3.Connection) -> None:
        pairs = cpk.load_pairs(
            conn, "cochrane", "sonnet_4_6_fulltext_pass2", "overall")
        # Only RCT999 survives; RCT030's (high, some_concerns) pair is gone.
        assert pairs == [("low", "low")]

    def test_load_judgments_drops_wrong_paper_rct(
            self, conn: sqlite3.Connection) -> None:
        judgments = cpk.load_judgments(
            conn, "sonnet_4_6_fulltext_pass2", "overall")
        assert "RCT030" not in judgments
        assert judgments == {"RCT999": "low"}
