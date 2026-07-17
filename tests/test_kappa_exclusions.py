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

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_STUDY_DIR = (
    Path(__file__).resolve().parents[1]
    / "studies" / "eisele_metzger_replication"
)


def _load(module_name: str):
    """Load a study module by file path (mirrors test_recover_parse_failures).

    Registers the module in ``sys.modules`` before executing it: ``@dataclass``
    (used by ``compute_phase6_kappa.KappaRow``) looks the class's module up in
    ``sys.modules`` during class creation and fails if it is absent.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, _STUDY_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


exclusions = _load("exclusions")
# compute_phase6_kappa adds the study dir to sys.path at import time, so its
# own ``from exclusions import ...`` / ``from sanity_check_kappa import ...``
# resolve when exec'd here.
cpk = _load("compute_phase6_kappa")


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
