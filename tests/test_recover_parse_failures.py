"""Tests for the parse-failure recovery script's wrong-paper exclusion.

Regression context (2026-07-17): ``recover_parse_failures.py`` recovered
two qwen3.6 fulltext domain rows for RCT030 — the wrong-paper acquisition
(Phase 1 resolved the parent Cochrane review instead of the primary
trial). The model's signalling answers describe a different document, so
algorithmic recovery injected wrong-paper judgements into the benchmark.
The rows were reverted and ``WRONG_PAPER_RCTS`` added as a guard; these
tests pin that guard.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "studies" / "eisele_metzger_replication" / "recover_parse_failures.py"
)
_spec = importlib.util.spec_from_file_location(
    "recover_parse_failures", _MODULE_PATH)
recover_parse_failures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recover_parse_failures)


RECOVERABLE_RESPONSE = json.dumps({
    "domain": "missing_outcome_data",
    "signalling_answers": {"3.1": "Y"},
    "justification": "Outcome data available for nearly all participants.",
    # Note: no "judgement" key — the known schema-drift mode.
})


@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with the minimal benchmark schema used by recovery."""
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
        CREATE TABLE evaluation_run (
            rct_id TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            model_id TEXT NOT NULL,
            protocol TEXT NOT NULL,
            pass_n INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            raw_response TEXT,
            parse_status TEXT NOT NULL,
            parse_attempts INTEGER NOT NULL DEFAULT 1,
            error TEXT,
            PRIMARY KEY (rct_id, source, domain)
        );
        """
    )
    yield conn
    conn.close()


def _insert_parse_failure(conn: sqlite3.Connection, rct_id: str,
                          source: str, domain: str) -> None:
    """Insert a parse-failure pair whose raw response IS recoverable."""
    conn.execute(
        "INSERT INTO benchmark_judgment "
        "(rct_id, source, domain, judgment, valid, raw_label) "
        "VALUES (?, ?, ?, NULL, 0, NULL)",
        (rct_id, source, domain),
    )
    conn.execute(
        "INSERT INTO evaluation_run "
        "(rct_id, source, domain, model_id, protocol, pass_n, started_at, "
        " raw_response, parse_status) "
        "VALUES (?, ?, ?, 'qwen3.6:35b', 'fulltext', 1, '2026-01-01', ?, "
        " 'parse_failure')",
        (rct_id, source, domain, RECOVERABLE_RESPONSE),
    )


class TestWrongPaperExclusion:
    def test_wrong_paper_rct_is_never_recovered(
            self, conn: sqlite3.Connection) -> None:
        excluded = next(iter(recover_parse_failures.WRONG_PAPER_RCTS))
        _insert_parse_failure(
            conn, excluded, "qwen3_6_35b_fulltext_pass1", "d3")

        counts = recover_parse_failures.run_recovery(conn, dry_run=False)

        assert counts["excluded_wrong_paper"] == 1
        assert counts["domain_recovered"] == 0
        row = conn.execute(
            "SELECT judgment, valid, raw_label FROM benchmark_judgment "
            "WHERE rct_id = ?", (excluded,),
        ).fetchone()
        assert row == (None, 0, None)
        # No FALLBACK row of any kind may exist for a wrong-paper RCT.
        n_fallback = conn.execute(
            "SELECT COUNT(*) FROM benchmark_judgment "
            "WHERE rct_id = ? AND raw_label = 'FALLBACK'", (excluded,),
        ).fetchone()[0]
        assert n_fallback == 0

    def test_other_rcts_still_recovered(
            self, conn: sqlite3.Connection) -> None:
        _insert_parse_failure(
            conn, "RCT999", "qwen3_6_35b_fulltext_pass1", "d3")

        counts = recover_parse_failures.run_recovery(conn, dry_run=False)

        assert counts["excluded_wrong_paper"] == 0
        assert counts["domain_recovered"] == 1
        judgment, valid, raw_label = conn.execute(
            "SELECT judgment, valid, raw_label FROM benchmark_judgment "
            "WHERE rct_id = 'RCT999'",
        ).fetchone()
        # D3 rule: 3.1 in {Y, PY} => low (algorithms.domain_3_missing_data).
        assert (judgment, valid, raw_label) == ("low", 1, "FALLBACK")

    def test_dry_run_writes_nothing(self, conn: sqlite3.Connection) -> None:
        _insert_parse_failure(
            conn, "RCT999", "qwen3_6_35b_fulltext_pass1", "d3")

        counts = recover_parse_failures.run_recovery(conn, dry_run=True)

        assert counts["domain_recovered"] == 1
        valid = conn.execute(
            "SELECT valid FROM benchmark_judgment WHERE rct_id = 'RCT999'",
        ).fetchone()[0]
        assert valid == 0
