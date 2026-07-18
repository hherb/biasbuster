"""Tests for the pre-manuscript spot-check analysis helpers.

Covers the pure functions and ``LowAuditRow`` logic in
``studies/eisele_metzger_replication/premanuscript_spotchecks.py`` (runbook §6:
the Sonnet ``low``-judgement audit and the per-domain run-to-run instability
audit). Most DB loaders are exercised end-to-end by running the script against
the benchmark DB; here we pin the chance-free computation so a refactor cannot
silently change what "instability" or a "correct low" means, plus a focused
in-memory test of ``load_low_audit`` proving algorithm-derived (``FALLBACK``)
judgements are excluded from the "model-emitted low" audit.
"""
from __future__ import annotations

import sqlite3

from tests.conftest import load_study_module

pms = load_study_module("premanuscript_spotchecks")


# --- disagreement_stats / disagreement_rate ---------------------------

class TestDisagreementStats:
    def test_empty(self) -> None:
        assert pms.disagreement_stats([]) == (0, 0)
        assert pms.disagreement_rate(0, 0) is None

    def test_all_agree(self) -> None:
        pairs = [("low", "low"), ("high", "high"), ("some_concerns", "some_concerns")]
        assert pms.disagreement_stats(pairs) == (0, 3)
        assert pms.disagreement_rate(0, 3) == 0.0

    def test_mixed(self) -> None:
        pairs = [("low", "high"), ("low", "low"), ("high", "some_concerns")]
        n_dis, n_tot = pms.disagreement_stats(pairs)
        assert (n_dis, n_tot) == (2, 3)
        assert pms.disagreement_rate(n_dis, n_tot) == 2 / 3


# --- dominant_domain --------------------------------------------------

class TestDominantDomain:
    def test_empty_and_all_zero_total(self) -> None:
        assert pms.dominant_domain({}) is None
        assert pms.dominant_domain({"d1": (0, 0), "d2": (0, 0)}) is None

    def test_highest_rate_wins_not_highest_count(self) -> None:
        # d1: 10/100 = 10%; d2: 3/10 = 30% -> d2 dominates despite fewer flips.
        assert pms.dominant_domain({"d1": (10, 100), "d2": (3, 10)}) == "d2"

    def test_zero_total_domain_ignored(self) -> None:
        assert pms.dominant_domain({"d1": (0, 0), "d2": (1, 10)}) == "d2"

    def test_tie_on_rate_broken_by_count(self) -> None:
        # Same 20% rate; d3 has the larger absolute count -> wins.
        assert pms.dominant_domain({"d2": (2, 10), "d3": (4, 20)}) == "d3"

    def test_tie_on_rate_and_count_broken_by_domain_order(self) -> None:
        # Identical (rate, count); earlier domain in SIGNALLING_DOMAINS wins.
        assert pms.dominant_domain({"d5": (2, 10), "d2": (2, 10)}) == "d2"


# --- LowAuditRow ------------------------------------------------------

def _row(cochrane_overall, pass_overall, cochrane_domains=None, pass_domains=None):
    return pms.LowAuditRow(
        rct_id="RCTxxx",
        cochrane_overall=cochrane_overall,
        cochrane_domains=cochrane_domains or {},
        pass_overall=pass_overall,
        pass_domains=pass_domains or {},
        low_pass_rationales={},
    )


class TestLowAuditRow:
    def test_low_passes_sorted(self) -> None:
        row = _row("low", {3: "low", 1: "some_concerns", 2: "low"})
        assert row.low_passes() == [2, 3]

    def test_correct_low_passes_when_cochrane_low(self) -> None:
        row = _row("low", {1: "low", 2: "low", 3: "some_concerns"})
        assert row.correct_low_passes() == [1, 2]

    def test_correct_low_passes_empty_when_cochrane_not_low(self) -> None:
        # RCT099 shape: model says low on every pass but Cochrane is some_concerns.
        row = _row("some_concerns", {1: "low", 2: "low", 3: "low"})
        assert row.low_passes() == [1, 2, 3]
        assert row.correct_low_passes() == []

    def test_differing_domains_ignores_none(self) -> None:
        # RCT099 shape: only D2 differs (model low, Cochrane some_concerns);
        # a NULL on either side is not counted as a difference.
        row = _row(
            "some_concerns",
            {1: "low"},
            cochrane_domains={"d1": "low", "d2": "some_concerns",
                              "d3": "low", "d4": "low", "d5": None},
            pass_domains={1: {"d1": "low", "d2": "low", "d3": "low",
                              "d4": None, "d5": "low"}},
        )
        assert row.differing_domains(1) == ["d2"]


# --- load_low_audit FALLBACK exclusion (in-memory DB) -----------------

_DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")


def _memory_db() -> sqlite3.Connection:
    """In-memory benchmark_judgment mirroring the real columns."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE benchmark_judgment (
               rct_id TEXT NOT NULL,
               source TEXT NOT NULL,
               domain TEXT NOT NULL,
               judgment TEXT,
               rationale TEXT,
               valid INTEGER NOT NULL DEFAULT 1,
               raw_label TEXT,
               PRIMARY KEY (rct_id, source, domain)
           )"""
    )
    return conn


def _seed_all_domains(
    conn: sqlite3.Connection, rct_id: str, source: str, judgment: str,
    *, raw_label: str | None, rationale: str | None = None,
) -> None:
    """Insert the five signalling domains + overall, all at ``judgment``."""
    for d in _DOMAINS:
        conn.execute(
            """INSERT INTO benchmark_judgment
               (rct_id, source, domain, judgment, rationale, valid, raw_label)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (rct_id, source, d, judgment,
             rationale if d == "overall" else None, raw_label),
        )


class TestLoadLowAuditFallbackExclusion:
    """``load_low_audit`` must audit only the model's OWN emitted ``low`` calls.

    Algorithm-derived judgements (``raw_label='FALLBACK'``) carry no model
    reasoning, so they must not be counted as ``low`` calls nor contribute a
    rationale — otherwise the §3.5 "right-for-the-right-reasons" precision would
    be inflated by rows the model never actually reasoned to.
    """

    def _db_with_mixed_passes(self) -> sqlite3.Connection:
        conn = _memory_db()
        # Cochrane gold: overall low (so a genuine model low is "correct").
        _seed_all_domains(conn, "RCTAUD", "cochrane", "low", raw_label=None)
        # pass1: genuine model low.
        _seed_all_domains(conn, "RCTAUD", "sonnet_4_6_fulltext_pass1", "low",
                          raw_label="low", rationale="genuine model reasoning")
        # pass2: FALLBACK-derived low — must be excluded from the audit.
        _seed_all_domains(conn, "RCTAUD", "sonnet_4_6_fulltext_pass2", "low",
                          raw_label="FALLBACK", rationale="algorithm-derived")
        # pass3: genuine some_concerns (not a low call).
        _seed_all_domains(conn, "RCTAUD", "sonnet_4_6_fulltext_pass3",
                          "some_concerns", raw_label="some_concerns")
        return conn

    def test_fallback_low_pass_not_counted(self) -> None:
        conn = self._db_with_mixed_passes()
        try:
            rows = pms.load_low_audit(conn, "sonnet_4_6")
        finally:
            conn.close()
        assert len(rows) == 1
        row = rows[0]
        # Only the genuine pass-1 low counts; the FALLBACK pass-2 low does not.
        assert row.low_passes() == [1]
        assert row.correct_low_passes() == [1]
        # The FALLBACK pass shows no model-emitted overall.
        assert row.pass_overall.get(2) is None
        # Only the genuine low contributes a rationale.
        assert row.low_pass_rationales == {1: "genuine model reasoning"}

    def test_rct_low_only_via_fallback_is_omitted(self) -> None:
        # An RCT whose only overall `low` is FALLBACK-derived must not surface
        # in the audit at all (the DISTINCT query excludes it).
        conn = _memory_db()
        _seed_all_domains(conn, "RCTFB", "cochrane", "low", raw_label=None)
        _seed_all_domains(conn, "RCTFB", "sonnet_4_6_fulltext_pass1", "low",
                          raw_label="FALLBACK")
        try:
            rows = pms.load_low_audit(conn, "sonnet_4_6")
        finally:
            conn.close()
        assert rows == []
