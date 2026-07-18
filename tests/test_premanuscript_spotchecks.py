"""Tests for the pre-manuscript spot-check analysis helpers.

Covers the pure functions and ``LowAuditRow`` logic in
``studies/eisele_metzger_replication/premanuscript_spotchecks.py`` (runbook §6:
the Sonnet ``low``-judgement audit and the per-domain run-to-run instability
audit). The DB loaders are exercised end-to-end by running the script against
the benchmark DB; here we pin the chance-free computation so a refactor cannot
silently change what "instability" or a "correct low" means.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_STUDY_DIR = (
    Path(__file__).resolve().parents[1]
    / "studies" / "eisele_metzger_replication"
)


def _load(module_name: str):
    """Load a study module by file path (mirrors test_kappa_exclusions)."""
    spec = importlib.util.spec_from_file_location(
        module_name, _STUDY_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pms = _load("premanuscript_spotchecks")


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
