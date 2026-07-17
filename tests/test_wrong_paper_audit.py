"""Tests for the wrong-paper acquisition audit (issue #29, step 5).

The audit detects RCTs whose Phase-1 acquisition fetched a different document
than the intended primary trial (the RCT030 failure mode). These tests pin the
pure detection helpers on synthetic inputs (no dependency on the gitignored
benchmark DB) plus the ``audit`` loader against an in-memory database.
"""
from __future__ import annotations

import importlib.util
import sqlite3
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


audit_mod = _load("audit_wrong_paper_acquisitions")


# --- title_tokens -----------------------------------------------------

class TestTitleTokens:
    def test_lowercases_and_drops_short_and_stopwords(self) -> None:
        toks = audit_mod.title_tokens("A Randomized Trial of Aspirin in ICU")
        # "a"/"of"/"in" glue, "randomized"/"trial" design boilerplate, and the
        # 2-char nothing are all dropped; content words survive lowercased.
        assert toks == {"aspirin", "icu"}

    def test_punctuation_split(self) -> None:
        assert "sotrovimab" in audit_mod.title_tokens("Antibody Sotrovimab.")


# --- reference_coverage -----------------------------------------------

class TestReferenceCoverage:
    def test_full_coverage_when_title_is_substring_of_reference(self) -> None:
        acquired = "Aspirin for hospitalised COVID-19 patients"
        ref = "Horby PW, et al. Aspirin for hospitalised COVID-19 patients. Lancet 2022."
        assert audit_mod.reference_coverage(acquired, ref) == 1.0

    def test_zero_coverage_for_disjoint_topics(self) -> None:
        acquired = "Stainless steel slag waste in self-compacting concrete"
        ref = "Entrenas Castillo E. Calcifediol treatment for COVID-19. J Steroid Biochem 2020."
        assert audit_mod.reference_coverage(acquired, ref) == 0.0

    def test_empty_acquired_title_is_zero_not_error(self) -> None:
        assert audit_mod.reference_coverage("", "Some reference text here.") == 0.0

    def test_partial_coverage_between_zero_and_one(self) -> None:
        # "systematic"/"review" are extra words absent from the trial ref.
        acquired = "Systematic review of self management in COPD"
        ref = "Jolly K. Self management of patients with COPD. BMJ 2018."
        cov = audit_mod.reference_coverage(acquired, ref)
        assert 0.0 < cov < 1.0


# --- count_mismatch_rationales ----------------------------------------

class TestCountMismatchRationales:
    def test_counts_only_mismatch_phrases(self) -> None:
        rationales = [
            "The source materials describe an entirely unrelated study on X.",
            "The abstract provides no information about sequence generation.",  # legit
            "This is a study protocol registered before results.",
            None,
            "",
        ]
        assert audit_mod.count_mismatch_rationales(rationales) == 2

    def test_absent_protocol_is_not_a_mismatch(self) -> None:
        # A tightened phrase list must NOT flag the legitimate Domain-5 note
        # that the protocol is simply unavailable (the RCT053 false positive).
        rationales = [
            "The protocol and trial registration are not available, so "
            "selective reporting cannot be assessed.",
        ]
        assert audit_mod.count_mismatch_rationales(rationales) == 0


# --- is_suspected -----------------------------------------------------

class TestIsSuspected:
    def test_low_coverage_fires(self) -> None:
        assert audit_mod.is_suspected(0.1, 0) is True

    def test_enough_mismatch_hits_fires_even_with_high_coverage(self) -> None:
        assert audit_mod.is_suspected(0.9, 3) is True

    def test_clean_row_not_suspected(self) -> None:
        assert audit_mod.is_suspected(0.95, 0) is False


# --- audit (loader) ---------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE benchmark_rct (
            rct_id TEXT PRIMARY KEY, cr_id TEXT, pmid TEXT, doi TEXT,
            nct_nr TEXT, title TEXT, authors_text TEXT, publication_year INTEGER,
            condition TEXT, intervention TEXT, outcome_text TEXT,
            has_abstract INTEGER, has_fulltext INTEGER, has_registration INTEGER,
            fulltext_source TEXT, em_rct_ref TEXT NOT NULL, notes TEXT
        );
        CREATE TABLE benchmark_judgment (
            rct_id TEXT NOT NULL, source TEXT NOT NULL, domain TEXT NOT NULL,
            judgment TEXT, rationale TEXT, valid INTEGER NOT NULL DEFAULT 1,
            raw_label TEXT, PRIMARY KEY (rct_id, source, domain)
        );
        """
    )
    conn.executemany(
        "INSERT INTO benchmark_rct (rct_id, cr_id, title, em_rct_ref) "
        "VALUES (?, ?, ?, ?)",
        [
            # wrong paper: acquired title disjoint from the intended reference
            ("RCTW", "CR1", "Concrete slag waste manufacturing",
             "Author A. Calcifediol treatment for COVID-19. Journal 2020."),
            # correct paper: acquired title is a substring of its reference
            ("RCTOK", "CR2", "Aspirin for COVID-19 patients",
             "Horby PW. Aspirin for COVID-19 patients. Lancet 2022."),
            # never fetched: no acquired title -> must be skipped by audit()
            ("RCTNF", "CR3", "", "Author C. Some trial. Journal 2019."),
        ],
    )
    conn.executemany(
        "INSERT INTO benchmark_judgment "
        "(rct_id, source, domain, judgment, rationale, valid) VALUES (?,?,?,?,?,?)",
        [
            ("RCTW", "cochrane", "overall", "high", "", 1),
            ("RCTW", "sonnet_fulltext_pass1", "d1",
             "The source materials are entirely from a different study.", "", 1),
            ("RCTW", "sonnet_fulltext_pass1", "overall", "some_concerns",
             "The source materials describe an entirely unrelated study.", 1),
            ("RCTOK", "cochrane", "overall", "low", "", 1),
            ("RCTOK", "sonnet_fulltext_pass1", "overall", "low",
             "Web-based randomisation with allocation concealment.", 1),
        ],
    )
    conn.commit()
    return conn


class TestAudit:
    def test_skips_never_fetched_and_ranks_wrong_paper_first(self) -> None:
        conn = _make_db()
        try:
            rows = audit_mod.audit(conn)
        finally:
            conn.close()
        ids = [r.rct_id for r in rows]
        assert "RCTNF" not in ids            # never-fetched row skipped
        assert ids[0] == "RCTW"              # lowest coverage sorts first

        wrong = rows[0]
        assert wrong.suspected is True
        assert wrong.coverage == 0.0
        assert wrong.mismatch_hits >= 1
        assert wrong.model_valid_rows == 2   # cochrane excluded from the count

        ok = next(r for r in rows if r.rct_id == "RCTOK")
        assert ok.suspected is False
        assert ok.coverage == 1.0
