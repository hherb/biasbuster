"""Tests for the wrong-paper recovery tool (issue #29 recovery path).

Covers the pure verdict logic, the EMRow-from-DB adapter, and — most
importantly — the surgical safety of ``apply_recovery``: it must delete only
the stale *model* rows and never the ``cochrane`` / ``em_claude2_*`` ground
truth, and must leave other RCTs untouched. No network is exercised here.
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
    spec = importlib.util.spec_from_file_location(
        module_name, _STUDY_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


recover = _load("recover_wrong_papers")


# --- classify_verdict -------------------------------------------------

class TestClassifyVerdict:
    def test_confident_match_with_oa_is_full_recovery(self) -> None:
        assert recover.classify_verdict(0.9, "123", True) == "recover_fulltext"

    def test_confident_match_without_oa_is_abstract_recovery(self) -> None:
        assert recover.classify_verdict(0.9, "123", False) == "recover_abstract"

    def test_below_gate_is_exclude_even_with_oa(self) -> None:
        assert recover.classify_verdict(0.3, "123", True) == "exclude"

    def test_no_pmid_is_exclude(self) -> None:
        assert recover.classify_verdict(0.99, "", True) == "exclude"


# --- em_row_from_db ---------------------------------------------------

def _row(**cols) -> sqlite3.Row:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    keys = ["rct_id", "cr_id", "pmid", "nct_nr", "authors_text",
            "condition", "intervention", "em_rct_ref"]
    conn.execute(f"CREATE TABLE t ({', '.join(keys)})")
    conn.execute(
        f"INSERT INTO t ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
        [cols.get(k, "") for k in keys],
    )
    return conn.execute("SELECT * FROM t").fetchone()


class TestEmRowFromDb:
    def test_derives_surname_year_and_nct(self) -> None:
        row = _row(
            rct_id="RCT093", authors_text="Horby, 2021", nct_nr="NCT04381936",
            em_rct_ref="Horby PW. Aspirin in patients admitted with COVID-19 "
                       "(RECOVERY). Lancet 2022.",
        )
        em = recover.em_row_from_db(row)
        assert em.rct_id == "RCT093"
        assert em.first_author_surname == "Horby"
        assert em.publication_year == "2021"
        assert em.extracted_nct == "NCT04381936"


# --- apply_recovery surgical safety -----------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
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
            rct_id TEXT, source TEXT, domain TEXT, judgment TEXT,
            rationale TEXT, valid INTEGER DEFAULT 1, raw_label TEXT,
            PRIMARY KEY (rct_id, source, domain)
        );
        CREATE TABLE evaluation_run (
            rct_id TEXT, source TEXT, domain TEXT, model_id TEXT, protocol TEXT,
            pass_n INTEGER, started_at TEXT, parse_status TEXT,
            PRIMARY KEY (rct_id, source, domain)
        );
        """
    )
    conn.execute(
        "INSERT INTO benchmark_rct (rct_id, cr_id, pmid, title, has_abstract, "
        "has_fulltext, em_rct_ref, notes) VALUES "
        "('RCTX','CR','99999','Wrong concrete paper',1,1,'Author. Correct trial. J 2020.','old note')"
    )
    judg = [
        ("RCTX", "cochrane", "overall", "high"),
        ("RCTX", "em_claude2_abstract", "overall", "some_concerns"),
        ("RCTX", "sonnet_4_6_fulltext_pass1", "overall", "low"),
        ("RCTX", "gemma4_26b_abstract_pass2", "d1", "low"),
        ("RCTY", "cochrane", "overall", "low"),
        ("RCTY", "sonnet_4_6_fulltext_pass1", "overall", "low"),
    ]
    conn.executemany(
        "INSERT INTO benchmark_judgment (rct_id, source, domain, judgment) "
        "VALUES (?,?,?,?)", judg,
    )
    conn.executemany(
        "INSERT INTO evaluation_run (rct_id, source, domain, model_id, protocol, "
        "pass_n, started_at, parse_status) VALUES (?,?,?,?,?,?,?,?)",
        [
            ("RCTX", "sonnet_4_6_fulltext_pass1", "overall", "m", "fulltext", 1, "t", "ok"),
            ("RCTY", "sonnet_4_6_fulltext_pass1", "overall", "m", "fulltext", 1, "t", "ok"),
        ],
    )
    conn.commit()
    return conn


class TestParseApplyTargets:
    def test_plain_and_override_syntax(self) -> None:
        targets = recover.parse_apply_targets(["RCT088", "RCT093=34800427"])
        assert targets == {"RCT088": None, "RCT093": "34800427"}

    def test_empty_override_is_none(self) -> None:
        assert recover.parse_apply_targets(["RCT017="]) == {"RCT017": None}


class TestPlanRecoveryOverride:
    def test_override_rejected_when_title_mismatches_reference(
        self, monkeypatch
    ) -> None:
        # A supplied PMID whose title is unrelated to em_rct_ref must be rejected
        # so a typo cannot silently recover the wrong paper.
        monkeypatch.setattr(
            recover, "fetch_pubmed_record",
            lambda client, pmid: {"title": "Stainless steel slag concrete", "abstract": "x"},
        )
        conn = _make_db()
        row = conn.execute("SELECT * FROM benchmark_rct WHERE rct_id='RCTX'").fetchone()
        assert recover.plan_recovery(conn, row, client=None, override_pmid="00000") is None

    def test_override_accepted_when_title_matches_reference(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            recover, "fetch_pubmed_record",
            lambda client, pmid: {"title": "Correct trial", "abstract": "abs",
                                  "doi": "10.1/x", "pmcid": ""},
        )
        conn = _make_db()
        row = conn.execute("SELECT * FROM benchmark_rct WHERE rct_id='RCTX'").fetchone()
        plan = recover.plan_recovery(conn, row, client=None, override_pmid="32871238")
        assert plan is not None
        assert plan.new_pmid == "32871238"
        assert plan.has_abstract is True
        assert plan.has_fulltext is False  # no pmcid -> no OA full text


class TestApplyRecovery:
    def test_deletes_model_rows_keeps_ground_truth_and_other_rcts(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(recover, "FULLTEXT_DIR", tmp_path)
        conn = _make_db()
        plan = recover.RecoveryPlan(
            rct_id="RCTX", old_pmid="99999", new_pmid="32871238",
            new_title="Correct calcifediol trial", new_doi="10.1/x",
            has_abstract=True, has_fulltext=True, fulltext_source="europepmc_xml",
            model_judgment_rows=2, eval_run_rows=1,
            abstract_text="Correct abstract text", jats_xml="<article/>",
        )
        recover.apply_recovery(conn, plan)

        # benchmark_rct updated + provenance appended, old note preserved
        rct = conn.execute(
            "SELECT pmid, title, has_fulltext, notes FROM benchmark_rct WHERE rct_id='RCTX'"
        ).fetchone()
        assert rct["pmid"] == "32871238"
        assert rct["title"] == "Correct calcifediol trial"
        assert rct["has_fulltext"] == 1
        assert "old note" in rct["notes"] and "32871238" in rct["notes"]

        # RCTX model rows gone; ground truth kept
        srcs = {r["source"] for r in conn.execute(
            "SELECT source FROM benchmark_judgment WHERE rct_id='RCTX'")}
        assert srcs == {"cochrane", "em_claude2_abstract"}

        # RCTY completely untouched
        assert conn.execute(
            "SELECT COUNT(*) FROM benchmark_judgment WHERE rct_id='RCTY'"
        ).fetchone()[0] == 2

        # evaluation_run: RCTX cleared, RCTY kept
        assert conn.execute(
            "SELECT COUNT(*) FROM evaluation_run WHERE rct_id='RCTX'").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM evaluation_run WHERE rct_id='RCTY'").fetchone()[0] == 1

        # correct document written to the file layout
        assert (tmp_path / "RCTX" / "abstract.txt").read_text() == "Correct abstract text"
        assert (tmp_path / "RCTX" / "paper.jats.xml").exists()

    def test_removes_stale_fulltext_when_recovery_is_abstract_only(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(recover, "FULLTEXT_DIR", tmp_path)
        conn = _make_db()
        # a stale wrong-paper full text exists on disk
        (tmp_path / "RCTX").mkdir()
        (tmp_path / "RCTX" / "paper.jats.xml").write_text("<wrong/>")
        plan = recover.RecoveryPlan(
            rct_id="RCTX", old_pmid="99999", new_pmid="32871238",
            new_title="Correct trial", new_doi="",
            has_abstract=True, has_fulltext=False, fulltext_source="",
            model_judgment_rows=2, eval_run_rows=1,
            abstract_text="Correct abstract", jats_xml="",
        )
        recover.apply_recovery(conn, plan)
        # stale full text must be gone so the model does not re-read it
        assert not (tmp_path / "RCTX" / "paper.jats.xml").exists()
        assert conn.execute(
            "SELECT has_fulltext FROM benchmark_rct WHERE rct_id='RCTX'"
        ).fetchone()[0] == 0
