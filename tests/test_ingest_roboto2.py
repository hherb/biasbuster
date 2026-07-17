"""Tests for the ROBoto2 record parser and ingest orchestration (Stage A
seed pool 1).

``parse_roboto2_record`` is a pure function (no network I/O), so most of
it is exercised offline against the synthetic fixture. The async
``ingest_roboto2`` orchestration performs real network calls in production
(Europe PMC OA status, PubMed PublicationType, JATS fetch) — those are
covered end-to-end only from a terminal against the real ROBoto2 dataset
file (CLAUDE.md's >2-minute-process rule; see the module's ``__main__``).
``test_ingest_roboto2_end_to_end_stubbed`` below exercises the same
orchestration loop with the network helpers monkeypatched to no-network
stubs, so it stays fast and safe to run in-session.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

import scripts.fetch_fulltext_for_expert_ratings as fetch_fulltext_module
from biasbuster.collectors.oa_license import OAStatus, classify_license
import biasbuster.collectors.oa_license as oa_license_module
import biasbuster.utils.pubtype as pubtype_module
from studies.oa_rob_benchmark.ingest_roboto2 import ingest_roboto2, parse_roboto2_record
from studies.oa_rob_benchmark.store import BenchmarkStore
import studies.oa_rob_benchmark.store as store_module

RECS = json.loads(Path("tests/fixtures/oa_rob/roboto2_sample.json").read_text())


def test_parse_manual_record_extracts_pmid_and_answers():
    out = parse_roboto2_record(RECS[0])
    assert out is not None
    paper_id, domain_answers, overall = out
    assert paper_id == "111"
    assert domain_answers["randomization"] == {"1.1": "Y", "1.2": "N", "1.3": "N"}


def test_llm_assisted_only_record_is_dropped():
    assert parse_roboto2_record(RECS[1]) is None


def test_manual_assessment_with_list_signalling_is_dropped_not_raised():
    """A list-shaped ``signalling`` value must not raise (fails safe).

    Regression test for the reviewer-flagged bug: ``dict(d.get("signalling",
    {}))`` raised ``ValueError``/``TypeError`` when ``signalling`` was a
    flat list (e.g. ``["Y", "N", "N"]``, a plausible real-world encoding of
    per-question answers) instead of the expected ``{question: answer}``
    mapping. The malformed domain entry must simply be dropped, not crash
    the whole record.
    """
    rec = {
        "paper_id": "PMID:999",
        "manual_assessment": [
            {"domain": "randomization", "signalling": ["Y", "N", "N"]},
            {"domain": "deviations", "signalling": {"2.1": "N", "2.2": "N"}},
        ],
    }
    out = parse_roboto2_record(rec)
    assert out is not None
    pmid, domain_answers, _overall = out
    assert pmid == "999"
    assert "randomization" not in domain_answers  # malformed entry skipped
    assert domain_answers["deviations"] == {"2.1": "N", "2.2": "N"}  # good entry kept


def test_manual_assessment_with_non_dict_entry_is_dropped_not_raised():
    """A non-dict entry in ``manual_assessment`` must not raise (fails safe)."""
    rec = {"paper_id": "PMID:998", "manual_assessment": ["not_a_dict_entry"]}
    out = parse_roboto2_record(rec)
    assert out is not None
    pmid, domain_answers, _overall = out
    assert pmid == "998"
    assert domain_answers == {}


async def test_ingest_roboto2_end_to_end_stubbed(tmp_path, monkeypatch):
    """Exercise the async ingest loop with all network I/O stubbed out.

    Dataset has three records: one admissible manual assessment (all five
    RoB 2 domains present with signalling that yields a canonical tuple),
    one LLM-assisted-only record (dropped by the parser itself, no manual
    assessment), and one malformed record — a bare JSON ``null`` in place
    of a record dict, which trips ``parse_roboto2_record``'s very first
    line (``rec.get(...)``) before its internal try/except can catch it.
    That's exactly the gap Fix 1b's call-site guard closes: it must be
    logged and skipped, never allowed to crash the whole ingest run.

    The OA status stub also returns an unprefixed ``pmcid`` (no "PMC"
    lead) to exercise Fix 2 — ``fetch_jats`` must be called with the
    normalized ``PMC``-prefixed id.
    """
    dataset = [
        {  # admissible manual record — mirrors tests/fixtures/oa_rob/roboto2_sample.json's shape
            "paper_id": "PMID:301",
            "manual_assessment": [
                {"domain": "randomization", "signalling": {"1.1": "Y", "1.2": "N", "1.3": "N"}},
                {"domain": "deviations", "signalling": {"2.1": "N", "2.2": "N"}},
                {"domain": "missing_outcome", "signalling": {"3.1": "Y", "3.2": "N"}},
                {"domain": "measurement", "signalling": {"4.1": "N", "4.2": "N"}},
                {"domain": "reporting", "signalling": {"5.1": "N", "5.2": "N", "5.3": "N"}},
            ],
            "roboto2_assessment": [],
        },
        {  # LLM-assisted-only record — no manual_assessment, dropped
            "paper_id": "PMID:302",
            "manual_assessment": [],
            "roboto2_assessment": [{"domain": "randomization", "signalling": {"1.1": "Y"}}],
        },
        None,  # malformed: not a dict at all — must be logged & skipped, not crash
    ]
    dataset_path = tmp_path / "roboto2_integration.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    rejects_path = tmp_path / "rejects.jsonl"
    monkeypatch.setattr(store_module, "REJECTS_PATH", str(rejects_path))

    fetch_jats_calls: list[tuple[str, str]] = []

    async def fake_fetch_oa_status(client, pmid, *, base):
        assert pmid == "301"
        lic = classify_license("cc by")
        # Deliberately unprefixed pmcid — exercises Fix 2's normalization.
        return OAStatus(pmid=pmid, pmcid="1000301", in_oa_subset=True, license=lic)

    async def fake_fetch_publication_types(pmids, *, client=None, ncbi_api_key="", **_kw):
        return {pmid: ["Randomized Controlled Trial"] for pmid in pmids}

    async def fake_fetch_jats(client, pmid, pmcid, cache_dir):
        fetch_jats_calls.append((pmid, pmcid))
        return "ok", 1234

    monkeypatch.setattr(oa_license_module, "fetch_oa_status", fake_fetch_oa_status)
    monkeypatch.setattr(pubtype_module, "fetch_publication_types", fake_fetch_publication_types)
    monkeypatch.setattr(fetch_fulltext_module, "fetch_jats", fake_fetch_jats)

    store = BenchmarkStore(str(tmp_path / "bench.db"))
    config = SimpleNamespace(europmc_base="https://example.invalid/europmc", ncbi_api_key="")

    async with httpx.AsyncClient() as client:
        report = await ingest_roboto2(
            str(dataset_path), store, client=client, config=config,
        )

    assert report.seen == 3
    assert store.count() == 1  # only the admissible trial was admitted
    assert report.admitted == 1
    assert report.rejected >= 1  # LLM-only + malformed records both rejected, run did not crash

    # Fix 2: fetch_jats must see the PMC-normalized pmcid, not the raw one.
    assert fetch_jats_calls == [("301", "PMC1000301")]

    # Fix 1b: the malformed (None) record was logged, not silently dropped.
    reject_lines = [json.loads(line) for line in rejects_path.read_text().splitlines()]
    assert any(r["rule"] == "malformed_record" for r in reject_lines)
