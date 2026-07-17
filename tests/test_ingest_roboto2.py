"""Tests for the ROBoto2 record parser and ingest orchestration (Stage A
seed pool 1).

``parse_record`` is a pure function (no network I/O), exercised offline
against the normalized fixture (the shape ``convert_roboto2_csv`` emits).
The async ``ingest_roboto2`` orchestration performs real network calls in
production (title→PMID esearch, Europe PMC OA status, PubMed PublicationType,
JATS fetch) — those run end-to-end only from a terminal against the real
dataset file (CLAUDE.md's >2-minute-process rule; see the module's
``__main__``). ``test_ingest_roboto2_end_to_end_stubbed`` below exercises the
same orchestration loop with every network helper monkeypatched to a
no-network stub, so it stays fast and safe to run in-session.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

import scripts.fetch_fulltext_for_expert_ratings as fetch_fulltext_module
from biasbuster.collectors.oa_license import OAStatus, classify_license
import biasbuster.collectors.oa_license as oa_license_module
import biasbuster.utils.pubtype as pubtype_module
import studies.oa_rob_benchmark.ingest_roboto2 as ingest_module
from studies.oa_rob_benchmark.ingest_roboto2 import ingest_roboto2, parse_record
from studies.oa_rob_benchmark.store import BenchmarkStore
import studies.oa_rob_benchmark.store as store_module
from studies.oa_rob_benchmark.title_resolver import TitleResolution

RECS = json.loads(Path("tests/fixtures/oa_rob/roboto2_sample.json").read_text())


def test_parse_record_uses_recorded_labels():
    parsed = parse_record(RECS[0])
    assert parsed is not None
    assert parsed.paper_id == "29838"
    assert parsed.title.startswith("Efficacy of a brief school-based")
    # Ground truth is the recorded expert tuple, not re-derived from signalling.
    assert parsed.rob2.overall == "high"
    assert parsed.rob2.d4 == "high"  # measurement
    assert parsed.rob2.d1 == "low"


def test_record_with_none_domain_is_dropped():
    """A domain recorded as 'none' cannot form a canonical tuple → dropped."""
    assert parse_record(RECS[1]) is None


def test_record_without_title_is_dropped():
    rec = dict(RECS[0])
    rec["title"] = ""
    assert parse_record(rec) is None


def test_non_dict_record_is_dropped_not_raised():
    assert parse_record(None) is None
    assert parse_record("not_a_dict") is None


async def test_ingest_roboto2_end_to_end_stubbed(tmp_path, monkeypatch):
    """Exercise the async ingest loop with all network I/O stubbed out.

    Dataset has five records: one admissible with a PubMed-confirmed trial
    type, one admissible whose PubMed metadata lacks a trial tag (admitted as
    ``trial_source_asserted``, flagged for manual verification — ROBoto2 rows
    are RCTs by construction), one with a ``none`` domain (dropped by the
    parser), one whose title does not resolve (rejected, not guessed), and one
    bare JSON ``null`` (must be logged and skipped, never crash the run).

    The OA status stub returns an unprefixed ``pmcid`` (no "PMC" lead) to
    verify ``fetch_jats`` is still called with the normalized ``PMC``-prefixed
    id.
    """
    dataset = [
        {  # admissible, PubMed-confirmed trial type
            "paper_id": "29838",
            "title": "An admissible resolvable trial title",
            "rob2": {"overall": "low", "randomization": "low", "deviations": "low",
                     "missing_outcome": "low", "measurement": "low", "reporting": "low"},
            "signalling": {},
        },
        {  # admissible, but PubMed does not tag it a trial → source-asserted
            "paper_id": "555",
            "title": "A source-asserted trial PubMed did not tag",
            "rob2": {"overall": "some concerns", "randomization": "some concerns",
                     "deviations": "low", "missing_outcome": "low",
                     "measurement": "low", "reporting": "low"},
            "signalling": {},
        },
        RECS[1],  # 'none' domain → parser drops
        {  # title cannot be resolved confidently
            "paper_id": "999",
            "title": "A title with no confident PubMed match",
            "rob2": {"overall": "high", "randomization": "high", "deviations": "low",
                     "missing_outcome": "low", "measurement": "low", "reporting": "low"},
            "signalling": {},
        },
        None,  # malformed: not a dict — logged & skipped, not a crash
    ]
    dataset_path = tmp_path / "roboto2.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    rejects_path = tmp_path / "rejects.jsonl"
    monkeypatch.setattr(store_module, "REJECTS_PATH", str(rejects_path))

    fetch_jats_calls: list[tuple[str, str]] = []

    async def fake_resolve(client, title, *, pubmed_base, ncbi_api_key="", **_kw):
        if title.startswith("An admissible"):
            return TitleResolution("301", 0.97, "title_search")
        if title.startswith("A source-asserted"):
            return TitleResolution("555", 0.95, "title_search")
        return TitleResolution("", 0.42, "below_threshold")

    async def fake_fetch_oa_status(client, pmid, *, base):
        assert pmid in {"301", "555"}
        lic = classify_license("cc by")
        # Deliberately unprefixed pmcid — exercises PMC normalization.
        return OAStatus(pmid=pmid, pmcid=f"1000{pmid}", in_oa_subset=True, license=lic)

    async def fake_fetch_publication_types(pmids, *, client=None, ncbi_api_key="", **_kw):
        # 301 is a tagged RCT; 555 carries no trial PublicationType.
        tags = {"301": ["Randomized Controlled Trial"], "555": ["Journal Article"]}
        return {pmid: tags.get(pmid, []) for pmid in pmids}

    async def fake_fetch_jats(client, pmid, pmcid, cache_dir):
        fetch_jats_calls.append((pmid, pmcid))
        return "ok", 1234

    monkeypatch.setattr(ingest_module, "resolve_pmid_by_title", fake_resolve)
    monkeypatch.setattr(oa_license_module, "fetch_oa_status", fake_fetch_oa_status)
    monkeypatch.setattr(pubtype_module, "fetch_publication_types", fake_fetch_publication_types)
    monkeypatch.setattr(fetch_fulltext_module, "fetch_jats", fake_fetch_jats)

    store = BenchmarkStore(str(tmp_path / "bench.db"))
    config = SimpleNamespace(
        pubmed_base="https://example.invalid/eutils",
        europmc_base="https://example.invalid/europmc",
        ncbi_api_key="",
    )

    async with httpx.AsyncClient() as client:
        report = await ingest_roboto2(
            str(dataset_path), store, client=client, config=config,
        )

    assert report.seen == 5
    assert store.count() == 2  # confirmed-trial + source-asserted both admitted
    assert report.admitted == 2
    assert report.rejected == 3  # none-domain + unresolved-title + malformed

    # fetch_jats must see the PMC-normalized pmcid, not the raw one.
    assert set(fetch_jats_calls) == {("301", "PMC1000301"), ("555", "PMC1000555")}

    rows = {r["trial_pmid"]: r for r in store.all_items()}
    # The PubMed-confirmed trial is recorded as a plain "trial".
    assert rows["301"]["pubtype_check"] == "trial"
    assert rows["301"]["rob2_overall"] == "low"
    assert rows["301"]["resolution_method"] == "title_search"
    assert rows["301"]["similarity_score"] == 0.97
    assert rows["301"]["trial_title"] == "An admissible resolvable trial title"
    # The PubMed-untagged RCT is admitted but flagged source-asserted, awaiting
    # manual verification (never claimed as PubMed-confirmed).
    assert rows["555"]["pubtype_check"] == "trial_source_asserted"
    assert rows["555"]["manual_verified"] == 0

    # The unresolved-title record was logged, not silently dropped.
    reject_rules = {json.loads(line)["rule"] for line in rejects_path.read_text().splitlines()}
    assert "title_unresolved" in reject_rules
    # non_trial_pubtype must NOT appear — the source-asserted row was admitted.
    assert "non_trial_pubtype" not in reject_rules
