"""Tests for Stage A seed pool 2 — EM OA-candidate re-derivation.

The pure functions (``parse_reference_list``, ``derive_items_from_review``)
are exercised directly. The async ``ingest_em_candidates`` orchestration
performs real network I/O (Europe PMC OA status, Cochrane review DOI
resolution, PubMed PublicationType, JATS fetch) over a whole candidate list
— per CLAUDE.md's >2-minute-process rule it is terminal-only against the
real cohort (see the module's ``__main__``). To keep the orchestration
under test without network, ``test_ingest_em_candidates_end_to_end_stubbed``
drives the same loop with every network helper monkeypatched to a
no-network stub.
"""
from pathlib import Path
from types import SimpleNamespace

import httpx

import biasbuster.collectors.oa_license as oa_license_module
import biasbuster.utils.pubtype as pubtype_module
import scripts.fetch_fulltext_for_expert_ratings as fetch_fulltext_module
import studies.oa_rob_benchmark.ingest_em_candidates as em_module
import studies.oa_rob_benchmark.store as store_module
from biasbuster.collectors.oa_license import OAStatus, classify_license
from studies.oa_rob_benchmark.ingest_em_candidates import (
    derive_items_from_review,
    ingest_em_candidates,
    parse_reference_list,
)
from studies.oa_rob_benchmark.store import BenchmarkStore

JATS = Path("tests/fixtures/oa_rob/review_with_rob2.xml").read_bytes()


def test_reference_list_parsed_with_pmid():
    refs = parse_reference_list(JATS)
    assert any(r.pmid == "111" and r.first_author.lower() == "smith" for r in refs)


def test_derive_items_resolves_target_trial():
    items = derive_items_from_review(JATS, target_pmids={"111"})
    assert len(items) == 1
    it = items[0]
    assert it["trial_pmid"] == "111"
    assert it["rob2_overall"] == "high"
    assert it["rob2_d3"] == "some concerns"
    assert it["resolution_method"] in {"bracket_ref", "author_year_title"}


def test_non_target_pmid_not_emitted():
    assert derive_items_from_review(JATS, target_pmids={"999"}) == []


async def test_ingest_em_candidates_end_to_end_stubbed(tmp_path, monkeypatch):
    """Exercise the async EM ingest loop with all network I/O stubbed out.

    One candidate trial (PMID ``111``) whose parent review (PMID ``999``)
    carries the fixture's RoB 2 table. Every network helper is replaced by
    a no-network stub, and ``fetch_jats`` writes the review fixture into the
    (redirected) cache so ``derive_items_from_review`` re-derives the label
    from the review's own table — never a copy of EM's supplement.

    Doubles as the regression guard for the DOI-casing fix: the stubbed
    ID Converter returns its map keyed by the *uppercase* canonical Cochrane
    DOI (``...CD001159.pub3``) while the ingest queries with the lowercased
    form. Keying the lookup on the query string would miss and drop the
    candidate; taking the single resolved value resolves it.
    """
    import biasbuster.collectors.retraction_watch as retraction_watch_module

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(fetch_fulltext_module, "DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(store_module, "REJECTS_PATH", str(tmp_path / "rejects.jsonl"))

    def fake_load_em_benchmark_data(_db_path):
        return {"111": "CD001159.PUB3"}, set()

    monkeypatch.setattr(em_module, "_load_em_benchmark_data", fake_load_em_benchmark_data)

    async def fake_fetch_oa_status(client, pmid, *, base):
        if pmid == "111":  # the trial
            return OAStatus(pmid=pmid, pmcid="PMC111", in_oa_subset=True,
                            license=classify_license("cc by"))
        # the parent review — only its PMCID is used (labels stored as citation)
        return OAStatus(pmid=pmid, pmcid="PMC999", in_oa_subset=False,
                        license=classify_license(""))

    class _FakeRW:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def doi_to_pmid(self, dois):
            # NCBI echoes the DOI in canonical (uppercase-CD) casing, not the
            # lowercased form the ingest queried with — the casing fix must
            # still resolve this to the review PMID.
            return {"10.1002/14651858.CD001159.pub3": "999"}

    async def fake_fetch_publication_types(pmids, *, client=None, ncbi_api_key="", **_kw):
        return {pmid: ["Randomized Controlled Trial"] for pmid in pmids}

    async def fake_fetch_jats(client, pmid, pmcid, cache_dir):
        path = fetch_fulltext_module._cache_path(cache_dir, pmid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(JATS if pmid == "999" else b"<article/>")
        return "ok", len(JATS)

    monkeypatch.setattr(oa_license_module, "fetch_oa_status", fake_fetch_oa_status)
    monkeypatch.setattr(retraction_watch_module, "RetractionWatchCollector", _FakeRW)
    monkeypatch.setattr(pubtype_module, "fetch_publication_types", fake_fetch_publication_types)
    monkeypatch.setattr(fetch_fulltext_module, "fetch_jats", fake_fetch_jats)

    store = BenchmarkStore(str(tmp_path / "bench.db"))
    config = SimpleNamespace(
        europmc_base="https://example.invalid/europmc",
        crossref_mailto="test@example.invalid",
        ncbi_api_key="",
    )

    async with httpx.AsyncClient() as client:
        report = await ingest_em_candidates(["111"], store, client=client, config=config)

    assert report.seen == 1
    assert report.admitted == 1
    assert report.rejected == 0
    assert store.count() == 1

    item = store.all_items()[0]
    assert item["trial_pmid"] == "111"
    assert item["source_review_pmid"] == "999"  # DOI-casing fix resolved the review
    assert item["source_review_pmcid"] == "PMC999"
    assert item["rob2_overall"] == "high"
    assert item["rob2_d3"] == "some concerns"
    assert item["label_source"] == "cochrane_review"
    assert item["license_redistributable"] == 1  # SQLite stores the bool as int
