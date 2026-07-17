"""Tests for Stage A seed pool 2 — EM OA-candidate re-derivation.

Only the pure functions (``parse_reference_list``, ``derive_items_from_review``)
are exercised here. The async ``ingest_em_candidates`` orchestration performs
real network I/O (Europe PMC OA status, Cochrane review DOI resolution,
PubMed PublicationType, JATS fetch) over a whole candidate list — per
CLAUDE.md's >2-minute-process rule it is terminal-only (see the module's
``__main__``) and is not exercised here.
"""
from pathlib import Path

from studies.oa_rob_benchmark.ingest_em_candidates import (
    derive_items_from_review,
    parse_reference_list,
)

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
