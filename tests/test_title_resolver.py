"""Tests for the ROBoto2 title→PMID resolver.

``select_best_title_match`` is where the wrong-document guard lives, so it is
unit-tested exhaustively offline. The network wrapper ``resolve_pmid_by_title``
has its happy path covered via the stubbed ingest test (its live PubMed calls
are terminal-only); its error handling — the fail-closed contract the ingest
loop depends on — is covered here with ``fetch_with_retry`` stubbed to raise.
"""
import httpx

import studies.oa_rob_benchmark.title_resolver as title_resolver_module
from biasbuster.utils.retry import RetryExhaustedError
from studies.oa_rob_benchmark.title_resolver import (
    AMBIGUITY_MARGIN,
    TITLE_MATCH_THRESHOLD,
    _title_query,
    resolve_pmid_by_title,
    select_best_title_match,
)

_TITLE = "Efficacy of a brief school-based program for selective prevention of childhood anxiety"


def test_title_query_is_unquoted_field_search():
    """The esearch term is an UNQUOTED [Title] field search (the brittle
    "...phrase..."[Title] form recalls nothing for these titles)."""
    q = _title_query("Efficacy of a brief school-based program")
    assert q == "Efficacy of a brief school-based program[Title]"
    assert '"' not in q


def test_title_query_strips_query_syntax_chars():
    """Field-tag brackets and quotes in a title must not inject query syntax."""
    q = _title_query('A trial of [18F] "labelled" tracer')
    assert "[" not in q.replace("[Title]", "")
    assert '"' not in q
    assert q.endswith("[Title]")


def test_exact_title_match_is_accepted():
    res = select_best_title_match(_TITLE, {"301": _TITLE})
    assert res.pmid == "301"
    assert res.similarity == 1.0
    assert res.reason == "title_search"


def test_no_candidates_is_unresolved():
    res = select_best_title_match(_TITLE, {})
    assert res.pmid == ""
    assert res.reason == "no_candidates"


def test_empty_query_title_is_unresolved():
    assert select_best_title_match("", {"301": _TITLE}).reason == "no_candidates"


def test_below_threshold_is_rejected():
    res = select_best_title_match(_TITLE, {"301": "A completely unrelated cardiology trial"})
    assert res.pmid == ""
    assert res.reason == "below_threshold"
    assert res.similarity < TITLE_MATCH_THRESHOLD


def test_two_near_identical_candidates_are_ambiguous():
    """Two different PMIDs both matching the title (duplicate publication)
    must not be resolved to the arbitrary first — rejected as ambiguous."""
    res = select_best_title_match(_TITLE, {"301": _TITLE, "302": _TITLE})
    assert res.pmid == ""
    assert res.reason == "ambiguous_title"


def test_clear_winner_over_weak_runner_up_is_accepted():
    """A strong best plus a clearly weaker runner-up (gap > margin) resolves."""
    res = select_best_title_match(
        _TITLE,
        {"301": _TITLE, "302": "Something loosely about anxiety prevention in kids maybe"},
    )
    assert res.pmid == "301"
    # The runner-up must be far enough below the winner to clear the margin.
    assert res.similarity >= TITLE_MATCH_THRESHOLD


def test_margin_constant_is_respected_at_boundary():
    # Best clears threshold; a same-title second PMID within margin → ambiguous.
    near = {"301": _TITLE, "302": _TITLE[:-3] + "xyz"}
    res = select_best_title_match(_TITLE, near, margin=AMBIGUITY_MARGIN)
    assert res.reason == "ambiguous_title"


class _FakeJsonResponse:
    """Minimal stand-in for an httpx response carrying an esearch id-list."""

    def __init__(self, idlist: list[str]) -> None:
        self._idlist = idlist

    def json(self) -> dict:
        return {"esearchresult": {"idlist": self._idlist}}


async def test_esearch_retry_exhaustion_fails_closed(monkeypatch):
    """A spent-retries ``RetryExhaustedError`` on esearch must NOT propagate —
    it resolves to an unresolved ``esearch_failed`` so a transient PubMed
    outage never aborts the ingest run (the loop trusts this to fail closed)."""
    async def boom(*_a, **_kw):
        raise RetryExhaustedError(4, httpx.ConnectError("pubmed unreachable"))

    monkeypatch.setattr(title_resolver_module, "fetch_with_retry", boom)
    async with httpx.AsyncClient() as client:
        res = await resolve_pmid_by_title(
            client, _TITLE, pubmed_base="https://example.invalid/eutils",
        )
    assert res.pmid == ""
    assert res.reason == "esearch_failed"


async def test_efetch_retry_exhaustion_fails_closed(monkeypatch):
    """esearch succeeds but efetch exhausts its retries — still fail closed
    (``efetch_failed``), never raising into the ingest loop."""
    async def fake(client, method, url, **_kw):
        if "esearch" in url:
            return _FakeJsonResponse(["301"])
        raise RetryExhaustedError(4, httpx.ReadTimeout("efetch timed out"))

    monkeypatch.setattr(title_resolver_module, "fetch_with_retry", fake)
    async with httpx.AsyncClient() as client:
        res = await resolve_pmid_by_title(
            client, _TITLE, pubmed_base="https://example.invalid/eutils",
        )
    assert res.pmid == ""
    assert res.reason == "efetch_failed"
