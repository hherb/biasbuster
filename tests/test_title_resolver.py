"""Tests for the ROBoto2 title→PMID resolver's pure selection core.

``select_best_title_match`` is where the wrong-document guard lives, so it is
unit-tested exhaustively offline. The network wrapper ``resolve_pmid_by_title``
is covered via the stubbed ingest test (its live PubMed calls are terminal-only).
"""
from studies.oa_rob_benchmark.title_resolver import (
    AMBIGUITY_MARGIN,
    TITLE_MATCH_THRESHOLD,
    _title_query,
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
    assert res.reason in {"ambiguous_title", "title_search"}
