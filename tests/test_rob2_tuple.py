"""Tests for the RoB 2 six-field tuple assembler (studies/oa_rob_benchmark/rob2_tuple.py).

Covers both entry points:
* ``tuple_from_table_row`` — structural-extractor path (Stage B / EM candidates).
* ``tuple_from_signalling`` — signalling-answer path (ROBoto2), which derives
  domain judgements via the RoB 2 algorithm in
  ``biasbuster.methodologies.cochrane_rob2.algorithms``.
"""
from studies.oa_rob_benchmark.rob2_tuple import (
    normalise_level, tuple_from_table_row, tuple_from_signalling,
    CANONICAL_LEVELS,
)
from biasbuster.collectors.rob_table_extractor import (
    ExtractedStudyRow, ExtractedRating,
)


def _rating(domain, text):
    return ExtractedRating(domain=domain, rating_text=text, rating_colour="",
                           raw_text=text, raw_style="")


def test_normalise_level_maps_underscore_and_unclear():
    assert normalise_level("some_concerns") == "some concerns"
    assert normalise_level("unclear") == "some concerns"
    assert normalise_level("Low") == "low"
    assert normalise_level("n/a") == ""


def test_complete_row_yields_tuple():
    row = ExtractedStudyRow(
        study_id="Smith 2020",
        overall=_rating("overall", "high"),
        domains=[_rating("randomization", "low"), _rating("deviations", "low"),
                 _rating("missing_outcome", "some_concerns"),
                 _rating("measurement", "low"), _rating("reporting", "high")],
        row_index=0)
    t = tuple_from_table_row(row)
    assert t is not None
    assert t.overall == "high"
    assert t.d3 == "some concerns"
    assert all(v in CANONICAL_LEVELS for v in (t.d1, t.d2, t.d3, t.d4, t.d5, t.overall))


def test_partial_row_is_rejected():
    row = ExtractedStudyRow(
        study_id="Smith 2020", overall=_rating("overall", "high"),
        domains=[_rating("randomization", "low")],  # only 1 of 5 domains
        row_index=0)
    assert tuple_from_table_row(row) is None


def test_row_with_invalid_level_is_rejected():
    row = ExtractedStudyRow(
        study_id="Smith 2020", overall=_rating("overall", ""),   # blank overall
        domains=[_rating("randomization", "low"), _rating("deviations", "low"),
                 _rating("missing_outcome", "low"),
                 _rating("measurement", "low"), _rating("reporting", "low")],
        row_index=0)
    assert tuple_from_table_row(row) is None


def test_signalling_path_derives_judgements_via_algorithm():
    """End-to-end check that tuple_from_signalling calls derive_domain_judgement
    with RoB2 codes (d1..d5), not extractor slugs, keyed by the extractor's
    canonical domain names in ``domain_answers``.

    Signalling answers chosen per biasbuster/methodologies/cochrane_rob2/algorithms.py:
      D1 low:  1.1=Y, 1.2=Y, 1.3=N
      D2 high: 2.5=N (or 2.7=Y) -> use 2.5=N
      D3 some_concerns: none of the low/high triggers fire (3.1=N, 3.2=N, 3.4=N)
      D4 low:  4.1=N, 4.2=N, 4.3=N
      D5 high: 5.2=Y
    Worst-wins overall (no domain is "high"... wait D2 and D5 are high) -> "high".
    """
    domain_answers = {
        "randomization": {"1.1": "Y", "1.2": "Y", "1.3": "N"},          # -> low
        "deviations": {"2.1": "Y", "2.2": "Y", "2.5": "N", "2.6": "N", "2.7": "N"},  # -> high
        "missing_outcome": {"3.1": "N", "3.2": "N", "3.4": "N"},        # -> some_concerns
        "measurement": {"4.1": "N", "4.2": "N", "4.3": "N"},            # -> low
        "reporting": {"5.1": "N", "5.2": "Y", "5.3": "N"},              # -> high
    }
    t = tuple_from_signalling(domain_answers, None)
    assert t is not None
    assert t.d1 == "low"
    assert t.d2 == "high"
    assert t.d3 == "some concerns"
    assert t.d4 == "low"
    assert t.d5 == "high"
    assert all(v in CANONICAL_LEVELS for v in (t.d1, t.d2, t.d3, t.d4, t.d5, t.overall))
    # worst-wins: any "high" domain present -> overall "high"
    assert t.overall == "high"


def test_signalling_worst_wins_some_concerns_not_downgraded():
    """Regression: synthesis_worst_wins must see the raw underscore-form
    judgements (``some_concerns``), not the canonical space-form
    (``some concerns``) that tuple_from_signalling stores in the tuple
    fields. Before the fix, the canonical form was fed to
    synthesis_worst_wins, whose "some_concerns" in values check never
    matched, so the overall silently fell through to "low" whenever the
    worst domain was some-concerns and none was high.

    All five domains derive to "low" except missing_outcome, which derives
    to "some_concerns" (per algorithms.py: neither the low trigger
    (3.1/3.2 yes) nor the high trigger (3.4 yes) fires). No domain is
    "high", so worst-wins must yield "some concerns" overall, not "low".
    """
    domain_answers = {
        "randomization": {"1.1": "Y", "1.2": "Y", "1.3": "N"},          # -> low
        "deviations": {"2.1": "N", "2.2": "N", "2.6": "Y"},              # -> low
        "missing_outcome": {"3.1": "N", "3.2": "N", "3.4": "N"},        # -> some_concerns
        "measurement": {"4.1": "N", "4.2": "N", "4.3": "N"},            # -> low
        "reporting": {"5.1": "Y", "5.2": "N", "5.3": "N"},              # -> low
    }
    t = tuple_from_signalling(domain_answers, None)
    assert t is not None
    assert t.d1 == "low"
    assert t.d2 == "low"
    assert t.d3 == "some concerns"
    assert t.d4 == "low"
    assert t.d5 == "low"
    assert t.overall == "some concerns"


def test_signalling_missing_domain_rejected():
    """Regression: a domain_answers dict missing one of the five extractor
    domain names must be rejected with None, not silently derived to
    "some_concerns" (derive_domain_judgement(code, {}) does not fire any
    low/high trigger and falls through to "some_concerns" for an empty
    answers dict, which previously masked the missing domain entirely).
    """
    domain_answers = {
        "randomization": {"1.1": "Y", "1.2": "Y", "1.3": "N"},
        "deviations": {"2.1": "N", "2.2": "N", "2.6": "Y"},
        "missing_outcome": {"3.1": "N", "3.2": "N", "3.4": "N"},
        "measurement": {"4.1": "N", "4.2": "N", "4.3": "N"},
        # "reporting" domain is absent entirely
    }
    assert tuple_from_signalling(domain_answers, None) is None
