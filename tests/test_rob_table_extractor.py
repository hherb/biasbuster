"""Characterization tests for the reused structural bias-table extractor.

Pins the EXISTING behaviour of ``biasbuster.collectors.rob_table_extractor``
so that its reuse in the OA-first RoB benchmark (Stage A/B) is regression-safe.
Do NOT modify ``rob_table_extractor.py`` to make these tests pass — if the
extractor's real behaviour differs from what's asserted here, update the
assertions to match reality, not the other way around.

``tests/fixtures/cochrane_reviews/jcm-15-01829.xml`` (the only existing JATS
fixture at authoring time) uses QUADAS-2, not RoB 2 — confirmed by running
the extractor against it. Per the task brief's Step 6 note, a minimal
synthetic RoB 2 JATS fixture was added at ``tests/fixtures/oa_rob/rob2_table.xml``
to exercise RoB 2 detection specifically, without touching the extractor.
"""
from pathlib import Path

from biasbuster.collectors.rob_table_extractor import extract_bias_tables, ROB2

EXISTING_FIXTURE = Path("tests/fixtures/cochrane_reviews/jcm-15-01829.xml")
ROB2_FIXTURE = Path("tests/fixtures/oa_rob/rob2_table.xml")


def test_extractor_runs_on_existing_fixture_without_error():
    tables = extract_bias_tables(EXISTING_FIXTURE.read_bytes())
    # Characterization: pin current behaviour so Stage-A reuse is regression-safe.
    assert isinstance(tables, list)
    for t in tables:
        assert t.methodology.name in {"rob2", "quadas2", "robins_i"}


def test_existing_fixture_is_quadas2_not_rob2():
    """Characterization: the only pre-existing JATS fixture uses QUADAS-2.

    This is why a synthetic RoB 2 fixture (below) was added rather than
    relying on this fixture for RoB 2 detection coverage.
    """
    tables = extract_bias_tables(EXISTING_FIXTURE.read_bytes())
    assert len(tables) == 1
    assert tables[0].methodology.name == "quadas2"


def test_synthetic_fixture_is_detected_as_rob2():
    """The added synthetic fixture must be detected as RoB 2 with all five
    domains plus an overall column, and per-cell text + colour extracted."""
    tables = extract_bias_tables(ROB2_FIXTURE.read_bytes())
    assert len(tables) == 1
    table = tables[0]
    assert table.methodology is ROB2
    assert table.overall_col is not None
    assert set(table.domain_mapping.values()) == set(ROB2.domain_keywords.keys())
    assert len(table.studies) == 2

    first = table.studies[0]
    assert first.study_id == "Alpha (2021)"
    assert len(first.domains) == 5
    assert first.overall is not None
    assert first.overall.rating_text == "some_concerns"


def test_malformed_xml_returns_empty_not_raises():
    assert extract_bias_tables(b"<not-valid-xml") == []
