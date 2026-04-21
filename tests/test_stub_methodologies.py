"""Tests for the stub methodologies (robins_i, robins_e, probast, ...).

A stub methodology is registered for discoverability (shows in
``list_methodologies()``) but every behavioural callable raises
:class:`NotImplementedError` with a specific message. The pipeline
layer refuses to use them, so an accidental run fails at the entry
point rather than deep inside the annotator.
"""

from __future__ import annotations

import pytest

from biasbuster.methodologies import (
    Methodology,
    get_methodology,
    list_active_methodologies,
    list_methodologies,
)
from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

# Every methodology slug expected to be registered as a stub. This list
# is the single source of truth for the "eight stubs" invariant; if a
# future PR promotes one of these to active, the corresponding entry
# here must move to the active methodology registration check.
EXPECTED_STUBS: frozenset[str] = frozenset({
    "robins_i", "robins_e", "probast", "probast_plus_ai",
    "rob_me", "syrcle", "robis", "amstar_2",
})


class TestStubRegistry:
    def test_every_stub_is_registered(self) -> None:
        registered = set(list_methodologies())
        missing = EXPECTED_STUBS - registered
        assert not missing, f"stubs not registered: {missing}"

    def test_stubs_are_excluded_from_active_list(self) -> None:
        active = set(list_active_methodologies())
        overlap = EXPECTED_STUBS & active
        assert not overlap, (
            f"stub methodologies listed as active: {overlap}"
        )

    def test_active_methodologies_are_the_three_implemented_ones(self) -> None:
        assert set(list_active_methodologies()) == {
            "biasbuster", "cochrane_rob2", "quadas_2",
        }

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_every_stub_has_status_stub(self, name: str) -> None:
        m = get_methodology(name)
        assert m.status == "stub"

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_every_stub_has_display_name_and_version(self, name: str) -> None:
        m = get_methodology(name)
        assert m.display_name
        assert m.version
        # Stub modules tag their display names; useful so CLI --help
        # output clearly signals "registered but not implemented".
        assert "STUB" in m.display_name


class TestStubCallablesRaise:
    """Every behavioural callable on a stub must raise NotImplementedError."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_build_system_prompt_raises(self, name: str) -> None:
        m = get_methodology(name)
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            m.build_system_prompt("any_stage")

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_aggregate_raises(self, name: str) -> None:
        m = get_methodology(name)
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            m.aggregate({})

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_check_applicability_raises(self, name: str) -> None:
        m = get_methodology(name)
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            m.check_applicability({}, {}, False)

    @pytest.mark.parametrize("name", sorted(EXPECTED_STUBS))
    def test_evaluation_mapping_raises(self, name: str) -> None:
        m = get_methodology(name)
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            m.evaluation_mapping_to_ground_truth({})

    def test_stub_error_message_names_methodology_and_method(self) -> None:
        """The error message tells the user *which* stub and *which* method."""
        m = get_methodology("probast")
        with pytest.raises(NotImplementedError) as exc:
            m.build_system_prompt("x")
        assert "probast" in str(exc.value)
        assert "build_system_prompt" in str(exc.value)


class TestStubHelpers:
    """The stub factory is the single place stubs should be constructed."""

    def test_make_stub_methodology_produces_status_stub(self) -> None:
        m = make_stub_methodology(
            name="temp_test_stub",
            display_name="Temp — STUB",
            version="test-1",
            applicable_study_designs=frozenset({"unknown"}),
            requires_full_text=True,
            orchestration="decomposed_full_text",
            severity_rollup_levels=("low", "high"),
        )
        assert m.status == "stub"
        with pytest.raises(NotImplementedError):
            m.build_system_prompt("x")

    def test_register_stub_once_is_idempotent(self) -> None:
        from biasbuster.methodologies.registry import REGISTRY

        m = make_stub_methodology(
            name="temp_idempotent_stub",
            display_name="Temp — STUB",
            version="test-1",
            applicable_study_designs=frozenset({"unknown"}),
            requires_full_text=True,
            orchestration="decomposed_full_text",
            severity_rollup_levels=("low", "high"),
        )
        try:
            first = register_stub_once(m)
            second = register_stub_once(m)
            assert first is m
            assert second is m
            assert REGISTRY["temp_idempotent_stub"] is m
        finally:
            REGISTRY.pop("temp_idempotent_stub", None)


class TestPipelineRefusesStubs:
    """stage_annotate must reject a stub methodology up-front."""

    def test_stage_annotate_raises_valueerror_on_stub(self) -> None:
        """Called as a library call, stage_annotate raises for stub methodologies."""
        import asyncio
        from unittest.mock import MagicMock

        from biasbuster.pipeline import stage_annotate

        with pytest.raises(ValueError, match="not active"):
            # The DB/config are never touched because the methodology
            # check happens first.
            asyncio.run(stage_annotate(
                MagicMock(), MagicMock(),
                methodology="probast",
            ))
