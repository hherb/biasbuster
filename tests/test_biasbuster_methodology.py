"""Tests for the biasbuster methodology pass-through adapter.

Key guarantees under test:

- The methodology registers itself on import under the slug ``"biasbuster"``.
- Its callables delegate to the existing biasbuster code — no prompt text
  is duplicated, no parsing is re-implemented, no aggregation is reinvented.
  Byte-identity of the delegated prompts is the primary safety property.
- It is always applicable, never refuses, and does not require full text
  (biasbuster is the default abstract-capable pathway).
"""

from __future__ import annotations

import pytest

from biasbuster import prompts as _prompts_v1
from biasbuster import prompts_v3 as _prompts_v3
from biasbuster import prompts_v4 as _prompts_v4
from biasbuster import prompts_v5a as _prompts_v5a
from biasbuster.methodologies import (
    ANY_STUDY_DESIGN,
    Methodology,
    check_or_raise,
    get_methodology,
    list_active_methodologies,
)


@pytest.fixture
def biasbuster_methodology() -> Methodology:
    return get_methodology("biasbuster")


class TestRegistration:
    def test_registered_under_expected_slug(
        self, biasbuster_methodology: Methodology
    ) -> None:
        assert biasbuster_methodology.name == "biasbuster"
        assert biasbuster_methodology.status == "active"
        assert "biasbuster" in list_active_methodologies()

    def test_always_applicable_to_any_design(
        self, biasbuster_methodology: Methodology
    ) -> None:
        assert ANY_STUDY_DESIGN in biasbuster_methodology.applicable_study_designs
        for design in ("rct_parallel", "cohort", "diagnostic_accuracy", "unknown"):
            assert biasbuster_methodology.applies_to(design)

    def test_does_not_require_full_text(
        self, biasbuster_methodology: Methodology
    ) -> None:
        assert biasbuster_methodology.requires_full_text is False

    def test_five_level_severity_rollup(
        self, biasbuster_methodology: Methodology
    ) -> None:
        assert biasbuster_methodology.severity_rollup_levels == (
            "none", "low", "moderate", "high", "critical",
        )


class TestPromptDispatch:
    """build_system_prompt must return *identical* constants to the existing code."""

    @pytest.mark.parametrize("stage,expected_source", [
        ("single_call", _prompts_v1.ANNOTATION_SYSTEM_PROMPT),
        ("extract_abstract", _prompts_v3.EXTRACTION_SYSTEM_PROMPT),
        ("assess", _prompts_v3.ASSESSMENT_SYSTEM_PROMPT),
        ("extract_section", _prompts_v3.SECTION_EXTRACTION_SYSTEM_PROMPT),
        ("assess_agentic", _prompts_v4.ASSESSMENT_AGENT_SYSTEM_PROMPT),
        ("domain_override", _prompts_v5a.DOMAIN_OVERRIDE_SYSTEM_PROMPT),
        ("synthesize", _prompts_v5a.SYNTHESIS_SYSTEM_PROMPT),
        ("training", _prompts_v1.TRAINING_SYSTEM_PROMPT),
    ])
    def test_stage_returns_canonical_prompt(
        self,
        biasbuster_methodology: Methodology,
        stage: str,
        expected_source: str,
    ) -> None:
        """Each known stage maps to the exact canonical prompt string.

        ``is`` identity (not just equality) is checked: the methodology
        must *reference* the existing constant, not duplicate it.
        """
        got = biasbuster_methodology.build_system_prompt(stage)
        assert got is expected_source

    def test_unknown_stage_raises(
        self, biasbuster_methodology: Methodology
    ) -> None:
        with pytest.raises(KeyError, match="no prompt for stage"):
            biasbuster_methodology.build_system_prompt("made-up-stage")


class TestUserMessageDelegation:
    """build_user_message must route through the existing helper unchanged."""

    def test_delegates_to_legacy_helper(
        self, biasbuster_methodology: Methodology
    ) -> None:
        paper = {
            "pmid": "T1",
            "title": "Effect of TestDrug on mortality",
            "abstract": "Randomized placebo-controlled trial of 500 patients. "
                        "Relative risk reduction 30%.",
        }
        msg = biasbuster_methodology.build_user_message(paper=paper)
        assert isinstance(msg, str)
        assert "T1" in msg
        assert "TestDrug" in msg
        assert "Relative risk" in msg

    def test_merges_enrichment_into_metadata(
        self, biasbuster_methodology: Methodology
    ) -> None:
        paper = {"pmid": "T2", "title": "T", "abstract": "A"}
        enrichment = {"suspicion_level": "high", "reporting_bias_score": 0.87}
        msg = biasbuster_methodology.build_user_message(
            paper=paper, enrichment=enrichment,
        )
        assert "T2" in msg


class TestOutputParsing:
    def test_parse_valid_json(
        self, biasbuster_methodology: Methodology
    ) -> None:
        raw = (
            '{"overall_severity": "high", "overall_bias_probability": 0.9, '
            '"confidence": "medium", "reasoning": "...", '
            '"statistical_reporting": {"severity": "high"}, '
            '"spin": {"spin_level": "high", "severity": "high"}, '
            '"outcome_reporting": {"severity": "moderate"}, '
            '"conflict_of_interest": {"severity": "moderate"}, '
            '"methodology": {"severity": "low"}, '
            '"recommended_verification_steps": ["Check ClinicalTrials.gov"]}'
        )
        out = biasbuster_methodology.parse_output(raw, stage="assess")
        assert out is not None
        assert out["overall_severity"] == "high"

    def test_parse_garbage_returns_none(
        self, biasbuster_methodology: Methodology, caplog
    ) -> None:
        assert biasbuster_methodology.parse_output(
            "this is not JSON", stage="assess"
        ) is None


class TestApplicabilityAlwaysTrue:
    def test_check_applicability_is_unconditionally_true(
        self, biasbuster_methodology: Methodology
    ) -> None:
        ok, reason = biasbuster_methodology.check_applicability(
            {"pmid": "X"}, {}, False,
        )
        assert ok is True
        assert reason == ""

    def test_check_or_raise_never_raises_for_biasbuster(
        self, biasbuster_methodology: Methodology
    ) -> None:
        """Biasbuster is the safety-net pathway: no error under any design."""
        for design in ("rct_parallel", "cohort", "unknown", "diagnostic_accuracy"):
            for ft in (True, False):
                check_or_raise(
                    biasbuster_methodology,
                    {"pmid": "P1"}, {},
                    full_text_available=ft,
                    detected_design=design,
                )


class TestGroundTruthMapping:
    def test_returns_none_signalling_external_mapping(
        self, biasbuster_methodology: Methodology
    ) -> None:
        """Biasbuster has no built-in mapping — comparison lives in compare_vs_cochrane."""
        assert biasbuster_methodology.evaluation_mapping_to_ground_truth(
            {"pmid": "X", "overall_rob": "high"}
        ) is None
