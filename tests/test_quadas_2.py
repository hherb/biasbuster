"""Scaffold tests for the QUADAS-2 methodology.

Three layers of coverage, analogous to test_cochrane_rob2.py:

1. Pure-Python primitives — schema dataclass invariants (including the
   flow-and-timing asymmetry: the first three domains require an
   applicability rating, the fourth must not have one), algorithm
   rollup (2-D worst-wins), applicability gating.
2. Methodology registration — quadas_2 is active, declarative fields
   match the Whiting 2011 spec, stage dispatcher covers every domain.
3. End-to-end assessor — a scripted fake annotator walks through the
   four domain calls and produces a well-formed QUADAS2Assessment via
   the BaseAnnotator dispatch.

No LLM calls, no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import pytest

from biasbuster.annotators import BaseAnnotator
from biasbuster.methodologies import (
    ApplicabilityError,
    FullTextRequiredError,
    check_or_raise,
    get_methodology,
    study_design,
)
from biasbuster.methodologies.quadas_2 import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    evaluation_mapping_to_ground_truth,
)
from biasbuster.methodologies.quadas_2.algorithm import (
    worst_applicability,
    worst_bias,
)
from biasbuster.methodologies.quadas_2.applicability import (
    check_applicability,
)
from biasbuster.methodologies.quadas_2.assessor import (
    QUADAS2Assessor,
    _parse_domain_response,
)
from biasbuster.methodologies.quadas_2.prompts import (
    build_system_prompt,
    domain_stage_name,
)
from biasbuster.methodologies.quadas_2.schema import (
    QUADAS2_APPLICABILITY_DOMAINS,
    QUADAS2_DOMAIN_SLUGS,
    QUADAS2Assessment,
    QUADAS2DomainJudgement,
)


# ---- Helpers -----------------------------------------------------------

def _make_domain(
    slug: str, *,
    bias: str = "low",
    applicability: Optional[str] = None,
) -> QUADAS2DomainJudgement:
    """Build a minimal well-formed domain judgement.

    Applicability defaults to ``'low'`` on domains 1-3 and ``None`` on
    flow-and-timing, matching the tool's dimensionality rule.
    """
    if applicability is None and slug in QUADAS2_APPLICABILITY_DOMAINS:
        applicability = "low"
    return QUADAS2DomainJudgement(
        domain=slug,
        signalling_answers={"1.1": "yes"},
        bias_rating=bias,  # type: ignore[arg-type]
        applicability=applicability,  # type: ignore[arg-type]
    )


def _all_low_domains() -> dict[str, QUADAS2DomainJudgement]:
    return {slug: _make_domain(slug) for slug in QUADAS2_DOMAIN_SLUGS}


# ---- Schema invariants -------------------------------------------------

class TestSchemaInvariants:
    def test_unknown_domain_slug_rejected(self) -> None:
        with pytest.raises(ValueError, match="domain must be one of"):
            QUADAS2DomainJudgement(
                domain="not_a_real_domain",
                signalling_answers={},
                bias_rating="low",
            )

    def test_flow_and_timing_must_not_carry_applicability(self) -> None:
        with pytest.raises(ValueError, match="must not carry an applicability"):
            QUADAS2DomainJudgement(
                domain="flow_and_timing",
                signalling_answers={"4.1": "yes"},
                bias_rating="low",
                applicability="low",  # forbidden here
            )

    def test_domains_1_through_3_require_applicability(self) -> None:
        for slug in ("patient_selection", "index_test", "reference_standard"):
            with pytest.raises(ValueError, match="requires an applicability"):
                QUADAS2DomainJudgement(
                    domain=slug,
                    signalling_answers={"1.1": "yes"},
                    bias_rating="low",
                    # applicability intentionally omitted
                )

    def test_invalid_bias_rating_rejected(self) -> None:
        with pytest.raises(ValueError, match="bias_rating must be"):
            QUADAS2DomainJudgement(
                domain="patient_selection",
                signalling_answers={"1.1": "yes"},
                bias_rating="moderate",  # type: ignore[arg-type]
                applicability="low",
            )

    def test_invalid_signalling_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid signalling answers"):
            QUADAS2DomainJudgement(
                domain="patient_selection",
                signalling_answers={"1.1": "YES"},  # case-sensitive
                bias_rating="low",
                applicability="low",
            )

    def test_assessment_requires_all_four_domains(self) -> None:
        partial = _all_low_domains()
        del partial["flow_and_timing"]
        with pytest.raises(ValueError, match="missing domain"):
            QUADAS2Assessment(
                pmid="T1",
                domains=partial,
                methodology_version="quadas2-2011",
                worst_bias="low",
                worst_applicability="low",
            )

    def test_assessment_roundtrips_to_dict(self) -> None:
        domains = _all_low_domains()
        assessment = QUADAS2Assessment(
            pmid="T1",
            domains=domains,
            methodology_version="quadas2-2011",
            worst_bias="low",
            worst_applicability="low",
        )
        payload = assessment.to_dict()
        # Round-trip through JSON — no custom encoders.
        ser = json.loads(json.dumps(payload, default=str))
        assert ser["worst_bias"] == "low"
        assert ser["worst_applicability"] == "low"
        assert set(ser["domains"]) == set(QUADAS2_DOMAIN_SLUGS)
        # Flow and timing must serialise with applicability=None
        assert ser["domains"]["flow_and_timing"]["applicability"] is None


# ---- Algorithm ---------------------------------------------------------

class TestRollup:
    def test_all_low_rolls_up_to_low(self) -> None:
        domains = _all_low_domains()
        assert worst_bias(domains) == "low"
        assert worst_applicability(domains) == "low"

    def test_any_high_bias_rolls_up_to_high(self) -> None:
        domains = _all_low_domains()
        domains["reference_standard"] = _make_domain(
            "reference_standard", bias="high", applicability="low",
        )
        assert worst_bias(domains) == "high"

    def test_unclear_beats_low_but_loses_to_high(self) -> None:
        domains = _all_low_domains()
        domains["index_test"] = _make_domain(
            "index_test", bias="unclear", applicability="low",
        )
        assert worst_bias(domains) == "unclear"
        domains["patient_selection"] = _make_domain(
            "patient_selection", bias="high", applicability="low",
        )
        assert worst_bias(domains) == "high"

    def test_worst_applicability_ignores_flow_and_timing(self) -> None:
        """Flow and timing has no applicability; rollup must skip it."""
        domains = _all_low_domains()
        # Applicability on reference_standard is high → rollup is high
        domains["reference_standard"] = _make_domain(
            "reference_standard", bias="low", applicability="high",
        )
        assert worst_applicability(domains) == "high"

    def test_missing_domain_raises(self) -> None:
        partial = _all_low_domains()
        del partial["flow_and_timing"]
        with pytest.raises(ValueError, match="missing domain"):
            worst_bias(partial)


# ---- Applicability gate ------------------------------------------------

class TestApplicability:
    def test_diagnostic_accuracy_accepted(self) -> None:
        paper = {
            "title": "Diagnostic accuracy of biomarker X for disease Y",
            "abstract": "Sensitivity and specificity computed.",
            "mesh_terms": [],
        }
        ok, reason = check_applicability(paper, {}, True)
        assert ok is True
        assert reason == ""

    def test_systematic_review_refused_with_robis_hint(self) -> None:
        paper = {"mesh_terms": ["Systematic Review"]}
        ok, reason = check_applicability(paper, {}, True)
        assert ok is False
        assert "systematic review" in reason.lower()
        assert "ROBIS" in reason or "AMSTAR" in reason

    def test_rct_refused_with_rob2_hint(self) -> None:
        paper = {"mesh_terms": ["Randomized Controlled Trial"]}
        ok, reason = check_applicability(paper, {}, True)
        assert ok is False
        assert "cochrane_rob2" in reason

    def test_check_or_raise_refuses_cohort(self) -> None:
        paper = {"mesh_terms": ["Cohort Studies"]}
        with pytest.raises(ApplicabilityError):
            check_or_raise(
                METHODOLOGY, paper, {},
                full_text_available=True,
                detected_design=study_design.detect(paper),
            )

    def test_check_or_raise_refuses_abstract_only(self) -> None:
        paper = {"title": "Diagnostic accuracy of X",
                 "abstract": "Sensitivity computed."}
        with pytest.raises(FullTextRequiredError):
            check_or_raise(
                METHODOLOGY, paper, {},
                full_text_available=False,
                detected_design=study_design.detect(paper),
            )


# ---- Registration ------------------------------------------------------

class TestRegistration:
    def test_registered_as_active(self) -> None:
        assert METHODOLOGY.name == "quadas_2"
        assert METHODOLOGY.status == "active"

    def test_registry_lookup(self) -> None:
        assert get_methodology("quadas_2") is METHODOLOGY

    def test_declarative_fields(self) -> None:
        assert METHODOLOGY.requires_full_text is True
        assert METHODOLOGY.orchestration == "decomposed_full_text"
        assert METHODOLOGY.applicable_study_designs == frozenset(
            {"diagnostic_accuracy"}
        )
        assert METHODOLOGY.severity_rollup_levels == (
            "low", "unclear", "high",
        )

    def test_build_system_prompt_covers_all_four_domains(self) -> None:
        for slug in QUADAS2_DOMAIN_SLUGS:
            stage = domain_stage_name(slug)
            prompt = build_system_prompt(stage)
            assert prompt
            assert "signalling_answers" in prompt
            assert "bias_rating" in prompt

    def test_flow_and_timing_prompt_omits_applicability(self) -> None:
        prompt = build_system_prompt("domain_flow_and_timing")
        # The prompt explicitly tells the LLM NOT to emit an applicability field.
        assert "no applicability" in prompt.lower() \
            or "omit the applicability" in prompt.lower()

    def test_ground_truth_mapping_returns_none_until_ingestion_path_exists(
        self,
    ) -> None:
        paper = {"pmid": "T1", "abstract": "Diagnostic accuracy study"}
        assert evaluation_mapping_to_ground_truth(paper) is None


# ---- Fake annotator + end-to-end ---------------------------------------

def _domain_response(
    slug: str, *,
    bias: str = "low",
    applicability: Optional[str] = None,
) -> str:
    """Build a well-formed LLM domain-call response JSON string."""
    if applicability is None and slug in QUADAS2_APPLICABILITY_DOMAINS:
        applicability = "low"
    blob: dict[str, object] = {
        "domain": slug,
        "signalling_answers": {"1.1": "yes", "1.2": "yes"},
        "bias_rating": bias,
        "justification": f"Stub justification for {slug}.",
        "evidence_quotes": [{"text": "Quote", "section": "Methods"}],
    }
    if applicability is not None:
        blob["applicability"] = applicability
    return json.dumps(blob)


@dataclass
class FakeAnnotator(BaseAnnotator):
    """Scripts domain-call responses and bypasses the real extraction pipeline."""

    model: str = "fake-quadas2"
    max_retries: int = 1
    response_queue: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    fake_extraction: dict = field(default_factory=lambda: {
        "design": "diagnostic_accuracy",
        "index_test": "ultrasound",
        "reference_standard": "biopsy",
    })

    async def _extract_full_text_sections(
        self, pmid: str, title: str, sections: list,
    ):
        return self.fake_extraction, [], [], 0

    async def _call_llm(
        self, system_prompt: str, user_message: str, pmid: str = "",
    ):
        self.calls.append((system_prompt, user_message, pmid))
        if not self.response_queue:
            return None
        return self.response_queue.pop(0)

    async def __aenter__(self) -> "FakeAnnotator":
        return self

    async def __aexit__(self, *args) -> None:
        return None


class TestAssessorEndToEnd:
    @pytest.mark.asyncio
    async def test_happy_path_all_low(self) -> None:
        fa = FakeAnnotator(response_queue=[
            _domain_response(slug) for slug in QUADAS2_DOMAIN_SLUGS
        ])
        assessor = QUADAS2Assessor(fa)
        result = await assessor.assess(
            pmid="T1", title="Diagnostic accuracy of X",
            sections=[("Methods", "Index test compared to biopsy.")],
        )
        assert result is not None
        assert result.pmid == "T1"
        assert result.worst_bias == "low"
        assert result.worst_applicability == "low"
        assert set(result.domains) == set(QUADAS2_DOMAIN_SLUGS)
        # Flow and timing must have no applicability field.
        assert result.domains["flow_and_timing"].applicability is None

    @pytest.mark.asyncio
    async def test_mixed_ratings_rollup_correctly(self) -> None:
        """Bias and applicability rollups are computed independently."""
        fa = FakeAnnotator(response_queue=[
            _domain_response("patient_selection", bias="low", applicability="high"),
            _domain_response("index_test", bias="high", applicability="low"),
            _domain_response("reference_standard", bias="unclear", applicability="low"),
            _domain_response("flow_and_timing", bias="low"),
        ])
        assessor = QUADAS2Assessor(fa)
        result = await assessor.assess(
            pmid="T2", title="Another diagnostic study",
            sections=[("Methods", "…")],
        )
        assert result is not None
        assert result.worst_bias == "high"  # index_test
        assert result.worst_applicability == "high"  # patient_selection

    @pytest.mark.asyncio
    async def test_domain_parse_failure_aborts_assessment(self) -> None:
        queue = [_domain_response(slug) for slug in QUADAS2_DOMAIN_SLUGS]
        queue[1] = "not valid json"  # corrupt the index_test response
        fa = FakeAnnotator(response_queue=queue)
        assessor = QUADAS2Assessor(fa, max_retries_per_domain=1)
        result = await assessor.assess(
            pmid="T3", title="Doomed study",
            sections=[("Methods", "…")],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_end_to_end_via_base_annotator_dispatch(self) -> None:
        """annotate_full_text_decomposed routes quadas_2 to the right assessor."""
        fa = FakeAnnotator(response_queue=[
            _domain_response(slug) for slug in QUADAS2_DOMAIN_SLUGS
        ])
        result = await fa.annotate_full_text_decomposed(
            pmid="T4", title="Diagnostic study",
            sections=[("Methods", "…")],
            metadata=None,
            methodology=METHODOLOGY,
        )
        assert result is not None
        assert result["_methodology"] == "quadas_2"
        assert result["_methodology_version"] == METHODOLOGY_VERSION
        assert result["_annotation_mode"] == "decomposed_quadas2"
        assert result["overall_severity"] == "low"
        assert result["overall_applicability"] == "low"


# ---- Parser unit tests --------------------------------------------------

class TestDomainResponseParser:
    def test_well_formed_domain_1_parsed(self) -> None:
        raw = _domain_response("patient_selection", bias="high",
                               applicability="unclear")
        parsed = _parse_domain_response(raw, "patient_selection", pmid="T1")
        assert parsed is not None
        assert parsed.bias_rating == "high"
        assert parsed.applicability == "unclear"

    def test_flow_and_timing_parsed_without_applicability(self) -> None:
        raw = _domain_response("flow_and_timing", bias="low")
        parsed = _parse_domain_response(raw, "flow_and_timing", pmid="T1")
        assert parsed is not None
        assert parsed.applicability is None

    def test_missing_applicability_on_domain_1_returns_none(self) -> None:
        """Domain 1-3 without an applicability rating is invalid."""
        raw = json.dumps({
            "domain": "patient_selection",
            "signalling_answers": {"1.1": "yes"},
            "bias_rating": "low",
            # applicability missing
        })
        assert _parse_domain_response(raw, "patient_selection", pmid="T1") is None

    def test_capitalised_ratings_coerced(self) -> None:
        """LLMs often emit 'Yes' / 'Low' instead of 'yes' / 'low'.

        The coercion layer normalises these to the canonical lowercase
        form so valid assessments aren't rejected on casing alone.
        """
        raw = json.dumps({
            "domain": "index_test",
            "signalling_answers": {"2.1": "Yes"},
            "bias_rating": "Low",
            "applicability": "Unclear",
        })
        parsed = _parse_domain_response(raw, "index_test", pmid="T1")
        assert parsed is not None
        assert parsed.bias_rating == "low"
        assert parsed.applicability == "unclear"
        assert parsed.signalling_answers == {"2.1": "yes"}
