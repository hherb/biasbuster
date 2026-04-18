"""Step-7 scaffold tests for the Cochrane RoB 2 methodology.

Three layers of coverage:

1. **Pure-Python primitives** — schema dataclasses, algorithm rollup,
   applicability gating. No LLM, no annotator, no network.
2. **Methodology registration** — cochrane_rob2 is registered as
   ``status='active'``, carries the right declarative fields, and round-trips
   through the registry lookup.
3. **End-to-end assessor with a fake annotator** — a subclass of
   :class:`BaseAnnotator` returns canned domain-call responses; the
   assessor orchestrates the five domain calls, runs the deterministic
   rollup, and emits a well-formed :class:`RoB2Assessment`.

Step 8 iterates on the prompts against real Cochrane-reviewed papers;
this file is the scaffold guarantee that the plumbing is correct.
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
from biasbuster.methodologies.cochrane_rob2 import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    evaluation_mapping_to_ground_truth,
)
from biasbuster.methodologies.cochrane_rob2.algorithm import (
    aggregate_outcome,
    worst_case_across_outcomes,
)
from biasbuster.methodologies.cochrane_rob2.applicability import (
    check_applicability,
)
from biasbuster.methodologies.cochrane_rob2.assessor import (
    CochraneRoB2Assessor,
    _parse_domain_response,
)
from biasbuster.methodologies.cochrane_rob2.prompts import (
    build_system_prompt,
    domain_stage_name,
)
from biasbuster.methodologies.cochrane_rob2.schema import (
    ROB2_DOMAIN_SLUGS,
    EvidenceQuote,
    RoB2Assessment,
    RoB2DomainJudgement,
    RoB2OutcomeJudgement,
)


# ---- Dataclass invariants -----------------------------------------------

class TestSchemaInvariants:
    def test_domain_slug_rejected_if_unknown(self) -> None:
        with pytest.raises(ValueError, match="domain must be one of"):
            RoB2DomainJudgement(
                domain="not_a_real_domain",
                signalling_answers={},
                judgement="low",
            )

    def test_judgement_must_be_valid(self) -> None:
        with pytest.raises(ValueError, match="judgement must be one of"):
            RoB2DomainJudgement(
                domain="randomization",
                signalling_answers={"1.1": "Y"},
                judgement="maybe",  # type: ignore[arg-type]
            )

    def test_signalling_answer_must_be_valid(self) -> None:
        with pytest.raises(ValueError, match="Invalid signalling answers"):
            RoB2DomainJudgement(
                domain="randomization",
                signalling_answers={"1.1": "yes"},  # lower-case invalid
                judgement="low",
            )

    def test_outcome_requires_all_five_domains(self) -> None:
        d_rand = RoB2DomainJudgement(
            domain="randomization", signalling_answers={"1.1": "Y"},
            judgement="low",
        )
        with pytest.raises(ValueError, match="missing domain"):
            RoB2OutcomeJudgement(
                outcome_label="primary",
                result_label="ITT",
                domains={"randomization": d_rand},
                overall_judgement="low",
            )

    def test_assessment_requires_at_least_one_outcome(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            RoB2Assessment(
                pmid="T1",
                outcomes=[],
                methodology_version="rob2-2019",
                worst_across_outcomes="low",
            )

    def test_assessment_roundtrips_to_dict(self) -> None:
        domains = _make_all_low_domains()
        outcome = RoB2OutcomeJudgement(
            outcome_label="primary outcome",
            result_label="as reported",
            domains=domains,
            overall_judgement="low",
            overall_rationale="All five domains were low.",
        )
        assessment = RoB2Assessment(
            pmid="T1",
            outcomes=[outcome],
            methodology_version="rob2-2019",
            worst_across_outcomes="low",
        )
        payload = assessment.to_dict()
        assert payload["pmid"] == "T1"
        assert payload["worst_across_outcomes"] == "low"
        assert len(payload["outcomes"]) == 1
        # Round-trippable through JSON without custom encoders.
        roundtripped = json.loads(json.dumps(payload, default=str))
        assert roundtripped["outcomes"][0]["overall_judgement"] == "low"


# ---- Algorithm ---------------------------------------------------------

def _make_all_low_domains() -> dict[str, RoB2DomainJudgement]:
    return {
        slug: RoB2DomainJudgement(
            domain=slug,
            signalling_answers={"1.1": "Y"},
            judgement="low",
        )
        for slug in ROB2_DOMAIN_SLUGS
    }


def _make_domains_with_overrides(
    **per_slug_judgements,
) -> dict[str, RoB2DomainJudgement]:
    """Build a 5-domain dict, defaulting missing slugs to 'low'."""
    domains = _make_all_low_domains()
    for slug, j in per_slug_judgements.items():
        domains[slug] = RoB2DomainJudgement(
            domain=slug, signalling_answers={"1.1": "Y"}, judgement=j,
        )
    return domains


class TestAggregateOutcome:
    def test_all_low_rolls_up_to_low(self) -> None:
        assert aggregate_outcome(_make_all_low_domains()) == "low"

    def test_any_high_rolls_up_to_high(self) -> None:
        domains = _make_domains_with_overrides(randomization="high")
        assert aggregate_outcome(domains) == "high"

    def test_any_some_concerns_and_no_high_rolls_up_to_some_concerns(
        self,
    ) -> None:
        domains = _make_domains_with_overrides(
            deviations_from_interventions="some_concerns",
        )
        assert aggregate_outcome(domains) == "some_concerns"

    def test_high_beats_some_concerns(self) -> None:
        domains = _make_domains_with_overrides(
            deviations_from_interventions="some_concerns",
            outcome_measurement="high",
        )
        assert aggregate_outcome(domains) == "high"

    def test_missing_domain_raises(self) -> None:
        partial = _make_all_low_domains()
        del partial["outcome_measurement"]
        with pytest.raises(ValueError, match="missing domain"):
            aggregate_outcome(partial)


class TestWorstCaseAcrossOutcomes:
    def _outcome(self, judgement: str) -> RoB2OutcomeJudgement:
        return RoB2OutcomeJudgement(
            outcome_label=f"o-{judgement}",
            result_label="r",
            domains=_make_all_low_domains(),
            overall_judgement=judgement,  # type: ignore[arg-type]
        )

    def test_single_outcome_passthrough(self) -> None:
        assert worst_case_across_outcomes(
            [self._outcome("some_concerns")]
        ) == "some_concerns"

    def test_worst_across_mixed_outcomes(self) -> None:
        assert worst_case_across_outcomes([
            self._outcome("low"),
            self._outcome("high"),
            self._outcome("some_concerns"),
        ]) == "high"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no outcome judgements"):
            worst_case_across_outcomes([])


# ---- Applicability -----------------------------------------------------

class TestApplicability:
    def test_parallel_rct_accepted(self) -> None:
        paper = {"randomization_bias": "low"}  # triggers rct_parallel detection
        ok, reason = check_applicability(paper, {}, True)
        assert ok is True
        assert reason == ""

    def test_cohort_refused_with_specific_reason(self) -> None:
        paper = {"mesh_terms": ["Cohort Studies"]}
        ok, reason = check_applicability(paper, {}, True)
        assert ok is False
        assert "parallel-group" in reason
        assert "cohort" in reason

    def test_cluster_rct_refused_with_variant_hint(self) -> None:
        paper = {"title": "A cluster-randomized trial of X"}
        ok, reason = check_applicability(paper, {}, True)
        assert ok is False
        assert "cluster-RCT" in reason or "cluster-randomized" in reason

    def test_crossover_refused_with_variant_hint(self) -> None:
        paper = {"title": "A crossover trial of TestDrug"}
        ok, reason = check_applicability(paper, {}, True)
        assert ok is False
        assert "crossover" in reason

    def test_check_or_raise_refuses_cohort(self) -> None:
        paper = {"mesh_terms": ["Cohort Studies"]}
        with pytest.raises(ApplicabilityError):
            check_or_raise(
                METHODOLOGY, paper, {},
                full_text_available=True,
                detected_design=study_design.detect(paper),
            )

    def test_check_or_raise_refuses_abstract_only(self) -> None:
        paper = {"randomization_bias": "low"}
        with pytest.raises(FullTextRequiredError):
            check_or_raise(
                METHODOLOGY, paper, {},
                full_text_available=False,
                detected_design=study_design.detect(paper),
            )


# ---- Registration ------------------------------------------------------

class TestRegistration:
    def test_registered_as_active(self) -> None:
        assert METHODOLOGY.name == "cochrane_rob2"
        assert METHODOLOGY.status == "active"

    def test_registry_lookup(self) -> None:
        assert get_methodology("cochrane_rob2") is METHODOLOGY

    def test_declarative_fields(self) -> None:
        assert METHODOLOGY.requires_full_text is True
        assert METHODOLOGY.orchestration == "decomposed_full_text"
        assert METHODOLOGY.applicable_study_designs == frozenset({"rct_parallel"})
        assert METHODOLOGY.severity_rollup_levels == (
            "low", "some_concerns", "high",
        )

    def test_build_system_prompt_covers_all_five_domains(self) -> None:
        for slug in ROB2_DOMAIN_SLUGS:
            stage = domain_stage_name(slug)
            prompt = build_system_prompt(stage)
            assert prompt, f"empty prompt for stage {stage!r}"
            # Every domain prompt must describe its output shape.
            assert "signalling_answers" in prompt
            assert "judgement" in prompt

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(KeyError, match="no prompt for stage"):
            build_system_prompt("made-up-stage")


class TestGroundTruthMapping:
    def test_full_cochrane_columns_yield_mapping(self) -> None:
        paper = {
            "overall_rob": "some_concerns",
            "randomization_bias": "low",
            "deviation_bias": "some_concerns",
            "missing_outcome_bias": "low",
            "measurement_bias": "low",
            "reporting_bias": "low",
        }
        gt = evaluation_mapping_to_ground_truth(paper)
        assert gt is not None
        assert gt["overall"] == "some_concerns"
        assert gt["domains"]["randomization"] == "low"
        assert gt["domains"]["deviations_from_interventions"] == "some_concerns"

    def test_missing_domain_yields_none(self) -> None:
        paper = {
            "overall_rob": "low",
            "randomization_bias": "low",
            "deviation_bias": "low",
            "missing_outcome_bias": "low",
            "measurement_bias": "low",
            # reporting_bias missing
        }
        assert evaluation_mapping_to_ground_truth(paper) is None


# ---- Assessor end-to-end with a fake LLM -------------------------------

def _domain_response(domain_slug: str, judgement: str = "low") -> str:
    """Build a well-formed domain-call response JSON string."""
    return json.dumps({
        "domain": domain_slug,
        "signalling_answers": {"1.1": "Y", "1.2": "PY"},
        "judgement": judgement,
        "justification": f"Stub justification for {domain_slug}.",
        "evidence_quotes": [
            {"text": "Example quote", "section": "Methods"}
        ],
    })


@dataclass
class FakeAnnotator(BaseAnnotator):
    """Annotator that scripts domain-call responses and an extraction."""

    model: str = "fake-rob2"
    max_retries: int = 1
    # Response queue — one canned string per _call_llm invocation. Used
    # sequentially so tests can script the 5 domain calls' outputs (plus
    # whatever the extraction call emits, if any).
    response_queue: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    # Fixed extraction blob returned from _extract_full_text_sections;
    # bypasses the real extraction pipeline which would make its own
    # LLM calls.
    fake_extraction: dict = field(default_factory=lambda: {
        "design": "parallel_rct",
        "participants": 500,
    })

    async def _extract_full_text_sections(
        self, pmid: str, title: str, sections: list,
    ):
        # Return the 4-tuple shape the real method produces:
        # (extraction, section_extractions, merge_conflicts, failed_count)
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
        """Five low-risk domain responses produce an all-low RoB2Assessment."""
        fa = FakeAnnotator(response_queue=[
            _domain_response(slug, "low") for slug in ROB2_DOMAIN_SLUGS
        ])
        assessor = CochraneRoB2Assessor(fa)
        result = await assessor.assess(
            pmid="T1", title="Trial of X",
            sections=[("Methods", "Randomized to treatment or placebo.")],
        )
        assert result is not None
        assert result.pmid == "T1"
        assert result.worst_across_outcomes == "low"
        assert len(result.outcomes) == 1
        outcome = result.outcomes[0]
        assert outcome.overall_judgement == "low"
        assert set(outcome.domains) == set(ROB2_DOMAIN_SLUGS)
        for slug in ROB2_DOMAIN_SLUGS:
            assert outcome.domains[slug].judgement == "low"

    @pytest.mark.asyncio
    async def test_mixed_domains_rollup_to_high(self) -> None:
        """One 'high' domain forces the overall judgement to 'high'."""
        judgements = ["low", "low", "high", "some_concerns", "low"]
        fa = FakeAnnotator(response_queue=[
            _domain_response(slug, j)
            for slug, j in zip(ROB2_DOMAIN_SLUGS, judgements)
        ])
        assessor = CochraneRoB2Assessor(fa)
        result = await assessor.assess(
            pmid="T2", title="Another trial",
            sections=[("Methods", "...")],
        )
        assert result is not None
        assert result.worst_across_outcomes == "high"
        assert result.outcomes[0].overall_judgement == "high"

    @pytest.mark.asyncio
    async def test_domain_parse_failure_aborts_assessment(self) -> None:
        """An unparseable response on any domain causes the whole assess to fail.

        The assessor doesn't fabricate judgements for missing domains —
        an incomplete RoB 2 is worse than no RoB 2 because it would be
        silently taken as authoritative.
        """
        queue = [_domain_response(slug) for slug in ROB2_DOMAIN_SLUGS]
        # Corrupt the third response so the third domain fails to parse.
        queue[2] = "not valid json"
        fa = FakeAnnotator(response_queue=queue)
        assessor = CochraneRoB2Assessor(fa, max_retries_per_domain=1)
        result = await assessor.assess(
            pmid="T3", title="Doomed trial",
            sections=[("Methods", "...")],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_end_to_end_via_base_annotator_dispatch(self) -> None:
        """annotate_full_text_decomposed routes cochrane_rob2 to the right assessor.

        This is the integration point the pipeline and single-paper CLI
        both flow through. We verify the dispatch hands off to the
        Cochrane assessor (not biasbuster's DecomposedAssessor) and
        stamps the annotation with the right methodology metadata.
        """
        fa = FakeAnnotator(response_queue=[
            _domain_response(slug, "low") for slug in ROB2_DOMAIN_SLUGS
        ])
        result = await fa.annotate_full_text_decomposed(
            pmid="T4", title="Trial integrated",
            sections=[("Methods", "...")],
            metadata=None,
            methodology=METHODOLOGY,
        )
        assert result is not None
        assert result["_methodology"] == "cochrane_rob2"
        assert result["_methodology_version"] == METHODOLOGY_VERSION
        assert result["_annotation_mode"] == "decomposed_rob2"
        assert result["overall_severity"] == "low"
        assert result["worst_across_outcomes"] == "low"


# ---- Parser unit tests --------------------------------------------------

class TestDomainResponseParser:
    def test_well_formed_response_parsed(self) -> None:
        raw = _domain_response("randomization", "some_concerns")
        parsed = _parse_domain_response(raw, "randomization", pmid="T1")
        assert parsed is not None
        assert parsed.domain == "randomization"
        assert parsed.judgement == "some_concerns"
        assert parsed.signalling_answers == {"1.1": "Y", "1.2": "PY"}

    def test_markdown_fence_tolerated(self) -> None:
        raw = "```json\n" + _domain_response("randomization", "low") + "\n```"
        parsed = _parse_domain_response(raw, "randomization", pmid="T1")
        assert parsed is not None
        assert parsed.judgement == "low"

    def test_invalid_judgement_rejected(self) -> None:
        raw = json.dumps({
            "domain": "randomization",
            "signalling_answers": {"1.1": "Y"},
            "judgement": "maybe",
        })
        assert _parse_domain_response(raw, "randomization", pmid="T1") is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_domain_response(
            "definitely not json", "randomization", pmid="T1",
        ) is None
