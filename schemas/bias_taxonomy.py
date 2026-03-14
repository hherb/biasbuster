"""
Bias Taxonomy Schema for BMLibrarian Bias Detection

Defines the structured label format used for training data annotation.
Each abstract receives a multi-dimensional assessment across these domains.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json

def _get_system_prompt() -> str:
    """Lazy import to avoid circular dependency with export.py."""
    from export import SYSTEM_PROMPT
    return SYSTEM_PROMPT


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SpinLevel(str, Enum):
    """Boutron classification for spin in conclusions."""
    NONE = "none"
    LOW = "low"          # Uncertainty + recommendation for further trials OR acknowledgment of NS
    MODERATE = "moderate" # Uncertainty OR further trials, but no acknowledgment of NS primary
    HIGH = "high"        # No uncertainty, no further trials recommended, no acknowledgment of NS


class FundingType(str, Enum):
    INDUSTRY = "industry"
    PUBLIC = "public"
    MIXED = "mixed"
    NOT_REPORTED = "not_reported"
    UNCLEAR = "unclear"


class ComparatorType(str, Enum):
    PLACEBO = "placebo"
    ACTIVE = "active"
    STANDARD_OF_CARE = "standard_of_care"
    NO_TREATMENT = "no_treatment"
    UNCLEAR = "unclear"


class OutcomeType(str, Enum):
    PATIENT_CENTRED = "patient_centred"  # mortality, QoL, functional status
    SURROGATE = "surrogate"              # lab values, imaging markers
    COMPOSITE = "composite"
    UNCLEAR = "unclear"


@dataclass
class StatisticalReportingBias:
    """
    YOUR KEY HYPOTHESIS: Emphasis on relative measures is a strong bias predictor.

    This domain captures how effect sizes are reported in the abstract.
    The model should learn that relative-only reporting inflates perceived benefit.
    """
    relative_only: bool = False           # Only RRR/OR/HR reported, no ARR/NNT
    absolute_reported: bool = False        # ARR, absolute risk, or NNT present
    nnt_reported: bool = False            # Number needed to treat/harm
    baseline_risk_reported: bool = False   # Control group event rate given
    confidence_intervals_reported: bool = False
    selective_p_values: bool = False       # Only favourable p-values highlighted
    subgroup_emphasis: bool = False        # Subgroup results emphasised over primary
    severity: Severity = Severity.NONE
    evidence_quotes: list[str] = field(default_factory=list)  # Specific text spans


@dataclass
class SpinAssessment:
    """Boutron-derived spin classification."""
    spin_level: SpinLevel = SpinLevel.NONE
    conclusion_matches_results: bool = True
    causal_language_from_observational: bool = False
    focus_on_secondary_when_primary_ns: bool = False
    inappropriate_extrapolation: bool = False   # Applying results to unstudied population
    title_spin: bool = False                     # Title implies benefit not shown
    severity: Severity = Severity.NONE
    evidence_quotes: list[str] = field(default_factory=list)


@dataclass
class OutcomeReporting:
    """Outcome selection and reporting assessment."""
    primary_outcome_type: OutcomeType = OutcomeType.UNCLEAR
    surrogate_without_validation: bool = False  # Surrogate used without established link
    outcome_switching_detected: bool = False     # Differs from registry
    composite_not_disaggregated: bool = False
    registry_id: Optional[str] = None           # NCT number or equivalent
    registered_primary_outcome: Optional[str] = None
    published_primary_outcome: Optional[str] = None
    severity: Severity = Severity.NONE
    evidence_quotes: list[str] = field(default_factory=list)


@dataclass
class ConflictOfInterest:
    """
    COI signals the model should learn to flag.
    Also encodes WHERE to verify (verification_sources).
    """
    funding_type: FundingType = FundingType.NOT_REPORTED
    funding_disclosed_in_abstract: bool = False
    industry_author_affiliations: bool = False   # Authors employed by sponsor
    coi_disclosed: bool = False
    coi_details: Optional[str] = None
    open_payments_flag: bool = False              # Known payments in Open Payments
    payment_amount_usd: Optional[float] = None
    ghost_authorship_indicators: bool = False     # Medical writer acknowledged but not author
    severity: Severity = Severity.NONE

    # Verification sources - the model should learn to suggest these
    verification_sources: list[str] = field(default_factory=lambda: [
        "CMS Open Payments (openpaymentsdata.cms.gov)",
        "ClinicalTrials.gov funding section",
        "EuroPMC Grant Finder",
        "ORCID author profiles",
        "Journal disclosure statements",
    ])


@dataclass
class MethodologicalRedFlags:
    """Design-level issues detectable from the abstract."""
    comparator_type: ComparatorType = ComparatorType.UNCLEAR
    inappropriate_comparator: bool = False       # Placebo when active standard exists
    enrichment_design: bool = False              # Pre-selected responders
    per_protocol_only: bool = False              # No ITT analysis mentioned
    premature_stopping: bool = False             # Stopped early for benefit
    inadequate_blinding: bool = False
    short_follow_up: bool = False                # Relative to condition studied
    severity: Severity = Severity.NONE
    evidence_quotes: list[str] = field(default_factory=list)


@dataclass
class BiasAssessment:
    """
    Complete multi-dimensional bias assessment for a single abstract.
    This is the target label structure for fine-tuning.
    """
    pmid: str = ""
    doi: Optional[str] = None
    title: str = ""
    abstract_text: str = ""

    # The five bias domains
    statistical_reporting: StatisticalReportingBias = field(
        default_factory=StatisticalReportingBias
    )
    spin: SpinAssessment = field(default_factory=SpinAssessment)
    outcome_reporting: OutcomeReporting = field(default_factory=OutcomeReporting)
    conflict_of_interest: ConflictOfInterest = field(default_factory=ConflictOfInterest)
    methodology: MethodologicalRedFlags = field(default_factory=MethodologicalRedFlags)

    # Overall assessment
    overall_severity: Severity = Severity.NONE
    overall_bias_probability: float = 0.0  # 0.0 to 1.0
    reasoning: str = ""  # Free-text reasoning chain (for <think> training)

    # Provenance
    source: str = ""           # How this example entered the dataset
    human_validated: bool = False
    validator_notes: str = ""

    # Verification guidance (what the model should suggest)
    recommended_verification_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "BiasAssessment":
        """Reconstruct from dict, handling nested dataclasses and unknown keys."""
        from dataclasses import fields as dc_fields

        def _filter_fields(dc_cls, d: dict) -> dict:
            known = {f.name for f in dc_fields(dc_cls)}
            return {k: v for k, v in d.items() if k in known}

        data = data.copy()
        nested = {
            "statistical_reporting": StatisticalReportingBias,
            "spin": SpinAssessment,
            "outcome_reporting": OutcomeReporting,
            "conflict_of_interest": ConflictOfInterest,
            "methodology": MethodologicalRedFlags,
        }
        for key, dc_cls in nested.items():
            if key in data and isinstance(data[key], dict):
                data[key] = dc_cls(**_filter_fields(dc_cls, data[key]))

        return cls(**_filter_fields(cls, data))


# ---- Training format templates ----

# Canonical SYSTEM_PROMPT lives in export.py to avoid circular imports.
# Loaded lazily on first use via format_training_example().


def format_training_example(assessment: BiasAssessment) -> dict:
    """
    Format a BiasAssessment as an Alpaca-style training example
    with <think> reasoning chain for Qwen3.5 thinking mode.
    """
    system_prompt = _get_system_prompt()

    user_msg = f"""Assess the following clinical trial abstract for potential bias:

Title: {assessment.title}
PMID: {assessment.pmid}

Abstract:
{assessment.abstract_text}"""

    # Build the thinking chain
    think_chain = f"""<think>
{assessment.reasoning}
</think>

"""

    # Build the structured response
    response_parts = [think_chain]

    # Statistical reporting domain
    sr = assessment.statistical_reporting
    if sr.severity != Severity.NONE:
        response_parts.append(f"## Statistical Reporting Bias: {sr.severity.value.upper()}")
        if sr.relative_only:
            response_parts.append(
                "- Only relative measures reported (RRR/OR/HR) without absolute risk "
                "reduction or NNT. This inflates perceived treatment benefit."
            )
        if sr.evidence_quotes:
            for q in sr.evidence_quotes:
                response_parts.append(f'  Evidence: "{q}"')
        response_parts.append("")

    # Spin domain
    sp = assessment.spin
    if sp.severity != Severity.NONE:
        response_parts.append(f"## Spin Assessment: {sp.spin_level.value.upper()}")
        if not sp.conclusion_matches_results:
            response_parts.append("- Conclusions do not match reported results.")
        if sp.focus_on_secondary_when_primary_ns:
            response_parts.append(
                "- Primary outcome was not significant; conclusions focus on secondary analyses."
            )
        if sp.evidence_quotes:
            for q in sp.evidence_quotes:
                response_parts.append(f'  Evidence: "{q}"')
        response_parts.append("")

    # COI domain
    coi = assessment.conflict_of_interest
    if coi.severity != Severity.NONE:
        response_parts.append(f"## Conflict of Interest: {coi.severity.value.upper()}")
        response_parts.append(f"- Funding: {coi.funding_type.value}")
        if not coi.coi_disclosed:
            response_parts.append("- No conflict of interest disclosure found in abstract.")
        response_parts.append("")

    # Verification steps
    if assessment.recommended_verification_steps:
        response_parts.append("## Recommended Verification Steps")
        for step in assessment.recommended_verification_steps:
            response_parts.append(f"- {step}")
        response_parts.append("")

    # Overall
    response_parts.append(
        f"## Overall Assessment: {assessment.overall_severity.value.upper()} "
        f"(bias probability: {assessment.overall_bias_probability:.0%})"
    )

    assistant_msg = "\n".join(response_parts)

    return {
        "system": system_prompt,
        "instruction": user_msg,
        "output": assistant_msg,
        "pmid": assessment.pmid,
        "source": assessment.source,
        "human_validated": assessment.human_validated,
    }


if __name__ == "__main__":
    # Demo: create a sample assessment
    sample = BiasAssessment(
        pmid="12345678",
        title="Wonderdrug reduces cardiovascular events by 50% in high-risk patients",
        abstract_text="[Abstract text would go here]",
        statistical_reporting=StatisticalReportingBias(
            relative_only=True,
            absolute_reported=False,
            nnt_reported=False,
            baseline_risk_reported=False,
            severity=Severity.HIGH,
            evidence_quotes=["reduced cardiovascular events by 50% (HR 0.50, p<0.001)"],
        ),
        spin=SpinAssessment(
            spin_level=SpinLevel.MODERATE,
            conclusion_matches_results=False,
            severity=Severity.MODERATE,
        ),
        conflict_of_interest=ConflictOfInterest(
            funding_type=FundingType.INDUSTRY,
            industry_author_affiliations=True,
            coi_disclosed=False,
            severity=Severity.HIGH,
        ),
        overall_severity=Severity.HIGH,
        overall_bias_probability=0.85,
        reasoning=(
            "The abstract reports a 50% relative risk reduction without providing "
            "absolute risk reduction, NNT, or baseline event rates. If the baseline "
            "risk is 2%, the ARR is only 1% (NNT=100), which is far less impressive "
            "than the headline 50% figure suggests. The study is industry-funded with "
            "authors affiliated with the sponsor, yet COI is not disclosed in the abstract. "
            "The conclusion claims the drug 'should be considered for all high-risk patients' "
            "which goes beyond what a single trial can establish."
        ),
        recommended_verification_steps=[
            "Check ClinicalTrials.gov for registered primary outcome and baseline risk",
            "Search CMS Open Payments for author-sponsor payment history",
            "Compare abstract conclusions with full-text results section",
            "Check Retraction Watch / Crossref for any post-publication notices",
        ],
        source="synthetic_demo",
    )
    print(json.dumps(format_training_example(sample), indent=2))
