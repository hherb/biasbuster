"""
Programmatic verification step synthesis.

Generates verification tool recommendations from annotation flags rather than
relying on the LLM to produce them. This frees model capacity for core bias
detection while ensuring verification is always triggered.

Used by both the agent harness (inference) and export.py (training data).
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def synthesize_verification_steps(annotation: dict) -> list[str]:
    """Generate verification steps programmatically from annotation flags.

    Verification recommendations are deterministic logic — they don't require
    LLM judgment. This function is the sole source of verification steps;
    the LLM prompt no longer asks the model to generate them.

    Args:
        annotation: Parsed annotation dict with domain assessments
            (statistical_reporting, spin, outcome_reporting,
            conflict_of_interest, methodology).

    Returns:
        List of verification step strings matching tool_router patterns.

    Rules:
        - ClinicalTrials.gov: always (every RCT should be registry-checked)
        - CMS Open Payments: when any COI concern exists
        - ORCID: when COI concerns or industry involvement detected
        - Europe PMC: always (full-text access for disclosure verification)
        - Retraction Watch: always (post-publication notice check)
    """
    steps: list[str] = []
    coi = annotation.get("conflict_of_interest", {})
    coi_severity = coi.get("severity", "none")

    # ClinicalTrials.gov — always relevant for RCTs
    steps.append(
        "Verify registered primary outcomes and sponsor on ClinicalTrials.gov."
    )

    # CMS Open Payments — for any COI concern
    if (coi.get("funding_type") == "industry"
            or coi.get("industry_author_affiliations")
            or coi_severity != "none"):
        steps.append(
            "Check CMS Open Payments (openpaymentsdata.cms.gov) for "
            "author payment records — authors may have consulting or "
            "speaker relationships with industry even if the study is "
            "not industry-funded."
        )

    # ORCID — for COI concerns or industry involvement
    if (coi.get("industry_author_affiliations")
            or not coi.get("coi_disclosed")
            or coi.get("funding_type") == "industry"):
        steps.append(
            "Check ORCID profiles for author affiliation histories "
            "and undisclosed industry ties."
        )

    # Effect size audit — always relevant, local heuristic (no network)
    steps.append(
        "Run effect size reporting bias audit on the abstract."
    )

    # Europe PMC — always useful for full-text verification
    steps.append(
        "Search Europe PMC (europepmc.org) for full-text funding "
        "and COI disclosures."
    )

    # Retraction Watch — always relevant
    steps.append(
        "Check Retraction Watch database for post-publication notices "
        "or corrections."
    )

    return steps


# Default steps used when model output cannot be parsed as structured JSON.
# These are the "always" tools that apply to any RCT.
_DEFAULT_STEPS: list[str] = [
    "Verify registered primary outcomes and sponsor on ClinicalTrials.gov.",
    "Run effect size reporting bias audit on the abstract.",
    "Search Europe PMC (europepmc.org) for full-text funding and COI disclosures.",
    "Check Retraction Watch database for post-publication notices or corrections.",
]


def get_default_verification_steps() -> list[str]:
    """Return the minimum set of verification steps for any abstract.

    Used as a fallback when the model's initial assessment cannot be parsed
    into structured annotation flags.

    Returns:
        List of verification step strings for the 'always-run' tools.
    """
    return list(_DEFAULT_STEPS)


def plan_verification(
    model_output: str,
    parse_json_fn: Optional[Callable[[str], Optional[dict]]] = None,
) -> tuple[list[str], Optional[dict]]:
    """Plan verification steps from model output.

    Attempts to parse the model's initial assessment as JSON, then
    synthesizes verification steps from the annotation flags. Falls back
    to default steps if parsing fails.

    Args:
        model_output: Raw model output (JSON or markdown).
        parse_json_fn: JSON parser function (default: annotators.parse_llm_json).

    Returns:
        Tuple of (verification_steps, parsed_annotation_or_None).
    """
    if parse_json_fn is None:
        from biasbuster.annotators import parse_llm_json
        parse_json_fn = parse_llm_json

    parsed = parse_json_fn(model_output)

    if parsed and isinstance(parsed, dict):
        steps = synthesize_verification_steps(parsed)
        logger.info(
            "Synthesized %d verification steps from parsed assessment", len(steps),
        )
        return steps, parsed

    logger.warning(
        "Could not parse model output as JSON; using default verification steps",
    )
    return get_default_verification_steps(), None
