"""Top-level assessment orchestration.

Takes a v3 extraction JSON blob, runs each domain's rule evaluation,
computes per-domain severities, computes overall severity via the
max-domain rule, and computes overall bias probability from the
domain-severity distribution.

The assessment output is shaped to match the v3 assessment stage's
JSON schema so it can be stored under the same `annotations` DB
column and compared against existing annotations directly.

Additional key: `_provenance` — a list of every Rule that fired,
with its inputs and result, for audit.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from biasbuster.assessment.rules import (
    DomainSeverity,
    OverallSeverity,
    Rule,
    RuleOutcome,
    _count_endpoints,
    conflict_of_interest_flags,
    conflict_of_interest_severity,
    methodology_flags,
    methodology_severity,
    outcome_reporting_flags,
    outcome_reporting_severity,
    spin_flags,
    spin_severity,
    statistical_reporting_flags,
    statistical_reporting_severity,
)


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------
# Per-severity contribution to overall probability. Tuned against the
# Round 10 reliability runs where Claude produced 0.82 for high and
# 0.35-0.42 for moderate-at-boundary. Adjust here (not in the prompt)
# if future calibration runs show systematic drift.

SEVERITY_RANK = {
    DomainSeverity.NONE: 0,
    DomainSeverity.LOW: 1,
    DomainSeverity.MODERATE: 2,
    DomainSeverity.HIGH: 3,
    DomainSeverity.CRITICAL: 4,
}

PROBABILITY_ANCHORS: dict[OverallSeverity, tuple[float, float]] = {
    OverallSeverity.NONE:     (0.00, 0.10),
    OverallSeverity.LOW:      (0.15, 0.35),
    OverallSeverity.MODERATE: (0.40, 0.65),
    OverallSeverity.HIGH:     (0.70, 0.85),
    OverallSeverity.CRITICAL: (0.90, 1.00),
}


def _domain_to_overall(sev: DomainSeverity) -> OverallSeverity:
    return OverallSeverity(sev.value)


def _compute_overall_severity(
    domain_severities: dict[str, DomainSeverity],
) -> OverallSeverity:
    """max-domain rule — overall is the worst single domain."""
    worst = max(domain_severities.values(), key=lambda s: SEVERITY_RANK[s])
    return _domain_to_overall(worst)


def _compute_overall_probability(
    overall: OverallSeverity,
    domain_severities: dict[str, DomainSeverity],
) -> float:
    """Map the domain distribution to a probability within the anchor range.

    Logic:
      1. Start at the midpoint of the overall-severity anchor range.
      2. Count how many other domains are at MODERATE or higher.
      3. Each additional concerning domain shifts toward the top of the range.
      4. Clamp to the anchor band.

    This reproduces Claude's behaviour on Round 10: a single HIGH domain
    with everything else LOW/NONE yields ~0.68 (bottom of HIGH), while
    multiple HIGH domains together yield 0.80+ (middle-top of HIGH).
    """
    lo, hi = PROBABILITY_ANCHORS[overall]
    band = hi - lo

    # Count concerning domains (MODERATE or higher)
    n_concerns = sum(
        1 for s in domain_severities.values()
        if SEVERITY_RANK[s] >= SEVERITY_RANK[DomainSeverity.MODERATE]
    )

    # Position within the band based on how many concerns accumulate.
    # 1 concern → bottom quarter of band (0.25)
    # 2 concerns → middle (0.50)
    # 3 concerns → three-quarters (0.75)
    # 4-5 concerns → top (1.0)
    if n_concerns <= 1:
        position = 0.15
    elif n_concerns == 2:
        position = 0.45
    elif n_concerns == 3:
        position = 0.70
    else:
        position = 0.95

    return round(lo + band * position, 2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    """Run the full algorithmic assessment on a v3 extraction JSON blob.

    Args:
        extraction: A dict matching the v3 extraction schema (the output
            of Stage 1 in the two-call pipeline, or the 'extraction' key
            inside a stored annotation blob).

    Returns:
        An assessment dict matching the v3 assessment schema plus a
        `_provenance` key containing the full audit trail.
    """
    # --- Run each domain ---
    stat_flags, stat_rules = statistical_reporting_flags(extraction)
    stat_severity, stat_rationale = statistical_reporting_severity(stat_flags)

    spin_flag_dict, spin_rules = spin_flags(extraction)
    spin_sev, spin_rationale = spin_severity(spin_flag_dict)

    outcome_flag_dict, outcome_rules = outcome_reporting_flags(extraction)
    outcome_sev, outcome_rationale = outcome_reporting_severity(outcome_flag_dict)

    coi_flag_dict, coi_rules = conflict_of_interest_flags(extraction)
    coi_sev, coi_rationale = conflict_of_interest_severity(coi_flag_dict, outcome_flag_dict)

    total_endpoints, endpoints_prov = _count_endpoints(extraction)
    meth_flag_dict, meth_rules = methodology_flags(extraction, total_endpoints)
    meth_sev, meth_rationale = methodology_severity(meth_flag_dict, total_endpoints)

    # --- Aggregate ---
    domain_severities = {
        "statistical_reporting": stat_severity,
        "spin": spin_sev,
        "outcome_reporting": outcome_sev,
        "conflict_of_interest": coi_sev,
        "methodology": meth_sev,
    }
    overall_severity = _compute_overall_severity(domain_severities)
    overall_probability = _compute_overall_probability(overall_severity, domain_severities)

    # --- Shape output to match v3 assessment JSON schema ---
    def _clean_flags(d: dict) -> dict:
        """Drop the internal pass-throughs with None values for cleaner output."""
        return {k: v for k, v in d.items()}

    out: dict[str, Any] = {
        "statistical_reporting": {
            **_clean_flags(stat_flags),
            "severity": stat_severity.value,
            "evidence_quotes": [stat_rationale],
        },
        "spin": {
            "spin_level": spin_sev.value,
            **_clean_flags(spin_flag_dict),
            "severity": spin_sev.value,
            "evidence_quotes": [spin_rationale],
        },
        "outcome_reporting": {
            **_clean_flags(outcome_flag_dict),
            "severity": outcome_sev.value,
            "evidence_quotes": [outcome_rationale],
        },
        "conflict_of_interest": {
            **_clean_flags(coi_flag_dict),
            "severity": coi_sev.value,
        },
        "methodology": {
            **_clean_flags(meth_flag_dict),
            "severity": meth_sev.value,
            "evidence_quotes": [meth_rationale],
        },
        "overall_severity": overall_severity.value,
        "overall_bias_probability": overall_probability,
        "recommended_verification_steps": [],
        "reasoning": _build_reasoning(domain_severities, {
            "statistical_reporting": stat_rationale,
            "spin": spin_rationale,
            "outcome_reporting": outcome_rationale,
            "conflict_of_interest": coi_rationale,
            "methodology": meth_rationale,
        }),
        "confidence": _compute_confidence(domain_severities, meth_flag_dict, coi_flag_dict),
        "_annotation_mode": "algorithmic_v4_option_b",
        "_provenance": {
            "total_endpoints": total_endpoints,
            "total_endpoints_source": endpoints_prov,
            "domain_severities": {k: v.value for k, v in domain_severities.items()},
            "overall_severity_rationale": f"max of domain severities = {overall_severity.value}",
            "rules": {
                "statistical_reporting": [asdict(r) for r in stat_rules],
                "spin": [asdict(r) for r in spin_rules],
                "outcome_reporting": [asdict(r) for r in outcome_rules],
                "conflict_of_interest": [asdict(r) for r in coi_rules],
                "methodology": [asdict(r) for r in meth_rules],
            },
            "domain_rationales": {
                "statistical_reporting": stat_rationale,
                "spin": spin_rationale,
                "outcome_reporting": outcome_rationale,
                "conflict_of_interest": coi_rationale,
                "methodology": meth_rationale,
            },
        },
    }
    return out


def _build_reasoning(
    domain_severities: dict[str, DomainSeverity],
    rationales: dict[str, str],
) -> str:
    """Assemble a human-readable reasoning string from the domain rationales."""
    parts = []
    for dom in ["statistical_reporting", "spin", "outcome_reporting",
                "conflict_of_interest", "methodology"]:
        sev = domain_severities[dom].value.upper()
        rationale = rationales[dom]
        parts.append(f"{dom.replace('_', ' ').upper()}: {sev} — {rationale}.")
    overall = max(domain_severities.values(), key=lambda s: SEVERITY_RANK[s])
    parts.append(
        f"OVERALL: {overall.value.upper()} (= max of domain severities). "
        "Algorithmic assessment — see _provenance for full rule trace."
    )
    return " ".join(parts)


def _compute_confidence(
    domain_severities: dict[str, DomainSeverity],
    meth_flags: dict[str, Any],
    coi_flags: dict[str, Any],
) -> str:
    """Confidence heuristic based on how much the extraction populated.

    - HIGH: all methodology-dependent flags are non-None AND COI has
            either funding_type or author affiliations populated
    - LOW: more than half of the methodology flags are None (unknown)
    - MEDIUM: otherwise
    """
    method_keys = ["high_attrition", "differential_attrition", "per_protocol_only",
                   "no_multiplicity_correction", "inadequate_sample_size"]
    unknowns = sum(1 for k in method_keys if meth_flags.get(k) is None)
    coi_populated = bool(coi_flags.get("funding_type")) or bool(coi_flags.get("industry_author_affiliations"))

    if unknowns == 0 and coi_populated:
        return "high"
    if unknowns >= 3:
        return "low"
    return "medium"
