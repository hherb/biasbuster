"""
Export Module

Converts validated bias assessments to fine-tuning data formats:
- Alpaca (instruction/input/output) with <think> chains for Qwen3.5
- ShareGPT (multi-turn conversation format)
- JSONL chat format (OpenAI-compatible)
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical prompt imported from the single source of truth.
# See docs/MISTAKES_ROUND_1_AND_FIXES.md for why prompt unification matters.
from annotators import _ensure_parsed
from prompts import TRAINING_SYSTEM_PROMPT as SYSTEM_PROMPT  # noqa: E402


def build_thinking_chain(annotation: dict) -> str:
    """Build a <think> reasoning chain from the annotation.

    Evidence-grounded: references specific flags, quotes, and concern counts
    to justify severity assignments per the canonical boundary definitions.
    See docs/MISTAKES_ROUND_1_AND_FIXES.md (Phase 4) for the rationale.
    """
    parts = ["<think>"]
    reasoning = annotation.get("reasoning", "")
    if reasoning:
        parts.append(reasoning)

    # Domain-level reasoning with evidence-grounded severity justifications
    _build_statistical_reasoning(parts, annotation)
    _build_spin_reasoning(parts, annotation)
    _build_outcome_reasoning(parts, annotation)
    _build_coi_reasoning(parts, annotation)
    _build_methodology_reasoning(parts, annotation)

    # Cross-domain calibration step
    _build_calibration_summary(parts, annotation)

    # Retraction floor reasoning (if applicable)
    _build_retraction_reasoning(parts, annotation)

    # Verification database summary
    _build_verification_summary(parts, annotation)

    parts.append("</think>")
    return "\n".join(parts)


def _quote_evidence(parts: list[str], evidence_quotes: list) -> None:
    """Append the first evidence quote if available."""
    if evidence_quotes:
        parts.append(f'Evidence from abstract: "{evidence_quotes[0]}"')


def _build_statistical_reasoning(parts: list[str], annotation: dict) -> None:
    """Add statistical reporting reasoning to thinking chain."""
    stat = annotation.get("statistical_reporting", {})
    severity = stat.get("severity", "none")

    if severity == "none":
        positives = []
        if stat.get("absolute_reported"):
            positives.append("absolute measures are reported")
        if stat.get("baseline_risk_reported"):
            positives.append("baseline risk is provided")
        if stat.get("nnt_reported"):
            positives.append("NNT is reported")
        if positives:
            parts.append(
                "Statistical reporting appears adequate: "
                + "; ".join(positives) + ". Severity: NONE."
            )
        else:
            parts.append(
                "No statistical reporting concerns identified. Severity: NONE."
            )
        return

    # Collect specific issues with their names for counting
    issues = []
    if stat.get("relative_only"):
        issues.append("relative_only (no absolute measures)")
    if stat.get("selective_p_values"):
        issues.append("selective p-value reporting")
    if stat.get("subgroup_emphasis"):
        issues.append("subgroup emphasis over primary analysis")
    if not stat.get("baseline_risk_reported", True):
        issues.append("baseline risk not reported")

    if issues:
        parts.append(
            f"Statistical reporting: {len(issues)} concern(s) identified: "
            + "; ".join(issues) + "."
        )

    _quote_evidence(parts, stat.get("evidence_quotes", []))

    # Severity justification grounded in concern count + boundary definitions
    if severity == "low":
        parts.append(
            f"Severity: LOW — {len(issues)} minor concern(s). Per boundary "
            "definition: reader can still assess clinical significance "
            "(e.g., both arm rates or raw counts available). "
            "Not MODERATE because the reader does not need external data "
            "to interpret the findings."
        )
    elif severity == "moderate":
        parts.append(
            f"Severity: MODERATE — {len(issues)} concern(s). Per boundary "
            "definition: reader cannot assess clinical significance without "
            "external data (relative measures only or selective p-values). "
            "Not HIGH because no evidence of multiple concerns suggesting "
            "intentional obfuscation."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Severity: {severity.upper()} — {len(issues)} concern(s). "
            "Per boundary definition: multiple reporting concerns together "
            "suggest a pattern of selective or misleading presentation. "
            "Exceeds MODERATE because more than one significant concern "
            "is present."
        )


def _build_spin_reasoning(parts: list[str], annotation: dict) -> None:
    """Add spin reasoning with Boutron classification to thinking chain."""
    spin = annotation.get("spin", {})
    spin_level = spin.get("spin_level", spin.get("severity", "none"))

    reasons = []
    if not spin.get("conclusion_matches_results", True):
        reasons.append("conclusions do not match reported results")
    if spin.get("focus_on_secondary_when_primary_ns"):
        reasons.append("focus on secondary outcomes despite NS primary")
    if spin.get("inappropriate_extrapolation"):
        reasons.append("inappropriate extrapolation beyond studied population")
    if spin.get("causal_language_from_observational"):
        reasons.append("causal language from observational data")
    if spin.get("title_spin"):
        reasons.append("spin present in the title")

    if spin_level in ("moderate", "high"):
        reason_text = "; ".join(reasons) if reasons else "conclusions overstate findings"
        parts.append(
            f"Spin: {spin_level.upper()} (Boutron taxonomy). "
            f"Indicators: {reason_text}."
        )
        if spin_level == "moderate":
            parts.append(
                "Per boundary: MODERATE — some uncertainty OR further trials "
                "recommended, but NS primary outcome not acknowledged. "
                "Not HIGH because the authors do not recommend clinical use "
                "despite weak evidence."
            )
        elif spin_level == "high":
            parts.append(
                "Per boundary: HIGH — no uncertainty expressed, no "
                "recommendation for further trials, no acknowledgment of "
                "NS primary outcome, or clinical use recommended despite "
                "weak evidence. Exceeds MODERATE because none of the "
                "mitigating factors (uncertainty, further trials) are present."
            )
    elif spin_level == "low":
        parts.append(
            "Spin: LOW (Boutron) — some overstatement but conclusions include "
            "uncertainty or acknowledge limitations. LOW not MODERATE because "
            "authors acknowledge NS primary or express uncertainty AND recommend "
            "further trials."
        )
    else:
        parts.append(
            "Spin: NONE (Boutron) — conclusions accurately reflect results."
        )

    _quote_evidence(parts, spin.get("evidence_quotes", []))


def _build_outcome_reasoning(parts: list[str], annotation: dict) -> None:
    """Add outcome reporting reasoning to thinking chain."""
    outcome = annotation.get("outcome_reporting", {})
    severity = outcome.get("severity", "none")
    outcome_type = outcome.get("primary_outcome_type", "unclear")

    if severity == "none":
        parts.append(
            f"Outcome reporting: primary outcome is {outcome_type}. "
            "No surrogate or composite concerns. Severity: NONE."
        )
        return

    issues = []
    if outcome.get("surrogate_without_validation"):
        issues.append("surrogate endpoint without established patient-centred validation")
    if outcome.get("composite_not_disaggregated"):
        issues.append("composite endpoint not disaggregated")

    if issues:
        parts.append(
            f"Outcome reporting: primary is {outcome_type}. "
            f"Concerns: {'; '.join(issues)}. "
            "Verify against ClinicalTrials.gov for outcome switching."
        )

    _quote_evidence(parts, outcome.get("evidence_quotes", []))

    if severity == "low":
        parts.append(
            "Per boundary: LOW — patient-centred primary but secondary surrogate "
            "given undue prominence, or well-validated surrogate used. "
            "Not MODERATE because the primary outcome is patient-centred or "
            "the surrogate is well-validated."
        )
    elif severity == "moderate":
        parts.append(
            "Per boundary: MODERATE — primary is surrogate without patient-centred "
            "validation, or composite not disaggregated. "
            "Not HIGH because no evidence of outcome switching from a "
            "registered patient-centred endpoint."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Per boundary: {severity.upper()} — surrogate without validation AND "
            "evidence of outcome switching from registered primary. "
            "Exceeds MODERATE because outcome switching compounds the "
            "surrogate concern."
        )


def _build_coi_reasoning(parts: list[str], annotation: dict) -> None:
    """Add conflict of interest reasoning to thinking chain."""
    coi = annotation.get("conflict_of_interest", {})
    severity = coi.get("severity", "none")
    funding_type = coi.get("funding_type", "unclear")

    if severity == "none":
        if funding_type == "public":
            parts.append(
                "COI: publicly funded, no industry conflicts identified. "
                "Severity: NONE."
            )
        else:
            parts.append(
                f"COI: funding type is {funding_type}. No significant "
                "conflicts identified. Severity: NONE."
            )
        return

    # Count specific COI indicators
    indicators = []
    if funding_type == "industry":
        indicators.append(f"industry-funded ({funding_type})")
    if coi.get("industry_author_affiliations"):
        indicators.append("author(s) affiliated with pharma/device company")
    if not coi.get("coi_disclosed"):
        indicators.append("no author-level COI disclosure in abstract")
    if not coi.get("funding_disclosed_in_abstract", True):
        indicators.append("funding not disclosed in abstract")

    parts.append(
        f"COI: {len(indicators)} indicator(s): {'; '.join(indicators)}."
    )

    # Verification actions based on specific indicators
    if funding_type == "industry" or coi.get("industry_author_affiliations"):
        parts.append(
            "Check CMS Open Payments for author payment records. "
            "Verify sponsor on ClinicalTrials.gov."
        )
    if coi.get("industry_author_affiliations"):
        parts.append("Check ORCID for undisclosed industry affiliations.")
    if not coi.get("coi_disclosed"):
        parts.append(
            "Search Europe PMC for full-text COI disclosure section."
        )

    # Severity justification grounded in indicator count + boundary definitions
    if severity == "low":
        parts.append(
            f"Per boundary: LOW — {len(indicators)} indicator(s). Industry "
            "involvement present but fully disclosed and transparent. "
            "Not MODERATE because COI is disclosed and transparency gaps "
            "do not warrant verification."
        )
    elif severity == "moderate":
        parts.append(
            f"Per boundary: MODERATE — {len(indicators)} indicator(s). Industry "
            "funding or affiliations present but COI not fully disclosed. "
            "Transparency gaps warrant verification. "
            "Not HIGH because undisclosed COI is not combined with "
            "author-sponsor affiliations suggesting systematic non-disclosure."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Per boundary: {severity.upper()} — {len(indicators)} indicator(s). "
            "Industry funding with undisclosed COI AND author affiliations "
            "with sponsor — multiple undisclosed conflicts suggest "
            "systematic non-disclosure. Exceeds MODERATE because COI "
            "non-disclosure is combined with author-sponsor affiliations."
        )


def _build_methodology_reasoning(parts: list[str], annotation: dict) -> None:
    """Add methodology reasoning to thinking chain."""
    meth = annotation.get("methodology", {})
    severity = meth.get("severity", "none")

    if severity == "none":
        parts.append(
            "Methodology: no red flags. Design appears appropriate. "
            "Severity: NONE."
        )
        return

    issues = []
    if meth.get("per_protocol_only"):
        issues.append("per-protocol only (no ITT)")
    if meth.get("short_follow_up"):
        issues.append("insufficient follow-up duration")
    if meth.get("inappropriate_comparator"):
        issues.append("potentially inappropriate comparator")
    if meth.get("enrichment_design"):
        issues.append("enrichment design limiting generalisability")
    if meth.get("premature_stopping"):
        issues.append("possible premature stopping")

    parts.append(
        f"Methodology: {len(issues)} concern(s): {'; '.join(issues)}."
    )

    _quote_evidence(parts, meth.get("evidence_quotes", []))

    if severity == "low":
        parts.append(
            "Per boundary: LOW — a single minor concern (e.g., slightly "
            "short follow-up) that does not invalidate the primary analysis. "
            "Not MODERATE because no significant concern that could "
            "meaningfully affect interpretation."
        )
    elif severity == "moderate":
        if len(issues) >= 2:
            parts.append(
                f"Per boundary: MODERATE — {len(issues)} minor concerns "
                "together. Not HIGH because no single concern likely "
                "invalidates the primary analysis."
            )
        else:
            parts.append(
                "Per boundary: MODERATE — one significant concern that "
                "could meaningfully affect interpretation. Not HIGH because "
                "only one significant concern, not multiple."
            )
    elif severity in ("high", "critical"):
        parts.append(
            f"Per boundary: {severity.upper()} — {len(issues)} significant "
            "concern(s) that likely invalidate the primary analysis. "
            "Exceeds MODERATE because multiple significant concerns are "
            "present, or a single concern invalidates the primary analysis."
        )


def _build_calibration_summary(parts: list[str], annotation: dict) -> None:
    """Add cross-domain calibration reasoning."""
    domain_keys = [
        ("statistical_reporting", "Statistical reporting"),
        ("spin", "Spin"),
        ("outcome_reporting", "Outcome reporting"),
        ("conflict_of_interest", "COI"),
        ("methodology", "Methodology"),
    ]

    severities = {}
    for key, label in domain_keys:
        domain = annotation.get(key, {})
        sev = domain.get("severity", "none")
        severities[label] = sev

    non_none = {k: v for k, v in severities.items() if v != "none"}
    overall = annotation.get("overall_severity", "none")

    if not non_none:
        parts.append(
            "Cross-domain calibration: 0/5 domains have concerns. "
            f"Overall severity: {overall.upper()}."
        )
    else:
        highest = max(non_none.values(), key=lambda s: _SEV_ORDER.get(s, 0))
        domain_list = ", ".join(f"{k}={v.upper()}" for k, v in non_none.items())
        parts.append(
            f"Cross-domain calibration: {len(non_none)}/5 domains have concerns "
            f"({domain_list}). Highest domain severity: {highest.upper()}. "
            f"Overall severity reflects the highest domain concern, "
            f"tempered by breadth: {overall.upper()}."
        )


def _build_retraction_reasoning(parts: list[str], annotation: dict) -> None:
    """Add retraction-specific reasoning to the thinking chain."""
    reasons = _ensure_parsed(annotation.get("retraction_reasons"))
    if not reasons:
        return

    from enrichers.retraction_classifier import classify_retraction
    floor, category, detectable = classify_retraction(
        reasons, title=annotation.get("title", ""),
    )

    if not detectable:
        # Abstract-undetectable: note that severity reflects text only
        parts.append(
            f"RETRACTION NOTE: This paper was retracted ({category.replace('_', ' ')}). "
            f"This retraction reason is not detectable from the abstract text. "
            f"The severity rating reflects only what is visible in the abstract."
        )
    elif floor is not None:
        # Abstract-detectable with floor: enforce floor in reasoning
        parts.append(
            f"RETRACTION NOTE: This paper was retracted ({category.replace('_', ' ')}). "
            f"Severity floor: {floor.upper()}. The overall severity must be at least "
            f"{floor.upper()} because this type of issue may be visible in the text."
        )
    else:
        parts.append(
            f"RETRACTION NOTE: This paper was retracted ({category.replace('_', ' ')}). "
            f"This is not a bias-relevant retraction — no severity floor applied."
        )


# Severity ordering for calibration summary
_SEV_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}


def _build_verification_summary(parts: list[str], annotation: dict) -> None:
    """Add a verification database summary to the end of the thinking chain."""
    databases = []
    coi = annotation.get("conflict_of_interest", {})

    # ClinicalTrials.gov — always relevant for RCTs
    databases.append("ClinicalTrials.gov (registered outcomes, sponsor, amendments)")

    # CMS Open Payments — for any COI concern
    coi_severity = coi.get("severity", "none")
    if (coi.get("funding_type") == "industry"
            or coi.get("industry_author_affiliations")
            or coi_severity not in ("none",)):
        databases.append(
            "CMS Open Payments (openpaymentsdata.cms.gov) for author payment records"
        )

    # ORCID — for COI concerns
    if (coi.get("industry_author_affiliations")
            or not coi.get("coi_disclosed")
            or coi.get("funding_type") == "industry"):
        databases.append("ORCID for author affiliation histories")

    # Europe PMC — for full-text access
    databases.append("Europe PMC (europepmc.org) for full-text and funder metadata")

    # Retraction Watch
    databases.append("Retraction Watch for post-publication notices")

    parts.append(
        "Databases to check: " + "; ".join(databases) + "."
    )


def _synthesize_verification_steps(annotation: dict) -> list[str]:
    """
    Generate verification steps from annotation flags.

    Ensures every training example has comprehensive verification database
    citations, filling gaps where the original annotation lacks them.
    """
    steps = list(annotation.get("recommended_verification_steps", []))
    existing_lower = " ".join(steps).lower()

    coi = annotation.get("conflict_of_interest", {})

    # CMS Open Payments — for any COI concern, not just industry-funded studies.
    # Authors may have personal consulting/speaker relationships even when the
    # study itself is publicly funded.
    coi_severity = coi.get("severity", "none")
    if ("open payments" not in existing_lower
            and "openpaymentsdata" not in existing_lower
            and (coi.get("funding_type") == "industry"
                 or coi.get("industry_author_affiliations")
                 or coi_severity not in ("none",))):
        steps.append(
            "Check CMS Open Payments (openpaymentsdata.cms.gov) for "
            "author payment records — authors may have consulting or "
            "speaker relationships with industry even if the study is "
            "not industry-funded."
        )

    # ClinicalTrials.gov — always relevant
    if ("clinicaltrials.gov" not in existing_lower
            and "trial registry" not in existing_lower):
        steps.append(
            "Verify registered primary outcomes and sponsor on ClinicalTrials.gov."
        )

    # ORCID — for COI concerns
    if ("orcid" not in existing_lower
            and (coi.get("industry_author_affiliations")
                 or not coi.get("coi_disclosed")
                 or coi.get("funding_type") == "industry")):
        steps.append(
            "Check ORCID profiles for author affiliation histories "
            "and undisclosed industry ties."
        )

    # Europe PMC — for full-text access
    if ("europe pmc" not in existing_lower
            and "europepmc" not in existing_lower):
        steps.append(
            "Search Europe PMC (europepmc.org) for full-text funding "
            "and COI disclosures."
        )

    # Retraction Watch — always relevant
    if ("retraction" not in existing_lower):
        steps.append(
            "Check Retraction Watch database for post-publication notices "
            "or corrections."
        )

    return steps


def build_structured_response(annotation: dict) -> str:
    """Build the structured response portion (after thinking)."""
    parts = []

    # Statistical reporting — always present
    stat = annotation.get("statistical_reporting", {})
    stat_sev = stat.get("severity", "none")
    parts.append(f"## Statistical Reporting: {stat_sev.upper()}")
    if stat_sev != "none":
        if stat.get("relative_only"):
            parts.append(
                "- Only relative measures reported without absolute risk reduction or NNT"
            )
        if not stat.get("baseline_risk_reported", True):
            parts.append("- Baseline risk / control group event rate not provided")
        if stat.get("evidence_quotes"):
            for q in stat["evidence_quotes"][:2]:
                parts.append(f'  > "{q}"')
    else:
        parts.append("- No significant statistical reporting concerns identified")

    # Spin — always present
    spin = annotation.get("spin", {})
    spin_sev = spin.get("severity", "none")
    spin_label = spin.get("spin_level", spin_sev)
    parts.append(f"\n## Spin: {spin_label.upper()}")
    if spin_sev != "none":
        if not spin.get("conclusion_matches_results", True):
            parts.append("- Conclusions do not accurately reflect the reported results")
        if spin.get("focus_on_secondary_when_primary_ns"):
            parts.append("- Primary outcome not significant; emphasis on secondary/subgroup analyses")
        if spin.get("inappropriate_extrapolation"):
            parts.append("- Results extrapolated beyond the studied population")
    else:
        parts.append("- Conclusions appear to accurately reflect the reported results")

    # Outcome reporting — always present
    outcome = annotation.get("outcome_reporting", {})
    outcome_sev = outcome.get("severity", "none")
    parts.append(f"\n## Outcome Reporting: {outcome_sev.upper()}")
    if outcome_sev != "none":
        if outcome.get("surrogate_without_validation"):
            parts.append("- Surrogate endpoint used without established link to patient outcomes")
        if outcome.get("composite_not_disaggregated"):
            parts.append("- Composite endpoint not disaggregated")
    else:
        parts.append("- Primary outcomes appear patient-centred with no reporting concerns")

    # COI — always present
    coi = annotation.get("conflict_of_interest", {})
    coi_sev = coi.get("severity", "none")
    parts.append(f"\n## Conflict of Interest: {coi_sev.upper()}")
    if coi_sev != "none":
        parts.append(f"- Funding: {coi.get('funding_type', 'unclear')}")
        if coi.get("industry_author_affiliations"):
            parts.append("- Author(s) affiliated with study sponsor")
        if not coi.get("coi_disclosed"):
            parts.append("- COI not disclosed in abstract")
    else:
        parts.append("- Funding and COI disclosures appear adequate")

    # Methodology — always present
    meth = annotation.get("methodology", {})
    meth_sev = meth.get("severity", "none")
    parts.append(f"\n## Methodology: {meth_sev.upper()}")
    if meth_sev != "none":
        if meth.get("inappropriate_comparator"):
            parts.append("- Comparator may be inappropriate (placebo when active standard exists)")
        if meth.get("per_protocol_only"):
            parts.append("- Only per-protocol analysis reported (no ITT)")
    else:
        parts.append("- No significant methodological red flags identified")

    # Verification steps (synthesize missing database citations)
    steps = _synthesize_verification_steps(annotation)
    if steps:
        parts.append("\n## Recommended Verification Steps")
        for step in steps:
            parts.append(f"- {step}")

    # Overall
    parts.append(
        f"\n## Overall: {annotation.get('overall_severity', 'unknown').upper()} "
        f"(bias probability: {annotation.get('overall_bias_probability', 0):.0%})"
    )

    return "\n".join(parts)


def build_json_response(annotation: dict) -> str:
    """Build JSON-structured response from annotation dict.

    Outputs the schema defined in prompts._JSON_SCHEMA, matching what the
    evaluation scorer expects via _parse_from_json().  Used for V2+ training
    data so the model learns to produce JSON instead of markdown.
    """
    stat = annotation.get("statistical_reporting", {})
    spin = annotation.get("spin", {})
    outcome = annotation.get("outcome_reporting", {})
    coi = annotation.get("conflict_of_interest", {})
    meth = annotation.get("methodology", {})

    steps = _synthesize_verification_steps(annotation)

    result = {
        "statistical_reporting": {
            "severity": stat.get("severity", "none"),
            "relative_only": bool(stat.get("relative_only", False)),
            "absolute_reported": bool(stat.get("absolute_reported", False)),
            "nnt_reported": bool(stat.get("nnt_reported", False)),
            "baseline_risk_reported": bool(stat.get("baseline_risk_reported", False)),
            "selective_p_values": bool(stat.get("selective_p_values", False)),
            "subgroup_emphasis": bool(stat.get("subgroup_emphasis", False)),
            "evidence_quotes": list(stat.get("evidence_quotes", [])),
        },
        "spin": {
            "severity": spin.get("severity", "none"),
            "spin_level": spin.get("spin_level", spin.get("severity", "none")),
            "conclusion_matches_results": bool(
                spin.get("conclusion_matches_results", True)
            ),
            "causal_language_from_observational": bool(
                spin.get("causal_language_from_observational", False)
            ),
            "focus_on_secondary_when_primary_ns": bool(
                spin.get("focus_on_secondary_when_primary_ns", False)
            ),
            "inappropriate_extrapolation": bool(
                spin.get("inappropriate_extrapolation", False)
            ),
            "title_spin": bool(spin.get("title_spin", False)),
            "evidence_quotes": list(spin.get("evidence_quotes", [])),
        },
        "outcome_reporting": {
            "severity": outcome.get("severity", "none"),
            "primary_outcome_type": outcome.get(
                "primary_outcome_type", "unclear"
            ),
            "surrogate_without_validation": bool(
                outcome.get("surrogate_without_validation", False)
            ),
            "composite_not_disaggregated": bool(
                outcome.get("composite_not_disaggregated", False)
            ),
            "evidence_quotes": list(outcome.get("evidence_quotes", [])),
        },
        "conflict_of_interest": {
            "severity": coi.get("severity", "none"),
            "funding_type": coi.get("funding_type", "not_reported"),
            "funding_disclosed_in_abstract": bool(
                coi.get("funding_disclosed_in_abstract",
                         coi.get("funding_disclosed", False))
            ),
            "industry_author_affiliations": bool(
                coi.get("industry_author_affiliations", False)
            ),
            "coi_disclosed": bool(coi.get("coi_disclosed", False)),
        },
        "methodology": {
            "severity": meth.get("severity", "none"),
            "inappropriate_comparator": bool(
                meth.get("inappropriate_comparator", False)
            ),
            "enrichment_design": bool(meth.get("enrichment_design", False)),
            "per_protocol_only": bool(meth.get("per_protocol_only", False)),
            "premature_stopping": bool(meth.get("premature_stopping", False)),
            "short_follow_up": bool(meth.get("short_follow_up", False)),
            "evidence_quotes": list(meth.get("evidence_quotes", [])),
        },
        "overall_severity": annotation.get("overall_severity", "none"),
        "overall_bias_probability": float(
            annotation.get("overall_bias_probability", 0.0)
        ),
        "reasoning": annotation.get("reasoning", ""),
        "recommended_verification_steps": steps,
        "confidence": annotation.get("confidence", "medium"),
    }

    return json.dumps(result, indent=2)


def to_alpaca_format(
    annotation: dict,
    include_thinking: bool = True,
) -> dict:
    """
    Convert annotation to Alpaca instruction-tuning format.
    Compatible with Unsloth/TRL SFTTrainer.
    """
    instruction = (
        f"Assess the following clinical trial abstract for potential bias:\n\n"
        f"Title: {annotation.get('title', '')}\n"
        f"PMID: {annotation.get('pmid', '')}\n\n"
        f"Abstract:\n{annotation.get('abstract', annotation.get('abstract_text', ''))}"
    )

    if include_thinking:
        thinking = build_thinking_chain(annotation)
        response = thinking + "\n\n" + build_json_response(annotation)
    else:
        response = build_json_response(annotation)

    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "input": "",
        "output": response,
    }


def to_sharegpt_format(annotation: dict, include_thinking: bool = True) -> dict:
    """Convert to ShareGPT multi-turn format."""
    user_msg = (
        f"Assess the following clinical trial abstract for potential bias:\n\n"
        f"Title: {annotation.get('title', '')}\n"
        f"PMID: {annotation.get('pmid', '')}\n\n"
        f"Abstract:\n{annotation.get('abstract', annotation.get('abstract_text', ''))}"
    )

    if include_thinking:
        thinking = build_thinking_chain(annotation)
        assistant_msg = thinking + "\n\n" + build_json_response(annotation)
    else:
        assistant_msg = build_json_response(annotation)

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": assistant_msg},
        ]
    }


def to_openai_chat_format(annotation: dict, include_thinking: bool = True) -> dict:
    """Convert to OpenAI chat format (also works with many frameworks)."""
    user_msg = (
        f"Assess the following clinical trial abstract for potential bias:\n\n"
        f"Title: {annotation.get('title', '')}\n"
        f"PMID: {annotation.get('pmid', '')}\n\n"
        f"Abstract:\n{annotation.get('abstract', annotation.get('abstract_text', ''))}"
    )

    if include_thinking:
        thinking = build_thinking_chain(annotation)
        assistant_msg = thinking + "\n\n" + build_json_response(annotation)
    else:
        assistant_msg = build_json_response(annotation)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


_SEVERITY_RE = re.compile(
    r"## Overall:.*?(NONE|LOW|MODERATE|HIGH|CRITICAL)", re.IGNORECASE,
)


def _extract_overall_severity(example: dict) -> str:
    """Extract overall severity from a converted training example's output."""
    output = example.get("output", "")
    if not output:
        # ShareGPT or OpenAI chat format
        for msg in example.get("conversations", example.get("messages", [])):
            if msg.get("from") == "gpt" or msg.get("role") == "assistant":
                output = msg.get("value", msg.get("content", ""))
                break

    # Try JSON extraction first (V2+ format: JSON after </think>)
    json_text = output
    think_idx = output.find("</think>")
    if think_idx >= 0:
        json_text = output[think_idx + 8:].strip()
    try:
        data = json.loads(json_text)
        sev = data.get("overall_severity", "").lower()
        if sev in ("none", "low", "moderate", "high", "critical"):
            return sev
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Fall back to markdown regex (V1 format)
    match = _SEVERITY_RE.search(output)
    return match.group(1).lower() if match else "unknown"


def oversample_rare_severities(
    examples: list[dict],
    min_fraction: float = 0.05,
) -> list[dict]:
    """Duplicate rare severity examples so every class has at least ``min_fraction``.

    Only apply to training data — never to val/test.

    Args:
        examples: Converted training examples (alpaca, sharegpt, or openai_chat).
        min_fraction: Minimum fraction of total for each severity class.

    Returns:
        Augmented list with rare classes oversampled.
    """
    from collections import Counter

    # Count current distribution
    severity_map: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        sev = _extract_overall_severity(ex)
        severity_map.setdefault(sev, []).append(i)

    counts = {sev: len(idxs) for sev, idxs in severity_map.items()}
    total = len(examples)
    min_count = max(1, int(total * min_fraction))

    augmented = list(examples)
    for sev, idxs in severity_map.items():
        if sev == "unknown":
            continue
        current = len(idxs)
        if current < min_count:
            # Duplicate to reach min_count
            needed = min_count - current
            dupes = (idxs * ((needed // current) + 2))[:needed]
            for idx in dupes:
                augmented.append(examples[idx])
            logger.info(
                f"Oversampled {sev.upper()}: {current} -> {current + needed} "
                f"(+{needed} duplicates)"
            )

    # Log final distribution
    final_counts = Counter(_extract_overall_severity(ex) for ex in augmented)
    logger.info(
        f"Training severity distribution after oversampling: "
        f"{dict(sorted(final_counts.items()))}"
    )
    return augmented


def _apply_retraction_floors(annotations: list[dict]) -> list[dict]:
    """Apply retraction severity floors to annotations (non-mutating).

    For retracted papers, classifies the retraction reason and bumps
    overall_severity up to the floor if it's too low.  Non-retracted
    papers pass through unchanged.
    """
    from enrichers.retraction_classifier import (
        classify_retraction,
        enforce_severity_floor,
    )

    result = []
    floors_applied = 0
    floors_skipped = 0
    for ann in annotations:
        reasons = _ensure_parsed(ann.get("retraction_reasons"))
        if reasons:
            floor, category, detectable = classify_retraction(
                reasons, title=ann.get("title", ""),
            )
            if not detectable:
                # Abstract-undetectable retraction: do NOT enforce floor.
                # Let the annotation's own severity stand — it reflects
                # what the text actually shows.
                floors_skipped += 1
                result.append(ann)
            else:
                adjusted = enforce_severity_floor(ann, floor)
                if adjusted is not ann:
                    floors_applied += 1
                result.append(adjusted)
        else:
            result.append(ann)

    if floors_applied:
        logger.info(
            "Retraction severity floors applied to %d annotations", floors_applied,
        )
    if floors_skipped:
        logger.info(
            "Retraction floors skipped for %d abstract-undetectable papers",
            floors_skipped,
        )
    return result


def _stratified_split(
    examples: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into train/val/test with stratification by severity.

    Ensures each severity class is represented proportionally in all splits,
    rather than relying on random shuffle which can skew small classes.
    """
    from collections import defaultdict

    # Group by severity
    by_severity: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        sev = _extract_overall_severity(ex)
        by_severity[sev].append(ex)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for sev, items in sorted(by_severity.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_frac)) if n >= 3 else n
        n_val = max(1, int(n * val_frac)) if n - n_train >= 2 else 0
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle within each split so severity classes aren't grouped
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def _validate_json_outputs(path: Path, include_thinking: bool) -> int:
    """Validate that every exported example has valid JSON output.

    For thinking-enabled exports, checks that valid JSON follows </think>.
    Returns the number of validation errors found.
    """
    errors = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            example = json.loads(line)
            # Extract the assistant/model output text
            output = example.get("output", "")
            if not output:
                for msg in example.get("conversations",
                                       example.get("messages", [])):
                    if msg.get("from") == "gpt" or msg.get("role") == "assistant":
                        output = msg.get("value", msg.get("content", ""))
                        break

            json_text = output
            if include_thinking:
                idx = output.find("</think>")
                if idx < 0:
                    logger.warning(f"Validation: line {i} in {path.name}: missing </think>")
                    errors += 1
                    continue
                json_text = output[idx + 8:].strip()

            try:
                data = json.loads(json_text)
                # Check required top-level fields
                if "overall_severity" not in data:
                    logger.warning(
                        f"Validation: line {i} in {path.name}: "
                        "missing overall_severity in JSON"
                    )
                    errors += 1
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Validation: line {i} in {path.name}: invalid JSON: {e}"
                )
                errors += 1
    return errors


def export_dataset(
    annotations: list[dict],
    output_dir: Path,
    fmt: str = "alpaca",  # alpaca, sharegpt, openai_chat
    include_thinking: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Export annotations to training data files with train/val/test splits.

    Changes from Round 1 (see docs/MISTAKES_ROUND_1_AND_FIXES.md):
    - Retraction severity floors applied before conversion
    - No oversampling — natural distribution preserved
    - Stratified split by severity class
    - Severity distribution stats in metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply retraction severity floors before conversion
    annotations = _apply_retraction_floors(annotations)

    # Convert format
    converter = {
        "alpaca": to_alpaca_format,
        "sharegpt": to_sharegpt_format,
        "openai_chat": to_openai_chat_format,
    }[fmt]

    converted = []
    for ann in annotations:
        try:
            converted.append(converter(ann, include_thinking=include_thinking))
        except Exception as e:
            logger.warning(f"Failed to convert PMID {ann.get('pmid', '?')}: {e}")

    # Stratified split (no oversampling — natural distribution preserved)
    train_data, val_data, test_data = _stratified_split(
        converted, train_split, val_split, seed,
    )

    # Log severity distribution
    from collections import Counter
    train_dist = Counter(_extract_overall_severity(ex) for ex in train_data)
    logger.info(
        f"Training severity distribution (natural): "
        f"{dict(sorted(train_dist.items()))}"
    )

    # Write files and validate JSON output
    total_validation_errors = 0
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Exported {len(split_data)} examples to {path}")
        errs = _validate_json_outputs(path, include_thinking)
        total_validation_errors += errs

    if total_validation_errors > 0:
        logger.warning(
            f"JSON VALIDATION: {total_validation_errors} error(s) found — "
            "inspect warnings above before training"
        )
    else:
        logger.info("JSON validation: all outputs parse successfully")

    # Write metadata with severity distribution stats
    all_dist = Counter(_extract_overall_severity(ex) for ex in converted)
    meta = {
        "format": fmt,
        "include_thinking": include_thinking,
        "total_examples": len(converted),
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "seed": seed,
        "severity_distribution": dict(sorted(all_dist.items())),
        "train_severity_distribution": dict(sorted(train_dist.items())),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Dataset exported: {len(train_data)} train / {len(val_data)} val / "
        f"{len(test_data)} test ({fmt} format)"
    )
