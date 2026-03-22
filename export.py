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
            "(e.g., both arm rates or raw counts available)."
        )
    elif severity == "moderate":
        parts.append(
            f"Severity: MODERATE — {len(issues)} concern(s). Per boundary "
            "definition: reader cannot assess clinical significance without "
            "external data (relative measures only or selective p-values)."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Severity: {severity.upper()} — {len(issues)} concern(s). "
            "Per boundary definition: multiple reporting concerns together "
            "suggest a pattern of selective or misleading presentation."
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
                "MODERATE not HIGH because some uncertainty is expressed or "
                "further trials recommended, but NS primary not acknowledged."
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
            "Severity: LOW — patient-centred primary but secondary surrogate "
            "given undue prominence, or well-validated surrogate used."
        )
    elif severity == "moderate":
        parts.append(
            "Severity: MODERATE — primary is surrogate without patient-centred "
            "validation, or composite not disaggregated."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Severity: {severity.upper()} — surrogate without validation AND "
            "evidence of outcome switching from registered primary."
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

    # Severity justification grounded in indicator count
    if severity == "low":
        parts.append(
            f"Severity: LOW — {len(indicators)} indicator(s). Industry "
            "involvement present but fully disclosed and transparent."
        )
    elif severity == "moderate":
        parts.append(
            f"Severity: MODERATE — {len(indicators)} indicator(s). Industry "
            "funding or affiliations present but COI not fully disclosed. "
            "Transparency gaps warrant verification."
        )
    elif severity in ("high", "critical"):
        parts.append(
            f"Severity: {severity.upper()} — {len(indicators)} indicator(s). "
            "Multiple undisclosed conflicts suggest systematic non-disclosure."
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
            f"Severity: LOW — 1 minor concern that does not invalidate "
            "the primary analysis."
        )
    elif severity == "moderate":
        if len(issues) >= 2:
            parts.append(
                f"Severity: MODERATE — {len(issues)} minor concerns together."
            )
        else:
            parts.append(
                "Severity: MODERATE — 1 significant concern that could "
                "meaningfully affect interpretation."
            )
    elif severity in ("high", "critical"):
        parts.append(
            f"Severity: {severity.upper()} — {len(issues)} significant "
            "concern(s) suggesting primary analysis may not be reliable."
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
    """Add retraction-specific severity floor reasoning if applicable."""
    reasons = _ensure_parsed(annotation.get("retraction_reasons"))
    if not reasons:
        return

    from enrichers.retraction_classifier import classify_retraction
    floor, category = classify_retraction(
        reasons, title=annotation.get("title", ""),
    )

    if floor is not None:
        parts.append(
            f"RETRACTION NOTE: This paper was retracted ({category.replace('_', ' ')}). "
            f"Severity floor: {floor.upper()}. The overall severity must be at least "
            f"{floor.upper()} regardless of abstract content, because the retraction "
            f"indicates {category.replace('_', ' ')} that may not be visible in the text."
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
        response = thinking + "\n\n" + build_structured_response(annotation)
    else:
        response = build_structured_response(annotation)

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
        assistant_msg = thinking + "\n\n" + build_structured_response(annotation)
    else:
        assistant_msg = build_structured_response(annotation)

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
        assistant_msg = thinking + "\n\n" + build_structured_response(annotation)
    else:
        assistant_msg = build_structured_response(annotation)

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
    for ann in annotations:
        reasons = _ensure_parsed(ann.get("retraction_reasons"))
        if reasons:
            floor, category = classify_retraction(
                reasons, title=ann.get("title", ""),
            )
            adjusted = enforce_severity_floor(ann, floor)
            if adjusted is not ann:
                floors_applied += 1
            result.append(adjusted)
        else:
            result.append(ann)

    if floors_applied:
        logger.info(
            f"Retraction severity floors applied to {floors_applied} annotations"
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

    # Write files
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
