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
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a biomedical research integrity analyst. Given a clinical trial abstract,
assess it for potential bias across five domains:

1. STATISTICAL REPORTING: Does the abstract report only relative measures (RRR, OR, HR)
   without absolute measures (ARR, NNT, baseline risk)? Sole reliance on relative measures
   inflates perceived benefit and is a strong indicator of potential bias.
   - "relative_only" = TRUE only when effect sizes are expressed SOLELY as relative measures
     AND no absolute information appears anywhere in the abstract.
   - "relative_only" = FALSE if raw event counts in both arms, percentages in both arms,
     absolute risk difference, NNT, or baseline/control event rate appear.

2. SPIN: Do the conclusions match the actual results? Classify using the Boutron taxonomy:
   - HIGH: No uncertainty, no recommendation for further trials, no acknowledgment of
     non-significant primary outcome, or recommends clinical use despite weak evidence.
   - MODERATE: Some uncertainty OR further trials recommended, but non-significant
     primary outcome not acknowledged.
   - LOW: Uncertainty AND further trials recommended, OR acknowledges non-significant primary.
   - NONE: Conclusions accurately reflect results.

3. OUTCOME REPORTING: Are outcomes patient-centred or surrogate?
   - Patient-centred: mortality, overall survival, quality of life, functional status,
     symptom burden, major clinical events (MI, stroke), patient-reported outcomes.
   - Surrogate: lab values, imaging markers, biomarkers, process measures, physiological
     parameters, response rates without survival data.
   - Flag composite endpoints that are not disaggregated.
   - Check ClinicalTrials.gov for evidence of outcome switching from the registered protocol.

4. CONFLICT OF INTEREST: Is funding disclosed? Are authors affiliated with the sponsor?
   - Naming a funding source alone (e.g., "Funded by Amgen") does NOT count as COI
     disclosure. COI disclosure requires author-level conflict statements.
   - Industry author affiliations = TRUE if any author is affiliated with a pharmaceutical,
     device, or biotech company.

5. METHODOLOGICAL RED FLAGS: Inappropriate comparator? Enrichment design (run-in
   responders, prior-use requirement)? Per-protocol only without ITT? Premature stopping?
   Short follow-up (chronic disease <12 months, acute <4 weeks)?

VERIFICATION DATABASES — always recommend specific checks based on the study:
- CMS Open Payments (openpaymentsdata.cms.gov): For industry-funded studies, check author
  payment records from the study sponsor.
- ClinicalTrials.gov: Verify registered outcomes, sponsor identity, protocol amendments.
  Always recommend for any RCT.
- ORCID: Check author affiliation histories for undisclosed industry ties.
- Europe PMC (europepmc.org): Access full-text for funding and COI disclosure sections.
- Medicines Australia / EFPIA (betransparent.eu): For non-US physician payment data.
- Retraction Watch / Crossref: Check for post-publication notices, corrections, retractions.

CALIBRATION: Not every industry-funded study is biased. Not every study reporting only
relative measures is intentionally misleading. Assess the totality of evidence.

Provide your reasoning step by step, then a structured assessment with recommended
verification steps citing specific databases and URLs."""


def build_thinking_chain(annotation: dict) -> str:
    """
    Build a <think> reasoning chain from the annotation.

    This teaches the model to reason through bias assessment step by step,
    including which verification databases to check and why.
    """
    parts = ["<think>"]
    reasoning = annotation.get("reasoning", "")
    if reasoning:
        parts.append(reasoning)
    else:
        # Build reasoning from individual domain assessments
        _build_statistical_reasoning(parts, annotation)
        _build_spin_reasoning(parts, annotation)
        _build_outcome_reasoning(parts, annotation)
        _build_coi_reasoning(parts, annotation)
        _build_methodology_reasoning(parts, annotation)

    # Always end with a verification summary
    _build_verification_summary(parts, annotation)

    parts.append("</think>")
    return "\n".join(parts)


def _build_statistical_reasoning(parts: list[str], annotation: dict) -> None:
    """Add statistical reporting reasoning to thinking chain."""
    stat = annotation.get("statistical_reporting", {})
    if stat.get("relative_only"):
        parts.append(
            "The abstract reports effect sizes using only relative measures "
            "without providing absolute risk reduction or NNT. This makes it "
            "impossible to assess the clinical significance of the findings "
            "without knowing the baseline risk."
        )
    elif stat.get("severity", "none") != "none":
        issues = []
        if stat.get("selective_p_values"):
            issues.append("selective reporting of favourable p-values")
        if stat.get("subgroup_emphasis"):
            issues.append("emphasis on subgroup results over primary analysis")
        if not stat.get("baseline_risk_reported", True):
            issues.append("baseline risk not reported")
        if issues:
            parts.append(
                "Statistical reporting concerns: " + "; ".join(issues) + "."
            )
    if stat.get("evidence_quotes"):
        parts.append(f"Key text: {stat['evidence_quotes'][0]}")


def _build_spin_reasoning(parts: list[str], annotation: dict) -> None:
    """Add spin reasoning with Boutron classification to thinking chain."""
    spin = annotation.get("spin", {})
    spin_level = spin.get("spin_level", spin.get("severity", "none"))
    if spin_level in ("moderate", "high"):
        reasons = []
        if not spin.get("conclusion_matches_results", True):
            reasons.append("conclusions do not accurately reflect the reported results")
        if spin.get("focus_on_secondary_when_primary_ns"):
            reasons.append(
                "the primary outcome was not significant but conclusions "
                "focus on secondary or subgroup analyses"
            )
        if spin.get("inappropriate_extrapolation"):
            reasons.append("results are extrapolated beyond the studied population")
        if spin.get("causal_language_from_observational"):
            reasons.append("causal language used for observational data")
        reason_text = "; ".join(reasons) if reasons else "conclusions overstate the findings"
        parts.append(
            f"Spin is classified as {spin_level.upper()} (Boutron taxonomy) because "
            f"{reason_text}."
        )
    elif spin_level == "low":
        parts.append(
            "Spin is LOW — some overstatement present but conclusions include "
            "appropriate uncertainty or acknowledge limitations."
        )


def _build_outcome_reasoning(parts: list[str], annotation: dict) -> None:
    """Add outcome reporting reasoning to thinking chain."""
    outcome = annotation.get("outcome_reporting", {})
    if outcome.get("severity", "none") == "none":
        return
    issues = []
    if outcome.get("surrogate_without_validation"):
        issues.append(
            "the primary outcome is a surrogate endpoint without established "
            "validation linking it to patient-centred outcomes"
        )
    if outcome.get("composite_not_disaggregated"):
        issues.append(
            "a composite endpoint is used but individual component results "
            "are not reported separately"
        )
    if issues:
        parts.append(
            "Outcome reporting concerns: " + "; ".join(issues) + ". "
            "Verify against ClinicalTrials.gov whether the registered primary "
            "outcome matches what was reported in the abstract."
        )


def _build_coi_reasoning(parts: list[str], annotation: dict) -> None:
    """Add conflict of interest reasoning with database selection to thinking chain."""
    coi = annotation.get("conflict_of_interest", {})
    if coi.get("severity", "none") == "none":
        return
    funding_type = coi.get("funding_type", "unclear")
    if funding_type == "industry":
        parts.append(
            "This is an industry-funded study. Check CMS Open Payments "
            "(openpaymentsdata.cms.gov) for author payment records from the "
            "study sponsor. Verify sponsor identity on ClinicalTrials.gov."
        )
    if coi.get("industry_author_affiliations"):
        parts.append(
            "At least one author is affiliated with a pharmaceutical or device "
            "company. Check ORCID for undisclosed industry affiliations."
        )
    if not coi.get("coi_disclosed"):
        parts.append(
            "No author-level conflict of interest disclosure found in the "
            "abstract. Search Europe PMC (europepmc.org) for the full-text "
            "article to review the COI disclosures section."
        )


def _build_methodology_reasoning(parts: list[str], annotation: dict) -> None:
    """Add methodology reasoning to thinking chain."""
    meth = annotation.get("methodology", {})
    if meth.get("severity", "none") == "none":
        return
    issues = []
    if meth.get("per_protocol_only"):
        issues.append(
            "only per-protocol results reported without ITT analysis — "
            "check ClinicalTrials.gov for the pre-registered statistical analysis plan"
        )
    if meth.get("short_follow_up"):
        issues.append(
            "follow-up duration appears insufficient for the primary outcome"
        )
    if meth.get("inappropriate_comparator"):
        issues.append(
            "the comparator may be inappropriate (e.g., placebo when an active "
            "standard of care exists)"
        )
    if meth.get("enrichment_design"):
        issues.append(
            "enrichment design (run-in responders or prior-use requirement) "
            "limits generalisability"
        )
    if meth.get("premature_stopping"):
        issues.append("the trial may have been stopped prematurely")
    if issues:
        parts.append("Methodological concerns: " + "; ".join(issues) + ".")


def _build_verification_summary(parts: list[str], annotation: dict) -> None:
    """Add a verification database summary to the end of the thinking chain."""
    databases = []
    coi = annotation.get("conflict_of_interest", {})

    # ClinicalTrials.gov — always relevant for RCTs
    databases.append("ClinicalTrials.gov (registered outcomes, sponsor, amendments)")

    # CMS Open Payments — for industry-funded studies
    if coi.get("funding_type") == "industry":
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

    # CMS Open Payments — for industry-funded studies
    if (coi.get("funding_type") == "industry"
            and "open payments" not in existing_lower
            and "openpaymentsdata" not in existing_lower):
        steps.append(
            "Check CMS Open Payments (openpaymentsdata.cms.gov) for "
            "author payment records from the study sponsor."
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

    # Statistical reporting
    stat = annotation.get("statistical_reporting", {})
    if stat.get("severity", "none") != "none":
        parts.append(f"## Statistical Reporting: {stat['severity'].upper()}")
        if stat.get("relative_only"):
            parts.append(
                "- Only relative measures reported without absolute risk reduction or NNT"
            )
        if not stat.get("baseline_risk_reported", True):
            parts.append("- Baseline risk / control group event rate not provided")
        if stat.get("evidence_quotes"):
            for q in stat["evidence_quotes"][:2]:
                parts.append(f'  > "{q}"')

    # Spin
    spin = annotation.get("spin", {})
    if spin.get("severity", "none") != "none":
        parts.append(f"\n## Spin: {spin.get('spin_level', spin['severity']).upper()}")
        if not spin.get("conclusion_matches_results", True):
            parts.append("- Conclusions do not accurately reflect the reported results")
        if spin.get("focus_on_secondary_when_primary_ns"):
            parts.append("- Primary outcome not significant; emphasis on secondary/subgroup analyses")
        if spin.get("inappropriate_extrapolation"):
            parts.append("- Results extrapolated beyond the studied population")

    # Outcome reporting
    outcome = annotation.get("outcome_reporting", {})
    if outcome.get("severity", "none") != "none":
        parts.append(f"\n## Outcome Reporting: {outcome['severity'].upper()}")
        if outcome.get("surrogate_without_validation"):
            parts.append("- Surrogate endpoint used without established link to patient outcomes")
        if outcome.get("composite_not_disaggregated"):
            parts.append("- Composite endpoint not disaggregated")

    # COI
    coi = annotation.get("conflict_of_interest", {})
    if coi.get("severity", "none") != "none":
        parts.append(f"\n## Conflict of Interest: {coi['severity'].upper()}")
        parts.append(f"- Funding: {coi.get('funding_type', 'unclear')}")
        if coi.get("industry_author_affiliations"):
            parts.append("- Author(s) affiliated with study sponsor")
        if not coi.get("coi_disclosed"):
            parts.append("- COI not disclosed in abstract")

    # Methodology
    meth = annotation.get("methodology", {})
    if meth.get("severity", "none") != "none":
        parts.append(f"\n## Methodology: {meth['severity'].upper()}")
        if meth.get("inappropriate_comparator"):
            parts.append("- Comparator may be inappropriate (placebo when active standard exists)")
        if meth.get("per_protocol_only"):
            parts.append("- Only per-protocol analysis reported (no ITT)")

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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Shuffle and split
    random.seed(seed)
    random.shuffle(converted)

    n = len(converted)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_data = converted[:n_train]
    val_data = converted[n_train : n_train + n_val]
    test_data = converted[n_train + n_val :]

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

    # Also write metadata
    meta = {
        "format": fmt,
        "include_thinking": include_thinking,
        "total_examples": n,
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Dataset exported: {len(train_data)} train / {len(val_data)} val / "
        f"{len(test_data)} test ({fmt} format)"
    )
