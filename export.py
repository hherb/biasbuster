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

2. SPIN: Do the conclusions match the actual results? Look for claims of benefit when
   the primary outcome was not statistically significant, inappropriate causal language,
   and emphasis on secondary analyses.

3. OUTCOME REPORTING: Are outcomes patient-centred or surrogate? Is there evidence of
   outcome switching from the trial registry? Are composite endpoints disaggregated?

4. CONFLICT OF INTEREST: Is funding disclosed? Are authors affiliated with the sponsor?
   Suggest verification steps using CMS Open Payments, ClinicalTrials.gov, ORCID,
   Europe PMC, and country-specific databases (Medicines Australia, EFPIA).

5. METHODOLOGICAL RED FLAGS: Inappropriate comparator? Enrichment design?
   Per-protocol only? Premature stopping?

Provide your reasoning step by step, then a structured assessment with recommended
verification steps citing specific databases and URLs."""


def build_thinking_chain(annotation: dict) -> str:
    """
    Build a <think> reasoning chain from the annotation.
    This teaches the model to reason through bias assessment step by step.
    """
    parts = ["<think>"]
    reasoning = annotation.get("reasoning", "")
    if reasoning:
        parts.append(reasoning)
    else:
        # Build reasoning from individual domain assessments
        stat = annotation.get("statistical_reporting", {})
        if stat.get("relative_only"):
            parts.append(
                "The abstract reports effect sizes using only relative measures "
                "without providing absolute risk reduction or NNT. This makes it "
                "impossible to assess the clinical significance of the findings "
                "without knowing the baseline risk."
            )
        if stat.get("evidence_quotes"):
            parts.append(f"Key text: {stat['evidence_quotes'][0]}")

        spin = annotation.get("spin", {})
        if spin.get("spin_level") in ("moderate", "high"):
            parts.append(
                "The conclusions appear to overstate the findings. "
                + ("Primary outcome was not significant but conclusions focus on "
                   "secondary analyses." if spin.get("focus_on_secondary_when_primary_ns") else "")
            )

        coi = annotation.get("conflict_of_interest", {})
        if coi.get("funding_type") == "industry":
            parts.append(
                "This is an industry-funded study. Verification of author payments "
                "via CMS Open Payments is recommended."
            )

    parts.append("</think>")
    return "\n".join(parts)


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

    # Verification steps
    steps = annotation.get("recommended_verification_steps", [])
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
