"""
Annotator utilities shared across backends (Anthropic, OpenAI-compatible).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REVIEW_CSV_COLUMNS = [
    "pmid", "title", "overall_severity", "overall_bias_probability",
    "statistical_severity", "relative_only", "spin_level",
    "funding_type", "confidence",
    "reasoning_summary",
    "HUMAN_VALIDATED", "HUMAN_OVERRIDE_SEVERITY", "HUMAN_NOTES",
]


def build_user_message(
    pmid: str,
    title: str,
    abstract: str,
    metadata: Optional[dict] = None,
) -> str:
    """Build the user message from abstract + metadata.

    Shared across all annotator backends to ensure identical prompts and
    comparable outputs.
    """
    user_parts = [
        f"PMID: {pmid}",
        f"Title: {title}",
        f"\nAbstract:\n{abstract}",
    ]

    if metadata:
        if metadata.get("authors"):
            author_str = "; ".join(
                f"{a.get('last', '')}, {a.get('first', '')} "
                f"({', '.join(a.get('affiliations', [])[:2])})"
                for a in metadata["authors"][:5]
            )
            user_parts.append(f"\nAuthors: {author_str}")

        if metadata.get("grants"):
            grant_str = "; ".join(
                f"{g.get('agency', '')} ({g.get('id', '')})"
                for g in metadata["grants"]
            )
            user_parts.append(f"Funding: {grant_str}")

        if metadata.get("journal"):
            user_parts.append(f"Journal: {metadata['journal']}")

        if metadata.get("mesh_terms"):
            user_parts.append(f"MeSH: {', '.join(metadata['mesh_terms'][:10])}")

        if metadata.get("retraction_reasons"):
            user_parts.append(
                f"NOTE: This paper has been RETRACTED. "
                f"Reasons: {', '.join(metadata['retraction_reasons'])}"
            )

        if metadata.get("effect_size_audit"):
            audit = metadata["effect_size_audit"]
            user_parts.append(
                f"\nHeuristic pre-screen: {audit.get('pattern', 'unknown')} "
                f"(score: {audit.get('reporting_bias_score', 0):.2f})"
            )
            if audit.get("flags"):
                user_parts.append(f"Flags: {'; '.join(audit['flags'])}")

    return "\n".join(user_parts)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def save_annotations(annotations: list[dict], output_path: Path) -> None:
    """Save annotations as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    logger.info(f"Saved {len(annotations)} annotations to {output_path}")


def generate_review_csv(annotations: list[dict], output_path: Path) -> None:
    """Generate a CSV for human review in a spreadsheet.

    Includes key fields and empty columns for human validation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REVIEW_CSV_COLUMNS)

        for ann in annotations:
            stat = ann.get("statistical_reporting", {})
            spin = ann.get("spin", {})
            coi = ann.get("conflict_of_interest", {})

            writer.writerow([
                ann.get("pmid", ""),
                ann.get("title", "")[:100],
                ann.get("overall_severity", ""),
                ann.get("overall_bias_probability", ""),
                stat.get("severity", ""),
                stat.get("relative_only", ""),
                spin.get("spin_level", ""),
                coi.get("funding_type", ""),
                ann.get("confidence", ""),
                ann.get("reasoning", "")[:200],
                "",  # HUMAN_VALIDATED
                "",  # HUMAN_OVERRIDE_SEVERITY
                "",  # HUMAN_NOTES
            ])

    logger.info(f"Generated review CSV at {output_path}")
