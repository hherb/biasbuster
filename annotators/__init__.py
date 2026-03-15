"""
Annotator utilities shared across backends (Anthropic, OpenAI-compatible).
"""

import json
import logging
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
            authors = metadata["authors"]
            if isinstance(authors, str):
                try:
                    authors = json.loads(authors)
                except (json.JSONDecodeError, TypeError):
                    authors = []
            if authors:
                author_str = "; ".join(
                    f"{a.get('last', '')}, {a.get('first', '')} "
                    f"({', '.join(a.get('affiliations', [])[:2])})"
                    for a in authors[:5]
                )
                user_parts.append(f"\nAuthors: {author_str}")

        if metadata.get("grants"):
            grants = metadata["grants"]
            if isinstance(grants, str):
                try:
                    grants = json.loads(grants)
                except (json.JSONDecodeError, TypeError):
                    grants = []
            if grants:
                grant_str = "; ".join(
                    f"{g.get('agency', '')} ({g.get('id', '')})"
                    for g in grants
                )
                user_parts.append(f"Funding: {grant_str}")

        if metadata.get("journal"):
            user_parts.append(f"Journal: {metadata['journal']}")

        if metadata.get("mesh_terms"):
            mesh = metadata["mesh_terms"]
            if isinstance(mesh, str):
                try:
                    mesh = json.loads(mesh)
                except (json.JSONDecodeError, TypeError):
                    mesh = []
            if mesh:
                user_parts.append(f"MeSH: {', '.join(mesh[:10])}")

        if metadata.get("retraction_reasons"):
            reasons = metadata["retraction_reasons"]
            if isinstance(reasons, str):
                try:
                    reasons = json.loads(reasons)
                except (json.JSONDecodeError, TypeError):
                    reasons = []
            if reasons:
                user_parts.append(
                    f"NOTE: This paper has been RETRACTED. "
                    f"Reasons: {', '.join(reasons)}"
                )

        if metadata.get("effect_size_audit"):
            audit = metadata["effect_size_audit"]
            if isinstance(audit, str):
                try:
                    audit = json.loads(audit)
                except (json.JSONDecodeError, TypeError):
                    audit = {}
            if audit:
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


def repair_json(text: str) -> str:
    """Attempt to repair common JSON malformations from LLM output.

    Handles:
    - Missing commas between object fields or array elements
    - Trailing commas before closing braces/brackets
    - Unescaped newlines inside string values
    - Truncated output (unclosed braces/brackets)

    Returns the repaired text. Callers should still try json.loads()
    and handle failure — this is best-effort.
    """
    import re

    # Strip any remaining markdown/preamble before the first {
    first_brace = text.find("{")
    if first_brace > 0:
        text = text[first_brace:]

    # Remove trailing text after the last }
    last_brace = text.rfind("}")
    if last_brace >= 0 and last_brace < len(text) - 1:
        text = text[: last_brace + 1]

    # Fix missing commas between fields:
    # Pattern: value (end of line) followed by a quoted key on the next line
    # e.g.  "field": true\n  "next_field": ...
    text = re.sub(
        r'(true|false|null|\d+\.?\d*|"[^"]*"|\]|\})\s*\n(\s*")',
        r"\1,\n\2",
        text,
    )

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Close unclosed braces/brackets (truncated output)
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0 or open_brackets > 0:
        # Try to close any open string first
        # Count unescaped quotes
        in_string = False
        for ch in text:
            if ch == '"' and (not text or text[text.index(ch) - 1] != "\\"):
                in_string = not in_string
        if in_string:
            text += '"'
        text += "]" * max(0, open_brackets)
        text += "}" * max(0, open_braces)

    return text


def parse_llm_json(text: str, pmid: str = "") -> dict | None:
    """Parse JSON from LLM output with repair and logging.

    Shared across all annotator backends. Tries direct parse first,
    then attempts repair if that fails.

    Returns parsed dict or None if unrecoverable.
    """
    text = strip_markdown_fences(text)

    if not text.strip():
        logger.warning(f"PMID {pmid}: model returned empty response")
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt repair
    repaired = repair_json(text)
    try:
        result = json.loads(repaired)
        logger.info(f"JSON repaired successfully for PMID {pmid}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON repair failed for PMID {pmid}: {e}\n"
            f"  Raw response (first 300 chars): {text[:300]}"
        )
        return None
