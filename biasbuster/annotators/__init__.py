"""
Annotator utilities shared across backends (Anthropic, OpenAI-compatible).
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_parsed(value, default=None):
    """Deserialize a value that may be a JSON string (from SQLite) or already parsed."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


REVIEW_CSV_COLUMNS = [
    "pmid", "title", "overall_severity", "overall_bias_probability",
    "statistical_severity", "relative_only", "spin_level",
    "funding_type", "confidence",
    "reasoning_summary",
    "HUMAN_VALIDATED", "HUMAN_OVERRIDE_SEVERITY", "HUMAN_NOTES",
]


import re as _re

# Patterns that indicate the text is a retraction/withdrawal notice,
# not an original research abstract with assessable content.
_RETRACTION_NOTICE_PATTERNS = [
    _re.compile(r"^retract(ed|ion)\b", _re.IGNORECASE),
    _re.compile(r"^withdraw(n|al)\b", _re.IGNORECASE),
    _re.compile(r"^this article has been retracted", _re.IGNORECASE),
    _re.compile(r"^this paper has been retracted", _re.IGNORECASE),
    _re.compile(r"^retraction notice", _re.IGNORECASE),
    _re.compile(r"^retraction:", _re.IGNORECASE),
    _re.compile(r"^expression of concern", _re.IGNORECASE),
]

# Minimum abstract length (characters) below which a retracted paper is
# almost certainly a bare notice rather than original content.
_MIN_ABSTRACT_LENGTH_FOR_RETRACTED = 200


def is_retraction_notice(
    title: str, abstract: str, metadata: dict | None = None,
) -> bool:
    """Detect whether an item is a bare retraction/withdrawal notice.

    These have no assessable research content and should be excluded from
    annotation. Original papers that were *later* retracted (with their
    full abstract intact) return False and should be annotated normally.

    Args:
        title: Paper title.
        abstract: Abstract text.
        metadata: Optional metadata dict (checked for retraction_reasons).

    Returns:
        True if this looks like a bare retraction notice.
    """
    title_stripped = title.strip()
    abstract_stripped = abstract.strip()

    # Title-only signals (e.g. "Retraction: ..." or "WITHDRAWN: ...")
    for pat in _RETRACTION_NOTICE_PATTERNS:
        if pat.search(title_stripped):
            # Title says retraction — is the abstract just the notice too?
            if len(abstract_stripped) < _MIN_ABSTRACT_LENGTH_FOR_RETRACTED:
                return True
            # Long abstract with retraction title: check if abstract itself
            # is also just a notice (vs. original content still present).
            for apat in _RETRACTION_NOTICE_PATTERNS:
                if apat.search(abstract_stripped):
                    return True
            # Long abstract, doesn't start with retraction language —
            # likely original content, keep it.
            return False

    # No abstract at all
    if not abstract_stripped:
        return True

    # Very short abstract starting with retraction language
    if len(abstract_stripped) < _MIN_ABSTRACT_LENGTH_FOR_RETRACTED:
        for pat in _RETRACTION_NOTICE_PATTERNS:
            if pat.search(abstract_stripped):
                return True

    return False


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
        authors = _ensure_parsed(metadata.get("authors"))
        if authors:
            author_str = "; ".join(
                f"{a.get('last', '')}, {a.get('first', '')} "
                f"({', '.join(a.get('affiliations', [])[:2])})"
                for a in authors[:5]
            )
            user_parts.append(f"\nAuthors: {author_str}")

        grants = _ensure_parsed(metadata.get("grants"))
        if grants:
            grant_str = "; ".join(
                f"{g.get('agency', '')} ({g.get('id', '')})"
                for g in grants
            )
            user_parts.append(f"Funding: {grant_str}")

        if metadata.get("journal"):
            user_parts.append(f"Journal: {metadata['journal']}")

        mesh = _ensure_parsed(metadata.get("mesh_terms"))
        if mesh:
            user_parts.append(f"MeSH: {', '.join(mesh[:10])}")

        reasons = _ensure_parsed(metadata.get("retraction_reasons"))
        if reasons:
            # Classify the retraction to determine detectability
            from biasbuster.enrichers.retraction_classifier import (
                classify_retraction,
                format_retraction_context,
            )
            floor, category, detectable = classify_retraction(
                reasons,
                title=title,
                abstract="",  # Don't pass the original abstract as notice text
            )

            if not detectable:
                # Abstract-undetectable retraction (fabrication, fraud, etc.)
                # Do NOT tell the LLM about the retraction — assess abstract
                # on its own merits for training purposes.  We intentionally
                # skip format_retraction_context() here so no retraction info
                # leaks into the prompt at all.
                pass
            else:
                # Abstract-detectable retraction — include context and floor
                user_parts.append(
                    f"NOTE: This paper has been RETRACTED. "
                    f"Reasons: {', '.join(reasons)}"
                )
                retraction_context = format_retraction_context(
                    floor, category, detectable,
                )
                if retraction_context:
                    user_parts.append(retraction_context)

        # Cochrane Risk of Bias 2 expert judgments (when available)
        overall_rob = metadata.get("overall_rob")
        if overall_rob:
            rob_parts = [f"overall={overall_rob}"]
            for rob_field in ("randomization_bias", "deviation_bias",
                              "missing_outcome_bias", "measurement_bias",
                              "reporting_bias"):
                val = metadata.get(rob_field)
                if val:
                    rob_parts.append(f"{rob_field.replace('_bias', '')}={val}")
            user_parts.append(
                f"\nCochrane RoB 2 expert assessment: {', '.join(rob_parts)}. "
                "Use these expert judgments to calibrate your severity ratings."
            )

        audit = _ensure_parsed(metadata.get("effect_size_audit"), default={})
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


# Fields that must be present in a complete annotation.
# If any are missing, the response was likely truncated.
REQUIRED_ANNOTATION_FIELDS = {
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
    "overall_severity",
    "overall_bias_probability",
    "recommended_verification_steps",
    "confidence",
}


def validate_annotation(annotation: dict, pmid: str = "") -> list[str]:
    """Check that a parsed annotation has all required fields.

    Returns a list of missing field names (empty if valid).
    """
    missing = []
    for field in REQUIRED_ANNOTATION_FIELDS:
        if field not in annotation:
            missing.append(field)
        elif field in (
            "statistical_reporting", "spin", "outcome_reporting",
            "conflict_of_interest", "methodology",
        ):
            # Domain fields must be dicts with a severity key
            val = annotation[field]
            if not isinstance(val, dict) or "severity" not in val:
                missing.append(f"{field}.severity")
    if missing:
        logger.warning(
            f"PMID {pmid}: incomplete annotation, missing: {', '.join(missing)}"
        )
    return missing


def parse_llm_json(text: str, pmid: str = "") -> dict | None:
    """Parse JSON from LLM output with repair and logging.

    Shared across all annotator backends. Tries direct parse first,
    then attempts repair if that fails. Validates that all required
    annotation fields are present (rejects truncated responses).

    Returns parsed dict or None if unrecoverable or incomplete.
    """
    text = strip_markdown_fences(text)

    if not text.strip():
        logger.warning(f"PMID {pmid}: model returned empty response")
        return None

    # Try direct parse first
    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt repair if direct parse failed
    if result is None:
        repaired = repair_json(text)
        try:
            result = json.loads(repaired)
            logger.info(f"JSON repaired successfully for PMID {pmid}")
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON repair failed for PMID {pmid}: {e}\n"
                f"  Raw response (first 300 chars): {text[:300]}"
            )
            return None

    # Validate completeness — reject truncated responses so they get retried
    if not isinstance(result, dict):
        logger.warning(f"PMID {pmid}: parsed result is not a dict")
        return None

    missing = validate_annotation(result, pmid=pmid)
    if missing:
        logger.warning(
            f"PMID {pmid}: rejecting truncated response "
            f"(missing {len(missing)} fields)"
        )
        return None

    return result
