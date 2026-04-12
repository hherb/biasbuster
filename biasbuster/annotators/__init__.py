"""
Annotator utilities shared across backends (Anthropic, OpenAI-compatible).

Supports both single-call (v1) and two-call (v3) annotation modes:
- Single-call: one LLM call does extraction + assessment together
- Two-call: Stage 1 extracts facts, Stage 2 assesses bias from those facts
"""

import abc
import asyncio
import json
import logging
from typing import Optional

from tqdm import tqdm

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


# ============================================================================
# Two-call (v3) support: extraction parsing and assessment message building
# ============================================================================

REQUIRED_EXTRACTION_FIELDS = {
    "paper_metadata",
    "sample",
    "analysis",
    "outcomes",
    "subgroups",
    "conflicts",
    "methodology_details",
    "conclusions",
}


def validate_extraction(extraction: dict, pmid: str = "") -> list[str]:
    """Check that a parsed extraction has all required top-level fields.

    Returns a list of missing field names (empty if valid).
    """
    missing = []
    for field in REQUIRED_EXTRACTION_FIELDS:
        if field not in extraction:
            missing.append(field)
        elif not isinstance(extraction[field], dict):
            actual_type = type(extraction[field]).__name__
            missing.append(f"{field} (expected dict, got {actual_type})")
    if missing:
        logger.warning(
            f"PMID {pmid}: incomplete extraction, missing: {', '.join(missing)}"
        )
    return missing


def parse_extraction_json(text: str, pmid: str = "") -> dict | None:
    """Parse Stage 1 extraction JSON from LLM output.

    Same repair logic as parse_llm_json but validates against
    REQUIRED_EXTRACTION_FIELDS instead of annotation fields.

    Returns parsed dict or None if unrecoverable or incomplete.
    """
    text = strip_markdown_fences(text)

    if not text.strip():
        logger.warning(f"PMID {pmid}: extraction returned empty response")
        return None

    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        pass

    if result is None:
        repaired = repair_json(text)
        try:
            result = json.loads(repaired)
            logger.info(f"Extraction JSON repaired successfully for PMID {pmid}")
        except json.JSONDecodeError as e:
            logger.warning(
                f"Extraction JSON repair failed for PMID {pmid}: {e}\n"
                f"  Raw response (first 300 chars): {text[:300]}"
            )
            return None

    if not isinstance(result, dict):
        logger.warning(f"PMID {pmid}: extraction result is not a dict")
        return None

    missing = validate_extraction(result, pmid=pmid)
    if missing:
        logger.warning(
            f"PMID {pmid}: rejecting incomplete extraction "
            f"(missing {len(missing)} fields)"
        )
        return None

    return result


def build_assessment_user_message(
    extraction: dict,
    metadata: Optional[dict] = None,
) -> str:
    """Build the Stage 2 user message from Stage 1 extraction JSON.

    The assessment model receives the structured extraction as JSON,
    plus any retraction context and enrichment hints that affect
    severity judgment.

    Args:
        extraction: Parsed Stage 1 extraction dict.
        metadata: Original paper metadata (for retraction context,
                  Cochrane RoB, effect size audit hints).

    Returns:
        Formatted string for the Stage 2 LLM call.
    """
    parts = [
        "EXTRACTED FACTS (from Stage 1):",
        json.dumps(extraction, indent=2),
    ]

    if metadata:
        # Retraction context affects severity floors in assessment
        reasons = _ensure_parsed(metadata.get("retraction_reasons"))
        if reasons:
            from biasbuster.enrichers.retraction_classifier import (
                classify_retraction,
                format_retraction_context,
            )
            floor, category, detectable = classify_retraction(
                reasons, title="", abstract="",
            )
            if detectable:
                parts.append(
                    f"\nRETRACTION CONTEXT: This paper has been RETRACTED. "
                    f"Reasons: {', '.join(reasons)}"
                )
                retraction_context = format_retraction_context(
                    floor, category, detectable,
                )
                if retraction_context:
                    parts.append(retraction_context)

        # Cochrane RoB 2 expert judgments for calibration
        overall_rob = metadata.get("overall_rob")
        if overall_rob:
            rob_parts = [f"overall={overall_rob}"]
            for rob_field in ("randomization_bias", "deviation_bias",
                              "missing_outcome_bias", "measurement_bias",
                              "reporting_bias"):
                val = metadata.get(rob_field)
                if val:
                    rob_parts.append(f"{rob_field.replace('_bias', '')}={val}")
            parts.append(
                f"\nCochrane RoB 2 expert assessment: {', '.join(rob_parts)}. "
                "Use these expert judgments to calibrate your severity ratings."
            )

        # Heuristic pre-screen hints
        audit = _ensure_parsed(metadata.get("effect_size_audit"), default={})
        if audit:
            parts.append(
                f"\nHeuristic pre-screen: {audit.get('pattern', 'unknown')} "
                f"(score: {audit.get('reporting_bias_score', 0):.2f})"
            )
            if audit.get("flags"):
                parts.append(f"Flags: {'; '.join(audit['flags'])}")

    return "\n".join(parts)


# ============================================================================
# Extraction merge — combine partial extractions from map-reduce chunks
# ============================================================================
# See docs/two_step_approach/MERGE_STRATEGY.md for rationale and limitations.
# ============================================================================

# Section-name → priority rank (higher = more authoritative for its fields).
# Matching is substring, case-insensitive; first match wins.
_SECTION_PRIORITY = {
    "abstract": 1,
    "introduction": 1,
    "background": 1,
    "method": 5,            # "Methods", "Materials and Methods"
    "result": 5,            # "Results"
    "finding": 5,           # "Findings"
    "discussion": 3,
    "conclusion": 3,
    "limitation": 3,
    "funding": 6,           # COI / Funding back-matter — most authoritative
    "coi": 6,
    "conflict": 6,          # "Conflict of interest"
    "disclosure": 6,
    "acknowledg": 4,        # "Acknowledgments" (covers US + UK spelling)
}


def _section_priority(section_name: str | None) -> int:
    """Return the authority rank of a section name (higher = more authoritative).

    Returns 2 (default, ahead of Abstract/Intro) if no keyword matches, so
    unlabeled chunks don't get zero priority.
    """
    if not section_name:
        return 2
    lowered = section_name.lower()
    for keyword, rank in _SECTION_PRIORITY.items():
        if keyword in lowered:
            return rank
    return 2


# Presence booleans — use OR logic during merge.
_PRESENCE_BOOLEANS_BY_PATH = {
    ("sample", "attrition_stated"),
    ("subgroups", "subgroup_analyses_present"),
    ("conflicts", "coi_statement_present"),
    ("conflicts", "funding_disclosed_in_abstract"),
    ("methodology_details", "run_in_or_enrichment"),
    ("methodology_details", "early_stopping"),
    ("conclusions", "clinical_language_in_conclusions"),
    ("conclusions", "further_research_recommended"),
    ("conclusions", "uncertainty_language_present"),
}

# List-valued fields — union with dedup by a stable key.
# Value is the dedup key within each list item; None means "compare whole dict".
_LIST_FIELDS_BY_PATH = {
    ("sample", "attrition_quotes"): None,
    ("outcomes", "primary_outcomes_stated"): "name",
    ("outcomes", "secondary_outcomes_stated"): "name",
    ("outcomes", "effect_size_quotes"): None,
    ("subgroups", "subgroups"): "name",
    ("conflicts", "authors_with_industry_affiliation"): "name",
    ("conclusions", "conclusion_quotes"): None,
    ("conclusions", "limitations_acknowledged"): None,
}


def _is_null_like(value) -> bool:
    """True if the value is null / empty / zero-length."""
    if value is None:
        return True
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return True
    return False


def _merge_list_field(
    merged: list, incoming: list, dedup_key: str | None,
) -> list:
    """Union two lists with dedup by key (or whole-item if key is None)."""
    if not incoming:
        return merged
    if dedup_key is None:
        seen = {json.dumps(item, sort_keys=True) for item in merged
                if isinstance(item, (dict, list))}
        seen_scalars = {item for item in merged
                        if not isinstance(item, (dict, list))}
        for item in incoming:
            if isinstance(item, (dict, list)):
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            else:
                if item not in seen_scalars:
                    seen_scalars.add(item)
                    merged.append(item)
        return merged

    seen_keys = {
        item.get(dedup_key) for item in merged
        if isinstance(item, dict) and item.get(dedup_key)
    }
    for item in incoming:
        if isinstance(item, dict):
            key = item.get(dedup_key)
            if key and key not in seen_keys:
                seen_keys.add(key)
                merged.append(item)
            elif not key:
                merged.append(item)  # no key → always include
        else:
            merged.append(item)
    return merged


def _merge_singleton(
    merged_value,
    merged_priority: int,
    incoming_value,
    incoming_priority: int,
    path: tuple[str, ...],
    conflicts: list[dict],
) -> tuple[object, int]:
    """Merge a singleton (scalar or nested dict) with priority-based conflict detection.

    Returns (chosen_value, chosen_priority). Records a conflict when two
    non-null values from equally-authoritative sections disagree.
    """
    if _is_null_like(incoming_value):
        return merged_value, merged_priority
    if _is_null_like(merged_value):
        return incoming_value, incoming_priority

    # Both non-null. Compare.
    if merged_value == incoming_value:
        return merged_value, max(merged_priority, incoming_priority)

    if incoming_priority > merged_priority:
        # Higher authority wins; log conflict
        conflicts.append({
            "path": ".".join(path),
            "kept": incoming_value,
            "dropped": merged_value,
            "reason": "higher-priority section overrode lower-priority",
        })
        return incoming_value, incoming_priority

    if incoming_priority < merged_priority:
        conflicts.append({
            "path": ".".join(path),
            "kept": merged_value,
            "dropped": incoming_value,
            "reason": "lower-priority section dropped",
        })
        return merged_value, merged_priority

    # Equal priority disagreement — preserve both, keep first
    conflicts.append({
        "path": ".".join(path),
        "kept": merged_value,
        "dropped": incoming_value,
        "reason": "equal-priority sections disagree — kept first-seen",
    })
    return merged_value, merged_priority


def _merge_dict_recursive(
    merged: dict,
    priorities: dict,
    incoming: dict,
    incoming_priority: int,
    conflicts: list[dict],
    path: tuple[str, ...] = (),
) -> None:
    """Recursively merge ``incoming`` into ``merged`` using path-specific rules.

    ``priorities`` tracks the per-path authority rank of the current merged
    value so later chunks can override it when they come from a more
    authoritative section.
    """
    for key, incoming_val in incoming.items():
        field_path = path + (key,)
        existing_val = merged.get(key)

        # Rule 1: presence booleans — OR
        if field_path in _PRESENCE_BOOLEANS_BY_PATH:
            merged[key] = bool(existing_val) or bool(incoming_val)
            continue

        # Rule 2: list fields — union with dedup
        if field_path in _LIST_FIELDS_BY_PATH:
            if not isinstance(existing_val, list):
                existing_val = []
            if isinstance(incoming_val, list):
                merged[key] = _merge_list_field(
                    list(existing_val),
                    incoming_val,
                    _LIST_FIELDS_BY_PATH[field_path],
                )
            else:
                merged[key] = existing_val
            continue

        # Rule 3: nested dicts — recurse
        if isinstance(incoming_val, dict) and isinstance(existing_val, dict):
            _merge_dict_recursive(
                existing_val, priorities, incoming_val,
                incoming_priority, conflicts, field_path,
            )
            continue
        if isinstance(incoming_val, dict) and existing_val is None:
            merged[key] = {}
            _merge_dict_recursive(
                merged[key], priorities, incoming_val,
                incoming_priority, conflicts, field_path,
            )
            continue

        # Rule 4: singleton — priority-based
        existing_priority = priorities.get(field_path, -1)
        new_val, new_prio = _merge_singleton(
            existing_val, existing_priority,
            incoming_val, incoming_priority,
            field_path, conflicts,
        )
        merged[key] = new_val
        priorities[field_path] = new_prio


def merge_section_extractions(
    partials: list[tuple[str, dict]],
) -> dict:
    """Merge per-section partial extractions into a single extraction dict.

    Args:
        partials: list of (section_name, partial_extraction_dict) tuples,
                  one per successfully extracted chunk.

    Returns:
        Merged extraction dict with the same schema as whole-paper extraction,
        plus a ``_merge_conflicts`` key listing any detected conflicts.

    See docs/two_step_approach/MERGE_STRATEGY.md for the merge rules and
    the list of fields that cannot be synthesised from per-section views
    (the coherence-pass limitation).
    """
    if not partials:
        return {}

    merged: dict = {}
    priorities: dict[tuple[str, ...], int] = {}
    conflicts: list[dict] = []

    for section_name, partial in partials:
        if not isinstance(partial, dict):
            logger.warning(
                f"merge_section_extractions: skipping non-dict partial "
                f"from section '{section_name}'"
            )
            continue
        prio = _section_priority(section_name)
        _merge_dict_recursive(
            merged, priorities, partial, prio, conflicts,
        )

    if conflicts:
        merged["_merge_conflicts"] = conflicts
    return merged


# ============================================================================
# Base annotator — shared annotation logic for all LLM backends
# ============================================================================

class BaseAnnotator(abc.ABC):
    """Abstract base for annotator backends.

    Subclasses must implement ``_call_llm`` (the transport layer) and the
    async-context-manager lifecycle (``__aenter__``/``__aexit__``).
    Everything else — single-call, two-call, and batch orchestration — is
    shared.
    """

    model: str
    max_retries: int

    @abc.abstractmethod
    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        pmid: str = "",
    ) -> Optional[str]:
        """Send a system+user message pair and return raw text.

        Must handle retries, rate limiting, and transient errors internally.
        Returns the raw text response or None on unrecoverable failure.
        """

    @abc.abstractmethod
    async def __aenter__(self) -> "BaseAnnotator":
        ...

    @abc.abstractmethod
    async def __aexit__(self, *args) -> None:
        ...

    # ------------------------------------------------------------------
    # Single-call annotation (v1)
    # ------------------------------------------------------------------

    async def annotate_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Single-call annotation (v1): extraction + assessment in one LLM call.

        Returns parsed JSON assessment or None on failure.
        """
        # Deferred import avoids circular dependency at module load time
        from biasbuster.prompts import ANNOTATION_SYSTEM_PROMPT

        user_message = build_user_message(pmid, title, abstract, metadata)

        for attempt in range(self.max_retries):
            text = await self._call_llm(
                ANNOTATION_SYSTEM_PROMPT, user_message, pmid=pmid,
            )
            if text is None:
                return None

            assessment = parse_llm_json(text, pmid=pmid)
            if assessment is not None:
                assessment["pmid"] = pmid
                assessment["title"] = title
                assessment["_annotation_model"] = self.model
                return assessment

            logger.warning(
                f"PMID {pmid}: JSON parse failed "
                f"(attempt {attempt + 1}/{self.max_retries}), retrying"
            )
            await asyncio.sleep(2 ** attempt)

        logger.error(
            f"All {self.max_retries} parse attempts failed for PMID {pmid}"
        )
        return None

    # ------------------------------------------------------------------
    # Full-text two-call annotation (v3, map-reduce)
    # ------------------------------------------------------------------

    async def _extract_full_text_sections(
        self,
        pmid: str,
        title: str,
        sections: list[tuple[str, str]],
    ) -> Optional[tuple[dict, list[dict], list[dict], int]]:
        """Run Stage 1 extraction (map+reduce) on a sectioned full-text paper.

        Shared by ``annotate_full_text_two_call`` (v3) and
        ``annotate_full_text_agentic`` (v4). The Stage 2 assessment
        differs between the two but the extraction phase is identical:
        per-section sequential extraction → merge → return the merged
        extraction plus per-section debugging metadata.

        Args:
            pmid: Paper identifier (used for logging).
            title: Paper title (passed in the per-section user message).
            sections: List of (section_name, section_text) tuples.

        Returns:
            Tuple of:
              - merged extraction dict (with _merge_conflicts removed)
              - per-section partial extractions for debugging
              - merge_conflicts list (may be empty)
              - failed_count of sections that did not extract cleanly
            None if every section failed and we can't proceed.
        """
        from biasbuster.prompts_v3 import SECTION_EXTRACTION_SYSTEM_PROMPT

        if not sections:
            logger.error(f"PMID {pmid}: no sections to annotate")
            return None

        # Stage 1 (map): extract per-section *sequentially*.
        # We previously ran sections in parallel via asyncio.gather, but
        # observed non-deterministic content loss on long pipelines (e.g.
        # 120b dropping a chunk's sample/attrition fields one run, then
        # extracting them perfectly when re-run in isolation). Sequential
        # execution matches what local Ollama actually does (single GPU,
        # serialised queue) and removes any chance of in-flight contention
        # while costing only wall-clock time.
        async def extract_section(
            section_name: str, section_text: str,
        ) -> Optional[tuple[str, dict]]:
            user_msg = (
                f"Paper: {title}\n"
                f"Section: {section_name}\n\n"
                f"{section_text}"
            )
            for attempt in range(self.max_retries):
                raw = await self._call_llm(
                    SECTION_EXTRACTION_SYSTEM_PROMPT, user_msg, pmid=pmid,
                )
                if raw is None:
                    return None
                partial = parse_extraction_json(raw, pmid=pmid)
                if partial is not None:
                    return (section_name, partial)
                logger.warning(
                    f"PMID {pmid}/{section_name}: section extraction parse "
                    f"failed (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(2 ** attempt)
            logger.error(
                f"PMID {pmid}/{section_name}: all extraction attempts failed"
            )
            return None

        partials: list[tuple[str, dict]] = []
        for idx, (name, text) in enumerate(sections):
            logger.info(
                f"PMID {pmid}: extracting section {idx + 1}/{len(sections)} "
                f"({name!r})"
            )
            result = await extract_section(name, text)
            if result is not None:
                partials.append(result)

        if not partials:
            logger.error(
                f"PMID {pmid}: every section extraction failed, aborting"
            )
            return None

        failed_count = len(sections) - len(partials)
        if failed_count:
            logger.warning(
                f"PMID {pmid}: {failed_count}/{len(sections)} sections "
                f"failed extraction; assessing from the remaining "
                f"{len(partials)} sections"
            )

        # Preserve the raw per-section partials for retrospective debugging.
        # When the merged extraction looks suspect, this lets you inspect
        # exactly which chunk produced which value without re-running the
        # model. Stored as a list of {section, extraction} dicts so it
        # serialises to JSON cleanly when saved to the database.
        section_extractions = [
            {"section": section_name, "extraction": partial_data}
            for section_name, partial_data in partials
        ]

        # Stage 1 (reduce): merge partials
        extraction = merge_section_extractions(partials)
        merge_conflicts = extraction.pop("_merge_conflicts", [])
        if merge_conflicts:
            logger.info(
                f"PMID {pmid}: {len(merge_conflicts)} merge conflicts "
                f"(preserved in extraction metadata for assessment)"
            )

        return extraction, section_extractions, merge_conflicts, failed_count

    async def annotate_full_text_two_call(
        self,
        pmid: str,
        title: str,
        sections: list[tuple[str, str]],
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Map-reduce two-call annotation for full-text papers.

        Stage 1 (map): extract facts from each section independently using
            the section-level extraction prompt. **Sections are processed
            sequentially**, not in parallel — local LLM backends like
            Ollama serialise requests anyway, and parallel calls can
            cause non-deterministic content drops on some chunks. Cost
            is wall-clock time; benefit is reproducibility.
        Stage 1 (reduce): merge the partial extractions into a single
            extraction dict via field-type-specific rules. See
            ``merge_section_extractions`` and
            ``docs/two_step_approach/MERGE_STRATEGY.md``.
        Stage 2: assess bias from the merged extraction.

        Args:
            pmid: Paper identifier (used for logging and result metadata).
            title: Paper title (used for logging and result metadata).
            sections: List of (section_name, section_text) tuples. Typically
                      produced by ``biasbuster.cli.chunking.chunk_jats_article``
                      or ``chunk_plain_text``.
            metadata: Optional paper metadata (retraction context, Cochrane
                      RoB, effect-size audit hints). Passed through to
                      Stage 2 assessment for severity calibration.

        Returns:
            Parsed assessment dict with nested ``extraction`` and
            ``_annotation_mode: "two_call_full_text_v3"``. The full
            per-section partial extractions are also preserved under
            ``_section_extractions`` for retrospective debugging without
            having to re-run the model. None if any stage fails after
            retries.
        """
        from biasbuster.prompts_v3 import ASSESSMENT_SYSTEM_PROMPT

        extraction_result = await self._extract_full_text_sections(
            pmid, title, sections,
        )
        if extraction_result is None:
            return None
        extraction, section_extractions, merge_conflicts, failed_count = extraction_result

        # Stage 2: assessment from merged extraction
        assessment_message = build_assessment_user_message(extraction, metadata)
        if merge_conflicts:
            # Surface conflicts to the assessment model as a reporting signal
            assessment_message += (
                "\n\nMERGE CONFLICTS DETECTED during section-level extraction "
                "(the paper reported different values for the same field in "
                "different sections). Treat these as a reporting-consistency "
                "concern under statistical_reporting:\n"
                + json.dumps(merge_conflicts, indent=2)
            )

        for attempt in range(self.max_retries):
            raw_assessment = await self._call_llm(
                ASSESSMENT_SYSTEM_PROMPT, assessment_message, pmid=pmid,
            )
            if raw_assessment is None:
                return None

            assessment = parse_llm_json(raw_assessment, pmid=pmid)
            if assessment is not None:
                assessment["pmid"] = pmid
                assessment["title"] = title
                assessment["_annotation_model"] = self.model
                assessment["_annotation_mode"] = "two_call_full_text_v3"
                assessment["extraction"] = extraction
                assessment["_section_extractions"] = section_extractions
                if merge_conflicts:
                    assessment["_merge_conflicts"] = merge_conflicts
                if failed_count:
                    assessment["_failed_sections"] = failed_count
                return assessment

            logger.warning(
                f"PMID {pmid}: assessment parse failed "
                f"(attempt {attempt + 1}/{self.max_retries}), retrying"
            )
            await asyncio.sleep(2 ** attempt)

        logger.error(
            f"PMID {pmid}: all assessment parse attempts failed"
        )
        return None

    # ------------------------------------------------------------------
    # v4 Agentic annotation (tool-calling assessment agent)
    # ------------------------------------------------------------------

    async def annotate_full_text_agentic(
        self,
        pmid: str,
        title: str,
        sections: list[tuple[str, str]],
        metadata: Optional[dict] = None,
        api_key: Optional[str] = None,
        agent_provider: Optional[str] = None,
        agent_model: Optional[str] = None,
    ) -> Optional[dict]:
        """v4 agentic annotation: extraction → tool-calling assessment agent.

        Stage 1: identical to ``annotate_full_text_two_call`` — per-section
            sequential extraction → merge. Shared via
            ``_extract_full_text_sections``.
        Stage 2: hands the merged extraction to an ``AssessmentAgent``
            that calls ``run_mechanical_assessment`` (Python aggregator),
            reviews the draft, optionally overrides with contextual
            judgment, and optionally calls verification tools.

        Two provider modes for the assessment agent (Stage 2 only —
        Stage 1 always uses this annotator's ``_call_llm``):

        - ``agent_provider="anthropic"`` (default for LLMAnnotator):
          direct Anthropic SDK. Requires ``api_key``.
        - ``agent_provider="bmlib"`` (for BmlibAnnotator / local models):
          bmlib's ``LLMClient.chat(tools=...)``. Requires ``agent_model``
          in bmlib format (e.g. ``"ollama:gemma4:26b-a4b-it-q8_0"``).

        Args:
            pmid: Paper identifier.
            title: Paper title.
            sections: Chunked (section_name, section_text) tuples.
            metadata: Optional paper metadata (authors, DOI, etc.)
                for verification tool dispatch.
            api_key: Anthropic API key (for agent_provider="anthropic").
                If None, tries ``self.api_key``.
            agent_provider: ``"anthropic"`` or ``"bmlib"``. Defaults to
                ``"anthropic"`` unless the annotator is a BmlibAnnotator.
            agent_model: Model string for the assessment agent. Defaults
                to ``self.model``. For bmlib, pass the full bmlib model
                string (e.g. ``"ollama:gemma4:26b-a4b-it-q8_0"``).

        Returns:
            Assessment dict with ``_annotation_mode: "agentic_v4"``
            and ``_overrides: [...]``. None on failure.
        """
        from biasbuster.assessment_agent import AssessmentAgent

        # Resolve provider — default to anthropic for LLMAnnotator,
        # bmlib for BmlibAnnotator
        if agent_provider is None:
            from biasbuster.annotators.bmlib_backend import BmlibAnnotator
            agent_provider = "bmlib" if isinstance(self, BmlibAnnotator) else "anthropic"

        # Resolve API key for Anthropic path
        resolved_key = api_key or getattr(self, "api_key", "") or ""
        if agent_provider == "anthropic" and not resolved_key:
            logger.error(
                f"PMID {pmid}: annotate_full_text_agentic with provider=anthropic "
                f"requires an API key. Pass api_key= or use LLMAnnotator."
            )
            return None

        # Resolve model for the assessment agent
        resolved_model = agent_model or getattr(self, "model", "claude-sonnet-4-5-20250929")

        # Stage 1: extraction (shared with v3)
        extraction_result = await self._extract_full_text_sections(
            pmid, title, sections,
        )
        if extraction_result is None:
            return None
        extraction, section_extractions, merge_conflicts, failed_count = extraction_result

        # Stage 2: agentic assessment
        logger.info(
            f"PMID {pmid}: starting v4 assessment agent "
            f"(provider={agent_provider}, model={resolved_model}, "
            f"{len(section_extractions)} sections extracted, "
            f"{len(merge_conflicts)} merge conflicts)"
        )
        async with AssessmentAgent(
            api_key=resolved_key,
            model=resolved_model,
            provider=agent_provider,
        ) as agent:
            agent_result = await agent.assess(
                pmid=pmid,
                title=title,
                extraction=extraction,
                paper_metadata=metadata,
            )

        if agent_result.assessment is None:
            logger.error(
                f"PMID {pmid}: v4 agent did not produce a valid assessment "
                f"after {agent_result.iterations} iterations "
                f"(hit_cap={agent_result.hit_iteration_cap})"
            )
            return None

        # Decorate the result in the same shape as v3 for DB storage
        assessment = agent_result.assessment
        assessment["pmid"] = pmid
        assessment["title"] = title
        assessment["_annotation_model"] = self.model
        assessment["_annotation_mode"] = "agentic_v4"
        assessment["extraction"] = extraction
        assessment["_section_extractions"] = section_extractions
        if merge_conflicts:
            assessment["_merge_conflicts"] = merge_conflicts
        if failed_count:
            assessment["_failed_sections"] = failed_count
        # v4-specific metadata
        assessment["_agent_iterations"] = agent_result.iterations
        assessment["_agent_tool_calls"] = agent_result.tool_calls_made
        if agent_result.enforcement_notes:
            assessment["_enforcement_notes"] = agent_result.enforcement_notes

        return assessment

    # ------------------------------------------------------------------
    # Two-call annotation (v3)
    # ------------------------------------------------------------------

    async def annotate_abstract_two_call(
        self,
        pmid: str,
        title: str,
        abstract: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Two-call annotation (v3): Stage 1 extraction, then Stage 2 assessment.

        Stage 1 extracts structured facts from the paper text.
        Stage 2 assesses bias from those facts.
        The extraction is nested inside the final annotation for traceability.

        Returns parsed JSON assessment (with nested extraction) or None.
        """
        from biasbuster.prompts_v3 import (
            EXTRACTION_SYSTEM_PROMPT,
            ASSESSMENT_SYSTEM_PROMPT,
        )

        user_message = build_user_message(pmid, title, abstract, metadata)

        # Stage 1: Extraction
        for attempt in range(self.max_retries):
            raw_extraction = await self._call_llm(
                EXTRACTION_SYSTEM_PROMPT, user_message, pmid=pmid,
            )
            if raw_extraction is None:
                return None

            extraction = parse_extraction_json(raw_extraction, pmid=pmid)
            if extraction is not None:
                break

            logger.warning(
                f"PMID {pmid}: extraction parse failed "
                f"(attempt {attempt + 1}/{self.max_retries}), retrying"
            )
            await asyncio.sleep(2 ** attempt)
        else:
            logger.error(
                f"PMID {pmid}: all extraction parse attempts failed"
            )
            return None

        # Stage 2: Assessment from extracted facts
        assessment_message = build_assessment_user_message(extraction, metadata)

        for attempt in range(self.max_retries):
            raw_assessment = await self._call_llm(
                ASSESSMENT_SYSTEM_PROMPT, assessment_message, pmid=pmid,
            )
            if raw_assessment is None:
                return None

            assessment = parse_llm_json(raw_assessment, pmid=pmid)
            if assessment is not None:
                assessment["pmid"] = pmid
                assessment["title"] = title
                assessment["_annotation_model"] = self.model
                assessment["_annotation_mode"] = "two_call_v3"
                assessment["extraction"] = extraction
                return assessment

            logger.warning(
                f"PMID {pmid}: assessment parse failed "
                f"(attempt {attempt + 1}/{self.max_retries}), retrying"
            )
            await asyncio.sleep(2 ** attempt)

        logger.error(
            f"PMID {pmid}: all assessment parse attempts failed"
        )
        return None

    # ------------------------------------------------------------------
    # Batch orchestration
    # ------------------------------------------------------------------

    async def annotate_batch(
        self,
        items: list[dict],
        concurrency: int = 3,
        delay: float = 1.0,
        already_done: Optional[set[str]] = None,
        on_result: Optional[callable] = None,
        two_call: bool = True,
    ) -> list[dict]:
        """Annotate a batch of abstracts with rate limiting.

        Skips PMIDs already in *already_done* (for resume support).
        Each successful annotation is passed to *on_result* immediately
        for incremental persistence (e.g. saving to database).

        Args:
            items: List of dicts with pmid, title, abstract, metadata keys.
            concurrency: Max concurrent API requests.
            delay: Seconds between requests.
            already_done: Set of PMIDs to skip (already annotated).
            on_result: Optional callback(annotation_dict) called immediately
                       on each successful annotation for incremental save.
            two_call: If True (default), use v3 two-call pipeline
                      (extraction -> assessment). If False, use single-call v1.

        Returns:
            List of successful annotations.
        """
        if already_done is None:
            already_done = set()

        # Deduplicate by PMID
        seen_pmids: set[str] = set(already_done)
        remaining = []
        for it in items:
            pmid = it["pmid"]
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                remaining.append(it)
        if not remaining:
            logger.info("All items already annotated, nothing to do")
            return []

        semaphore = asyncio.Semaphore(concurrency)
        successful: list[dict] = []
        failed = 0

        pbar = tqdm(
            total=len(remaining),
            desc=f"Annotating ({self.model})",
            unit="paper",
            dynamic_ncols=True,
        )

        annotate_fn = (
            self.annotate_abstract_two_call if two_call
            else self.annotate_abstract
        )
        mode_label = "two-call v3" if two_call else "single-call v1"
        logger.info(f"Using {mode_label} annotation mode ({self.model})")

        async def process_one(item):
            nonlocal failed
            async with semaphore:
                result = await annotate_fn(
                    pmid=item["pmid"],
                    title=item["title"],
                    abstract=item["abstract"],
                    metadata=item.get("metadata"),
                )
                # Save incrementally as each result arrives
                if result is not None:
                    successful.append(result)
                    if on_result:
                        on_result(result)
                else:
                    failed += 1
                pbar.set_postfix(ok=len(successful), fail=failed, refresh=False)
                pbar.update(1)
                await asyncio.sleep(delay)
                return result

        await asyncio.gather(
            *(process_one(item) for item in remaining)
        )

        pbar.close()
        logger.info(
            f"Annotated {len(successful)}/{len(remaining)} abstracts successfully "
            f"(model: {self.model})"
        )
        return successful
