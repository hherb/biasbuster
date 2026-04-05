"""
Verification step router.

Maps verification step strings (natural language) to concrete tool calls
using keyword/regex matching. Step strings are produced by
``agent.verification_planner.synthesize_verification_steps()`` and use
consistent database names, so pattern matching is reliable without
needing an additional LLM call.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# --- NCT ID extraction ---
_NCT_RE = re.compile(r"NCT\d{7,8}", re.IGNORECASE)

# --- Author name extraction ---
# Matches "Dr. LastName", "Dr. First Last", "First Last" patterns
# Excludes database names that happen to look like proper nouns
_AUTHOR_NAME_RE = re.compile(
    r"(?:Dr\.?\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([A-Z][a-z]+)"
    r"|"
    r"(?:for|by|of)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)"
)

# Words that look like names but are actually part of database/tool names
_NAME_STOPWORDS = {
    "open", "payments", "clinical", "trials", "retraction", "watch",
    "europe", "pubmed", "crossref", "google", "scholar", "semantic",
    "cochrane", "medicines", "australia",
}


@dataclass
class AbstractContext:
    """Context from the original abstract, used to extract tool parameters."""

    pmid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    authors: list[dict] = field(default_factory=list)
    fulltext: str = ""


@dataclass
class ToolCall:
    """A resolved tool call ready for execution."""

    tool_name: str
    params: dict = field(default_factory=dict)
    original_step: str = ""


# --- Tool matchers ---

_TOOL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("clinicaltrials_gov", re.compile(
        r"ClinicalTrials\.gov|clinicaltrials\.gov|"
        r"trial\s+regist|registered\s+(?:primary\s+)?outcome|"
        r"NCT\d{7,8}|protocol\s+amendment",
        re.IGNORECASE,
    )),
    ("open_payments", re.compile(
        r"CMS\s+Open\s+Payments|open\s*payments|openpaymentsdata|"
        r"physician\s+payment",
        re.IGNORECASE,
    )),
    ("orcid", re.compile(
        r"ORCID|author\s+affiliation\s+histor|employment\s+history|"
        r"undisclosed\s+industry\s+ties",
        re.IGNORECASE,
    )),
    ("europmc", re.compile(
        r"Europe\s+PMC|europepmc|full[- ]text\s+fund|"
        r"funder\s+metadata|funding.*disclosure",
        re.IGNORECASE,
    )),
    ("retraction_watch", re.compile(
        r"Retraction\s+Watch|retraction(?:watch)?|"
        r"post[- ]publication\s+notice|expression\s+of\s+concern|"
        r"Crossref.*retraction",
        re.IGNORECASE,
    )),
    ("effect_size_audit", re.compile(
        r"effect\s+size|reporting\s+bias\s+audit|"
        r"absolute\s+(?:risk|measure)|relative\s+(?:only|measure)",
        re.IGNORECASE,
    )),
]


def _extract_nct_id(step: str, context: AbstractContext) -> Optional[str]:
    """Extract NCT ID from verification step text, abstract, or full text."""
    match = _NCT_RE.search(step)
    if match:
        return match.group(0)
    # Search abstract first (cheapest), then full text (NCT IDs are often
    # in the methods or ethics/IRB section rather than the abstract).
    for text in (context.abstract, context.fulltext):
        if text:
            match = _NCT_RE.search(text)
            if match:
                return match.group(0)
    return None


def _extract_author_names(
    step: str, context: AbstractContext,
) -> list[dict[str, str]]:
    """Extract author names from step text, falling back to context."""
    names = []
    for match in _AUTHOR_NAME_RE.finditer(step):
        # The regex has two alternatives; pick whichever matched
        first = match.group(1) or match.group(3) or ""
        last = match.group(2) or match.group(4) or ""
        if first.lower() in _NAME_STOPWORDS or last.lower() in _NAME_STOPWORDS:
            continue
        if first and last:
            names.append({"first": first, "last": last})

    if not names and context.authors:
        # Use first 3 authors from abstract metadata
        for author in context.authors[:3]:
            first = author.get("first", "")
            last = author.get("last", "")
            if last:
                names.append({"first": first, "last": last})
    return names


def _build_params(
    tool_name: str, step: str, context: AbstractContext,
) -> dict:
    """Build tool-specific parameters from step text and context."""
    if tool_name == "clinicaltrials_gov":
        nct_id = _extract_nct_id(step, context)
        return {
            "nct_id": nct_id or "",
            "abstract": context.abstract,
            "title": context.title,
            "pmid": context.pmid,
        }
    elif tool_name == "open_payments":
        authors = _extract_author_names(step, context)
        return {"authors": authors}
    elif tool_name == "orcid":
        authors = _extract_author_names(step, context)
        return {"authors": authors}
    elif tool_name == "europmc":
        return {"pmid": context.pmid}
    elif tool_name == "retraction_watch":
        return {"pmid": context.pmid, "doi": context.doi}
    elif tool_name == "effect_size_audit":
        return {
            "pmid": context.pmid,
            "title": context.title,
            "abstract": context.abstract,
        }
    return {}


def route_verification_steps(
    steps: list[str],
    context: AbstractContext,
) -> list[ToolCall]:
    """Map verification step strings to concrete tool calls.

    Args:
        steps: List of recommended verification steps from the model.
        context: Abstract metadata for parameter extraction.

    Returns:
        List of ToolCall objects, one per matched step. Unrecognised
        steps get ``tool_name="unsupported"``.
    """
    calls: list[ToolCall] = []
    seen_tools: set[str] = set()

    for step in steps:
        matched = False
        for tool_name, pattern in _TOOL_PATTERNS:
            if pattern.search(step):
                # Deduplicate: one call per tool type (aggregate params later)
                if tool_name not in seen_tools:
                    params = _build_params(tool_name, step, context)
                    calls.append(ToolCall(
                        tool_name=tool_name,
                        params=params,
                        original_step=step,
                    ))
                    seen_tools.add(tool_name)
                matched = True
                break

        if not matched:
            calls.append(ToolCall(
                tool_name="unsupported",
                params={},
                original_step=step,
            ))
            logger.debug("No tool match for step: %s", step[:100])

    return calls


def parse_verification_steps_from_output(model_output: str) -> list[str]:
    """Extract verification step strings from model output (legacy).

    This function parsed ``recommended_verification_steps`` from the model's
    JSON output. Since models no longer produce this field, verification
    steps are now synthesized programmatically via
    ``agent.verification_planner.plan_verification()``.

    Retained for backward compatibility with older model outputs that may
    still include the field.

    Tries JSON parsing first (``recommended_verification_steps`` field),
    then falls back to regex extraction of bullet points under a
    'Verification' heading in markdown output.

    Args:
        model_output: Raw model output (markdown or JSON).

    Returns:
        List of verification step strings.
    """
    from biasbuster.annotators import parse_llm_json

    # Try JSON parse first
    parsed = parse_llm_json(model_output)
    if parsed and isinstance(parsed, dict):
        steps = parsed.get("recommended_verification_steps", [])
        if isinstance(steps, list) and steps:
            return [s for s in steps if isinstance(s, str) and s.strip()]

    # Fallback: regex extraction from markdown
    return _extract_steps_from_markdown(model_output)


def _extract_steps_from_markdown(text: str) -> list[str]:
    """Extract bullet-point verification steps from markdown output."""
    lines = text.split("\n")
    steps: list[str] = []
    in_verification_section = False

    verification_header_re = re.compile(
        r"^#{1,4}\s+.*[Vv]erification", re.IGNORECASE,
    )
    bullet_re = re.compile(r"^\s*[-*]\s+(.+)")
    numbered_re = re.compile(r"^\s*\d+[.)]\s+(.+)")
    heading_re = re.compile(r"^#{1,4}\s+")

    for line in lines:
        if verification_header_re.match(line):
            in_verification_section = True
            continue

        if in_verification_section:
            # Stop at the next non-verification heading
            if heading_re.match(line) and not verification_header_re.match(line):
                break

            bullet_match = bullet_re.match(line)
            if bullet_match:
                steps.append(bullet_match.group(1).strip())
                continue

            numbered_match = numbered_re.match(line)
            if numbered_match:
                steps.append(numbered_match.group(1).strip())

    return steps
