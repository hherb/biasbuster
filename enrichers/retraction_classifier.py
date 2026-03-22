"""
Retraction Reason Classifier

Classifies retracted papers by inferred reason and assigns severity floors.
Since Crossref provides only the generic label "Retraction" (no structured
reason codes), this module infers reason categories from:
  1. The retraction notice text (abstract of the notice itself)
  2. Title patterns (e.g., "RETRACTED: ...")
  3. Any structured reasons if available from the Retraction Watch database

See docs/MISTAKES_ROUND_1_AND_FIXES.md (Root Cause 2) for why this matters:
136/368 retracted papers in Round 1 were labelled NONE severity because the
original abstract content showed no visible bias, even though the paper was
retracted for data integrity issues.

Severity floor semantics:
  - CRITICAL: Evidence of intentional fraud (fabrication, falsification)
  - HIGH: Serious data/analytical issues (unreliable results, manipulation)
  - MODERATE: Default floor for retracted papers where reason is unclear
    but the paper WAS retracted (something was wrong)
  - None: Non-bias retractions (authorship, plagiarism, consent, duplicate)
    — assess abstract content normally, no floor imposed
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reason patterns → (severity_floor, category)
#
# Patterns are matched case-insensitively against:
#   1. retraction_reasons list items (from Crossref/Retraction Watch)
#   2. Retraction notice abstract text (if available)
#   3. Paper title (for "RETRACTED:" prefix papers)
#
# Order matters: first match wins. More specific patterns come first.
# ---------------------------------------------------------------------------

_REASON_PATTERNS: list[tuple[re.Pattern, Optional[str], str]] = [
    # --- Bias-relevant: CRITICAL floor ---
    (re.compile(r"fabricat", re.I), "critical", "data_fabrication"),
    (re.compile(r"falsif", re.I), "critical", "data_falsification"),
    (re.compile(r"fraud", re.I), "critical", "fraud"),
    (re.compile(r"fak(e|ed|ing)\s+(data|results|images?)", re.I), "critical", "data_fabrication"),
    (re.compile(r"paper\s*mill", re.I), "critical", "paper_mill"),
    (re.compile(r"misconduct", re.I), "critical", "misconduct"),

    # --- Bias-relevant: HIGH floor ---
    (re.compile(r"manipulat(e|ed|ion|ing)", re.I), "high", "manipulation"),
    (re.compile(r"unreliable", re.I), "high", "unreliable_results"),
    (re.compile(r"concern.{0,30}(data|results|integrity|reliab|conclusions)", re.I),
     "high", "data_concerns"),
    (re.compile(r"(data|image|figure).{0,10}(manipulat|doctor|alter)", re.I), "high", "manipulation"),
    (re.compile(r"misrepresent", re.I), "high", "misrepresentation"),
    (re.compile(r"flawed\s*(data|analysis|methodology)", re.I), "high", "flawed_analysis"),
    (re.compile(r"original\s*data.{0,20}not\s*(provided|available)", re.I),
     "high", "data_not_available"),
    (re.compile(r"computer.{0,10}(aided|generated)\s*content", re.I), "high", "ai_generated"),

    # --- Bias-relevant: MODERATE floor ---
    (re.compile(r"(statistic|analytic)al?\s*(error|mistake|flaw)", re.I),
     "moderate", "statistical_errors"),
    (re.compile(r"error.{0,20}(data|analysis|result|method)", re.I),
     "moderate", "analytical_errors"),
    (re.compile(r"irreproduci", re.I), "moderate", "irreproducible"),
    (re.compile(r"cannot\s*be\s*(reprod|replic)", re.I), "moderate", "irreproducible"),
    (re.compile(r"conflict\s*of\s*interest", re.I), "moderate", "conflict_of_interest"),
    (re.compile(r"concern.{0,30}image", re.I), "moderate", "image_concerns"),
    (re.compile(r"concern.{0,30}peer\s*review", re.I), "moderate", "peer_review_concerns"),
    (re.compile(r"compromised\s*peer\s*review", re.I), "moderate", "peer_review_concerns"),
    (re.compile(r"lack\s*of\s*(IRB|IACUC|ethic)", re.I), "moderate", "ethics_violation"),
    (re.compile(r"breach\s*of\s*policy", re.I), "moderate", "policy_breach"),

    # --- NOT bias-relevant: no severity floor ---
    (re.compile(r"concern.{0,30}authorship", re.I), None, "authorship_dispute"),
    (re.compile(r"authorship\s*(dispute|issue|concern)", re.I), None, "authorship_dispute"),
    (re.compile(r"objections?\s*by\s*author", re.I), None, "author_objection"),
    (re.compile(r"author\s*unresponsive", re.I), None, "author_unresponsive"),
    (re.compile(r"plagiari", re.I), None, "plagiarism"),
    (re.compile(r"duplicat", re.I), None, "duplication"),
    (re.compile(r"euphemism.{0,10}duplicat", re.I), None, "duplication"),
    (re.compile(r"copyright", re.I), None, "copyright"),
    (re.compile(r"consent\s*(issue|violation|concern|not\s*obtained)", re.I), None, "consent_issues"),
    (re.compile(r"publisher\s*error", re.I), None, "publisher_error"),
    (re.compile(r"author\s*request", re.I), None, "author_request"),
    (re.compile(r"withdraw(n|al)?\s*(at|by|per)\s*author", re.I), None, "author_request"),
    (re.compile(r"overlap(ping)?\s*(with|article|publication)", re.I), None, "duplication"),
    (re.compile(r"referenc", re.I), None, "referencing_issues"),
    (re.compile(r"removed$", re.I), None, "removed"),
    (re.compile(r"notice.{0,10}(limited|no)\s*information", re.I), None, "no_information"),
    (re.compile(r"upgrade.{0,10}prior\s*notice", re.I), None, "notice_update"),
    (re.compile(r"date.{0,20}unknown", re.I), None, "date_unknown"),
    (re.compile(r"investigation\s*by", re.I), None, "under_investigation"),
]


def classify_retraction(
    reasons: list[str],
    title: str = "",
    abstract: str = "",
) -> tuple[Optional[str], str]:
    """Classify a retraction and determine the severity floor.

    Searches through the retraction reasons, title, and abstract text for
    patterns indicating why the paper was retracted.

    Args:
        reasons: Retraction reason strings (from Crossref ``update-to.label``
            or Retraction Watch structured fields).
        title: Paper or retraction notice title.
        abstract: Retraction notice abstract text (NOT the original paper
            abstract — this is the notice explaining why it was retracted).

    Returns:
        Tuple of ``(severity_floor, category)``:
        - ``severity_floor``: ``"critical"`` / ``"high"`` / ``"moderate"`` /
          ``None``. ``None`` means no floor — assess normally.
        - ``category``: Human-readable reason category (e.g.
          ``"data_fabrication"``, ``"authorship_dispute"``, ``"unknown"``).
    """
    # Build the text corpus to search
    search_texts = [r.strip() for r in reasons if r.strip()]
    if title:
        search_texts.append(title)
    if abstract:
        search_texts.append(abstract)

    combined = " ".join(search_texts)

    # Try each pattern
    for pattern, floor, category in _REASON_PATTERNS:
        if pattern.search(combined):
            logger.debug(
                "Retraction classified as %s (floor=%s) from pattern: %s",
                category, floor, pattern.pattern,
            )
            return floor, category

    # No specific pattern matched. If the paper IS retracted (we know it is
    # if this function was called), apply a default MODERATE floor — the paper
    # was retracted for *some* reason, we just can't determine what.
    if reasons or "retract" in combined.lower():
        return "moderate", "unknown_retraction"

    return None, "not_retracted"


def format_retraction_context(
    severity_floor: Optional[str],
    category: str,
) -> str:
    """Format retraction classification as context text for the annotator.

    Returns a string to append to the user message, or empty string if
    no retraction context applies.
    """
    if severity_floor is None and category in ("not_retracted", "authorship_dispute",
                                                "plagiarism", "duplicate_publication",
                                                "copyright", "consent_issues",
                                                "ethics_violation", "publisher_error",
                                                "author_request"):
        if category == "not_retracted":
            return ""
        return (
            f"RETRACTION CLASSIFICATION: This paper was retracted for "
            f"{category.replace('_', ' ')}. This is NOT a bias-relevant "
            f"retraction — assess the abstract content normally with no "
            f"severity floor."
        )

    floor_label = (severity_floor or "none").upper()
    return (
        f"RETRACTION CLASSIFICATION: This paper was retracted. "
        f"Reason category: {category.replace('_', ' ')}. "
        f"Severity floor: {floor_label}. "
        f"The overall severity MUST be at least {floor_label}, regardless "
        f"of how the abstract reads. Apply the floor to the most relevant "
        f"domain(s) based on the retraction reason."
    )


# ---------------------------------------------------------------------------
# Severity ordering for floor enforcement at export time
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}


def enforce_severity_floor(
    annotation: dict,
    severity_floor: Optional[str],
) -> dict:
    """Apply a severity floor to an annotation dict (non-mutating).

    If the annotation's ``overall_severity`` is below the floor, bump it up.
    Returns a shallow copy with the adjusted severity. The original annotation
    in the database is never modified — this is applied at export time.

    Args:
        annotation: The annotation dict (from DB).
        severity_floor: The floor to enforce, or None to skip.

    Returns:
        Annotation dict (may be the original if no change needed).
    """
    if severity_floor is None:
        return annotation

    current = annotation.get("overall_severity", "none").lower()
    floor_rank = _SEVERITY_ORDER.get(severity_floor, 0)
    current_rank = _SEVERITY_ORDER.get(current, 0)

    if current_rank >= floor_rank:
        return annotation

    # Need to bump — create a shallow copy
    adjusted = dict(annotation)
    adjusted["overall_severity"] = severity_floor
    logger.info(
        "PMID %s: severity floor applied (%s → %s) due to retraction",
        annotation.get("pmid", "?"), current, severity_floor,
    )
    return adjusted


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        (["Retraction"], "RETRACTED: Some paper title", ""),
        (["Retraction"], "", "This article was retracted due to data fabrication"),
        (["Retraction"], "", "Retracted due to authorship dispute"),
        (["Retraction"], "", "Concerns about the reliability of the data"),
        (["Retraction"], "", "Unreliable results identified by institution"),
        (["Retraction"], "", "Statistical errors in the analysis"),
        ([], "Some normal title", "Normal abstract"),
    ]

    for reasons, title, abstract in test_cases:
        floor, category = classify_retraction(reasons, title, abstract)
        context = format_retraction_context(floor, category)
        print(f"  reasons={reasons}, title={title[:50]!r}")
        print(f"  → floor={floor}, category={category}")
        if context:
            print(f"  → {context[:100]}...")
        print()
