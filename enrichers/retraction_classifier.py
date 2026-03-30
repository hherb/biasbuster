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

Abstract detectability:
  Each retraction reason is also classified as abstract-detectable or not.
  Reasons like data fabrication produce clean-looking abstracts — the fraud
  is invisible in the text.  For training a text-based bias detector, these
  papers should be assessed on abstract merits only (no severity floor).
  They remain valuable for testing the full agent harness that checks
  external databases (Retraction Watch, Crossref).  See docs/ANNOTATED_DATA_SET.md.
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

# Each tuple: (pattern, severity_floor, category, abstract_detectable)
# abstract_detectable=True means the retraction reason MAY produce visible
# bias signals in the abstract text (e.g. statistical errors, flawed analysis).
# abstract_detectable=False means fraud is invisible in the abstract — the paper
# should be assessed on text merits only for training purposes.
_REASON_PATTERNS: list[tuple[re.Pattern, Optional[str], str, bool]] = [
    # --- Bias-relevant: CRITICAL floor (all undetectable from abstract) ---
    (re.compile(r"fabricat", re.I), "critical", "data_fabrication", False),
    (re.compile(r"falsif", re.I), "critical", "data_falsification", False),
    (re.compile(r"fraud", re.I), "critical", "fraud", False),
    (re.compile(r"fak(e|ed|ing)\s+(data|results|images?)", re.I), "critical", "data_fabrication", False),
    (re.compile(r"paper\s*mill", re.I), "critical", "paper_mill", False),
    (re.compile(r"misconduct", re.I), "critical", "misconduct", False),

    # --- Bias-relevant: HIGH floor ---
    (re.compile(r"manipulat(e|ed|ion|ing)", re.I), "high", "manipulation", False),
    (re.compile(r"unreliable", re.I), "high", "unreliable_results", False),
    (re.compile(r"concern.{0,30}(data|results|integrity|reliab|conclusions)", re.I),
     "high", "data_concerns", False),
    (re.compile(r"(data|image|figure).{0,10}(manipulat|doctor|alter)", re.I), "high", "manipulation", False),
    (re.compile(r"misrepresent", re.I), "high", "misrepresentation", False),
    (re.compile(r"flawed\s*(data|analysis|methodology)", re.I), "high", "flawed_analysis", True),
    (re.compile(r"original\s*data.{0,20}not\s*(provided|available)", re.I),
     "high", "data_not_available", False),
    (re.compile(r"computer.{0,10}(aided|generated)\s*content", re.I), "high", "ai_generated", False),

    # --- Bias-relevant: MODERATE floor ---
    (re.compile(r"(statistic|analytic)al?\s*(error|mistake|flaw)", re.I),
     "moderate", "statistical_errors", True),
    (re.compile(r"error.{0,20}(data|analysis|result|method)", re.I),
     "moderate", "analytical_errors", True),
    (re.compile(r"irreproduci", re.I), "moderate", "irreproducible", False),
    (re.compile(r"cannot\s*be\s*(reprod|replic)", re.I), "moderate", "irreproducible", False),
    (re.compile(r"conflict\s*of\s*interest", re.I), "moderate", "conflict_of_interest", True),
    (re.compile(r"concern.{0,30}image", re.I), "moderate", "image_concerns", False),
    (re.compile(r"concern.{0,30}peer\s*review", re.I), "moderate", "peer_review_concerns", False),
    (re.compile(r"compromised\s*peer\s*review", re.I), "moderate", "peer_review_concerns", False),
    (re.compile(r"lack\s*of\s*(IRB|IACUC|ethic)", re.I), "moderate", "ethics_violation", True),
    (re.compile(r"breach\s*of\s*policy", re.I), "moderate", "policy_breach", True),

    # --- NOT bias-relevant: no severity floor ---
    (re.compile(r"concern.{0,30}authorship", re.I), None, "authorship_dispute", False),
    (re.compile(r"authorship\s*(dispute|issue|concern)", re.I), None, "authorship_dispute", False),
    (re.compile(r"objections?\s*by\s*author", re.I), None, "author_objection", False),
    (re.compile(r"author\s*unresponsive", re.I), None, "author_unresponsive", False),
    (re.compile(r"plagiari", re.I), None, "plagiarism", False),
    (re.compile(r"duplicat", re.I), None, "duplication", False),
    (re.compile(r"euphemism.{0,10}duplicat", re.I), None, "duplication", False),
    (re.compile(r"copyright", re.I), None, "copyright", False),
    (re.compile(r"consent\s*(issue|violation|concern|not\s*obtained)", re.I), None, "consent_issues", False),
    (re.compile(r"publisher\s*error", re.I), None, "publisher_error", False),
    (re.compile(r"author\s*request", re.I), None, "author_request", False),
    (re.compile(r"withdraw(n|al)?\s*(at|by|per)\s*author", re.I), None, "author_request", False),
    (re.compile(r"overlap(ping)?\s*(with|article|publication)", re.I), None, "duplication", False),
    (re.compile(r"referenc", re.I), None, "referencing_issues", False),
    (re.compile(r"removed$", re.I), None, "removed", False),
    (re.compile(r"notice.{0,10}(limited|no)\s*information", re.I), None, "no_information", False),
    (re.compile(r"upgrade.{0,10}prior\s*notice", re.I), None, "notice_update", False),
    (re.compile(r"date.{0,20}unknown", re.I), None, "date_unknown", False),
    (re.compile(r"investigation\s*by", re.I), None, "under_investigation", False),
]


def classify_retraction(
    reasons: list[str],
    title: str = "",
    abstract: str = "",
) -> tuple[Optional[str], str, bool]:
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
        Tuple of ``(severity_floor, category, abstract_detectable)``:
        - ``severity_floor``: ``"critical"`` / ``"high"`` / ``"moderate"`` /
          ``None``. ``None`` means no floor — assess normally.
        - ``category``: Human-readable reason category (e.g.
          ``"data_fabrication"``, ``"authorship_dispute"``, ``"unknown"``).
        - ``abstract_detectable``: Whether the retraction reason could
          produce visible bias signals in the abstract text.  When False,
          the paper should be assessed on abstract merits only for training.
    """
    # Build the text corpus to search
    search_texts = [r.strip() for r in reasons if r.strip()]
    if title:
        search_texts.append(title)
    if abstract:
        search_texts.append(abstract)

    combined = " ".join(search_texts)

    # Try each pattern
    for pattern, floor, category, detectable in _REASON_PATTERNS:
        if pattern.search(combined):
            logger.debug(
                "Retraction classified as %s (floor=%s, detectable=%s) "
                "from pattern: %s",
                category, floor, detectable, pattern.pattern,
            )
            return floor, category, detectable

    # No specific pattern matched. If the paper IS retracted (we know it is
    # if this function was called), apply a default MODERATE floor — the paper
    # was retracted for *some* reason, we just can't determine what.
    # Unknown retractions are treated as undetectable (conservative for training).
    if reasons or "retract" in combined.lower():
        return "moderate", "unknown_retraction", False

    return None, "not_retracted", False


def format_retraction_context(
    severity_floor: Optional[str],
    category: str,
    abstract_detectable: bool = True,
) -> str:
    """Format retraction classification as context text for the annotator.

    Args:
        severity_floor: The severity floor, or None for non-bias retractions.
        category: The retraction reason category.
        abstract_detectable: Whether the reason produces visible bias signals
            in the abstract.  When False, no severity floor is communicated
            to the annotator — the abstract should be assessed on its merits.

    Returns a string to append to the user message, or empty string if
    no retraction context applies.
    """
    if category == "not_retracted":
        return ""

    # Non-bias retractions (authorship, plagiarism, etc.) — no floor
    if severity_floor is None:
        return (
            f"RETRACTION CLASSIFICATION: This paper was retracted for "
            f"{category.replace('_', ' ')}. This is NOT a bias-relevant "
            f"retraction — assess the abstract content normally with no "
            f"severity floor."
        )

    # Abstract-undetectable retractions — assess on text merits only
    if not abstract_detectable:
        return (
            f"RETRACTION CLASSIFICATION: This paper was retracted "
            f"({category.replace('_', ' ')}). This retraction reason is "
            f"NOT expected to be detectable from the abstract text alone. "
            f"Assess the abstract on its own merits — do NOT apply a "
            f"severity floor. Rate only what the text actually shows."
        )

    # Abstract-detectable retractions — enforce severity floor
    floor_label = severity_floor.upper()
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
        floor, category, detectable = classify_retraction(reasons, title, abstract)
        context = format_retraction_context(floor, category, detectable)
        print(f"  reasons={reasons}, title={title[:50]!r}")
        print(f"  → floor={floor}, category={category}, abstract_detectable={detectable}")
        if context:
            print(f"  → {context[:120]}...")
        print()
