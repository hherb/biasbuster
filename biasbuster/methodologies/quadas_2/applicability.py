"""Study-design gate for the QUADAS-2 methodology.

QUADAS-2 applies to primary diagnostic-accuracy studies — studies that
compare an index test against a reference standard and report
sensitivity / specificity / predictive values. Systematic reviews of
diagnostic accuracy are out of scope (those need a review-level tool
like ROBIS); so are studies that merely report a test without
benchmarking against a reference.

As with the other methodology applicability modules, the decision is
two-tier:

1. The declarative ``applicable_study_designs={"diagnostic_accuracy"}``
   on the Methodology dataclass is the first filter — it keys off the
   :func:`biasbuster.methodologies.study_design.detect` slug.
2. This function is the second filter, giving the methodology a chance
   to refuse based on finer-grained paper state. For the MVP it stays
   lightweight — refusing *clearly wrong* designs (RCT, systematic
   review, cohort, etc.) with a specific reason, but accepting
   ``unknown`` because the regex/MeSH detector frequently misses
   diagnostic-accuracy studies that don't use the canonical phrasing.
   When the user explicitly opts into QUADAS-2 via ``--methodology=quadas_2``
   we trust that opt-in for ambiguous designs.
"""

from __future__ import annotations

import logging

from biasbuster.methodologies import study_design

logger = logging.getLogger(__name__)


#: Study-design slugs that are clearly *not* diagnostic-accuracy and
#: must be refused even when the user opts into ``--methodology=quadas_2``.
#: Anything outside this set (cohort, case_control, case_series,
#: diagnostic_accuracy, unknown) is plausibly framable as a primary
#: diagnostic-accuracy study and is accepted with a warning.
_REFUSED_DESIGNS: frozenset[str] = frozenset({
    "rct_parallel", "rct_cluster", "rct_crossover",
    "systematic_review", "meta_analysis", "narrative_review",
})


def check_applicability(
    paper: dict, enrichment: dict, full_text_available: bool,
) -> tuple[bool, str]:
    """Return ``(True, "")`` if QUADAS-2 applies; ``(False, reason)`` otherwise."""
    del enrichment, full_text_available  # unused in the MVP check
    detected = study_design.detect(paper)
    if detected == "diagnostic_accuracy":
        return True, ""
    # Refuse only *clearly wrong* designs (intervention trials and
    # review-level tools). Other slugs (cohort, case_control,
    # case_series, unknown) are plausibly diagnostic-accuracy framings:
    # case-control diagnostic studies are explicitly addressed by
    # Q1.2 of QUADAS-2 ("Was a case-control design avoided?"), and
    # cohort designs are common for prospective accuracy studies.
    if detected in ("systematic_review", "meta_analysis"):
        return False, (
            "paper appears to be a systematic review of diagnostic "
            "accuracy; QUADAS-2 applies to primary studies, not reviews. "
            "The review-level tool (ROBIS / AMSTAR 2) is not yet "
            "registered in biasbuster."
        )
    if detected in ("rct_parallel", "rct_cluster", "rct_crossover"):
        return False, (
            "paper appears to be a randomized controlled trial; QUADAS-2 "
            "applies to primary diagnostic-accuracy studies, not "
            "intervention trials. Use --methodology=cochrane_rob2 for RCT "
            "risk-of-bias assessment."
        )
    if detected == "narrative_review":
        return False, (
            "paper appears to be a narrative review; QUADAS-2 applies "
            "to primary diagnostic-accuracy studies, not reviews."
        )
    # detected ∈ {cohort, case_control, case_series, unknown}: trust
    # the user's --methodology=quadas_2 opt-in but log for audit.
    logger.warning(
        "PMID %s: QUADAS-2 accepted on detected design %r — verify this "
        "is a primary diagnostic-accuracy study (case-control diagnostic "
        "designs are scored as high bias on Q1.2 by the assessor).",
        paper.get("pmid", "?"), detected,
    )
    return True, ""
