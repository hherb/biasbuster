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
   lightweight — refusing non-diagnostic designs with a specific
   reason. Step 10 prompt-iteration can add heuristics (e.g. detecting
   "likelihood ratio" / "AUROC" in the abstract to confirm).
"""

from __future__ import annotations

from biasbuster.methodologies import study_design


def check_applicability(
    paper: dict, enrichment: dict, full_text_available: bool,
) -> tuple[bool, str]:
    """Return ``(True, "")`` if QUADAS-2 applies; ``(False, reason)`` otherwise."""
    del enrichment, full_text_available  # unused in the MVP check
    detected = study_design.detect(paper)
    if detected == "diagnostic_accuracy":
        return True, ""
    # Common mistakes worth naming: systematic reviews of diagnostic
    # accuracy (which need ROBIS not QUADAS-2) and RCTs (which need RoB 2).
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
    return False, (
        "QUADAS-2 applies only to primary diagnostic-accuracy studies; "
        f"detected study design was {detected!r}. Use "
        "--methodology=biasbuster for design-agnostic assessment."
    )
