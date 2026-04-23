"""Lightweight study-design detection for methodology applicability gating.

Each paper gets classified into one of the slugs in
:data:`biasbuster.methodologies.base.STUDY_DESIGN_SLUGS`. The classification
is used by :func:`biasbuster.methodologies.base.check_or_raise` to reject
mismatches early (e.g. running RoB 2 on a cohort study).

Design philosophy for v1:

- **MVP heuristic, not ML**: regex + MeSH publication-type matching. Fast,
  deterministic, auditable. False negatives (returning ``"unknown"``) are
  acceptable — callers fail loud, the user sees the detected slug in the
  refusal message, and can either choose a different methodology or
  correct the paper metadata.
- **Strongest-signal-wins priority order**: stored Cochrane RoB fields
  beat MeSH terms beat title/abstract keywords. This keeps expert-assigned
  signals (Cochrane reviewers already classified the trial as a
  parallel-group RCT) authoritative.
- **No ML, no API calls**: so the detector can run inside test fixtures
  and CI without network access.

The detector is *not* meant to replace human design classification; it
gates obvious methodology mismatches.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .base import STUDY_DESIGN_SLUGS

# MeSH publication-type cues. Matching is case-insensitive; keys are the
# detected slug, values are the MeSH phrases that imply it.
#
# Order matters: the first rule whose phrase appears in the paper's MeSH
# terms wins. More *specific* designs must come before more *general*
# ones so (e.g.) a paper tagged both "Meta-Analysis" and "Systematic
# Review" is classified as the former — the stronger, more specific
# design — not the latter. Likewise "Review" must come last so it only
# matches narrative reviews that lack the more specific tags.
#
# Note: MeSH has no dedicated publication type for cluster-randomised
# trials; the keyword regex below (_KEYWORD_RULES) is the canonical path.
# We deliberately DON'T map MeSH "Cluster Analysis" here — that term
# refers to the statistical technique (k-means, hierarchical clustering,
# etc.), not the trial design, and papers using cluster analysis for
# subgrouping would be mis-classified.
_MESH_DESIGN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rct_crossover", ("Cross-Over Studies",)),
    ("rct_parallel", (
        "Randomized Controlled Trial",
        "Randomized Controlled Trials as Topic",
        "Pragmatic Clinical Trial",
    )),
    # meta_analysis before systematic_review: a paper tagged with both is a
    # meta-analysis (systematic review + statistical synthesis).
    ("meta_analysis", ("Meta-Analysis", "Meta-Analysis as Topic")),
    ("systematic_review", ("Systematic Review", "Systematic Reviews as Topic")),
    ("cohort", ("Cohort Studies", "Prospective Studies", "Longitudinal Studies")),
    ("case_control", ("Case-Control Studies",)),
    ("case_series", ("Case Reports",)),
    ("diagnostic_accuracy", (
        "Sensitivity and Specificity",
        "Diagnostic Tests, Routine",
        "Reproducibility of Results",
    )),
    ("narrative_review", ("Review",)),  # catch-all; must be last
)

# Keyword cues on title + abstract (lower priority than MeSH). Regexes are
# word-boundary anchored where useful.
_KEYWORD_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rct_cluster",
        re.compile(r"\bcluster[- ]?randomi[sz]ed\b", re.IGNORECASE),
    ),
    (
        "rct_crossover",
        re.compile(r"\bcross[- ]?over\b", re.IGNORECASE),
    ),
    (
        "diagnostic_accuracy",
        re.compile(
            r"\b(diagnostic accuracy|sensitivity and specificity|"
            r"likelihood ratio|area under (the )?ROC|AUROC)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "systematic_review",
        re.compile(
            r"\bsystematic(ally)? (review|search)\b|PRISMA", re.IGNORECASE,
        ),
    ),
    (
        "meta_analysis",
        re.compile(r"\bmeta[- ]?analys[ei]s\b", re.IGNORECASE),
    ),
    (
        "rct_parallel",
        re.compile(
            r"\b(randomi[sz]ed[ ,-]+(double[- ]blind|placebo[- ]controlled|"
            r"controlled) trial|parallel[- ]group (randomi[sz]ed )?trial)\b",
            re.IGNORECASE,
        ),
    ),
)


def _normalise_mesh(mesh_terms: Any) -> list[str]:
    """Return MeSH terms as a list of strings, tolerating None / mixed types."""
    if not mesh_terms:
        return []
    if isinstance(mesh_terms, str):
        return [mesh_terms]
    if isinstance(mesh_terms, (list, tuple)):
        return [str(t) for t in mesh_terms if t]
    return []


def _cochrane_signal(paper: dict) -> str | None:
    """Return ``'rct_parallel'`` if the paper carries Cochrane RoB 2 ratings.

    Cochrane reviewers assess parallel-group RCTs; the presence of non-empty
    per-domain ratings (randomization_bias, deviation_bias, etc.) is strong
    evidence of that design. Returns None otherwise.

    ``overall_rob`` is intentionally excluded: that column holds the
    worst rating from any methodology stored on ``papers`` (RoB 2,
    QUADAS-2, ROBINS-I), so a non-empty value is not RoB 2-specific.
    Only the five per-domain columns are uniquely RoB 2.
    """
    cochrane_fields = (
        "randomization_bias", "deviation_bias", "missing_outcome_bias",
        "measurement_bias", "reporting_bias",
    )
    for f in cochrane_fields:
        val = paper.get(f)
        if isinstance(val, str) and val.strip():
            return "rct_parallel"
    return None


def _match_mesh(mesh_terms: Iterable[str]) -> str | None:
    """Return the first design slug matched by any MeSH term, else None."""
    lowered = [m.lower() for m in mesh_terms]
    for slug, phrases in _MESH_DESIGN_RULES:
        for phrase in phrases:
            if phrase.lower() in lowered:
                return slug
    return None


def _match_keywords(haystack: str) -> str | None:
    """Return the first design slug matched by a title/abstract keyword, else None."""
    for slug, pattern in _KEYWORD_RULES:
        if pattern.search(haystack):
            return slug
    return None


def detect(paper: dict) -> str:
    """Classify a paper into a study-design slug.

    Priority:
        1. Stored Cochrane RoB 2 ratings imply ``rct_parallel``.
        2. MeSH publication-type phrases.
        3. Title + abstract keyword cues.
        4. Otherwise ``"unknown"``.

    The returned slug is guaranteed to be in
    :data:`biasbuster.methodologies.base.STUDY_DESIGN_SLUGS`.
    """
    cochrane = _cochrane_signal(paper)
    if cochrane is not None:
        return cochrane

    mesh_terms = _normalise_mesh(paper.get("mesh_terms"))
    mesh_hit = _match_mesh(mesh_terms)
    if mesh_hit is not None:
        return mesh_hit

    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    kw_hit = _match_keywords(f"{title}\n{abstract}")
    if kw_hit is not None:
        return kw_hit

    assert "unknown" in STUDY_DESIGN_SLUGS
    return "unknown"
