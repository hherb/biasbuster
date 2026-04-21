"""Rollup helpers for QUADAS-2 assessments.

QUADAS-2 deliberately does not publish a deterministic per-domain
algorithm (unlike Cochrane RoB 2). The signalling questions inform
expert judgement; the tool authors want users to synthesise, not
mechanically rollup. We therefore only expose the *paper-level*
rollup here — across the four domain ratings — which is a pragmatic
worst-wins summary used in common QUADAS-2 reporting practice.

Two rollups:

- :func:`worst_bias` — across the four domains' ``bias_rating``.
- :func:`worst_applicability` — across the three domains that carry
  an applicability rating (patient selection, index test, reference
  standard). Flow-and-timing has no applicability dimension.

The nominal-with-unclear vocabulary (low / high / unclear) is treated
as partially ordered:

- ``high`` dominates ``unclear`` dominates ``low``.

Rationale: for reporting purposes "any domain is high risk" is the
most conservative summary; "all low and one unclear" is noticeably
worse than "all low" because unclear hides something, even if it
doesn't confirm a problem. This ordering is used only for the rollup
summary — per-domain metrics in the faithfulness harness treat the
three values as nominal (no distance between low and unclear vs.
low and high).
"""

from __future__ import annotations

from typing import Iterable

from .schema import (
    QUADAS2_APPLICABILITY_DOMAINS,
    QUADAS2_DOMAIN_SLUGS,
    QUADAS2DomainJudgement,
    QUADASRating,
)

# Partial-order rank for the rollup: high (2) > unclear (1) > low (0).
_RATING_RANK: dict[str, int] = {
    "low": 0,
    "unclear": 1,
    "high": 2,
}
_INV_RANK: dict[int, QUADASRating] = {
    v: k for k, v in _RATING_RANK.items()  # type: ignore[misc]
}


def _max_rating(ratings: Iterable[QUADASRating]) -> QUADASRating:
    """Return the worst (highest-rank) rating in the iterable."""
    ranks = [_RATING_RANK[r] for r in ratings]
    if not ranks:
        raise ValueError("cannot aggregate an empty rating iterable")
    return _INV_RANK[max(ranks)]


def worst_bias(
    domains: dict[str, QUADAS2DomainJudgement],
) -> QUADASRating:
    """Return the worst bias rating across the four QUADAS-2 domains."""
    missing = set(QUADAS2_DOMAIN_SLUGS) - set(domains)
    if missing:
        raise ValueError(
            f"worst_bias missing domain(s): {sorted(missing)}"
        )
    return _max_rating(d.bias_rating for d in domains.values())


def worst_applicability(
    domains: dict[str, QUADAS2DomainJudgement],
) -> QUADASRating:
    """Return the worst applicability rating across domains 1-3.

    Flow-and-timing (domain 4) carries no applicability rating and is
    deliberately excluded from this rollup — see
    :data:`schema.QUADAS2_APPLICABILITY_DOMAINS`.
    """
    relevant = [
        domains[slug].applicability
        for slug in QUADAS2_APPLICABILITY_DOMAINS
        if slug in domains and domains[slug].applicability is not None
    ]
    if len(relevant) != len(QUADAS2_APPLICABILITY_DOMAINS):
        raise ValueError(
            "worst_applicability requires all three applicability-"
            "carrying domains (patient_selection, index_test, "
            "reference_standard) to have non-None applicability ratings; "
            f"got {len(relevant)}."
        )
    return _max_rating(relevant)  # type: ignore[arg-type]
