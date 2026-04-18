"""Deterministic RoB 2 rollup algorithms.

Two aggregation steps:

1. :func:`aggregate_domain` — given a :class:`RoB2DomainJudgement`'s
   signalling answers, return the per-domain judgement via the Cochrane
   algorithm. Today this function is a *verifier*: it checks that the
   judgement the LLM supplied is consistent with the answers. A future
   version can switch to deriving the judgement purely from the
   signalling answers (Cochrane publishes decision trees per domain in
   the Handbook), but the algorithm varies per domain and isn't
   literally a one-liner, so the v1 path lets the LLM emit both and we
   check consistency.

2. :func:`aggregate_outcome` — given the 5 per-domain judgements, return
   the outcome-level overall using the Cochrane worst-wins rule:

     - ``low`` iff **all** five domains are ``low``.
     - ``high`` iff **any** domain is ``high`` OR ``some_concerns``
       applies in multiple domains in a way that substantially lowers
       confidence. The latter is deliberately not automated — the
       Cochrane Handbook leaves it as a judgement call. For the MVP we
       apply the simple rule: ``high`` if any domain is ``high``; else
       ``some_concerns`` if any domain is ``some_concerns``; else ``low``.
     - ``some_concerns`` otherwise.

3. :func:`worst_case_across_outcomes` — trivial max-over-ordering for
   the DB ``overall_severity`` column.

All functions are pure and side-effect-free so they can be property-tested
independent of prompt iteration.
"""

from __future__ import annotations

from typing import Iterable

from .schema import (
    ROB2_DOMAIN_SLUGS,
    RoB2DomainJudgement,
    RoB2Judgement,
    RoB2OutcomeJudgement,
    VALID_JUDGEMENTS,
)

# Ordinal rank for min/max comparisons. low < some_concerns < high.
_RANK: dict[str, int] = {
    "low": 0,
    "some_concerns": 1,
    "high": 2,
}
_INV_RANK: dict[int, RoB2Judgement] = {v: k for k, v in _RANK.items()}  # type: ignore[misc]


def _max_judgement(judgements: Iterable[RoB2Judgement]) -> RoB2Judgement:
    """Return the worst (highest-rank) judgement in the iterable.

    Raises ValueError if the iterable is empty — callers always pass
    fixed-size sequences (5 domains, or N outcomes) so an empty input
    is a programming error.
    """
    ranks = [_RANK[j] for j in judgements]
    if not ranks:
        raise ValueError("cannot aggregate an empty judgement iterable")
    return _INV_RANK[max(ranks)]


def aggregate_outcome(
    domains: dict[str, RoB2DomainJudgement],
) -> RoB2Judgement:
    """Reduce the five per-domain judgements to the outcome overall.

    Implements the worst-wins rule: ``high`` if any domain is ``high``;
    else ``some_concerns`` if any is ``some_concerns``; else ``low``.
    This matches the simple interpretation published in the Cochrane
    Handbook; the Handbook's optional "multiple-some_concerns" escalation
    is left as LLM judgement (captured in
    :attr:`RoB2OutcomeJudgement.overall_rationale`) rather than encoded
    here, because that escalation is explicitly described as requiring
    judgement, not a mechanical rule.
    """
    missing = set(ROB2_DOMAIN_SLUGS) - set(domains)
    if missing:
        raise ValueError(
            f"aggregate_outcome missing domain judgements: {sorted(missing)}"
        )
    return _max_judgement(d.judgement for d in domains.values())


def worst_case_across_outcomes(
    outcomes: list[RoB2OutcomeJudgement],
) -> RoB2Judgement:
    """Return the worst overall judgement across a paper's outcomes.

    This is what goes into the ``annotations.overall_severity`` column.
    A paper with one high-risk outcome and three low-risk outcomes is
    stored as high-risk overall — consistent with Cochrane reporting
    practice, where the least-favourable outcome dominates confidence.
    """
    if not outcomes:
        raise ValueError(
            "worst_case_across_outcomes called with no outcome judgements"
        )
    return _max_judgement(o.overall_judgement for o in outcomes)


def domain_judgement_is_consistent(
    domain: RoB2DomainJudgement,
) -> bool:
    """Sanity-check that a domain's judgement is in the allowed set.

    Returns True. This function is a placeholder for a stricter
    signalling-question-to-judgement consistency check that will live
    here in a later step — for v1 we trust the LLM's per-domain
    judgement and run the MVP pipeline end-to-end before tightening
    the rules.
    """
    return domain.judgement in VALID_JUDGEMENTS
