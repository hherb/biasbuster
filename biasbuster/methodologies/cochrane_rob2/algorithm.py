"""Deterministic RoB 2 *rollup* algorithms (domains → outcome → paper).

NOTE — two similarly named modules, do not confuse them:
  * ``algorithm.py`` (this file) — the **rollup** direction: reduce the
    five per-domain judgements to an outcome overall, and outcomes to a
    paper overall, via the worst-wins rule. Also verifies that a domain
    judgement is consistent with its signalling answers.
  * ``algorithms.py`` (plural) — the **per-domain decision rules**: map a
    single domain's signalling answers (Y/PY/PN/N/NI) to its judgement
    (:func:`.algorithms.derive_domain_judgement`).

Contents:

1. :func:`aggregate_outcome` — given the 5 per-domain judgements, return
   the outcome-level overall using the Cochrane worst-wins rule:

     - ``low`` iff **all** five domains are ``low``.
     - ``high`` iff **any** domain is ``high`` OR ``some_concerns``
       applies in multiple domains in a way that substantially lowers
       confidence. The latter is deliberately not automated — the
       Cochrane Handbook leaves it as a judgement call. For the MVP we
       apply the simple rule: ``high`` if any domain is ``high``; else
       ``some_concerns`` if any domain is ``some_concerns``; else ``low``.
     - ``some_concerns`` otherwise.

2. :func:`worst_case_across_outcomes` — trivial max-over-ordering for
   the DB ``overall_severity`` column.

3. :func:`domain_judgement_is_consistent` — verifies that a domain's
   emitted judgement matches the per-domain truth table
   (:func:`.algorithms.derive_domain_judgement`) applied to its
   signalling answers, per the schema contract that a domain judgement
   "must be reproducible from the signalling inputs alone."

All functions are pure and side-effect-free so they can be property-tested
independent of prompt iteration.
"""

from __future__ import annotations

from typing import Iterable

from .algorithms import derive_domain_judgement
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
    """Check a domain's judgement is reproducible from its signalling answers.

    Per the schema contract (:class:`.schema.RoB2DomainJudgement`) the
    per-domain judgement "must be reproducible from the signalling inputs
    alone." This recomputes the judgement from the signalling answers via
    the Cochrane per-domain truth table
    (:func:`.algorithms.derive_domain_judgement`) and compares.

    Returns False if the judgement is outside the allowed set, or if it
    contradicts the truth-table result. Returns True when they agree, or
    when the truth table cannot derive a judgement for this domain (an
    unrecognised domain slug) — there is then nothing to contradict.
    """
    if domain.judgement not in VALID_JUDGEMENTS:
        return False
    derived = derive_domain_judgement(domain.domain, domain.signalling_answers)
    if derived is None:
        return True
    return derived == domain.judgement
