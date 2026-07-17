"""Assemble and validate the complete six-field RoB 2 tuple (spec §4.2).

Two entry points, one output contract: a ``RoB2Tuple`` where every field is
one of ``CANONICAL_LEVELS`` — or ``None``, meaning the row is incomplete or
invalid and must be rejected (never persisted as a partial tuple).

* ``tuple_from_table_row`` — for the structural extractor path (EM candidates,
  Stage B). Normalises ``rob_table_extractor``'s ``some_concerns`` to the
  canonical ``some concerns`` and prefers cell-text over CSS colour.
* ``tuple_from_signalling`` — for ROBoto2, whose expert labels are recorded as
  signalling-question answers; domain judgements are derived deterministically
  by the RoB 2 algorithm (``derive_domain_judgement``), which faithfully
  represents the expert's assessment.

Two domain-key spaces meet here and must not be conflated:

* ``rob_table_extractor.ROB2.domain_keywords`` (and therefore every
  ``ExtractedRating.domain`` value) uses the canonical extractor names
  ``randomization`` / ``deviations`` / ``missing_outcome`` / ``measurement``
  / ``reporting``.
* ``biasbuster.methodologies.cochrane_rob2.algorithms.derive_domain_judgement``
  accepts either a RoB 2 code (``d1``..``d5``, via
  ``DOMAIN_ALGORITHMS_BY_CODE``) or its own slug vocabulary (via
  ``DOMAIN_ALGORITHMS_BY_SLUG``), and those slugs
  (``deviations_from_interventions``, ``missing_outcome_data``,
  ``outcome_measurement``, ``selection_of_reported_result``) do NOT match the
  extractor names above (only ``randomization`` happens to coincide).

``_DOMAINS`` pairs each extractor name with its RoB 2 code so callers can key
``domain_answers`` by the extractor name (as Stage A/B data naturally is) while
this module translates to the code internally before calling
``derive_domain_judgement``.
"""
from __future__ import annotations

from dataclasses import dataclass

from biasbuster.collectors.rob_table_extractor import ExtractedStudyRow
from biasbuster.methodologies.cochrane_rob2.algorithms import (
    derive_domain_judgement, synthesis_worst_wins,
)

CANONICAL_LEVELS: tuple[str, ...] = ("low", "some concerns", "high")

_ALIASES: dict[str, str] = {
    "low": "low", "low risk": "low",
    "some concerns": "some concerns", "some_concerns": "some concerns",
    "unclear": "some concerns",
    "high": "high", "high risk": "high",
}

# (extractor canonical domain name, RoB 2 domain code) — see module docstring
# for why these two key spaces must be kept separate.
_DOMAINS: tuple[tuple[str, str], ...] = (
    ("randomization", "d1"),
    ("deviations", "d2"),
    ("missing_outcome", "d3"),
    ("measurement", "d4"),
    ("reporting", "d5"),
)


@dataclass(frozen=True)
class RoB2Tuple:
    """A complete, canonical six-field RoB 2 judgement."""
    overall: str
    d1: str
    d2: str
    d3: str
    d4: str
    d5: str


def normalise_level(raw: str) -> str:
    """Map a raw rating to one of ``CANONICAL_LEVELS`` or ``""`` (invalid)."""
    t = (raw or "").strip().lower()
    return _ALIASES.get(t, "")


def tuple_from_table_row(row: ExtractedStudyRow) -> RoB2Tuple | None:
    """Build a complete tuple from an extractor row, or None if incomplete."""
    by_domain = {r.domain: (r.rating_text or r.rating_colour) for r in row.domains}
    levels = [normalise_level(by_domain.get(name, "")) for name, _code in _DOMAINS]
    overall = normalise_level(
        (row.overall.rating_text or row.overall.rating_colour) if row.overall else ""
    )
    if not overall or any(lv not in CANONICAL_LEVELS for lv in levels):
        return None
    return RoB2Tuple(overall, *levels)


def tuple_from_signalling(
    domain_answers: dict[str, dict[str, str]],
    overall_answers: dict[str, str] | None,
) -> RoB2Tuple | None:
    """Build a tuple from per-domain signalling answers via the RoB 2 algorithm.

    ``domain_answers`` maps an extractor canonical domain name (see
    ``_DOMAINS``) to that domain's signalling-question answers. Each domain's
    judgement is derived by calling ``derive_domain_judgement`` with the
    corresponding RoB 2 code (``d1``..``d5``) — not the extractor name and not
    ``algorithms.py``'s own (inconsistent) slug vocabulary. Overall is taken
    from ``overall_answers`` if the source records it, else derived by
    worst-wins over the five domains (RoB 2's own rule).

    A ``domain_answers`` missing any of the five domains (or holding an
    empty answers dict for one) is treated as an incomplete row and
    rejected with ``None`` — ``derive_domain_judgement`` would otherwise
    silently fall through to ``some_concerns`` for an absent domain
    (no signalling answer fires either its low or high trigger), masking
    the missing assessment as a plausible-looking judgement.

    ``synthesis_worst_wins`` is fed the RAW (underscore-form, e.g.
    ``"some_concerns"``) judgements returned by ``derive_domain_judgement``
    — its own native vocabulary — never the canonical space-form
    (``"some concerns"``) stored in the returned tuple's per-domain fields.
    Feeding it the canonical form would make its ``"some_concerns" in
    values`` membership check never match, silently downgrading the
    overall to ``"low"`` whenever the worst domain was some-concerns and
    none was high.
    """
    if any(not domain_answers.get(name) for name, _code in _DOMAINS):
        return None
    raw_judgements: list[str] = []
    levels: list[str] = []
    for name, code in _DOMAINS:
        judged = derive_domain_judgement(code, domain_answers[name])
        lv = normalise_level(judged or "")
        if lv not in CANONICAL_LEVELS:
            return None
        raw_judgements.append(judged or "")
        levels.append(lv)
    if overall_answers is not None:
        overall = normalise_level(overall_answers.get("overall", ""))
    else:
        overall = normalise_level(synthesis_worst_wins(raw_judgements))
    if overall not in CANONICAL_LEVELS:
        return None
    return RoB2Tuple(overall, *levels)
