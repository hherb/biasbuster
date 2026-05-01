"""Per-domain RoB 2 decision algorithms (Cochrane Handbook chapter 8).

Each function takes a dict of signalling-question answers
(``Y`` / ``PY`` / ``PN`` / ``N`` / ``NI``) and returns one of
``low`` | ``some_concerns`` | ``high``. These rules are the same as those
embedded in the system prompts (`prompts.py`); the functions exist so
they can be invoked in code as well — for example to derive a
judgement when an LLM emits valid signalling answers but omits the
``judgement`` field, a known schema-drift mode in the
:mod:`biasbuster.methodologies.cochrane_rob2.prompts` flow (see
CLAUDE.md note on PMID 36101416).

The encoded rules:

  D1 (randomization process)
    low: Q1.1 yes/PY AND Q1.2 yes/PY AND Q1.3 no/PN
    high: Q1.1 no/PN OR Q1.2 no/PN OR Q1.3 yes/PY
    some_concerns: everything else (including NI in key questions)

  D2 (deviations from intended interventions, ITT variant)
    low: Q2.1 no/PN AND Q2.2 no/PN AND Q2.6 yes/PY
    high: Q2.5 no/PN OR Q2.7 yes/PY
    some_concerns: otherwise

  D3 (missing outcome data)
    low: Q3.1 yes/PY OR Q3.2 yes/PY
    high: Q3.4 yes/PY
    some_concerns: otherwise

  D4 (measurement of the outcome)
    low: Q4.1 no/PN AND Q4.2 no/PN AND Q4.3 no/PN
    high: Q4.1 yes/PY OR Q4.2 yes/PY OR Q4.5 yes/PY
    some_concerns: otherwise

  D5 (selection of the reported result)
    low: Q5.1 yes/PY AND Q5.2 no/PN AND Q5.3 no/PN
    high: Q5.2 yes/PY OR Q5.3 yes/PY
    some_concerns: otherwise

  Synthesis (overall, worst-wins)
    high if any domain is "high"
    else some_concerns if any domain is "some_concerns"
    else low
"""

from __future__ import annotations

from typing import Iterable

# Sets used by every domain's rules. ``Y``/``PY`` count as affirmative;
# ``N``/``PN`` count as negative; ``NI`` is "no information" and falls
# through to the residual ``some_concerns`` bucket per the Cochrane
# rule "everything else (including NI in key questions)".
_YES_PY = frozenset({"Y", "PY"})
_NO_PN = frozenset({"N", "PN"})


def _norm(answer: str | None) -> str:
    """Uppercase + strip; missing/None becomes empty string (treated as NI)."""
    return (answer or "").strip().upper()


# --- Per-domain rules ----------------------------------------------------

def domain_1_randomization(a: dict[str, str]) -> str:
    """Domain 1 — bias arising from the randomization process."""
    q11 = _norm(a.get("1.1"))
    q12 = _norm(a.get("1.2"))
    q13 = _norm(a.get("1.3"))
    if q11 in _YES_PY and q12 in _YES_PY and q13 in _NO_PN:
        return "low"
    if q11 in _NO_PN or q12 in _NO_PN or q13 in _YES_PY:
        return "high"
    return "some_concerns"


def domain_2_deviations(a: dict[str, str]) -> str:
    """Domain 2 — deviations from intended interventions (ITT variant)."""
    q21 = _norm(a.get("2.1"))
    q22 = _norm(a.get("2.2"))
    q25 = _norm(a.get("2.5"))
    q26 = _norm(a.get("2.6"))
    q27 = _norm(a.get("2.7"))
    if q21 in _NO_PN and q22 in _NO_PN and q26 in _YES_PY:
        return "low"
    if q25 in _NO_PN or q27 in _YES_PY:
        return "high"
    return "some_concerns"


def domain_3_missing_data(a: dict[str, str]) -> str:
    """Domain 3 — missing outcome data."""
    q31 = _norm(a.get("3.1"))
    q32 = _norm(a.get("3.2"))
    q34 = _norm(a.get("3.4"))
    if q31 in _YES_PY or q32 in _YES_PY:
        return "low"
    if q34 in _YES_PY:
        return "high"
    return "some_concerns"


def domain_4_measurement(a: dict[str, str]) -> str:
    """Domain 4 — measurement of the outcome."""
    q41 = _norm(a.get("4.1"))
    q42 = _norm(a.get("4.2"))
    q43 = _norm(a.get("4.3"))
    q45 = _norm(a.get("4.5"))
    if q41 in _NO_PN and q42 in _NO_PN and q43 in _NO_PN:
        return "low"
    if q41 in _YES_PY or q42 in _YES_PY or q45 in _YES_PY:
        return "high"
    return "some_concerns"


def domain_5_reporting(a: dict[str, str]) -> str:
    """Domain 5 — selection of the reported result."""
    q51 = _norm(a.get("5.1"))
    q52 = _norm(a.get("5.2"))
    q53 = _norm(a.get("5.3"))
    if q51 in _YES_PY and q52 in _NO_PN and q53 in _NO_PN:
        return "low"
    if q52 in _YES_PY or q53 in _YES_PY:
        return "high"
    return "some_concerns"


# --- Dispatch maps -------------------------------------------------------

# Domain code as used in the benchmark_judgment.domain column
# (``d1`` .. ``d5``).
DOMAIN_ALGORITHMS_BY_CODE: dict[str, callable] = {
    "d1": domain_1_randomization,
    "d2": domain_2_deviations,
    "d3": domain_3_missing_data,
    "d4": domain_4_measurement,
    "d5": domain_5_reporting,
}

# Domain slug as emitted by the LLM in its JSON ``domain`` field
# (matches the slugs used in `prompts.py` ``_with_shape`` calls).
DOMAIN_ALGORITHMS_BY_SLUG: dict[str, callable] = {
    "randomization": domain_1_randomization,
    "deviations_from_interventions": domain_2_deviations,
    "missing_outcome_data": domain_3_missing_data,
    "outcome_measurement": domain_4_measurement,
    "selection_of_reported_result": domain_5_reporting,
}


# --- Synthesis (overall) -------------------------------------------------

def synthesis_worst_wins(domain_judgements: Iterable[str]) -> str:
    """Apply Cochrane's worst-wins rule to derive the overall judgement.

    Accepts any iterable of domain judgements (e.g. a list, the values
    of a ``{domain_code: judgement}`` dict). Returns ``low`` only if
    every domain is ``low``; ``high`` if any is ``high``; otherwise
    ``some_concerns``.
    """
    values = list(domain_judgements)
    if "high" in values:
        return "high"
    if "some_concerns" in values:
        return "some_concerns"
    return "low"


# --- Convenience wrapper -------------------------------------------------

def derive_domain_judgement(domain_code_or_slug: str,
                             signalling_answers: dict[str, str]) -> str | None:
    """Apply the appropriate algorithm and return the derived judgement.

    Returns None if the domain identifier is not recognised. The function
    accepts either the short code (``d1``) or the slug
    (``randomization``) so callers don't have to translate.
    """
    fn = DOMAIN_ALGORITHMS_BY_CODE.get(domain_code_or_slug)
    if fn is None:
        fn = DOMAIN_ALGORITHMS_BY_SLUG.get(domain_code_or_slug)
    if fn is None:
        return None
    return fn(signalling_answers)
