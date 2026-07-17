"""Single source of truth for RCTs excluded from the benchmark analysis.

**Wrong-paper acquisitions** are a class (see ``WRONG_PAPER_RCTS`` below), not
just RCT030: for each, Phase 1 resolved a *different document* than the primary
trial EM cited (a parent Cochrane review, a systematic review, a protocol, a
sub-analysis, or the wrong platform arm). RCT030 is the canonical example —
Phase 1 resolved the parent Cochrane review (PMID 37131928, CD013127.PUB2)
instead of the underlying trial (Hung MS et al. 2021, *Collegian*, not
PubMed-indexed). Every model judgement for a wrong-paper RCT describes a
different document than the Cochrane ground truth it is scored against, so every
model-vs-Cochrane pair for it is spurious and must not enter any κ computation.

Code paths that must honour this exclusion, all importing from here so the set
is defined once:

* ``recover_parse_failures.py`` — must never *recover* a wrong-paper
  parse-failure row into a valid judgement.
* the κ scripts that score **our four models** against Cochrane
  (``compute_phase6_kappa.py``, ``interim_analysis.py``,
  ``temperature_analysis.py``) — must never *count* a wrong-paper row in a
  model-vs-Cochrane or run-to-run pair.

Deliberately **not** applied in ``sanity_check_kappa.py``: that script
reproduces Eisele-Metzger 2025's *published* κ ≈ 0.22 from EM's own Claude 2
judgements (``em_claude2_*``) against the Cochrane ground truth. The
wrong-paper problem is specific to *our* Phase-1 full-text acquisition; EM's
Claude 2 data and the Cochrane RoB 2 assessment are both for the correct
primary trial, so excluding RCT030 there would deviate from EM's methodology
and corrupt the reproduction check.

See ``benchmark_rct.notes`` for RCT030 and GitHub issue #29 (the analysis-path
exclusion was previously enforced only in the recovery path). The wrong-paper
*rows* themselves remain in the DB tagged ``valid=1``; this exclusion is applied
at query time, so it is robust to whatever is stored.
"""
from __future__ import annotations

# RCTs whose Phase-1 acquisition resolved the WRONG DOCUMENT — a class, not just
# RCT030. Found by the 2026-07-17 completeness audit + manual review (issue #29;
# see audit_wrong_paper_acquisitions.py and recovery_obtainability.md). Every
# model judgement for these describes a different document than the Cochrane
# ground truth it is scored against, so all model-vs-Cochrane pairs for them are
# spurious. The **primary** analysis (hybrid design, owner-approved 2026-07-17)
# excludes ALL of these. Kept as a frozenset so callers can do O(1) membership
# tests (recovery guard) and build a SQL fragment (analysis) from one source.
WRONG_PAPER_RCTS: frozenset[str] = frozenset({
    "RCT008",  # systematic review, not the Jolly COPD RCT
    "RCT009",  # TTM2 oxygen sub-analysis, not the temperature main paper
    "RCT017",  # ERAS study protocol, not the results paper
    "RCT019",  # fluocinolone 3-year results, not the 12-month paper
    "RCT030",  # parent Cochrane review, not the Hung MHFA trial
    "RCT040",  # insulin antibody 104-wk study, not the 52-wk efficacy RCT
    "RCT064",  # PRET-PD cognition report, not the 2-year main RCT
    "RCT074",  # Steps Ahead study protocol, not the results paper
    "RCT080",  # Scandinavian mortality stats, not the kindergarten RCT
    "RCT088",  # concrete-engineering paper, not the calcifediol COVID RCT
    "RCT093",  # RECOVERY empagliflozin arm, not the intended aspirin arm
    "RCT095",  # STOIC mechanistic sub-study, not the budesonide RCT
    "RCT100",  # pooled four-trial analysis, not the single COVE trial
})

# Of the wrong-paper set, the RCTs whose CORRECT primary trial is not obtainable
# (not PubMed-indexed) — excluded even in the recovered-corpus **sensitivity**
# analysis. The rest are recoverable (see recover_wrong_papers.MANUAL_PMIDS and
# recovery_obtainability.md); recovering them and re-assessing is the owner-gated
# secondary analysis. Until that re-assessment exists, only the primary (exclude
# ALL of WRONG_PAPER_RCTS) is computed.
UNRECOVERABLE_WRONG_PAPER_RCTS: frozenset[str] = frozenset({
    "RCT030",  # Hung MS et al 2021, Collegian — not PubMed-indexed
    "RCT080",  # Rogde et al 2016, education journal — not PubMed-indexed
})


def wrong_paper_filter(*aliases: str) -> tuple[str, list[str]]:
    """SQL fragment + bound params excluding ``WRONG_PAPER_RCTS`` rows.

    Mirrors the alias convention of ``compute_phase6_kappa._fallback_filter``:
    pass the table alias used for each ``benchmark_judgment`` reference in the
    query (e.g. ``"a"``, ``"b"``); pass ``""`` for an unaliased table. The
    returned fragment starts with ``" AND "`` so it can be concatenated
    directly onto an existing ``WHERE`` clause.

    Returns ``("", [])`` when the exclusion set is empty (so callers stay
    correct if ``WRONG_PAPER_RCTS`` is ever cleared). Uses bound parameters —
    never string-interpolated ids — so the caller appends the returned params
    to its own parameter tuple in the same left-to-right order the aliases are
    passed.

    Example::

        sql, params = wrong_paper_filter("a", "b")
        conn.execute(base_sql + sql, (source_a, source_b, domain, *params))
    """
    if not WRONG_PAPER_RCTS:
        return "", []
    ids = sorted(WRONG_PAPER_RCTS)
    placeholders = ",".join("?" * len(ids))
    parts: list[str] = []
    params: list[str] = []
    for alias in aliases:
        col = f"{alias}.rct_id" if alias else "rct_id"
        parts.append(f"{col} NOT IN ({placeholders})")
        params.extend(ids)
    return " AND " + " AND ".join(parts), params
