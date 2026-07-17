"""Clean study-id → trial-PMID resolver for RoB table rows.

Implements REBUILD_DESIGN.md §5 / FORENSICS.md §6.3: resolve a table row's
study identifier to a trial PMID using only high-confidence evidence —
a bracketed reference number, or an author+year match confirmed by title
similarity. The anti-patterns that corrupted the earlier corpus are
deliberately absent: no surname-only matching (FORENSICS §3.3), no
"first of many" PubMed search result (FORENSICS §3.4). An ambiguous row
resolves to ``unresolved`` and is dropped by the caller, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

#: Minimum title similarity to accept an author+year candidate.
TITLE_SIMILARITY_THRESHOLD = 0.70

_STUDY_ID_RE = re.compile(r"([A-Z][A-Za-z'\-]+)\s+((?:19|20)\d{2})\s*(.*)")


@dataclass(frozen=True)
class Reference:
    """One entry from a review's reference list."""
    ref_number: str
    pmid: str
    first_author: str
    year: str
    title: str


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one study id to a PMID."""
    pmid: str
    method: str
    similarity: float


def _similarity(a: str, b: str) -> float:
    """Return a 0..1 fuzzy-match ratio between two strings, case/space-insensitive."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def resolve_study_pmid(
    study_id: str,
    ref_number: str,
    references: list[Reference],
    *,
    threshold: float = TITLE_SIMILARITY_THRESHOLD,
) -> Resolution:
    """Resolve ``study_id`` to a trial PMID with a recorded method+confidence.

    Priority: (1) bracket reference number → direct PMID; (2) author+year
    candidates disambiguated by title similarity ≥ ``threshold``. If neither
    yields a confident match, returns an ``unresolved`` Resolution.
    """
    by_number = {r.ref_number: r for r in references if r.ref_number}
    if ref_number and ref_number in by_number and by_number[ref_number].pmid:
        return Resolution(by_number[ref_number].pmid, "bracket_ref", 1.0)

    m = _STUDY_ID_RE.match(study_id.strip())
    if not m:
        return Resolution("", "unresolved", 0.0)
    author, year, tail = m.group(1).lower(), m.group(2), m.group(3).strip()

    candidates = [
        r for r in references
        if r.first_author.lower() == author and r.year == year and r.pmid
    ]
    if not candidates:
        return Resolution("", "unresolved", 0.0)
    if len(candidates) == 1 and tail:
        # single author+year hit, but still require title evidence to accept
        sim = _similarity(tail, candidates[0].title)
        if sim >= threshold:
            return Resolution(candidates[0].pmid, "author_year_title", sim)
        return Resolution("", "unresolved", sim)

    # Multiple candidates (or no title tail): pick best title match ≥ threshold
    best, best_sim = None, 0.0
    for c in candidates:
        sim = _similarity(tail, c.title)
        if sim > best_sim:
            best, best_sim = c, sim
    if best is not None and best_sim >= threshold:
        return Resolution(best.pmid, "author_year_title", best_sim)
    return Resolution("", "unresolved", best_sim)
