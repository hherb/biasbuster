"""
Outcome matching utilities for comparing registered vs published trial outcomes.

Pure text-processing functions extracted from clinicaltrials_gov.py for
detecting outcome switching between ClinicalTrials.gov registrations and
published abstracts.

These functions operate on strings and dataclass instances — they make no
network calls and can be tested without I/O.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)


def extract_outcome_terms(outcome_text: str) -> list[str]:
    """Extract clinically meaningful terms from an outcome description.

    Filters out common biomedical filler words and returns both single-word
    terms and short bigrams that are likely to carry clinical meaning.

    Args:
        outcome_text: Free-text outcome measure string, e.g.
            ``"Change from baseline in HbA1c at 24 weeks"``.

    Returns:
        A list of meaningful single-word tokens followed by up to three
        informative bigrams extracted from the input.
    """
    stopwords = {
        "change", "from", "baseline", "in", "at", "the", "of", "and",
        "or", "to", "as", "by", "with", "for", "on", "time", "rate",
        "proportion", "percentage", "number", "score", "level", "levels",
        "measured", "assessed", "defined", "weeks", "months", "years",
        "week", "month", "year", "day", "days",
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', outcome_text)
    terms = [w for w in words if w.lower() not in stopwords]

    # Also extract multi-word medical terms (simple bigrams)
    bigrams: list[str] = []
    word_list = outcome_text.split()
    for i in range(len(word_list) - 1):
        bigram = f"{word_list[i]} {word_list[i+1]}".strip(".,;:()")
        if len(bigram) > 5 and not all(w.lower() in stopwords for w in bigram.split()):
            bigrams.append(bigram)

    return terms + bigrams[:3]


def extract_published_primary(abstract: str) -> list[str]:
    """Attempt to extract what the abstract declares as its primary outcome.

    Uses regex heuristics to find phrases such as *"primary outcome was ..."*
    or *"primary endpoint: ..."*.

    Args:
        abstract: Full text of the published abstract.

    Returns:
        Up to three candidate primary-outcome strings found in the abstract.
        An empty list is returned when no pattern matches.
    """
    patterns = [
        r'[Pp]rimary\s+(?:outcome|endpoint|end\s*point|measure)\s+(?:was|were|is)\s+([^.]+)',
        r'[Pp]rimary\s+(?:outcome|endpoint|end\s*point)\s*[:\-]\s*([^.]+)',
        r'[Tt]he\s+primary\s+(?:outcome|endpoint)\s+(?:of\s+)?([^.]+)',
    ]

    results: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, abstract)
        results.extend(matches)

    return results[:3]


async def batch_check_outcome_switching(
    abstracts: list[dict],
) -> list[OutcomeSwitchingReport]:
    """Check a batch of abstracts for outcome switching.

    For each abstract the function first attempts to locate an NCT ID
    (embedded in the text or via a title-keyword search), then queries
    ClinicalTrials.gov and compares registered versus published outcomes.

    Imports :class:`ClinicalTrialsGovCollector` and
    :class:`OutcomeSwitchingReport` at call time to avoid a circular
    import with ``clinicaltrials_gov.py``.

    Args:
        abstracts: A list of dictionaries, each containing at least some of
            the keys ``pmid``, ``title``, ``abstract``, ``nct_id``, and
            ``doi``.

    Returns:
        A list of :class:`OutcomeSwitchingReport` instances — one per
        abstract for which a matching trial registration was found.
    """
    from collectors.clinicaltrials_gov import (
        ClinicalTrialsGovCollector,
        OutcomeSwitchingReport,
    )

    async with ClinicalTrialsGovCollector() as collector:
        reports: list[OutcomeSwitchingReport] = []

        for item in abstracts:
            nct_id = item.get("nct_id", "")
            if not nct_id:
                # Try to extract from abstract text
                nct_id = await collector.extract_nct_from_abstract(
                    item.get("abstract", "")
                )

            if not nct_id:
                # Try title-based search
                candidates = await collector.search_by_title_keywords(
                    item.get("title", "")
                )
                if candidates:
                    nct_id = candidates[0].nct_id

            if nct_id:
                report = await collector.detect_outcome_switching(
                    nct_id=nct_id,
                    published_abstract=item.get("abstract", ""),
                    published_title=item.get("title", ""),
                    pmid=item.get("pmid", ""),
                    doi=item.get("doi", ""),
                )
                reports.append(report)

            await asyncio.sleep(0.5)  # Rate limit

        return reports
