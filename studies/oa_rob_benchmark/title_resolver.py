"""Resolve a trial's PubMed PMID from its title (ROBoto2 ingest, spec §6.1 step 2).

ROBoto2 records carry no PMID/DOI — only a title (plus abstract/authors).
Admitting a trial therefore requires resolving its title to a PubMed record
*confidently*, because a wrong resolution silently binds the expert's RoB 2
labels to an unrelated paper — the exact wrong-document substitution the
OA-first benchmark exists to prevent (see ``project_oa_first_rob_benchmark``).

The resolver is deliberately conservative, mirroring
``study_pmid_resolver``'s "no first-of-many" rule:

* PubMed ``esearch`` on the ``[Title]`` field yields a short candidate list.
* Each candidate's *actual* PubMed title is fetched and compared to the query
  title by fuzzy ratio (``study_pmid_resolver._similarity``).
* A match is accepted only if the best candidate clears
  ``TITLE_MATCH_THRESHOLD`` **and** is unambiguous — i.e. no *other* candidate
  is within ``AMBIGUITY_MARGIN`` of it (two near-equal high scores mean a
  duplicate/near-duplicate publication we cannot safely disambiguate on title
  alone, so we reject rather than guess).

``select_best_title_match`` is the pure, unit-tested core (no network).
``resolve_pmid_by_title`` wraps it with PubMed I/O and is exercised only via
the stubbed ingest test (its live calls run terminal-only, per CLAUDE.md's
>2-minute rule). All network calls go through ``fetch_with_retry`` (retry +
exponential backoff, CLAUDE.md network rule).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from biasbuster.collectors.pubmed_xml import parse_pubmed_xml_batch
from biasbuster.collectors.study_pmid_resolver import _similarity
from biasbuster.utils.retry import RetryExhaustedError, fetch_with_retry

logger = logging.getLogger(__name__)

#: Minimum full-title fuzzy-match ratio to accept a candidate. Higher than
#: ``study_pmid_resolver.TITLE_SIMILARITY_THRESHOLD`` (0.70) because that
#: threshold compares a *title fragment* to a title, whereas here both sides
#: are complete titles, so a genuine match scores much closer to 1.0.
TITLE_MATCH_THRESHOLD = 0.90

#: If the second-best candidate is within this margin of the best, the match
#: is treated as ambiguous and rejected (never "first of many").
AMBIGUITY_MARGIN = 0.05

#: How many PubMed candidates to consider per title.
ESEARCH_RETMAX = 5

#: Characters that are PubMed query syntax (field-tag brackets, phrase quotes)
#: and would corrupt a title search if passed through literally.
_QUERY_SYNTAX_CHARS = str.maketrans({"[": " ", "]": " ", '"': " "})


def _title_query(title: str) -> str:
    """Build the PubMed ``esearch`` term for a title.

    Uses an UNQUOTED ``[Title]`` field search, NOT a ``"..."[Title]`` phrase
    match: the phrase index is too brittle for these titles (hyphenation and
    punctuation differences make the exact phrase miss — empirically 0 hits
    across the ROBoto2 set), whereas the word-level field search recalls the
    right paper. Loose recall is safe here because ``select_best_title_match``
    then verifies each candidate against the full title at
    ``TITLE_MATCH_THRESHOLD`` — real matches score ~0.99, so a loosely-related
    hit is still rejected. Field-tag brackets and quotes are stripped so a
    title can't inject query syntax.
    """
    cleaned = " ".join(title.translate(_QUERY_SYNTAX_CHARS).split())
    return f"{cleaned}[Title]"

#: Resolution method tag recorded on admitted items (store.resolution_method).
RESOLUTION_METHOD = "title_search"


@dataclass(frozen=True)
class TitleResolution:
    """Outcome of resolving one title to a PMID.

    ``pmid`` is empty when unresolved; ``similarity`` is the best score seen
    (0.0 when there were no candidates at all); ``reason`` names why an
    unresolved title failed, for reject logging.
    """
    pmid: str
    similarity: float
    reason: str


def select_best_title_match(
    query_title: str,
    candidates: dict[str, str],
    *,
    threshold: float = TITLE_MATCH_THRESHOLD,
    margin: float = AMBIGUITY_MARGIN,
) -> TitleResolution:
    """Pick the best PMID for ``query_title`` from ``{pmid: candidate_title}``.

    Pure (no I/O). Accepts the highest-scoring candidate only if it clears
    ``threshold`` and no other candidate is within ``margin`` of it. Returns
    an unresolved ``TitleResolution`` (empty ``pmid``) with a ``reason`` of
    ``no_candidates`` / ``below_threshold`` / ``ambiguous_title`` otherwise.
    """
    if not query_title.strip() or not candidates:
        return TitleResolution("", 0.0, "no_candidates")

    scored = sorted(
        ((pmid, _similarity(query_title, title)) for pmid, title in candidates.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best_pmid, best_sim = scored[0]
    if best_sim < threshold:
        return TitleResolution("", best_sim, "below_threshold")
    # ``candidates`` is keyed by PMID, so the runner-up is always a distinct
    # paper: a near-equal second score means a duplicate/near-duplicate
    # publication we cannot safely disambiguate on title alone → reject.
    if len(scored) > 1 and (best_sim - scored[1][1]) < margin:
        return TitleResolution("", best_sim, "ambiguous_title")
    return TitleResolution(best_pmid, best_sim, RESOLUTION_METHOD)


async def resolve_pmid_by_title(
    client: httpx.AsyncClient,
    title: str,
    *,
    pubmed_base: str,
    ncbi_api_key: str = "",
    threshold: float = TITLE_MATCH_THRESHOLD,
    retmax: int = ESEARCH_RETMAX,
) -> TitleResolution:
    """Resolve ``title`` to a PubMed PMID via ``esearch`` + title verification.

    Returns an unresolved ``TitleResolution`` (never raises) when the title
    is empty, PubMed returns no candidates, or no candidate clears the
    confidence bar — the ingest logs these as rejects and moves on.
    """
    if not title.strip():
        return TitleResolution("", 0.0, "empty_title")

    esearch_params = {
        "db": "pubmed",
        "term": _title_query(title),
        "retmax": retmax,
        "retmode": "json",
    }
    if ncbi_api_key:
        esearch_params["api_key"] = ncbi_api_key
    try:
        resp = await fetch_with_retry(
            client, "GET", f"{pubmed_base}/esearch.fcgi",
            params=esearch_params, max_retries=3, base_delay=1.0,
        )
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
    except (httpx.HTTPError, RetryExhaustedError, ValueError) as exc:
        # ``fetch_with_retry`` raises ``RetryExhaustedError`` (not an
        # ``httpx.HTTPError``) once retries are spent — the common transient
        # case (NCBI 429 rate-limit / 5xx) this resolver must fail closed on,
        # so one unreachable PubMed never aborts the whole ingest run.
        logger.warning("title esearch failed for %r: %s", title, exc)
        return TitleResolution("", 0.0, "esearch_failed")
    if not pmids:
        return TitleResolution("", 0.0, "no_candidates")

    efetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if ncbi_api_key:
        efetch_params["api_key"] = ncbi_api_key
    try:
        resp = await fetch_with_retry(
            client, "GET", f"{pubmed_base}/efetch.fcgi",
            params=efetch_params, max_retries=3, base_delay=1.0,
        )
        articles = parse_pubmed_xml_batch(resp.text)
    except (httpx.HTTPError, RetryExhaustedError) as exc:
        # See the esearch handler above: a spent-retries ``RetryExhaustedError``
        # must also fail closed here rather than propagate out of the ingest.
        logger.warning("candidate efetch failed for %r: %s", title, exc)
        return TitleResolution("", 0.0, "efetch_failed")

    candidates = {
        pmid: str(article.get("title") or "")
        for pmid, article in articles.items()
    }
    return select_best_title_match(title, candidates, threshold=threshold)
