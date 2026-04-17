"""PubMed PublicationType classification and batch fetch.

Single source of truth for trial / non-trial / ambiguous classification
of PubMed-indexed papers. Shared between:

- ``scripts/audit_publication_types.py`` — post-hoc audit tool
- ``biasbuster/collectors/cochrane_rob_v2.py`` — pre-persist gate in
  the rebuilt Cochrane collector (REBUILD_DESIGN.md §6)

The classification rule is deliberately conservative. A paper is only
admitted as a trial if PubMed explicitly tags it with an RCT or
clinical-trial PublicationType. Ambiguous cases (only ``Journal
Article``) are surfaced as ``ambiguous`` so callers can decide whether
to accept them — the rebuild collector rejects ambiguous rows when
the source is a CDSR RoB 2 table because a valid RoB 2 assessment
targets a trial, not a commentary.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterable

import httpx

from biasbuster.utils.retry import fetch_with_retry

logger = logging.getLogger(__name__)

PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# PubMed PublicationType strings that positively identify a randomised or
# controlled trial. Any one of these is sufficient.
TRIAL_TYPES: frozenset[str] = frozenset({
    "Randomized Controlled Trial",
    "Controlled Clinical Trial",
    "Clinical Trial",
    "Clinical Trial, Phase I",
    "Clinical Trial, Phase II",
    "Clinical Trial, Phase III",
    "Clinical Trial, Phase IV",
    "Pragmatic Clinical Trial",
    "Equivalence Trial",
    "Adaptive Clinical Trial",
})

# PublicationTypes that disqualify a paper from being treated as a trial.
# Any presence of these (without an overriding TRIAL_TYPES tag) → non_trial.
NON_TRIAL_TYPES: frozenset[str] = frozenset({
    "Letter",
    "Editorial",
    "Comment",
    "News",
    "Congresses",
    "Meeting Abstracts",
    "Personal Narratives",
    "Biography",
    "Autobiography",
    "Book",
    "Published Erratum",
    "Retracted Publication",
    "Retraction of Publication",
    "Address",
    "Lectures",
    "Portrait",
    "Case Reports",
    "Systematic Review",
    "Meta-Analysis",
    "Review",
})

# HTTP retry / rate-limit knobs. Callers may override via kwargs on
# ``fetch_publication_types``.
PUBMED_BATCH_SIZE = 200
HTTP_TIMEOUT_SECONDS = 60.0
DEFAULT_USER_AGENT = "biasbuster/pubtype"


def classify(publication_types: list[str] | None) -> str:
    """Classify a paper by its PubMed PublicationType list.

    Args:
        publication_types: List of PubMed PublicationType strings for one
            article. ``None`` or empty → ``"unknown"``.

    Returns:
        ``"trial"`` if any TRIAL_TYPES tag is present without a
        conflicting NON_TRIAL_TYPES tag; ``"non_trial"`` if only
        disqualifying tags are present; ``"ambiguous"`` if the list has
        no positive or negative indicator (e.g. only ``Journal
        Article``); ``"unknown"`` if the list is empty/missing.
    """
    if not publication_types:
        return "unknown"
    tset = set(publication_types)
    has_trial = bool(tset & TRIAL_TYPES)
    has_nontrial = bool(tset & NON_TRIAL_TYPES)
    if has_trial and not has_nontrial:
        return "trial"
    if has_trial and has_nontrial:
        # Conflicting tags (e.g. "RCT" + "Review") — treat as ambiguous
        # and let the caller decide. Rare in practice but real.
        return "ambiguous"
    if has_nontrial:
        return "non_trial"
    return "ambiguous"


def parse_publication_types(xml_text: str) -> dict[str, list[str]]:
    """Extract ``{pmid: [publication_types]}`` from a PubMed efetch XML.

    The XML may contain any number of ``<PubmedArticle>`` elements.
    Articles without a PMID element are silently skipped (PubMed should
    never emit those, but the parse remains tolerant).

    Args:
        xml_text: Raw response body from PubMed efetch with
            ``rettype=abstract&retmode=xml``.

    Returns:
        Dict mapping PMID to the list of PublicationType strings for
        that article. Empty dict on XML parse failure.
    """
    out: dict[str, list[str]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("PubMed XML parse error: %s", exc)
        return out
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()
        types = [
            (t.text or "").strip()
            for t in article.findall(".//PublicationTypeList/PublicationType")
            if t.text
        ]
        out[pmid] = types
    return out


async def fetch_publication_types(
    pmids: Iterable[str],
    *,
    client: httpx.AsyncClient | None = None,
    batch_size: int = PUBMED_BATCH_SIZE,
    user_agent: str = DEFAULT_USER_AGENT,
    ncbi_api_key: str = "",
) -> dict[str, list[str]]:
    """Fetch PubMed PublicationType lists for a set of PMIDs.

    Issues batched ``efetch`` calls with exponential-backoff retry.
    PMIDs that PubMed doesn't return are simply absent from the result
    dict — callers should treat a missing key as an ``unknown``
    classification.

    Args:
        pmids: PubMed IDs to fetch. Order is preserved in batching.
        client: Optional existing ``httpx.AsyncClient`` to reuse. If
            None, a client is created for the duration of the call.
        batch_size: Max PMIDs per PubMed request (PubMed's documented
            efetch limit is 200).
        user_agent: HTTP User-Agent header. Pass a project-identifying
            string so rate-limit issues are debuggable from NCBI's side.
        ncbi_api_key: Optional NCBI API key. With a key, NCBI permits
            ~10 req/s instead of ~3 req/s.

    Returns:
        Dict mapping PMID to the list of PublicationType strings.
    """
    pmid_list = [str(p) for p in pmids if p]
    if not pmid_list:
        return {}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": user_agent},
        )

    out: dict[str, list[str]] = {}
    try:
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
            if ncbi_api_key:
                params["api_key"] = ncbi_api_key
            try:
                resp = await fetch_with_retry(
                    client, "GET", PUBMED_EFETCH,
                    params=params, max_retries=3, base_delay=2.0,
                )
                resp.raise_for_status()
                out.update(parse_publication_types(resp.text))
            except Exception as exc:  # noqa: BLE001 — surface the error, keep going
                logger.warning(
                    "PubMed batch %d-%d failed: %s",
                    i + 1, i + len(batch), exc,
                )
    finally:
        if owns_client and client is not None:
            await client.aclose()

    return out
