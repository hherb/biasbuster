"""
Retraction Watch Collector

Downloads retracted papers from Crossref (which now hosts the Retraction Watch database).
Extracts PMIDs, DOIs, retraction reasons, and fetches abstracts from PubMed.

The Retraction Watch database is available:
1. As a CSV download from Crossref Labs
2. Via the Crossref REST API (filter=update-type:retraction)
3. On Kaggle (historical snapshot)

We use approach (2) for targeted queries and (1) for bulk access.
"""

import asyncio
import csv
import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from utils.retry import fetch_with_retry

logger = logging.getLogger(__name__)


# Re-export XML parsing functions for backwards compatibility
from collectors.pubmed_xml import parse_pubmed_xml, parse_pubmed_xml_batch

# Default timeouts (seconds)
DEFAULT_REQUEST_TIMEOUT = 30.0
BULK_DOWNLOAD_TIMEOUT = 120.0


@dataclass
class RetractedPaper:
    doi: str = ""
    pmid: str = ""
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: Optional[int] = None
    retraction_doi: Optional[str] = None
    retraction_reasons: list[str] = field(default_factory=list)
    retraction_source: str = ""  # "retraction_watch", "publisher", or "both"
    subjects: list[str] = field(default_factory=list)


class RetractionWatchCollector:
    """Collect retracted papers with their abstracts for training data."""

    def __init__(self, mailto: str, ncbi_api_key: str = "") -> None:
        """Initialise the collector with contact info and optional NCBI key.

        Args:
            mailto: Contact email required by Crossref and NCBI polite-access
                    policies.
            ncbi_api_key: Optional NCBI API key for higher PubMed rate limits.
        """
        self.mailto = mailto
        self.ncbi_api_key = ncbi_api_key
        self.crossref_base = "https://api.crossref.org/v1"
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "RetractionWatchCollector":
        """Create the shared ``httpx.AsyncClient`` used for all requests."""
        self.client = httpx.AsyncClient(timeout=DEFAULT_REQUEST_TIMEOUT)
        return self

    async def __aexit__(self, *args) -> None:
        """Close the underlying HTTP client gracefully."""
        if self.client:
            await self.client.aclose()

    async def fetch_retractions_from_crossref(
        self,
        max_results: int = 1000,
        subject_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch retracted papers from Crossref REST API.
        Uses filter=update-type:retraction to get Retraction Watch + publisher data.
        """
        papers = []
        cursor = "*"
        rows_per_page = 100

        while len(papers) < max_results:
            params = {
                "filter": "update-type:retraction",
                "rows": min(rows_per_page, max_results - len(papers)),
                "cursor": cursor,
                "mailto": self.mailto,
                "select": "DOI,title,subject,published-print,published-online,"
                          "container-title,update-to,relation",
            }
            if subject_filter:
                params["filter"] += f",subject:{subject_filter}"

            url = f"{self.crossref_base}/works"
            resp = await fetch_with_retry(
                self.client, "GET", url,
                params=params, max_retries=3, base_delay=1.0,
            )
            if resp.status_code != 200:
                logger.error(f"Crossref API error: {resp.status_code}")
                break

            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                break

            papers.extend(items)
            cursor = data.get("message", {}).get("next-cursor", "")
            if not cursor:
                break

            # Crossref polite-pool delay between paginated requests
            await asyncio.sleep(0.5)

        logger.info(f"Fetched {len(papers)} retracted papers from Crossref")
        return papers

    async def download_retraction_watch_csv(self) -> list[dict]:
        """
        Download the full Retraction Watch dataset as CSV from Crossref Labs.
        URL: https://api.labs.crossref.org/data/retractionwatch?{mailto}

        Returns parsed rows with retraction reasons, DOIs, etc.
        """
        url = f"https://api.labs.crossref.org/data/retractionwatch?{self.mailto}"
        logger.info(f"Downloading Retraction Watch CSV (this may take a while)...")

        resp = await fetch_with_retry(
            self.client, "GET", url,
            timeout=httpx.Timeout(BULK_DOWNLOAD_TIMEOUT),
            max_retries=3, base_delay=1.0,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to download RW CSV: {resp.status_code}")
            return []

        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        logger.info(f"Downloaded {len(rows)} entries from Retraction Watch CSV")
        return rows

    async def fetch_pubmed_abstract(self, pmid: str) -> Optional[dict]:
        """Fetch a single abstract from PubMed E-utilities."""
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key

        try:
            url = f"{self.pubmed_base}/efetch.fcgi"
            resp = await fetch_with_retry(
                self.client, "GET", url,
                params=params, max_retries=3, base_delay=1.0,
            )
            if resp.status_code == 200:
                return self._parse_pubmed_xml(resp.text)
        except Exception as e:
            logger.warning(f"PubMed fetch failed for {pmid}: {e}")
        return None

    async def fetch_pubmed_abstracts_batch(
        self, pmids: list[str], batch_size: int = 200
    ) -> dict[str, dict]:
        """Fetch abstracts in batches from PubMed."""
        results = {}
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            id_str = ",".join(batch)
            params = {
                "db": "pubmed",
                "id": id_str,
                "rettype": "abstract",
                "retmode": "xml",
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            try:
                url = f"{self.pubmed_base}/efetch.fcgi"
                resp = await fetch_with_retry(
                    self.client, "GET", url,
                    params=params, max_retries=3, base_delay=1.0,
                )
                if resp.status_code == 200:
                    batch_results = self._parse_pubmed_xml_batch(resp.text)
                    results.update(batch_results)
            except Exception as e:
                logger.warning(f"PubMed batch fetch failed: {e}")

            # NCBI rate-limit: ~3 req/s without key, ~10 req/s with key
            await asyncio.sleep(0.35)

        logger.info(f"Fetched {len(results)} abstracts from PubMed")
        return results

    async def doi_to_pmid(self, dois: list[str]) -> dict[str, str]:
        """Convert DOIs to PMIDs using the NCBI ID Converter API."""
        doi_to_pmid_map = {}
        for i in range(0, len(dois), 100):
            batch = dois[i : i + 100]
            ids_param = ",".join(batch)
            try:
                url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                resp = await fetch_with_retry(
                    self.client, "GET", url,
                    params={
                        "ids": ids_param,
                        "format": "json",
                        "tool": "biasbuster",
                        "email": self.mailto,
                    },
                    max_retries=3, base_delay=1.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for record in data.get("records", []):
                        doi = record.get("doi", "")
                        pmid = record.get("pmid", "")
                        if doi and pmid:
                            doi_to_pmid_map[doi] = pmid
            except Exception as e:
                logger.warning(f"DOI-to-PMID conversion failed: {e}")

            # NCBI rate-limit: ~3 req/s without key, ~10 req/s with key
            await asyncio.sleep(0.35)

        return doi_to_pmid_map

    def _parse_pubmed_xml(self, xml_text: str) -> Optional[dict]:
        """Parse a single PubMed XML response.

        Args:
            xml_text: Raw XML from PubMed efetch.

        Returns:
            Article dict or None on failure.
        """
        return parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml_batch(self, xml_text: str) -> dict[str, dict]:
        """Parse batch PubMed XML response.

        Args:
            xml_text: Raw XML from PubMed efetch (multiple articles).

        Returns:
            Dict mapping PMIDs to article dicts.
        """
        return parse_pubmed_xml_batch(xml_text)

    async def collect_retracted_with_abstracts(
        self,
        max_papers: int = 1000,
        require_abstract: bool = True,
    ) -> list[RetractedPaper]:
        """
        Full pipeline: get retracted DOIs, convert to PMIDs, fetch abstracts.
        Returns RetractedPaper objects ready for bias labelling.
        """

        # Step 1: Get retraction data from Crossref
        crossref_data = await self.fetch_retractions_from_crossref(max_results=max_papers)

        # Extract DOIs
        dois = [item.get("DOI", "") for item in crossref_data if item.get("DOI")]
        logger.info(f"Found {len(dois)} DOIs from retracted papers")

        # Step 2: Extract original paper DOIs from retraction notices.
        # Crossref update-type:retraction returns retraction NOTICES.
        # The `update-to` field points to the ORIGINAL paper we want.
        results = []
        pending_pmid_lookup = []  # Papers awaiting abstract fetch
        for i, item in enumerate(crossref_data):
            notice_doi = item.get("DOI", "")
            if not notice_doi:
                continue

            # Extract original paper DOI from update-to relationship.
            # Crossref update-type:retraction returns retraction NOTICES;
            # the `update-to` field points to the ORIGINAL paper we want.
            original_doi = ""
            retraction_reasons: list[str] = []
            update_to = item.get("update-to", [])
            for update in update_to:
                if update.get("type") == "retraction":
                    original_doi = update.get("DOI", "")
                    label = update.get("label", "")
                    if label:
                        retraction_reasons.append(label)

            # Use original paper DOI if available, fall back to notice DOI
            lookup_doi = original_doi or notice_doi

            # Search PubMed by the original paper's DOI
            params = {
                "db": "pubmed",
                "term": f'"{lookup_doi}"[doi]',
                "retmax": 1,
                "retmode": "json",
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            try:
                url = f"{self.pubmed_base}/esearch.fcgi"
                resp = await fetch_with_retry(
                    self.client, "GET", url,
                    params=params, max_retries=3, base_delay=1.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    if pmids:
                        paper = RetractedPaper(
                            doi=lookup_doi,
                            pmid=pmids[0],
                            title=(item.get("title", [""])[0]
                                   if isinstance(item.get("title"), list)
                                   else str(item.get("title", ""))),
                            journal=(item.get("container-title", [""])[0]
                                     if isinstance(item.get("container-title"), list)
                                     else ""),
                            retraction_doi=notice_doi,
                            retraction_reasons=retraction_reasons,
                        )

                        pending_pmid_lookup.append(paper)
            except Exception as e:
                logger.warning(f"Search failed for DOI {lookup_doi}: {e}")

            # Fetch abstracts in batches
            if len(pending_pmid_lookup) >= 50:
                batch_results = await self._fetch_and_merge_abstracts(
                    pending_pmid_lookup, require_abstract
                )
                results.extend(batch_results)
                pending_pmid_lookup = []

            if i % 10 == 0:
                # Longer pause every 10th request to stay within Crossref polite limits
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.15)

            if len(results) >= max_papers:
                break

        # Process remaining
        if pending_pmid_lookup:
            batch_results = await self._fetch_and_merge_abstracts(
                pending_pmid_lookup, require_abstract
            )
            results.extend(batch_results)

        logger.info(f"Collected {len(results)} retracted papers")
        return results

    async def _fetch_and_merge_abstracts(
        self,
        papers: list[RetractedPaper],
        require_abstract: bool,
    ) -> list[RetractedPaper]:
        """Batch fetch abstracts from PubMed and merge into paper objects."""
        pmids_to_fetch = [p.pmid for p in papers if p.pmid]
        abstract_data = await self.fetch_pubmed_abstracts_batch(pmids_to_fetch)

        merged = []
        for paper in papers:
            if paper.pmid in abstract_data:
                article = abstract_data[paper.pmid]
                paper.abstract = article.get("abstract", "")
                paper.title = article.get("title", "") or paper.title
                if require_abstract and not paper.abstract:
                    continue
                merged.append(paper)
        return merged

async def main():
    """Demo collection run."""
    import sys

    mailto = sys.argv[1] if len(sys.argv) > 1 else "demo@example.com"

    async with RetractionWatchCollector(mailto=mailto) as collector:
        papers = await collector.collect_retracted_with_abstracts(max_papers=10)
        for p in papers:
            print(f"PMID:{p.pmid} | {p.title[:80]}...")
            print(f"  Abstract length: {len(p.abstract)}")
            print(f"  Retraction source: {p.retraction_source}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
