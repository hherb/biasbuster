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

logger = logging.getLogger(__name__)


# ---- Standalone PubMed XML parsing functions ----

def parse_pubmed_xml(xml_text: str) -> Optional[dict]:
    """Parse a single PubMed XML response."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
        article = root.find(".//PubmedArticle")
        if article is None:
            return None
        return _extract_article_data(article)
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
        return None


def parse_pubmed_xml_batch(xml_text: str) -> dict[str, dict]:
    """Parse batch PubMed XML response. Returns {pmid: article_dict}."""
    import xml.etree.ElementTree as ET
    results = {}
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            data = _extract_article_data(article)
            if data and data.get("pmid"):
                results[data["pmid"]] = data
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
    return results


def _extract_article_data(article) -> Optional[dict]:
    """Extract key fields from a PubmedArticle XML element."""
    try:
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find(".//Article")
        if art is None:
            return None

        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Abstract - may have multiple AbstractText elements (structured)
        abstract_parts = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

        abstract = "\n".join(abstract_parts)

        # Journal
        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        # Year
        year_el = medline.find(".//DateCompleted/Year")
        if year_el is None:
            year_el = art.find(".//Journal/JournalIssue/PubDate/Year")
        year = int(year_el.text) if year_el is not None else None

        # Authors with affiliations
        authors = []
        for author in art.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            affils = [a.text for a in author.findall(".//Affiliation") if a.text]
            authors.append({
                "last": last.text if last is not None else "",
                "first": first.text if first is not None else "",
                "affiliations": affils,
            })

        # Funding / grants
        grants = []
        for grant in medline.findall(".//Grant"):
            gid = grant.find("GrantID")
            agency = grant.find("Agency")
            grants.append({
                "id": gid.text if gid is not None else "",
                "agency": agency.text if agency is not None else "",
            })

        # DOI
        doi = ""
        for eid in art.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""

        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        return {
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors,
            "grants": grants,
            "mesh_terms": mesh_terms,
        }
    except Exception as e:
        logger.warning(f"Article extraction error: {e}")
        return None


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

    def __init__(self, mailto: str, ncbi_api_key: str = ""):
        self.mailto = mailto
        self.ncbi_api_key = ncbi_api_key
        self.crossref_base = "https://api.crossref.org/v1"
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
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

            resp = await self.client.get(
                f"{self.crossref_base}/works", params=params
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

            # Rate limiting
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

        resp = await self.client.get(url, timeout=120.0)
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
            resp = await self.client.get(
                f"{self.pubmed_base}/efetch.fcgi", params=params
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
                resp = await self.client.get(
                    f"{self.pubmed_base}/efetch.fcgi", params=params
                )
                if resp.status_code == 200:
                    batch_results = self._parse_pubmed_xml_batch(resp.text)
                    results.update(batch_results)
            except Exception as e:
                logger.warning(f"PubMed batch fetch failed: {e}")

            await asyncio.sleep(0.35)  # Rate limit

        logger.info(f"Fetched {len(results)} abstracts from PubMed")
        return results

    async def doi_to_pmid(self, dois: list[str]) -> dict[str, str]:
        """Convert DOIs to PMIDs using the NCBI ID Converter API."""
        doi_to_pmid_map = {}
        for i in range(0, len(dois), 100):
            batch = dois[i : i + 100]
            ids_param = ",".join(batch)
            try:
                resp = await self.client.get(
                    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                    params={
                        "ids": ids_param,
                        "format": "json",
                        "tool": "biasbuster",
                        "email": self.mailto,
                    },
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

            await asyncio.sleep(0.35)

        return doi_to_pmid_map

    def _parse_pubmed_xml(self, xml_text: str) -> Optional[dict]:
        """Parse a single PubMed XML response."""
        return parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml_batch(self, xml_text: str) -> dict[str, dict]:
        """Parse batch PubMed XML response."""
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

        # Step 2: Search PubMed for these DOIs to get PMIDs and abstracts
        # (More reliable than the ID converter for batch lookups)
        results = []
        for i, item in enumerate(crossref_data):
            doi = item.get("DOI", "")
            if not doi:
                continue

            # Search PubMed by DOI
            params = {
                "db": "pubmed",
                "term": f'"{doi}"[doi]',
                "retmax": 1,
                "retmode": "json",
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            try:
                resp = await self.client.get(
                    f"{self.pubmed_base}/esearch.fcgi", params=params
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    if pmids:
                        paper = RetractedPaper(
                            doi=doi,
                            pmid=pmids[0],
                            title=(item.get("title", [""])[0]
                                   if isinstance(item.get("title"), list)
                                   else str(item.get("title", ""))),
                            journal=(item.get("container-title", [""])[0]
                                     if isinstance(item.get("container-title"), list)
                                     else ""),
                        )

                        # Parse retraction sources
                        update_to = item.get("update-to", [])
                        for update in update_to:
                            if update.get("type") == "retraction":
                                source = update.get("source", "")
                                paper.retraction_source = source

                        results.append(paper)
            except Exception as e:
                logger.warning(f"Search failed for DOI {doi}: {e}")

            if i % 10 == 0:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.15)

            if len(results) >= max_papers:
                break

        # Step 3: Batch fetch abstracts
        pmids_to_fetch = [p.pmid for p in results if p.pmid]
        abstract_data = await self.fetch_pubmed_abstracts_batch(pmids_to_fetch)

        # Merge abstracts back
        final_results = []
        for paper in results:
            if paper.pmid in abstract_data:
                article = abstract_data[paper.pmid]
                paper.abstract = article.get("abstract", "")
                paper.title = article.get("title", "") or paper.title
                if require_abstract and not paper.abstract:
                    continue
                final_results.append(paper)

        logger.info(
            f"Collected {len(final_results)} retracted papers with abstracts"
        )
        return final_results

    def save_results(self, papers: list[RetractedPaper], output_path: Path):
        """Save collected papers as JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for paper in papers:
                f.write(json.dumps(asdict(paper)) + "\n")
        logger.info(f"Saved {len(papers)} papers to {output_path}")


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
