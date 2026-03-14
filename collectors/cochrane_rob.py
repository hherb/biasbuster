"""
Cochrane Risk of Bias Extractor

Harvests expert Risk of Bias (RoB 2) assessments from Cochrane systematic reviews.
These are gold-standard expert judgments on individual trial bias.

Strategy:
1. Search Europe PMC for Cochrane reviews with RoB assessments
2. Extract included studies and their RoB ratings
3. Fetch the original trial abstracts from PubMed
4. Use the Cochrane RoB rating as a ground-truth label

This gives us expert-validated positive (high RoB) and negative (low RoB) examples.

Data sources:
- Europe PMC full-text API (Cochrane reviews are open access)
- PubMed for trial abstracts
- Cochrane Library (supplementary data tables)
"""

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from utils.retry import fetch_with_retry

logger = logging.getLogger(__name__)


@dataclass
class RoBAssessment:
    """Risk of Bias assessment for a single study from a Cochrane review."""
    study_id: str = ""           # Usually "Author Year" format
    pmid: str = ""
    doi: str = ""
    study_title: str = ""

    # RoB 2 domain ratings (low, some_concerns, high)
    randomization_bias: str = ""
    deviation_bias: str = ""       # Deviations from intended interventions
    missing_outcome_bias: str = ""
    measurement_bias: str = ""
    reporting_bias: str = ""       # Selective reporting
    overall_rob: str = ""          # low, some_concerns, high

    # Source Cochrane review
    cochrane_review_pmid: str = ""
    cochrane_review_doi: str = ""
    cochrane_review_title: str = ""

    # Additional context
    rob_notes: str = ""
    domain: str = ""              # Clinical domain (e.g., "cardiovascular")


@dataclass
class CochraneReview:
    """A Cochrane review with its included studies and RoB assessments."""
    pmid: str = ""
    doi: str = ""
    title: str = ""
    included_studies: list[RoBAssessment] = field(default_factory=list)
    total_studies: int = 0
    high_rob_count: int = 0
    low_rob_count: int = 0


class CochraneRoBCollector:
    """
    Extract RoB assessments from Cochrane reviews.

    Approach:
    1. Search Europe PMC for recent Cochrane reviews
    2. Parse full-text for RoB tables and included study references
    3. Map studies to PubMed IDs
    4. Fetch original abstracts
    """

    EUROPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, ncbi_api_key: str = "") -> None:
        """Initialise the collector with an optional NCBI API key for PubMed lookups."""
        self.ncbi_api_key = ncbi_api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "CochraneRoBCollector":
        """Create the shared ``httpx.AsyncClient`` used for all HTTP requests."""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args) -> None:
        """Close the underlying HTTP client."""
        if self.client:
            await self.client.aclose()

    async def search_cochrane_reviews(
        self,
        domain: str = "",
        max_results: int = 100,
        min_year: int = 2018,
    ) -> list[dict]:
        """
        Search Europe PMC for Cochrane reviews with risk of bias assessments.
        """
        query_parts = [
            'SRC:MED',
            '"Cochrane Database of Systematic Reviews"[JOURNAL]',
            f'PUB_YEAR:[{min_year} TO 2026]',
        ]
        if domain:
            query_parts.append(f'"{domain}"')

        # Add terms likely to indicate RoB assessment
        query_parts.append('("risk of bias" OR "quality assessment" OR "RoB 2")')

        query = " AND ".join(query_parts)

        reviews = []
        cursor = "*"
        page_size = 25

        while len(reviews) < max_results:
            try:
                resp = await fetch_with_retry(
                    self.client, "GET",
                    f"{self.EUROPMC_BASE}/search",
                    params={
                        "query": query,
                        "resultType": "core",
                        "format": "json",
                        "pageSize": min(page_size, max_results - len(reviews)),
                        "cursorMark": cursor,
                    },
                )

                if resp.status_code != 200:
                    logger.warning(f"Europe PMC search error: {resp.status_code}")
                    break

                data = resp.json()
                results = data.get("resultList", {}).get("result", [])
                if not results:
                    break

                for result in results:
                    reviews.append({
                        "pmid": result.get("pmid", ""),
                        "doi": result.get("doi", ""),
                        "title": result.get("title", ""),
                        "journal": result.get("journalTitle", ""),
                        "year": result.get("pubYear", ""),
                        "pmcid": result.get("pmcid", ""),
                    })

                cursor = data.get("nextCursorMark", "")
                if not cursor or cursor == "*":
                    break

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Cochrane review search error: {e}")
                break

        logger.info(f"Found {len(reviews)} Cochrane reviews")
        return reviews

    async def extract_included_study_refs(self, pmcid: str) -> list[dict]:
        """
        Extract references to included studies from a Cochrane review's full text.

        Cochrane reviews typically list included studies in a structured format.
        We parse the references section to find cited trials.
        """
        if not pmcid:
            return []

        try:
            # Fetch full-text XML from Europe PMC
            resp = await fetch_with_retry(
                self.client, "GET",
                f"{self.EUROPMC_BASE}/{pmcid}/fullTextXML",
            )

            if resp.status_code != 200:
                logger.debug(f"Full text not available for {pmcid}")
                return []

            # Parse references
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)

            refs = []
            # Look for reference list
            for ref in root.findall(".//ref"):
                ref_data = {}

                # Extract citation details
                for element_id in ref.findall(".//element-citation") + ref.findall(".//mixed-citation"):
                    # Author names
                    authors = []
                    for name in element_id.findall(".//name"):
                        surname = name.find("surname")
                        given = name.find("given-names")
                        if surname is not None:
                            authors.append(
                                f"{surname.text or ''} {(given.text or '') if given is not None else ''}"
                            )

                    # Year
                    year_el = element_id.find("year")
                    year = year_el.text if year_el is not None else ""

                    # Article title
                    title_el = element_id.find("article-title")
                    title = title_el.text if title_el is not None else ""

                    # DOI
                    doi = ""
                    for pub_id in element_id.findall("pub-id"):
                        if pub_id.get("pub-id-type") == "doi":
                            doi = pub_id.text or ""
                        elif pub_id.get("pub-id-type") == "pmid":
                            ref_data["pmid"] = pub_id.text or ""

                    ref_data.update({
                        "authors": authors,
                        "year": year,
                        "title": title,
                        "doi": doi,
                        "ref_id": ref.get("id", ""),
                    })

                if ref_data.get("title") or ref_data.get("doi"):
                    refs.append(ref_data)

            return refs

        except Exception as e:
            logger.warning(f"Reference extraction failed for {pmcid}: {e}")
            return []

    async def extract_rob_from_fulltext(self, pmcid: str) -> list[RoBAssessment]:
        """
        Parse RoB assessments from Cochrane review full text.

        Cochrane reviews include RoB tables/figures. We look for:
        - Structured RoB summary tables
        - Text mentioning "high risk of bias", "low risk of bias"
        - References to specific studies with RoB judgments
        """
        if not pmcid:
            return []

        assessments = []
        try:
            resp = await fetch_with_retry(
                self.client, "GET",
                f"{self.EUROPMC_BASE}/{pmcid}/fullTextXML",
            )

            if resp.status_code != 200:
                return []

            text = resp.text

            # Extract study-level RoB mentions
            # Pattern: "Author Year" followed by risk of bias judgment
            rob_patterns = [
                # "Smith 2020 was judged to be at high risk of bias"
                r'(\w+\s+(?:19|20)\d{2}[a-z]?)\s+(?:was|were)\s+(?:judged|rated|assessed|considered)\s+'
                r'(?:to\s+be\s+)?(?:at\s+)?(\w+)\s+risk\s+of\s+bias',
                # "high risk of bias (Smith 2020)"
                r'(\w+)\s+risk\s+of\s+bias\s*\(([^)]+(?:19|20)\d{2}[a-z]?)\)',
                # Table-style: "Smith 2020 ... High ... Low ..."
                r'(\w+\s+(?:19|20)\d{2}[a-z]?).*?(low|high|unclear|some\s+concerns)\s+risk',
            ]

            for pattern in rob_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) >= 2:
                        study_id = groups[0].strip()
                        rob_level = groups[1].strip().lower()

                        # Normalize
                        if "high" in rob_level:
                            overall = "high"
                        elif "low" in rob_level:
                            overall = "low"
                        elif "some" in rob_level or "unclear" in rob_level:
                            overall = "some_concerns"
                        else:
                            overall = rob_level

                        assessments.append(RoBAssessment(
                            study_id=study_id,
                            overall_rob=overall,
                            cochrane_review_pmid="",  # Will be filled by caller
                        ))

        except Exception as e:
            logger.warning(f"RoB extraction failed for {pmcid}: {e}")

        return assessments

    async def resolve_study_pmids(
        self, assessments: list[RoBAssessment]
    ) -> list[RoBAssessment]:
        """
        Resolve "Author Year" study IDs to PubMed IDs.
        Uses PubMed search by author + year.
        """
        for assessment in assessments:
            if assessment.pmid:
                continue

            # Parse "Author Year" format
            match = re.match(r'(\w+)\s+((?:19|20)\d{2})', assessment.study_id)
            if not match:
                continue

            author = match.group(1)
            year = match.group(2)

            try:
                params = {
                    "db": "pubmed",
                    "term": f"{author}[Author] AND {year}[Date - Publication] AND "
                            f"randomized controlled trial[Publication Type]",
                    "retmax": 5,
                    "retmode": "json",
                }
                if self.ncbi_api_key:
                    params["api_key"] = self.ncbi_api_key

                resp = await fetch_with_retry(
                    self.client, "GET",
                    f"{self.PUBMED_BASE}/esearch.fcgi", params=params,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    if len(pmids) == 1:
                        assessment.pmid = pmids[0]
                    elif pmids:
                        # Multiple matches - would need title matching for disambiguation
                        assessment.pmid = pmids[0]  # Take first as candidate
                        logger.debug(
                            f"Multiple PMID candidates for {assessment.study_id}: {pmids}"
                        )

                await asyncio.sleep(0.35)

            except Exception as e:
                logger.warning(f"PMID resolution failed for {assessment.study_id}: {e}")

        resolved = sum(1 for a in assessments if a.pmid)
        logger.info(f"Resolved {resolved}/{len(assessments)} study IDs to PMIDs")
        return assessments

    async def collect_rob_dataset(
        self,
        domains: Optional[list[str]] = None,
        max_reviews: int = 50,
        max_studies: int = 500,
    ) -> list[RoBAssessment]:
        """
        Full pipeline: search Cochrane reviews, extract RoB, resolve PMIDs.
        Returns RoBAssessments with PMIDs ready for abstract fetching.
        """
        if domains is None:
            domains = [
                "cardiovascular", "diabetes", "cancer",
                "mental health", "respiratory",
            ]

        all_assessments = []

        for domain in domains:
            logger.info(f"Searching Cochrane reviews for '{domain}'...")
            reviews = await self.search_cochrane_reviews(
                domain=domain,
                max_results=max_reviews // len(domains),
            )

            for review in reviews:
                pmcid = review.get("pmcid", "")
                if not pmcid:
                    continue

                # Extract RoB assessments
                assessments = await self.extract_rob_from_fulltext(pmcid)
                for a in assessments:
                    a.cochrane_review_pmid = review.get("pmid", "")
                    a.cochrane_review_doi = review.get("doi", "")
                    a.cochrane_review_title = review.get("title", "")
                    a.domain = domain

                all_assessments.extend(assessments)
                await asyncio.sleep(0.5)

                if len(all_assessments) >= max_studies:
                    break

            if len(all_assessments) >= max_studies:
                break

        # Resolve PMIDs
        all_assessments = await self.resolve_study_pmids(all_assessments)

        # Separate high and low RoB
        high_rob = [a for a in all_assessments if a.overall_rob == "high"]
        low_rob = [a for a in all_assessments if a.overall_rob == "low"]

        logger.info(
            f"Collected {len(all_assessments)} RoB assessments: "
            f"{len(high_rob)} high, {len(low_rob)} low"
        )

        return all_assessments

    def save_results(self, assessments: list[RoBAssessment], output_path: Path) -> None:
        """Save RoB assessments as JSONL to *output_path*."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for a in assessments:
                f.write(json.dumps(asdict(a)) + "\n")
        logger.info(f"Saved {len(assessments)} RoB assessments to {output_path}")


if __name__ == "__main__":
    async def demo():
        async with CochraneRoBCollector() as collector:
            # Search for a few Cochrane reviews
            reviews = await collector.search_cochrane_reviews(
                domain="cardiovascular",
                max_results=3,
            )
            for r in reviews:
                print(f"Review: {r['title'][:80]}...")
                print(f"  PMID: {r['pmid']}, PMC: {r['pmcid']}")

    asyncio.run(demo())
