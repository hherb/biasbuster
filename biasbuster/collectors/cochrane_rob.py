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
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx

from biasbuster.utils.retry import fetch_with_retry

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


def rob_assessment_to_paper_dict(assessment: RoBAssessment) -> dict:
    """Convert a RoBAssessment to a paper dict matching the DB papers schema.

    Pure function — no side effects.  All Cochrane save paths should use
    this to avoid divergent field mapping.

    Mapping:
    - ``study_title`` → ``title``
    - ``source`` set to ``"cochrane_rob"``
    - ``abstract`` defaults to ``""`` (Cochrane entries lack abstracts;
      PubMed fetch populates this later)
    - ``study_id`` and ``rob_notes`` are dropped (no DB columns)
    """
    paper = asdict(assessment)
    paper["source"] = "cochrane_rob"
    paper["title"] = paper.pop("study_title", "")
    paper.pop("rob_notes", None)
    paper.pop("study_id", None)
    paper.setdefault("abstract", "")
    return paper


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

    # Prompt for LLM-based RoB extraction from review full text
    _ROB_EXTRACTION_PROMPT = """\
You are a systematic review data extractor.  Given the full text of a
systematic review or meta-analysis that used the Cochrane Risk of Bias 2
(RoB 2) tool, extract every individual study that received a risk-of-bias
judgment.

For each study, return:
- "study_id": the first author surname and year (e.g. "Smith 2020").
  Strip "et al." and other co-author text — just the first surname and year.
- "overall_rob": one of "high", "low", or "some_concerns"
- "ref_number": the bracketed reference number if present (e.g. 28 for "[28]"),
  or null if not cited by number.
- "randomization_bias": one of "high", "low", "some_concerns", or null if not reported
- "deviation_bias": one of "high", "low", "some_concerns", or null if not reported
- "missing_outcome_bias": one of "high", "low", "some_concerns", or null if not reported
- "measurement_bias": one of "high", "low", "some_concerns", or null if not reported
- "reporting_bias": one of "high", "low", "some_concerns", or null if not reported

The five domain fields correspond to the standard RoB 2 domains:
  D1 = randomization_bias (bias arising from the randomization process)
  D2 = deviation_bias (bias due to deviations from intended interventions)
  D3 = missing_outcome_bias (bias due to missing outcome data)
  D4 = measurement_bias (bias in measurement of the outcome)
  D5 = reporting_bias (bias in selection of the reported result)

Return ONLY a JSON array.  Example:
[
  {"study_id": "Smith 2020", "overall_rob": "high", "ref_number": 12,
   "randomization_bias": "low", "deviation_bias": "some_concerns",
   "missing_outcome_bias": "low", "measurement_bias": "high",
   "reporting_bias": "some_concerns"},
  {"study_id": "Jones 2019", "overall_rob": "low", "ref_number": null,
   "randomization_bias": "low", "deviation_bias": "low",
   "missing_outcome_bias": "low", "measurement_bias": "low",
   "reporting_bias": "low"}
]

If per-domain ratings are not available for a study, set the domain
fields to null — but always include the overall_rob.

If the review does not contain individual study-level RoB judgments,
return an empty array [].

Respond ONLY with the JSON array. No preamble, no markdown fences."""

    # Default path for caching LLM extraction results (avoids re-spending tokens)
    DEFAULT_CACHE_PATH = Path("dataset/llm_rob_cache.json")

    # Skip full-text documents larger than this.  A typical Cochrane review
    # is 30-50 KB of XML; large papers with supplements can reach ~2 MB.
    # Documents far beyond this (e.g. 184 MB books like PMC9429973) would
    # waste enormous LLM tokens with no useful RoB data.
    MAX_FULLTEXT_BYTES = 2_400_000  # 2.4 MB

    def __init__(
        self,
        ncbi_api_key: str = "",
        llm_api_key: str = "",
        llm_api_base: str = "",
        llm_model: str = "",
        llm_max_tokens: int = 16000,
        max_retries: int = 3,
        llm_timeout: float = 180.0,
        cache_path: Optional[Path] = None,
    ) -> None:
        """Initialise the collector.

        Args:
            ncbi_api_key: NCBI E-utilities API key (optional, increases rate limit).
            llm_api_key: API key for the OpenAI-compatible LLM used for RoB extraction.
            llm_api_base: Base URL for the LLM API (e.g. https://api.deepseek.com).
            llm_model: Model ID (e.g. deepseek-reasoner).
            llm_max_tokens: Max output tokens for LLM extraction calls.
            max_retries: Max retry attempts for LLM calls on transient failures.
            llm_timeout: Timeout in seconds for each LLM API call.
            cache_path: Path for LLM extraction result cache.  Set to ``None``
                to use the default (``dataset/llm_rob_cache.json``).
        """
        self.ncbi_api_key = ncbi_api_key
        self.llm_api_key = llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.llm_api_base = llm_api_base.rstrip("/") if llm_api_base else ""
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.max_retries = max_retries
        self.llm_timeout = llm_timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._cache_path = cache_path if cache_path is not None else self.DEFAULT_CACHE_PATH
        self._llm_cache: dict[str, list[dict]] = self._load_cache()
        self._cache_dirty = 0  # number of unsaved entries

    @property
    def cached_pmcids(self) -> set[str]:
        """Return the set of PMCIDs present in the LLM extraction cache."""
        return set(self._llm_cache.keys())

    # Flush cache to disk every N new entries (avoids rewriting on every call)
    _CACHE_FLUSH_INTERVAL = 5

    def _load_cache(self) -> dict[str, list[dict]]:
        """Load the LLM extraction cache from disk."""
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text())
                logger.info(f"Loaded LLM RoB cache: {len(data)} entries from {self._cache_path}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load LLM cache: {e}")
        return {}

    def _save_cache(self, force: bool = False) -> None:
        """Persist the LLM extraction cache to disk.

        Writes are batched: the file is only rewritten every
        ``_CACHE_FLUSH_INTERVAL`` new entries, or when *force* is True
        (e.g. at shutdown).
        """
        self._cache_dirty += 1
        if not force and self._cache_dirty < self._CACHE_FLUSH_INTERVAL:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._llm_cache, indent=1))
            self._cache_dirty = 0
        except OSError as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    async def __aenter__(self) -> "CochraneRoBCollector":
        """Create the shared ``httpx.AsyncClient`` used for all HTTP requests."""
        self.client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, *args) -> None:
        """Close the underlying HTTP client and flush the LLM cache."""
        if self._cache_dirty:
            self._save_cache(force=True)
        if self.client:
            await self.client.aclose()

    async def search_cochrane_reviews(
        self,
        domain: str = "",
        max_results: int = 100,
        min_year: int = 2018,
    ) -> list[dict]:
        """Search Europe PMC for systematic reviews with RoB assessments.

        Searches both Cochrane reviews AND open-access systematic reviews
        in PubMed Central that used the RoB 2 tool.
        """
        query_parts = [
            'SRC:PMC',
            'OPEN_ACCESS:Y',
            'HAS_FT:Y',
            f'PUB_YEAR:[{min_year} TO 2026]',
            '("risk-of-bias" OR "risk of bias 2" OR "RoB 2")',
            '("included studies" OR "randomized controlled" OR "randomized")',
        ]
        if domain:
            query_parts.append(f'"{domain}"')

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

                # Extract the numeric label (e.g. <label>28</label> or id="CR28")
                label_el = ref.find("label")
                if label_el is not None and label_el.text:
                    ref_data["ref_label"] = re.sub(r'\D', '', label_el.text)
                else:
                    # Try to extract number from id attr (e.g. "CR28", "ref-28")
                    ref_id = ref.get("id", "")
                    num_match = re.search(r'(\d+)', ref_id)
                    ref_data["ref_label"] = num_match.group(1) if num_match else ""

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

    # Common words that the regex falsely captures as "author" names.
    _JUNK_AUTHORS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "had", "has", "have", "in", "is", "it", "no", "not", "of", "on",
        "or", "our", "per", "so", "the", "to", "up", "was", "we", "were",
        "with", "all", "also", "any", "been", "both", "but", "can", "did",
        "do", "each", "few", "get", "got", "her", "him", "his", "how",
        "its", "may", "more", "most", "new", "now", "old", "one", "only",
        "own", "pre", "see", "set", "she", "six", "ten", "two", "use",
        "via", "who", "yet",
        # Months
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        # Common non-author tokens found in Cochrane text
        "since", "until", "between", "during", "after", "before", "about",
        "over", "under", "into", "some", "such", "than", "that", "then",
        "them", "they", "this", "very", "what", "when", "here", "just",
        "like", "long", "much", "must", "near", "next", "once", "only",
        "same", "than", "used", "well", "will", "risk", "bias", "high",
        "low", "overall", "study", "trial", "total", "group", "data",
        "analysis", "outcome", "table", "figure", "section", "review",
        "included", "excluded", "results", "methods", "based", "using",
        "report", "compared", "evidence", "quality", "patients",
        "participants", "treatment", "intervention", "control", "placebo",
        "effect", "effects", "significant", "studies", "trials",
        "art", "candidates", "statement", "eular", "criteria",
        "removed", "assessed", "conducted", "reported", "performed",
    })

    # Regex to normalise "Author et al. 2020 [28]" → (author, year, ref_number)
    # Handles: "Smith 2020", "Smith et al. 2020", "Smith et al. [28]",
    #          "Hudson JL, 2020. [45]", "Smith 2020a"
    _STUDY_ID_RE = re.compile(
        r'(\w+)'                              # first-author surname
        r'(?:\s+et\s+al\.?,?)?'               # optional "et al." / "et al,"
        r'(?:\s+[A-Z]{1,3})?'                 # optional initials (e.g. "JL", "AB")
        r'(?:[,.\s]*\s*((?:19|20)\d{2})\w?)?'  # optional year (e.g. 2020, 2020a)
        r'(?:[.\s]*\[(\d+)\])?'               # optional bracket ref [28], skip "." or spaces
    )

    @classmethod
    def _normalize_study_id(cls, study_id: str) -> tuple[str, str, str]:
        """Parse a study ID into (surname, year, ref_number).

        Handles formats like:
          "Smith 2020", "Smith et al. 2020", "Smith et al. [28]",
          "Smith et al., 2020 [28]", "Smith 2020a"

        Returns empty strings for components that aren't present.
        """
        m = cls._STUDY_ID_RE.match(study_id.strip())
        if not m:
            return ("", "", "")
        return (m.group(1) or "", m.group(2) or "", m.group(3) or "")

    @classmethod
    def _is_valid_study_id(cls, study_id: str) -> bool:
        """Check whether a regex-captured study_id looks like a real Author Year."""
        author, year, ref_num = cls._normalize_study_id(study_id)
        if not author:
            return False
        # Need at least a year or a ref number to be useful
        if not year and not ref_num:
            # Fall back to old heuristic: check if there are at least 2 parts
            parts = study_id.strip().split()
            if len(parts) < 2:
                return False
        author_lower = author.lower().rstrip(",;:")
        # Must start with uppercase (proper noun) and not be a common word
        if not author[0].isupper():
            return False
        if author_lower in cls._JUNK_AUTHORS:
            return False
        # Author name should be at least 2 chars
        if len(author_lower) < 2:
            return False
        return True

    async def extract_rob_from_fulltext(self, pmcid: str) -> list[RoBAssessment]:
        """Parse RoB assessments from Cochrane review full text.

        Extracts study-level risk of bias judgments from Cochrane review XML.
        Filters out false-positive study IDs (common words, months, etc.).
        Deduplicates within a single review (same study_id keeps first match).
        """
        if not pmcid:
            return []

        assessments = []
        seen_ids: set[str] = set()
        try:
            resp = await fetch_with_retry(
                self.client, "GET",
                f"{self.EUROPMC_BASE}/{pmcid}/fullTextXML",
            )

            if resp.status_code != 200:
                return []

            text = resp.text

            rob_patterns = [
                # "Smith 2020 was judged to be at high risk of bias"
                r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)\s+(?:was|were)\s+(?:judged|rated|assessed|considered)\s+'
                r'(?:to\s+be\s+)?(?:at\s+)?(\w+)\s+risk\s+of\s+bias',
                # "high risk of bias (Smith 2020)"
                r'(\w+)\s+risk\s+of\s+bias\s*\(([A-Z][a-z]+[^)]*(?:19|20)\d{2}[a-z]?)\)',
                # "Smith 2020: high risk" or "Smith 2020 - high risk"
                r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)\s*[:–\-]\s*(low|high|unclear|some concerns)\s+risk',
                # "rated as high risk of bias: Smith 2020, Jones 2021"
                r'(high|low|unclear)\s+risk\s+of\s+bias[^.]*?([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)',
                # "Smith 2020 ... overall ... high" (table rows, up to 200 chars)
                r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?).{0,200}?overall.{0,50}?(low|high|some\s+concerns)',
            ]

            for pattern in rob_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) < 2:
                        continue

                    study_id = groups[0].strip()
                    rob_level = groups[1].strip().lower()

                    if not self._is_valid_study_id(study_id):
                        continue

                    # Deduplicate within this review
                    sid_key = study_id.lower()
                    if sid_key in seen_ids:
                        continue
                    seen_ids.add(sid_key)

                    # Normalize RoB level
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
                        cochrane_review_pmid="",
                    ))

        except Exception as e:
            logger.warning(f"RoB extraction failed for {pmcid}: {e}")

        return assessments

    async def extract_rob_via_llm(self, pmcid: str, full_text: str) -> list[RoBAssessment]:
        """Extract RoB assessments from review full text using an LLM.

        Sends the review text to an OpenAI-compatible LLM (e.g. DeepSeek
        reasoner) with a structured extraction prompt.  Falls back to
        regex-based extraction if the LLM is not configured.

        Args:
            pmcid: PMC identifier (for logging).
            full_text: Raw XML/text of the review.

        Returns:
            List of RoBAssessment objects extracted by the LLM.
        """
        # Try regex first (fast, free). Fall back to LLM only if regex
        # finds nothing and LLM is configured.
        regex_results = await self.extract_rob_from_fulltext(pmcid)
        if regex_results:
            logger.info(
                f"Regex extracted {len(regex_results)} RoB assessments from {pmcid}"
            )
            return regex_results

        if not self.llm_api_key or not self.llm_api_base or not self.llm_model:
            return []

        # Check LLM cache before spending tokens
        if pmcid in self._llm_cache:
            all_studies = self._llm_cache[pmcid]
            logger.info(
                f"Cache hit for {pmcid}: {len(all_studies)} raw study entries"
            )
        else:
            # Guard against oversized documents (books, supplements)
            ft_bytes = len(full_text.encode("utf-8"))
            if ft_bytes > self.MAX_FULLTEXT_BYTES:
                logger.warning(
                    f"Skipping LLM extraction for {pmcid}: full text too large "
                    f"({ft_bytes / 1_000_000:.1f} MB "
                    f"> {self.MAX_FULLTEXT_BYTES / 1_000_000:.1f} MB limit)"
                )
                return []

            logger.info(f"Regex found 0 in {pmcid}, trying LLM extraction...")

            # Strip XML tags — keep just the text content
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(full_text)
                plain = ET.tostring(root, encoding="unicode", method="text")
            except ET.ParseError:
                plain = re.sub(r"<[^>]+>", " ", full_text)

            # Chunk & map-reduce: split into overlapping chunks, extract from
            # each, merge results.  NEVER truncate — the RoB tables may be
            # anywhere in the document.
            chunk_size = 28_000  # chars per chunk (conservative for token limits)
            overlap = 2_000
            chunks: list[str] = []
            start = 0
            while start < len(plain):
                end = start + chunk_size
                chunks.append(plain[start:end])
                start = end - overlap
            if not chunks:
                chunks = [plain]

            logger.debug(
                f"{pmcid}: {len(plain)} chars -> {len(chunks)} chunk(s) for LLM"
            )

            all_studies: list[dict] = []
            for i, chunk in enumerate(chunks):
                chunk_label = f"{pmcid} chunk {i + 1}/{len(chunks)}"
                studies = await self._llm_extract_chunk(chunk, chunk_label)
                all_studies.extend(studies)

            # Cache raw LLM results for future re-resolution runs
            self._llm_cache[pmcid] = all_studies
            self._save_cache()
            logger.info(
                f"LLM extracted {len(all_studies)} raw study entries from "
                f"{pmcid} ({len(chunks)} chunk(s))"
            )

        # Deduplicate and convert to RoBAssessment objects
        assessments: list[RoBAssessment] = []
        seen: set[str] = set()
        for s in all_studies:
            if not isinstance(s, dict):
                continue
            sid = (s.get("study_id") or "").strip()
            rob = (s.get("overall_rob") or "").strip().lower()
            if not sid or not rob:
                continue

            # If LLM returned a ref_number, ensure it's in the study_id
            # so _normalize_study_id can extract it for bracket-ref lookup
            ref_num = s.get("ref_number")
            if ref_num is not None and f"[{ref_num}]" not in sid:
                sid = f"{sid} [{ref_num}]"

            if not self._is_valid_study_id(sid):
                continue
            sid_key = sid.lower()
            if sid_key in seen:
                continue
            seen.add(sid_key)

            if "high" in rob:
                overall = "high"
            elif "low" in rob:
                overall = "low"
            elif "some" in rob or "unclear" in rob:
                overall = "some_concerns"
            else:
                overall = rob

            # Extract per-domain RoB ratings (may be null/missing)
            domain_ratings = {}
            for dfield in (
                "randomization_bias", "deviation_bias",
                "missing_outcome_bias", "measurement_bias",
                "reporting_bias",
            ):
                raw = (s.get(dfield) or "").strip().lower()
                if not raw or raw == "null":
                    domain_ratings[dfield] = ""
                elif "high" in raw:
                    domain_ratings[dfield] = "high"
                elif "low" in raw:
                    domain_ratings[dfield] = "low"
                elif "some" in raw or "unclear" in raw:
                    domain_ratings[dfield] = "some_concerns"
                else:
                    domain_ratings[dfield] = ""

            assessments.append(RoBAssessment(
                study_id=sid,
                overall_rob=overall,
                **domain_ratings,
            ))

        return assessments

    async def _llm_extract_chunk(
        self, text: str, label: str,
    ) -> list[dict]:
        """Send one chunk to the LLM for RoB extraction.

        Returns a list of dicts with ``study_id`` and ``overall_rob`` keys,
        or an empty list on failure.  Retries on JSON parse errors and
        timeouts up to ``self.max_retries`` times.
        """
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": self._ROB_EXTRACTION_PROMPT},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.llm_max_tokens,
            "temperature": 0.0,
        }

        last_error: str = ""
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.post(
                    f"{self.llm_api_base}/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(self.llm_timeout),
                )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", 2 ** attempt))
                    logger.warning(f"Rate limited for {label}, retrying in {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"LLM RoB extraction failed for {label}: HTTP {resp.status_code}")
                    last_error = f"HTTP {resp.status_code}"
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {})
                text_out = message.get("content", "") or ""

                # DeepSeek reasoner may put the answer in reasoning_content
                # and leave content empty
                if not text_out.strip():
                    text_out = message.get("reasoning_content", "") or ""

                if not text_out:
                    logger.warning(f"LLM returned empty content for {label}")
                    last_error = "empty content"
                    await asyncio.sleep(2 ** attempt)
                    continue

                text_out = text_out.strip()
                if text_out.startswith("```"):
                    text_out = text_out.split("\n", 1)[1] if "\n" in text_out else text_out[3:]
                if text_out.endswith("```"):
                    text_out = text_out[:-3]
                text_out = text_out.strip()

                # If text contains mixed reasoning + JSON, extract the JSON array.
                # Search from the end — the JSON array is typically the last
                # bracket-enclosed block after any reasoning text.
                if not text_out.startswith("["):
                    for array_match in reversed(list(re.finditer(r'\[[\s\S]*?\]', text_out))):
                        candidate = array_match.group(0)
                        try:
                            json.loads(candidate)
                            text_out = candidate
                            break
                        except json.JSONDecodeError:
                            continue

                studies = json.loads(text_out)
                if not isinstance(studies, list):
                    logger.warning(f"LLM returned non-list for {label}: {type(studies)}")
                    last_error = "non-list response"
                    await asyncio.sleep(2 ** attempt)
                    continue
                return studies

            except json.JSONDecodeError as e:
                last_error = f"invalid JSON: {e}"
                logger.warning(
                    f"LLM returned invalid JSON for {label} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning(
                    f"LLM timeout for {label} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                continue
            except Exception as e:
                logger.warning(f"LLM extraction error for {label}: {e}")
                return []

        logger.warning(
            f"All {self.max_retries} attempts failed for {label}: {last_error}"
        )
        return []

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

            # Parse study ID (handles "Author Year", "Author et al. Year", etc.)
            author, year, _ = self._normalize_study_id(assessment.study_id)
            if not author or not year:
                continue

            try:
                # Search without Publication Type filter — Cochrane reviews
                # include studies that PubMed may not tag as RCTs.
                params = {
                    "db": "pubmed",
                    "term": f"{author}[Author] AND {year}[Date - Publication]",
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

    async def resolve_pmids_from_refs(
        self,
        assessments: list[RoBAssessment],
        refs: list[dict],
    ) -> None:
        """Match RoB assessments to reference PMIDs/DOIs extracted from XML.

        Uses three matching strategies (in priority order):
        1. Bracket reference number (e.g. ``[28]`` → ``<ref>`` with label 28)
        2. First-author surname + year
        3. Surname-only (if exactly one ref matches)
        """
        # Build lookups
        ref_by_label: dict[str, dict] = {}          # "28" → ref
        ref_by_author_year: dict[tuple[str, str], dict] = {}  # ("smith", "2020") → ref
        refs_by_author: dict[str, list[dict]] = {}   # "smith" → [ref, ...]

        for ref in refs:
            # Label lookup
            label = ref.get("ref_label", "")
            if label:
                ref_by_label[label] = ref

            year = str(ref.get("year", "")).strip()
            authors = ref.get("authors", [])
            if authors:
                surname = authors[0].split()[0].lower() if authors[0] else ""
                if surname:
                    if year:
                        ref_by_author_year[(surname, year)] = ref
                    refs_by_author.setdefault(surname, []).append(ref)

        def _apply_ref(assessment: RoBAssessment, ref: dict) -> bool:
            """Apply ref data to assessment. Returns True if PMID was set."""
            got_pmid = False
            if ref.get("pmid") and not assessment.pmid:
                assessment.pmid = ref["pmid"]
                got_pmid = True
            if ref.get("doi") and not assessment.doi:
                assessment.doi = ref["doi"]
            if ref.get("title") and not assessment.study_title:
                assessment.study_title = ref["title"]
            return got_pmid

        matched = 0
        for a in assessments:
            if a.pmid:
                continue

            author, year, ref_num = self._normalize_study_id(a.study_id)

            # Strategy 1: bracket reference number
            if ref_num and ref_num in ref_by_label:
                if _apply_ref(a, ref_by_label[ref_num]):
                    matched += 1
                continue

            author_lower = author.lower() if author else ""
            if not author_lower:
                continue

            # Strategy 2: surname + year
            if year:
                ref = ref_by_author_year.get((author_lower, year))
                if ref:
                    if _apply_ref(a, ref):
                        matched += 1
                    continue

            # Strategy 3: surname-only (only if unambiguous)
            candidates = refs_by_author.get(author_lower, [])
            if len(candidates) == 1:
                if _apply_ref(a, candidates[0]):
                    matched += 1

        if matched:
            logger.info(f"Matched {matched} PMIDs from Cochrane reference list")

    async def resolve_pmids_via_doi(
        self, assessments: list[RoBAssessment],
    ) -> None:
        """Resolve PMIDs for assessments that have a DOI but no PMID.

        Uses the NCBI ID Converter API (pmcids → pmids).
        """
        need_doi = [a for a in assessments if a.doi and not a.pmid]
        if not need_doi:
            return

        # PubMed can look up by DOI via esearch
        resolved = 0
        for a in need_doi:
            try:
                params = {
                    "db": "pubmed",
                    "term": f"{a.doi}[DOI]",
                    "retmax": 1,
                    "retmode": "json",
                }
                if self.ncbi_api_key:
                    params["api_key"] = self.ncbi_api_key
                resp = await fetch_with_retry(
                    self.client, "GET",
                    f"{self.PUBMED_BASE}/esearch.fcgi", params=params,
                )
                if resp.status_code == 200:
                    pmids = resp.json().get("esearchresult", {}).get("idlist", [])
                    if pmids:
                        a.pmid = pmids[0]
                        resolved += 1
                await asyncio.sleep(0.35)
            except Exception as e:
                logger.debug(f"DOI→PMID lookup failed for {a.doi}: {e}")

        if resolved:
            logger.info(f"Resolved {resolved} PMIDs via DOI lookup")

    async def collect_rob_dataset(
        self,
        domains: Optional[list[str]] = None,
        max_reviews: int = 50,
        max_studies: int = 500,
        on_result: Optional[Callable[[RoBAssessment], None]] = None,
        skip_pmids: Optional[set[str]] = None,
        skip_pmcids: Optional[set[str]] = None,
    ) -> list[RoBAssessment]:
        """Full pipeline: search reviews, extract RoB, resolve PMIDs.

        Saves results incrementally via ``on_result`` callback after each
        review is processed, so progress survives interruptions.

        Args:
            domains: Clinical domains to search.
            max_reviews: Max reviews to search across all domains.
            max_studies: Stop after this many total assessments.
            on_result: Optional callback(RoBAssessment) called for each
                resolved assessment.  Use this for incremental DB saves.
            skip_pmids: PMIDs to treat as already processed — they are
                added to the dedup set so they are never passed to
                ``on_result`` and don't count toward ``max_studies``.
            skip_pmcids: PMCIDs (review IDs) to skip entirely — no full
                text fetch, LLM extraction, or PMID resolution.

        Resolution strategy (per review, not batched):
        1. Extract PMIDs/DOIs directly from Cochrane reference XML
        2. Resolve DOIs → PMIDs via PubMed
        3. Fall back to author+year PubMed search for remaining
        """
        if domains is None:
            domains = [
                "cardiovascular", "diabetes", "cancer",
                "mental health", "respiratory",
            ]

        all_results: list[RoBAssessment] = []
        seen_pmcids: set[str] = set(skip_pmcids or ())
        seen_pmids: set[str] = set(skip_pmids or ())

        # Per-domain + broad search
        search_passes: list[tuple[str, int]] = [
            (d, max_reviews // (len(domains) + 1)) for d in domains
        ]
        search_passes.append(("", max_reviews // (len(domains) + 1)))

        for domain, per_domain_max in search_passes:
            label = f"'{domain}'" if domain else "(all domains)"
            logger.info(f"Searching systematic reviews for {label}...")
            reviews = await self.search_cochrane_reviews(
                domain=domain,
                max_results=per_domain_max,
            )

            for review in reviews:
                pmcid = review.get("pmcid", "")
                if not pmcid or pmcid in seen_pmcids:
                    continue
                seen_pmcids.add(pmcid)

                # Fetch full text
                try:
                    ft_resp = await fetch_with_retry(
                        self.client, "GET",
                        f"{self.EUROPMC_BASE}/{pmcid}/fullTextXML",
                    )
                    if ft_resp.status_code != 200:
                        await asyncio.sleep(0.5)
                        continue
                    full_text = ft_resp.text
                    ft_bytes = len(ft_resp.content)  # raw bytes, no re-encoding
                    if ft_bytes > self.MAX_FULLTEXT_BYTES:
                        logger.warning(
                            f"Skipping {pmcid}: full text too large "
                            f"({ft_bytes / 1_000_000:.1f} MB "
                            f"> {self.MAX_FULLTEXT_BYTES / 1_000_000:.1f} MB limit)"
                        )
                        await asyncio.sleep(0.5)
                        continue
                except Exception:
                    await asyncio.sleep(0.5)
                    continue

                # Extract RoB assessments (regex first, LLM fallback)
                assessments = await self.extract_rob_via_llm(pmcid, full_text)
                if not assessments:
                    await asyncio.sleep(0.5)
                    continue

                # Extract references for PMID matching
                refs = await self.extract_included_study_refs(pmcid)

                for a in assessments:
                    a.cochrane_review_pmid = review.get("pmid", "")
                    a.cochrane_review_doi = review.get("doi", "")
                    a.cochrane_review_title = review.get("title", "")
                    a.domain = domain

                # Resolve PMIDs per-review (not batched at end)
                # Layer 1: reference list
                await self.resolve_pmids_from_refs(assessments, refs)
                # Layer 2: DOI lookup
                await self.resolve_pmids_via_doi(assessments)
                # Layer 3: author+year search
                unresolved = [a for a in assessments if not a.pmid]
                if unresolved:
                    await self.resolve_study_pmids(assessments)

                # Save results incrementally — deduplicate by PMID
                for a in assessments:
                    if not a.pmid or a.pmid in seen_pmids:
                        continue
                    seen_pmids.add(a.pmid)
                    all_results.append(a)
                    if on_result:
                        on_result(a)

                saved_from_review = sum(
                    1 for a in assessments if a.pmid and a.pmid in seen_pmids
                )
                logger.info(
                    f"Review {pmcid}: {len(assessments)} extracted, "
                    f"{saved_from_review} with PMID. "
                    f"Running total: {len(all_results)}"
                )

                await asyncio.sleep(0.5)

                if len(all_results) >= max_studies:
                    break

            if len(all_results) >= max_studies:
                break

        high_rob = sum(1 for a in all_results if a.overall_rob == "high")
        low_rob = sum(1 for a in all_results if a.overall_rob == "low")

        logger.info(
            f"Collection complete: {len(all_results)} unique RoB assessments "
            f"({high_rob} high, {low_rob} low) from {len(seen_pmcids)} reviews."
        )

        return all_results

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
