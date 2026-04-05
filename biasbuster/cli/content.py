"""Content acquisition for the BiasBuster CLI.

Resolves identifiers (PMID, DOI, local file) to analysable content.
Uses bmlib for full-text retrieval and JATS parsing, falling back to
abstract-only when full text is unavailable.

Downloaded content is cached in ~/.biasbuster/downloads/ (keyed by PMID
or DOI) so repeated analyses of the same paper skip network calls.
Pass force_download=True to bypass the cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

from bmlib.fulltext.jats_parser import JATSParser
from bmlib.fulltext.models import JATSArticle
from bmlib.fulltext.service import FullTextService
from biasbuster.collectors.pubmed_xml import parse_pubmed_xml

from biasbuster.cli.settings import CLIConfig
from biasbuster.cli.pdf_extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

EUROPE_PMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ID_CONVERTER = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

DEFAULT_CACHE_DIR = Path.home() / ".biasbuster" / "downloads"


class ContentCache:
    """Disk cache for downloaded PubMed metadata and JATS full text.

    Layout::

        ~/.biasbuster/downloads/
            pmid/
                12345678.pubmed.xml     # Raw PubMed efetch XML
                12345678.jats.xml       # Raw JATS full-text XML
                12345678.meta.json      # Resolved identifiers (pmid↔doi↔pmcid)
            doi/
                10.1234_example.pubmed.xml
                10.1234_example.jats.xml
                10.1234_example.meta.json
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.root = cache_dir or DEFAULT_CACHE_DIR

    @staticmethod
    def _safe_key(identifier: str) -> str:
        """Turn an identifier into a filesystem-safe filename stem."""
        return re.sub(r"[^\w.\-]", "_", identifier)

    def _dir(self, id_type: str) -> Path:
        return self.root / id_type

    # -- PubMed XML ----------------------------------------------------------

    def get_pubmed_xml(self, id_type: str, identifier: str) -> str | None:
        """Return cached PubMed efetch XML text, or None."""
        path = self._dir(id_type) / f"{self._safe_key(identifier)}.pubmed.xml"
        if path.exists():
            logger.info("Cache hit: PubMed XML for %s %s", id_type, identifier)
            return path.read_text(encoding="utf-8")
        return None

    def save_pubmed_xml(self, id_type: str, identifier: str, xml_text: str) -> None:
        """Save PubMed efetch XML to cache."""
        d = self._dir(id_type)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self._safe_key(identifier)}.pubmed.xml"
        path.write_text(xml_text, encoding="utf-8")
        logger.debug("Cached PubMed XML → %s", path)

    # -- JATS XML ------------------------------------------------------------

    def get_jats_xml(self, id_type: str, identifier: str) -> bytes | None:
        """Return cached JATS full-text XML bytes, or None."""
        path = self._dir(id_type) / f"{self._safe_key(identifier)}.jats.xml"
        if path.exists():
            logger.info("Cache hit: JATS XML for %s %s", id_type, identifier)
            return path.read_bytes()
        return None

    def save_jats_xml(self, id_type: str, identifier: str, xml_bytes: bytes) -> None:
        """Save JATS full-text XML to cache."""
        d = self._dir(id_type)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self._safe_key(identifier)}.jats.xml"
        path.write_bytes(xml_bytes)
        logger.debug("Cached JATS XML → %s", path)

    # -- Identifier mapping --------------------------------------------------

    def get_meta(self, id_type: str, identifier: str) -> dict | None:
        """Return cached identifier mappings (pmid↔doi↔pmcid), or None."""
        path = self._dir(id_type) / f"{self._safe_key(identifier)}.meta.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def save_meta(self, id_type: str, identifier: str, meta: dict) -> None:
        """Save identifier mappings to cache."""
        d = self._dir(id_type)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self._safe_key(identifier)}.meta.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@dataclass
class AcquiredContent:
    """Result of content acquisition — everything the analysis stage needs."""

    # Identifiers
    pmid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""

    # Structured full text (from JATS)
    jats_article: JATSArticle | None = None
    jats_xml: bytes | None = None  # Raw JATS XML for back-matter extraction

    # Plain full text (from PDF or HTML stripping)
    plain_fulltext: str = ""

    # Metadata
    authors: list[dict] = field(default_factory=list)
    journal: str = ""
    year: str = ""

    # What we got
    content_type: str = "abstract"  # "abstract", "fulltext_jats", "fulltext_plain"

    @property
    def has_fulltext(self) -> bool:
        """Whether full text (structured or plain) is available."""
        return self.jats_article is not None or bool(self.plain_fulltext)


def _http_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
) -> httpx.Response:
    """HTTP GET with exponential backoff retry.

    Args:
        url: Request URL.
        params: Query parameters.
        headers: Request headers.
        timeout: Per-request timeout in seconds.

    Returns:
        Successful response.

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP error after retries.
        RuntimeError: After MAX_RETRIES exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.debug(
                "HTTP GET %s attempt %d failed (%s), retrying in %.1fs",
                url, attempt + 1, exc, delay,
            )
            time.sleep(delay)
        except httpx.HTTPStatusError as exc:
            # Retry on 5xx server errors, not on 4xx client errors
            if exc.response.status_code >= 500:
                last_exc = exc
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "HTTP GET %s attempt %d got %d, retrying in %.1fs",
                    url, attempt + 1, exc.response.status_code, delay,
                )
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(
        f"HTTP GET {url} failed after {MAX_RETRIES} retries: {last_exc}"
    )


def classify_identifier(raw: str) -> tuple[str, str]:
    """Classify an identifier string and return (type, normalised_value).

    Classification priority: DOI > PMID > DOI URL > local file.
    Local files are detected by extension (.pdf, .xml, .jats) or by
    containing a path separator.

    Returns:
        Tuple of ("pmid", "12345678"), ("doi", "10.1234/..."), or ("file", "/path/to/file").
    """
    stripped = raw.strip()

    # DOI — starts with "10." or "doi:" (check before file to avoid
    # false matches on directories named like DOIs)
    if stripped.lower().startswith("doi:"):
        return "doi", stripped[4:].strip()
    if stripped.startswith("10."):
        return "doi", stripped

    # DOI URL — https://doi.org/10.xxxx
    doi_url = re.match(r"https?://(?:dx\.)?doi\.org/(10\..+)", stripped)
    if doi_url:
        return "doi", doi_url.group(1)

    # PMID — numeric, possibly prefixed
    pmid_match = re.match(r"(?:PMID[:\s]*)?(\d{1,9})$", stripped, re.IGNORECASE)
    if pmid_match:
        return "pmid", pmid_match.group(1)

    # Local file — only match on known extensions or explicit path separators
    path = Path(raw)
    if path.suffix.lower() in (".pdf", ".xml", ".jats"):
        return "file", str(path.resolve())
    if os.sep in raw and path.exists():
        return "file", str(path.resolve())

    raise ValueError(f"Cannot classify identifier: {raw!r}. Expected PMID, DOI, or file path.")


def acquire_content(
    identifier: str,
    config: CLIConfig,
    *,
    force_download: bool = False,
) -> AcquiredContent:
    """Resolve an identifier and fetch the best available content.

    Attempts to get structured JATS full text first, then falls back to
    PDF text extraction, then to abstract-only. Downloaded content is
    cached in ~/.biasbuster/downloads/ for subsequent runs.

    Args:
        identifier: PMID, DOI, or local file path.
        config: CLI configuration.
        force_download: If True, bypass the cache and re-download.

    Returns:
        AcquiredContent with the best available text for analysis.
    """
    id_type, id_value = classify_identifier(identifier)
    cache = ContentCache() if not force_download else None

    if id_type == "file":
        return _acquire_from_file(id_value)
    elif id_type == "pmid":
        return _acquire_from_pmid(id_value, config, cache=cache)
    elif id_type == "doi":
        return _acquire_from_doi(id_value, config, cache=cache)
    else:
        raise ValueError(f"Unknown identifier type: {id_type}")


def _acquire_from_file(file_path: str) -> AcquiredContent:
    """Load content from a local PDF or JATS XML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in (".xml", ".jats"):
        data = path.read_bytes()
        parser = JATSParser(data)
        article = parser.parse()
        abstract = "\n".join(
            f"[{s.title}] {s.content}" if s.title else s.content
            for s in article.abstract_sections
        )
        authors = [
            {"name": a.full_name, "affiliations": a.affiliations}
            for a in article.authors
        ]
        return AcquiredContent(
            pmid=article.pmid,
            doi=article.doi,
            title=article.title,
            abstract=abstract,
            jats_article=article,
            jats_xml=data,
            authors=authors,
            journal=article.journal,
            year=article.year,
            content_type="fulltext_jats",
        )

    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {file_path}")
        return AcquiredContent(
            plain_fulltext=text,
            content_type="fulltext_plain",
        )

    raise ValueError(f"Unsupported file format: {suffix}")


def _acquire_from_pmid(
    pmid: str, config: CLIConfig, *, cache: ContentCache | None = None,
) -> AcquiredContent:
    """Fetch content for a PubMed ID — try full text, fall back to abstract."""
    content = AcquiredContent(pmid=pmid)

    # Step 1: Fetch abstract + metadata from PubMed
    _fetch_pubmed_metadata(content, config, cache=cache)

    # Step 2: Try to get structured JATS full text
    if _try_jats_fulltext(content, config, cache=cache):
        return content

    # Step 3: Try bmlib FullTextService for PDF fallback
    if _try_fulltext_service(content, config, pmid=pmid, doi=content.doi):
        return content

    # Step 4: Abstract-only
    if content.abstract:
        content.content_type = "abstract"
        return content

    raise ValueError(f"No content found for PMID {pmid}")


def _acquire_from_doi(
    doi: str, config: CLIConfig, *, cache: ContentCache | None = None,
) -> AcquiredContent:
    """Fetch content for a DOI — try full text, fall back to abstract."""
    content = AcquiredContent(doi=doi)

    # Step 1: Try to resolve DOI to PMID for metadata
    pmid = _doi_to_pmid(doi, email=config.email, cache=cache)
    if pmid:
        content.pmid = pmid
        _fetch_pubmed_metadata(content, config, cache=cache)

    # Step 2: Try structured JATS full text
    if _try_jats_fulltext(content, config, cache=cache):
        return content

    # Step 3: Try bmlib FullTextService for PDF fallback
    if _try_fulltext_service(content, config, pmid=content.pmid, doi=doi):
        return content

    # Step 4: Abstract-only
    if content.abstract:
        content.content_type = "abstract"
        return content

    raise ValueError(f"No content found for DOI {doi}")


def _fetch_pubmed_metadata(
    content: AcquiredContent,
    config: CLIConfig,
    *,
    cache: ContentCache | None = None,
) -> None:
    """Fetch title, abstract, and metadata from PubMed E-utilities."""
    if not content.pmid:
        return

    # Check cache first
    cached_xml = cache.get_pubmed_xml("pmid", content.pmid) if cache else None
    if cached_xml is not None:
        xml_text = cached_xml
    else:
        params = {
            "db": "pubmed",
            "id": content.pmid,
            "rettype": "xml",
            "retmode": "xml",
        }
        if config.ncbi_api_key:
            params["api_key"] = config.ncbi_api_key

        try:
            resp = _http_get_with_retry(PUBMED_EFETCH, params=params)
        except Exception:
            logger.warning("PubMed fetch failed for PMID %s", content.pmid, exc_info=True)
            return

        xml_text = resp.text

        # Save to cache
        if cache:
            cache.save_pubmed_xml("pmid", content.pmid, xml_text)

    parsed = parse_pubmed_xml(xml_text)
    if parsed is None:
        logger.warning("Failed to parse PubMed XML for PMID %s", content.pmid)
        return

    content.title = parsed.get("title", "")
    content.abstract = parsed.get("abstract", "")
    content.doi = parsed.get("doi", content.doi) or content.doi
    content.journal = parsed.get("journal", "")
    content.year = str(parsed.get("year", ""))
    content.authors = parsed.get("authors", [])


def _try_jats_fulltext(
    content: AcquiredContent,
    config: CLIConfig,
    *,
    cache: ContentCache | None = None,
) -> bool:
    """Try to fetch JATS XML from Europe PMC and parse into structured sections.

    Returns True if structured full text was obtained.
    """
    # Determine cache key: prefer PMID, fall back to DOI
    cache_id_type = "pmid" if content.pmid else "doi"
    cache_id = content.pmid or content.doi

    # Check JATS cache
    cached_jats = cache.get_jats_xml(cache_id_type, cache_id) if cache and cache_id else None
    if cached_jats is not None:
        return _parse_jats_into_content(cached_jats, content, "cached")

    pmc_id = _discover_pmc_id(content.doi, content.pmid)
    if not pmc_id:
        return False

    normalized = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    url = f"{EUROPE_PMC_REST}/{normalized}/fullTextXML"

    try:
        resp = _http_get_with_retry(url, headers={"Accept": "application/xml"})

        # Cache the raw JATS XML
        if cache and cache_id:
            cache.save_jats_xml(cache_id_type, cache_id, resp.content)

        return _parse_jats_into_content(resp.content, content, f"Europe PMC {pmc_id}")

    except Exception:
        logger.debug("JATS fetch/parse failed for %s", pmc_id, exc_info=True)
        return False


def _parse_jats_into_content(
    jats_bytes: bytes, content: AcquiredContent, source: str,
) -> bool:
    """Parse JATS XML bytes and populate content fields.

    Returns True if structured full text with body sections was obtained.
    """
    try:
        parser = JATSParser(jats_bytes)
        article = parser.parse()

        if not article.body_sections:
            logger.debug("JATS parsed but no body sections (%s)", source)
            return False

        content.jats_article = article
        content.jats_xml = jats_bytes
        content.content_type = "fulltext_jats"

        if not content.title and article.title:
            content.title = article.title
        if not content.doi and article.doi:
            content.doi = article.doi
        if not content.pmid and article.pmid:
            content.pmid = article.pmid

        logger.info("Got structured JATS full text (%s)", source)
        return True
    except Exception:
        logger.debug("JATS parse failed (%s)", source, exc_info=True)
        return False


def _try_fulltext_service(
    content: AcquiredContent,
    config: CLIConfig,
    *,
    pmid: str = "",
    doi: str = "",
) -> bool:
    """Try bmlib FullTextService for PDF fallback.

    Returns True if plain full text was extracted.
    """
    if not config.email:
        logger.debug("No email configured, skipping Unpaywall/DOI fallback")
        return False

    try:
        service = FullTextService(email=config.email)
        result = service.fetch_fulltext(
            pmid=pmid,
            doi=doi or None,
            identifier=doi or pmid,
        )
    except Exception:
        logger.debug("FullTextService failed", exc_info=True)
        return False

    # If we got a cached PDF file, extract text
    if result.file_path:
        try:
            text = extract_text_from_pdf(result.file_path)
            if text.strip():
                content.plain_fulltext = text
                content.content_type = "fulltext_plain"
                logger.info("Extracted text from cached PDF: %s", result.file_path)
                return True
        except Exception:
            logger.debug("PDF text extraction failed for %s", result.file_path, exc_info=True)

    # If we got HTML (JATS was already parsed by the service), use as plain text
    if result.html:
        text = _strip_html(result.html)
        if text.strip():
            content.plain_fulltext = text
            content.content_type = "fulltext_plain"
            logger.info("Got full text from HTML (source: %s)", result.source)
            return True

    return False


def _discover_pmc_id(doi: str | None, pmid: str) -> str | None:
    """Search Europe PMC to discover a PMC ID for a given DOI or PMID."""
    if not doi and not pmid:
        return None

    query = f"DOI:{doi}" if doi else f"EXT_ID:{pmid}"
    url = (
        f"{EUROPE_PMC_REST}/search"
        f"?query={quote(query, safe=':')}&format=json&resultType=core&pageSize=1"
    )

    try:
        resp = _http_get_with_retry(
            url, headers={"Accept": "application/json"}, timeout=15.0,
        )
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if results and results[0].get("inEPMC") == "Y":
            return results[0].get("pmcid")
    except Exception:
        logger.debug("Europe PMC search failed", exc_info=True)

    return None


def _doi_to_pmid(
    doi: str, email: str = "", *, cache: ContentCache | None = None,
) -> str | None:
    """Resolve DOI to PMID via NCBI ID Converter API, falling back to Europe PMC.

    Args:
        doi: Digital Object Identifier to resolve.
        email: Contact email for NCBI polite pool (required by NCBI).
        cache: Optional content cache for storing resolved identifiers.
    """
    # Check cache for previously resolved mapping
    if cache:
        meta = cache.get_meta("doi", doi)
        if meta and meta.get("pmid"):
            logger.info("Cache hit: DOI %s → PMID %s", doi, meta["pmid"])
            return meta["pmid"]

    def _cache_and_return(resolved_pmid: str) -> str:
        if cache:
            cache.save_meta("doi", doi, {"pmid": resolved_pmid, "doi": doi})
        return resolved_pmid

    # Try NCBI ID Converter (requires email)
    if email:
        try:
            params = {"ids": doi, "format": "json", "tool": "biasbuster", "email": email}
            resp = _http_get_with_retry(NCBI_ID_CONVERTER, params=params, timeout=15.0)
            data = resp.json()
            records = data.get("records", [])
            if records:
                pmid = records[0].get("pmid", "")
                if pmid:
                    return _cache_and_return(pmid)
        except Exception:
            logger.debug("NCBI ID Converter failed for %s", doi, exc_info=True)

    # Fallback: Europe PMC search (no email required)
    try:
        url = (
            f"{EUROPE_PMC_REST}/search"
            f"?query={quote(f'DOI:{doi}', safe=':')}&format=json&pageSize=1"
        )
        resp = _http_get_with_retry(url, timeout=15.0)
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if results:
            pmid = results[0].get("pmid", "")
            if pmid:
                return _cache_and_return(pmid)
    except Exception:
        logger.debug("Europe PMC DOI→PMID fallback failed for %s", doi, exc_info=True)

    return None


def _strip_html(html: str) -> str:
    """Strip HTML tags to get plain text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
