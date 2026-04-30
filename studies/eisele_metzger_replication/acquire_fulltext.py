"""Phase 1 full-text acquisition for the Eisele-Metzger 2025 replication study.

For each of the 100 RCTs in the EM benchmark, this script attempts to:

1. Resolve a PMID via NCBI E-utilities, using (in order):
   a) the trial registration number (NCT) as a PubMed [SI] query,
   b) the parent Cochrane review's referenced-trials list (skipped in v1
      because it requires Crossref API for the Cochrane review record),
   c) first-author + year + title-keyword search.

2. Fetch the abstract and DOI from PubMed via efetch.

3. Try Europe PMC for open-access full-text JATS XML.

4. Try Unpaywall for an OA PDF link by DOI as a secondary path.

5. Pull the trial registration record from ClinicalTrials.gov for any
   RCT with an NCT identifier.

Per the locked pre-analysis plan §3.2, the target is ≥80% full-text
coverage; the remainder will be evaluated abstract-only.

Per project CLAUDE.md guidance: this script is idempotent and resumable
— per-RCT metadata.json files act as checkpoints. Killing and
restarting the script picks up where the previous run left off.

Storage layout (gitignored):

  DATA/20240318_Data_for_analysis_full/fulltext/{rct_id}/
    metadata.json         always — attempt log + status
    abstract.txt          if PubMed record was retrieved
    paper.jats.xml        if Europe PMC has full-text XML
    paper.pdf             if Unpaywall yielded an OA PDF
    registration.json     if NCT resolved on ClinicalTrials.gov

Outputs at the study folder level:

  studies/eisele_metzger_replication/acquisition_report.md
  studies/eisele_metzger_replication/missing_fulltext.csv
    (the "still missing" list for OpenAthens manual fetching)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx

PROJECT_ROOT = Path("/Users/hherb/src/biasbuster")
EM_CSV = PROJECT_ROOT / "DATA/20240318_Data_for_analysis_full/Extracted_Data_Test_Data-Table 1.csv"
FULLTEXT_DIR = PROJECT_ROOT / "DATA/20240318_Data_for_analysis_full/fulltext"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
ACQUISITION_REPORT = STUDY_DIR / "acquisition_report.md"
MISSING_CSV = STUDY_DIR / "missing_fulltext.csv"

# CSV header is on row index 1 (0-based); row 0 is the Excel sheet marker.
EM_CSV_HEADER_ROW = 1

# Polite rate limit for NCBI without an API key: 3 req/s.
NCBI_RATE_LIMIT_SLEEP = 0.34
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Europe PMC has no published rate limit at our volume; we still pace.
EUROPEPMC_RATE_LIMIT_SLEEP = 0.10
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Unpaywall asks for email in URL.
UNPAYWALL_EMAIL = "horst.herb@gmail.com"

CT_GOV_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Network safety
HTTP_TIMEOUT = httpx.Timeout(20.0, read=30.0)
USER_AGENT = "biasbuster-replication-study/1.0 (mailto:horst.herb@gmail.com)"

# Identifier extraction (mirrors contamination_check.py)
_PMID_RE = re.compile(r"\bPMID[:\s]*(\d{6,9})\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
# Allow trailing disambiguator letter (e.g. "2013a") after the 4-digit year.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}(?:[a-z])?\b")

_PARTICLES = {"de", "del", "della", "di", "da", "do", "dos", "du",
              "el", "la", "le", "van", "von", "der", "den", "ter",
              "ten", "al", "bin", "ibn", "san"}


@dataclass
class EMRow:
    """One row of the EM CSV, with extracted identifiers."""
    rct_id: str
    cr_id: str
    rct_author: str
    rct_ref: str
    rct_regnr: str
    rct_condition: str
    rct_intervention: str
    extracted_pmid: str = ""
    extracted_doi: str = ""
    extracted_nct: str = ""
    first_author_surname: str = ""
    publication_year: str = ""


def load_em_rows() -> list[EMRow]:
    with open(EM_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[EM_CSV_HEADER_ROW]
    col_idx = {name: i for i, name in enumerate(header)}

    em_rows: list[EMRow] = []
    for raw in rows[EM_CSV_HEADER_ROW + 1:]:
        if not any(c.strip() for c in raw):
            continue
        em_rows.append(EMRow(
            rct_id=raw[col_idx["id"]].strip(),
            cr_id=raw[col_idx["cr_id"]].strip(),
            rct_author=raw[col_idx["rct_author"]].strip(),
            rct_ref=raw[col_idx["rct_ref"]].strip(),
            rct_regnr=raw[col_idx["rct_regnr"]].strip(),
            rct_condition=raw[col_idx["rct_condition"]].strip(),
            rct_intervention=raw[col_idx["rct_intervention"]].strip(),
        ))
    return em_rows


def extract_identifiers(row: EMRow) -> None:
    text = f"{row.rct_ref} {row.rct_regnr} {row.cr_id}"
    if m := _PMID_RE.search(text):
        row.extracted_pmid = m.group(1)
    if m := _DOI_RE.search(text):
        row.extracted_doi = m.group(0).rstrip(".,;:)")
    if m := _NCT_RE.search(text):
        row.extracted_nct = m.group(0).upper()
    if row.rct_author:
        for tok in row.rct_author.replace(",", " ").split():
            tok_lc = tok.lower().rstrip(".")
            # Accept ≥2-char tokens to allow Chinese/Korean-style surnames
            # (Hu, Ye, Wu, Li, ...). Filter pure digits and Latin particles.
            if len(tok_lc) >= 2 and not tok_lc.isdigit() and tok_lc not in _PARTICLES:
                row.first_author_surname = tok.rstrip(",.")
                break
    if m := _YEAR_RE.search(row.rct_author or row.rct_ref):
        # Strip any trailing disambiguator letter — PubMed [pdat] takes the year only.
        year_str = m.group(0)
        row.publication_year = year_str[:4] if year_str[-1].isalpha() else year_str


# --- Network helpers -----------------------------------------------------

class FetchError(Exception):
    """Wrap network errors so the caller can decide whether to retry."""


def http_get(client: httpx.Client, url: str, params: dict | None = None,
             max_retries: int = 3) -> httpx.Response:
    """GET with exponential backoff on 5xx and connection errors.

    4xx is returned without retry — those are typically "no result" or
    "bad request" and retrying won't help.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = client.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code < 500:
                return r
            last_exc = FetchError(f"HTTP {r.status_code} from {url}")
        except httpx.RequestError as exc:
            last_exc = exc
        time.sleep((2 ** attempt) * 1.0)
    raise FetchError(f"Failed after {max_retries} attempts: {last_exc}")


# --- PubMed resolution ---------------------------------------------------

def pubmed_search(client: httpx.Client, query: str) -> list[str]:
    """Return up to 10 PMIDs matching the query."""
    r = http_get(client, f"{NCBI_BASE}/esearch.fcgi", params={
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 10,
    })
    time.sleep(NCBI_RATE_LIMIT_SLEEP)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except (ValueError, KeyError):
        return []


def pubmed_summary(client: httpx.Client, pmids: list[str]) -> dict[str, dict]:
    """Return {pmid: summary_dict} for a batch of PMIDs (esummary)."""
    if not pmids:
        return {}
    r = http_get(client, f"{NCBI_BASE}/esummary.fcgi", params={
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    })
    time.sleep(NCBI_RATE_LIMIT_SLEEP)
    if r.status_code != 200:
        return {}
    try:
        result = r.json().get("result", {})
        return {p: result[p] for p in pmids if p in result}
    except (ValueError, KeyError):
        return {}


_REVIEW_PUB_TYPES = {
    "Systematic Review", "Meta-Analysis", "Review",
    "Practice Guideline", "Guideline",
}
_REVIEW_JOURNALS = {
    "Cochrane Database Syst Rev",
    "Cochrane database of systematic reviews",
}


def _is_review_or_meta(summary: dict) -> bool:
    """Heuristic: filter out systematic reviews / meta-analyses / Cochrane reviews."""
    if summary.get("source", "") in _REVIEW_JOURNALS:
        return True
    pub_types = set(summary.get("pubtype", []) or [])
    return bool(pub_types & _REVIEW_PUB_TYPES)


def _title_overlap_count(title: str, keywords: list[str]) -> int:
    """How many keywords (already lowercased) appear in the title (case-insensitive)?"""
    if not title or not keywords:
        return 0
    title_lc = title.lower()
    return sum(1 for kw in keywords if kw in title_lc)


def resolve_pmid_for(em: EMRow, client: httpx.Client) -> tuple[str, str]:
    """Resolve EM RCT to PubMed PMID. Multi-stage cascade.

    Returns (pmid, method) or ('', '').
    """
    keywords = _title_keywords(em.rct_ref, em.first_author_surname)[:8]

    # Stage 1: NCT-based via PubMed [si]. Filter out systematic reviews
    # and Cochrane review articles — those are the parent reviews, not the
    # original RCT. If multiple non-review candidates remain, prefer the
    # one whose title best matches the EM citation keywords.
    if em.extracted_nct:
        candidates = pubmed_search(client, f"{em.extracted_nct}[si]")
        if candidates:
            summaries = pubmed_summary(client, candidates)
            non_reviews = [
                p for p in candidates
                if not _is_review_or_meta(summaries.get(p, {}))
            ]
            if non_reviews:
                # Rank by title-keyword overlap with EM citation; if tie, take earliest year.
                def rank_key(pmid: str) -> tuple[int, int]:
                    s = summaries.get(pmid, {})
                    overlap = _title_overlap_count(s.get("title", ""), keywords)
                    year = int((s.get("pubdate", "") or "0")[:4] or 0)
                    return (-overlap, year)
                non_reviews.sort(key=rank_key)
                return non_reviews[0], f"nct_filtered:{em.extracted_nct}"

    # Stage 2: NCT via ClinicalTrials.gov references — sometimes the trial's
    # primary publication is listed there even when PubMed's [si] index
    # only knows about the Cochrane review. We accept ONLY references typed
    # as "RESULT" (publications reporting this trial's results). BACKGROUND
    # references are cited literature, NOT the trial's own paper, and
    # accepting them produces wrong matches (e.g. NCT02037633 lists Yun et
    # al. 2009 as BACKGROUND, but the trial's own paper is Diakomi 2014).
    # DERIVED references are typically Cochrane reviews and other secondary
    # works. If no RESULT-type reference exists, fall through to citation
    # search rather than guessing.
    if em.extracted_nct:
        if reg := fetch_ctgov_registration(client, em.extracted_nct):
            refs = (reg.get("protocolSection", {})
                       .get("referencesModule", {})
                       .get("references", []) or [])
            for ref in refs:
                if ref.get("type") == "RESULT" and ref.get("pmid"):
                    return ref["pmid"], f"ctgov_result:{em.extracted_nct}"

    # Stage 3: Citation-based — first author + year + distinctive title keywords.
    if em.first_author_surname and em.publication_year:
        if keywords:
            # Tight query first: surname + year + 3 keywords in title/abstract
            tight_kw = " AND ".join(f"{kw}[tiab]" for kw in keywords[:3])
            query = (f"{em.first_author_surname}[au] AND "
                     f"{em.publication_year}[pdat] AND {tight_kw}")
            pmids = pubmed_search(client, query)
            if pmids:
                # Pick the candidate with best title-keyword overlap.
                summaries = pubmed_summary(client, pmids[:5])
                ranked = sorted(
                    pmids[:5],
                    key=lambda p: -_title_overlap_count(
                        summaries.get(p, {}).get("title", ""), keywords),
                )
                return ranked[0], f"citation_tight:{query}"

        # Loose query: surname + year, accept only if exactly one hit.
        pmids = pubmed_search(
            client,
            f"{em.first_author_surname}[au] AND {em.publication_year}[pdat]"
        )
        if len(pmids) == 1:
            return pmids[0], f"citation_loose:{em.first_author_surname}+{em.publication_year}"

    return "", ""


def _title_keywords(rct_ref: str, surname: str) -> list[str]:
    """Distinctive ≥6-letter title/journal words from rct_ref.

    Citation format is roughly: ``Authors. Title. Journal year;vol:pages``.
    The first segment before ``". "`` is the author list, which is full of
    surnames that look like keywords but aren't (e.g. "Papaioannou" in a
    Diakomi-led paper). We strip that segment before extracting keywords.
    Stop-words are general citation noise; the surname itself is also
    excluded.
    """
    _STOP = {"randomized", "randomised", "trial", "study", "controlled",
             "double", "blind", "placebo", "patients", "effect", "effects",
             "treatment", "compared", "comparison", "versus", "abstract",
             "background", "methods", "results", "conclusion", "conclusions"}
    # Drop the author-list segment. The author list typically ends with one of:
    #   "et al."   (long lists; sometimes with no trailing space, e.g. "et al.Efficacy")
    #   ". "       (period followed by whitespace before the title)
    # We try "et al." first because some citations omit the space after it,
    # which would defeat the ". " split.
    if "et al." in rct_ref:
        body = rct_ref.split("et al.", 1)[1].lstrip()
    elif ". " in rct_ref:
        body = rct_ref.split(". ", 1)[1]
    else:
        body = rct_ref
    words = re.findall(r"[A-Za-z]{6,}", body.lower())
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _STOP or w == surname.lower() or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def fetch_pubmed_record(client: httpx.Client, pmid: str) -> dict:
    """Fetch the PubMed XML record and parse abstract + DOI + title."""
    r = http_get(client, f"{NCBI_BASE}/efetch.fcgi", params={
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
        "retmode": "xml",
    })
    time.sleep(NCBI_RATE_LIMIT_SLEEP)
    if r.status_code != 200:
        return {}
    xml = r.text
    out: dict[str, str] = {"pmid": pmid, "raw_xml_length": str(len(xml))}

    # Lightweight regex-based parsing — sufficient for our needs.
    if m := re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", xml, re.DOTALL):
        out["title"] = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    # Abstract may be split across multiple <AbstractText> elements (Background, Methods, etc.)
    abstract_parts: list[str] = []
    for m in re.finditer(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.DOTALL):
        label_match = re.search(r'Label="([^"]*)"', m.group(0))
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if label_match:
            abstract_parts.append(f"{label_match.group(1)}: {text}")
        else:
            abstract_parts.append(text)
    if abstract_parts:
        out["abstract"] = "\n\n".join(abstract_parts)
    if m := re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml):
        out["doi"] = m.group(1).strip()
    if m := re.search(r'<ArticleId IdType="pmc">([^<]+)</ArticleId>', xml):
        out["pmcid"] = m.group(1).strip()
    return out


# --- Europe PMC full text ------------------------------------------------

def europepmc_fulltext_xml(client: httpx.Client, pmcid: str) -> str:
    """JATS XML for an OA paper if available; '' otherwise."""
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    url = f"{EUROPEPMC_BASE}/{pmcid.upper()}/fullTextXML"
    try:
        r = http_get(client, url)
        time.sleep(EUROPEPMC_RATE_LIMIT_SLEEP)
        if r.status_code == 200 and r.text.strip().startswith("<"):
            return r.text
    except FetchError:
        pass
    return ""


def europepmc_pdf_url(pmcid: str) -> str:
    """Return the URL pattern that worked in our literature-folder fetch."""
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return f"https://europepmc.org/articles/{pmcid.upper()}?pdf=render"


def fetch_pdf(client: httpx.Client, url: str) -> bytes | None:
    try:
        r = http_get(client, url)
        time.sleep(EUROPEPMC_RATE_LIMIT_SLEEP)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            return r.content
        # Some endpoints redirect through the cache and end at PDF; check magic bytes too
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            return r.content
    except FetchError:
        pass
    return None


# --- Unpaywall ----------------------------------------------------------

def unpaywall_oa_pdf_url(client: httpx.Client, doi: str) -> str:
    """Return the best OA PDF URL for the DOI, or '' if none."""
    url = f"https://api.unpaywall.org/v2/{quote_plus(doi)}"
    try:
        r = http_get(client, url, params={"email": UNPAYWALL_EMAIL})
        time.sleep(0.10)
        if r.status_code == 200:
            data = r.json()
            best = data.get("best_oa_location") or {}
            return best.get("url_for_pdf", "") or best.get("url", "")
    except (FetchError, ValueError):
        pass
    return ""


# --- ClinicalTrials.gov -------------------------------------------------

def fetch_ctgov_registration(client: httpx.Client, nct: str) -> dict:
    """Get the v2 API study record for an NCT id.

    ClinicalTrials.gov sits behind a Cloudflare layer that blocks httpx's
    default TLS fingerprint with a 403 (verified empirically). Stdlib
    `urllib` produces a different fingerprint that does not trip the
    block, so we use it here despite the asymmetry with the rest of the
    fetch code. The ``client`` argument is unused but retained so the
    call signature stays uniform with the other fetchers.
    """
    import urllib.error
    import urllib.request
    url = f"{CT_GOV_BASE}/{nct}?format=json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            time.sleep(0.10)
            if resp.status != 200:
                return {}
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, json.JSONDecodeError):
        return {}


# --- Per-RCT acquisition orchestrator -----------------------------------

def acquire_one(em: EMRow, client: httpx.Client) -> dict:
    """Attempt full acquisition for one RCT. Idempotent via metadata.json."""
    rct_dir = FULLTEXT_DIR / em.rct_id
    rct_dir.mkdir(parents=True, exist_ok=True)
    meta_path = rct_dir / "metadata.json"

    if meta_path.exists():
        existing: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            return existing

    meta: dict[str, Any] = {
        "rct_id": em.rct_id,
        "rct_author": em.rct_author,
        "rct_regnr": em.rct_regnr,
        "cr_id": em.cr_id,
        "extracted": {
            "pmid": em.extracted_pmid,
            "doi": em.extracted_doi,
            "nct": em.extracted_nct,
            "first_author_surname": em.first_author_surname,
            "publication_year": em.publication_year,
        },
        "attempts": [],
        "complete": False,
    }

    # Step 1: PMID resolution
    pmid = em.extracted_pmid
    if not pmid:
        pmid, method = resolve_pmid_for(em, client)
        meta["pmid_resolution_method"] = method
    meta["pmid"] = pmid

    if not pmid:
        meta["attempts"].append("PMID resolution failed; cannot fetch abstract or full text.")
        meta["complete"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    # Step 2: Fetch PubMed record
    pm = fetch_pubmed_record(client, pmid)
    if pm:
        meta["title"] = pm.get("title", "")
        meta["doi"] = pm.get("doi", "")
        meta["pmcid"] = pm.get("pmcid", "")
        if abstract := pm.get("abstract"):
            (rct_dir / "abstract.txt").write_text(abstract, encoding="utf-8")
            meta["has_abstract"] = True
            meta["attempts"].append(f"Abstract fetched via PubMed efetch (pmid={pmid}).")
        else:
            meta["attempts"].append(f"PubMed efetch returned no abstract for pmid={pmid}.")
    else:
        meta["attempts"].append(f"PubMed efetch failed for pmid={pmid}.")

    # Step 3: Try Europe PMC for full-text JATS XML
    if pmcid := meta.get("pmcid"):
        xml = europepmc_fulltext_xml(client, pmcid)
        if xml:
            (rct_dir / "paper.jats.xml").write_text(xml, encoding="utf-8")
            meta["has_fulltext"] = True
            meta["fulltext_source"] = "europepmc_xml"
            meta["attempts"].append(f"Full-text XML fetched from Europe PMC ({pmcid}).")
        else:
            # Try PDF via Europe PMC
            if pdf := fetch_pdf(client, europepmc_pdf_url(pmcid)):
                (rct_dir / "paper.pdf").write_bytes(pdf)
                meta["has_fulltext"] = True
                meta["fulltext_source"] = "europepmc_pdf"
                meta["attempts"].append(f"Full-text PDF fetched from Europe PMC ({pmcid}).")

    # Step 4: Unpaywall fallback
    if not meta.get("has_fulltext") and meta.get("doi"):
        if pdf_url := unpaywall_oa_pdf_url(client, meta["doi"]):
            if pdf := fetch_pdf(client, pdf_url):
                (rct_dir / "paper.pdf").write_bytes(pdf)
                meta["has_fulltext"] = True
                meta["fulltext_source"] = f"unpaywall:{pdf_url}"
                meta["attempts"].append("Full-text PDF fetched via Unpaywall.")
            else:
                meta["attempts"].append(f"Unpaywall returned URL {pdf_url} but PDF download failed.")
        else:
            meta["attempts"].append("Unpaywall: no OA PDF available.")

    # Step 5: ClinicalTrials.gov registration
    if em.extracted_nct:
        reg = fetch_ctgov_registration(client, em.extracted_nct)
        if reg:
            (rct_dir / "registration.json").write_text(
                json.dumps(reg, indent=2), encoding="utf-8")
            meta["has_registration"] = True
            meta["attempts"].append(f"Registration fetched from ClinicalTrials.gov ({em.extracted_nct}).")
        else:
            meta["attempts"].append(f"ClinicalTrials.gov: no record for {em.extracted_nct}.")

    meta["complete"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --- Reporting ----------------------------------------------------------

def write_acquisition_report(metas: list[dict]) -> None:
    n = len(metas)
    n_pmid = sum(1 for m in metas if m.get("pmid"))
    n_abstract = sum(1 for m in metas if m.get("has_abstract"))
    n_fulltext = sum(1 for m in metas if m.get("has_fulltext"))
    n_registration = sum(1 for m in metas if m.get("has_registration"))

    fulltext_sources: dict[str, int] = {}
    for m in metas:
        if src := m.get("fulltext_source"):
            key = src.split(":")[0]
            fulltext_sources[key] = fulltext_sources.get(key, 0) + 1

    lines: list[str] = []
    lines.append("# Phase 1 Full-Text Acquisition Report")
    lines.append("")
    lines.append("**Generated:** by `studies/eisele_metzger_replication/acquire_fulltext.py`")
    lines.append("**Storage:** `DATA/20240318_Data_for_analysis_full/fulltext/{rct_id}/` (gitignored)")
    lines.append("**Pre-analysis plan reference:** §3.2 (target ≥80% full-text coverage)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total EM-100 RCTs:** {n}")
    lines.append(f"- **PMID resolved:** {n_pmid} ({100*n_pmid/n:.0f}%)")
    lines.append(f"- **Abstract obtained:** {n_abstract} ({100*n_abstract/n:.0f}%)")
    lines.append(f"- **Full-text obtained:** {n_fulltext} ({100*n_fulltext/n:.0f}%)")
    lines.append(f"- **Trial registration record obtained:** {n_registration} ({100*n_registration/n:.0f}%)")
    lines.append("")

    if fulltext_sources:
        lines.append("## Full-text sources")
        lines.append("")
        lines.append("| Source | Count |")
        lines.append("|---|---:|")
        for src in sorted(fulltext_sources, key=lambda k: -fulltext_sources[k]):
            lines.append(f"| {src} | {fulltext_sources[src]} |")
        lines.append("")

    n_threshold = round(n * 0.80)
    lines.append("## Pre-analysis plan threshold")
    lines.append("")
    if n_fulltext >= n_threshold:
        lines.append(f"✅ Full-text coverage **{n_fulltext}/{n}** meets the ≥80% threshold ({n_threshold}/{n}). Phase 5 may proceed with both abstract-only and full-text protocols.")
    else:
        lines.append(f"⚠️ Full-text coverage **{n_fulltext}/{n}** is below the ≥80% threshold ({n_threshold}/{n}). The 'still missing' list at `missing_fulltext.csv` should be fetched via institutional access (JCU + Queensland Health OpenAthens) before Phase 5.")
    lines.append("")

    lines.append("## Per-RCT detail")
    lines.append("")
    lines.append("| RCT | PMID | Abstract | Full text | Source | Registration |")
    lines.append("|---|---|:-:|:-:|---|:-:|")
    for m in metas:
        rct = m.get("rct_id", "?")
        pmid = m.get("pmid", "")
        ab = "✓" if m.get("has_abstract") else " "
        ft = "✓" if m.get("has_fulltext") else " "
        src = (m.get("fulltext_source") or "").split(":")[0]
        reg = "✓" if m.get("has_registration") else " "
        lines.append(f"| {rct} | {pmid} | {ab} | {ft} | {src} | {reg} |")
    lines.append("")

    ACQUISITION_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_missing_csv(metas: list[dict], em_rows: list[EMRow]) -> None:
    em_by_id = {r.rct_id: r for r in em_rows}
    rows: list[list[str]] = []
    for m in metas:
        if m.get("has_fulltext"):
            continue
        em = em_by_id.get(m.get("rct_id", ""))
        if em is None:
            continue
        rows.append([
            em.rct_id,
            m.get("pmid", ""),
            m.get("doi", ""),
            em.rct_regnr,
            em.rct_author,
            em.rct_ref,
            "; ".join(m.get("attempts", []))[:300],
        ])
    with open(MISSING_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rct_id", "pmid", "doi", "registration_nr",
                    "author", "citation", "acquisition_attempts"])
        w.writerows(rows)


# --- main ---------------------------------------------------------------

def main(limit: int | None = None) -> None:
    em_rows = load_em_rows()
    print(f"[load] {len(em_rows)} EM rows")

    for em in em_rows:
        extract_identifiers(em)

    if limit:
        em_rows = em_rows[:limit]
        print(f"[main] limited to first {limit} rows for smoke test")

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT},
                      follow_redirects=True) as client:
        for i, em in enumerate(em_rows, 1):
            print(f"[{i:>3}/{len(em_rows)}] {em.rct_id} {em.rct_author!r}", flush=True)
            try:
                meta = acquire_one(em, client)
            except Exception as exc:  # noqa: BLE001 — log and continue
                print(f"  [error] {exc!r}", flush=True)
                meta = {
                    "rct_id": em.rct_id,
                    "complete": False,
                    "error": repr(exc),
                }
            metas.append(meta)
            status_bits = []
            if meta.get("pmid"):
                status_bits.append(f"pmid={meta['pmid']}")
            if meta.get("has_abstract"):
                status_bits.append("abs")
            if meta.get("has_fulltext"):
                status_bits.append(f"ft={meta.get('fulltext_source','').split(':')[0]}")
            if meta.get("has_registration"):
                status_bits.append("reg")
            if not status_bits:
                status_bits.append("NO DATA")
            print(f"  → {', '.join(status_bits)}", flush=True)

    write_acquisition_report(metas)
    write_missing_csv(metas, em_rows)
    print(f"[write] {ACQUISITION_REPORT}")
    print(f"[write] {MISSING_CSV}")


if __name__ == "__main__":
    limit_arg: int | None = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--limit="):
        limit_arg = int(sys.argv[1].split("=", 1)[1])
    main(limit=limit_arg)
