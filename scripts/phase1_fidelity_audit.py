"""Phase 1 — Fidelity audit of the 23 back-linked Cochrane papers.

For each paper in the DB that has a ``cochrane_review_pmid`` back-link,
this diagnostic checks three things against the LLM extraction cache:

1. **Mapping**: does the source review's RoB table actually contain an
   entry whose first-author + year match the paper's PubMed metadata?
2. **Rating agreement**: if an entry is found, does its overall_rob /
   per-domain ratings match what is persisted in the DB?
3. **Coverage**: how many back-linked papers have any matching entry at
   all vs. are orphans (rating in DB but no matching entry in the
   review's extraction)?

This is a read-only diagnostic — no DB writes, no re-extraction. It
uses only:

- ``dataset/biasbuster.db`` — DB rows with back-links
- ``dataset/llm_rob_cache.json`` — cached Stage A LLM extractions
- Europe PMC search — PMID→PMCID resolution for the 4 source reviews
- PubMed E-utilities — titles/first-authors/years for trial PMIDs

Output: ``dataset/phase1_fidelity_audit.md`` (markdown report for
human inspection) plus a summary printed to stdout.

Usage:
    uv run python scripts/phase1_fidelity_audit.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biasbuster.utils.retry import fetch_with_retry  # noqa: E402

logger = logging.getLogger("phase1_fidelity_audit")

DEFAULT_DB = Path("dataset/biasbuster.db")
DEFAULT_CACHE = Path("dataset/llm_rob_cache.json")
DEFAULT_OUTPUT = Path("dataset/phase1_fidelity_audit.md")

EUROPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HTTP_TIMEOUT_SECONDS = 30.0
USER_AGENT = "biasbuster/phase1-fidelity-audit"


# -- Data acquisition ---------------------------------------------------

def load_backlinks(db_path: Path) -> dict[str, list[dict]]:
    """Return {review_pmid: [paper_rows,...]} for every back-linked paper."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT pmid, title, cochrane_review_pmid, cochrane_review_doi,
                   cochrane_review_title, overall_rob, randomization_bias,
                   deviation_bias, missing_outcome_bias, measurement_bias,
                   reporting_bias, COALESCE(excluded,0) AS excluded
            FROM papers
            WHERE source LIKE 'cochrane%' AND cochrane_review_pmid != ''
            """
        ).fetchall()
    by_review: dict[str, list[dict]] = {}
    for r in rows:
        by_review.setdefault(r["cochrane_review_pmid"], []).append(dict(r))
    return by_review


def load_llm_cache(cache_path: Path) -> dict[str, list[dict]]:
    """Load the LLM RoB extraction cache. Keys are PMCIDs."""
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


async def resolve_review_pmid_to_pmcid(
    client: httpx.AsyncClient, pmid: str
) -> tuple[str, str, str]:
    """Return (pmcid, title, year) for one review PMID via Europe PMC."""
    params = {
        "query": f"EXT_ID:{pmid} AND SRC:MED",
        "format": "json",
        "resultType": "lite",
        "pageSize": "1",
    }
    resp = await fetch_with_retry(
        client, "GET", EUROPMC_SEARCH, params=params, max_retries=3, base_delay=2.0,
    )
    resp.raise_for_status()
    data = resp.json()
    hits = (data.get("resultList") or {}).get("result") or []
    if not hits:
        return "", "", ""
    h = hits[0]
    return h.get("pmcid", ""), h.get("title", ""), h.get("pubYear", "")


async def fetch_paper_meta(
    client: httpx.AsyncClient, pmids: list[str]
) -> dict[str, dict]:
    """Fetch (title, first_author, year) from PubMed for each PMID."""
    if not pmids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = await fetch_with_retry(
        client, "GET", PUBMED_EFETCH, params=params, max_retries=3, base_delay=2.0,
    )
    resp.raise_for_status()
    out: dict[str, dict] = {}
    root = ET.fromstring(resp.text)
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        first_author = ""
        first = article.find(".//Author/LastName")
        if first is not None and first.text:
            first_author = first.text.strip()
        year_el = article.find(".//DateCompleted/Year")
        if year_el is None:
            year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
        year = year_el.text.strip() if year_el is not None and year_el.text else ""
        out[pmid] = {"title": title, "first_author": first_author, "year": year}
    return out


# -- Matching -----------------------------------------------------------

_YEAR = re.compile(r"(19|20)\d{2}")
_STRIP_SUFFIX = re.compile(r"\s+(et al\.?|and colleagues?)\s*", re.IGNORECASE)


def normalise_study_id(raw: str) -> tuple[str, str]:
    """Return (lastname_lower, year) from a 'Smith 2020' style study id."""
    if not raw:
        return "", ""
    s = _STRIP_SUFFIX.sub(" ", raw).strip()
    year_match = _YEAR.search(s)
    year = year_match.group(0) if year_match else ""
    # Drop year and take the first token as the last name
    without_year = _YEAR.sub("", s).strip(" ,;.-")
    first_token = without_year.split()[0] if without_year else ""
    return first_token.lower().strip(" ,;.-"), year


def find_matching_entry(
    paper_meta: dict, extractions: list[dict]
) -> tuple[dict | None, list[dict]]:
    """Find the extraction entry whose study_id matches this paper's author+year.

    Returns (best_match, other_same_year_candidates). If `best_match` is
    None, no entry matched at all.
    """
    target_last = (paper_meta.get("first_author") or "").lower()
    target_year = (paper_meta.get("year") or "")
    best = None
    same_year = []
    for ent in extractions:
        last, year = normalise_study_id(ent.get("study_id", ""))
        if not last:
            continue
        if last == target_last and year == target_year:
            best = ent
            break
        if year == target_year:
            same_year.append(ent)
    return best, same_year


# -- Comparison ---------------------------------------------------------

DOMAIN_FIELDS = [
    "overall_rob",
    "randomization_bias",
    "deviation_bias",
    "missing_outcome_bias",
    "measurement_bias",
    "reporting_bias",
]


def rating_match(extraction: dict, db_row: dict) -> dict:
    """Compare extraction ratings vs DB ratings field-by-field."""
    diffs: dict[str, tuple[str, str]] = {}
    for f in DOMAIN_FIELDS:
        ext = (extraction.get(f) or "")
        dbv = (db_row.get(f) or "")
        # Treat None, "", and "null" as the same empty sentinel
        if _norm(ext) != _norm(dbv):
            diffs[f] = (ext, dbv)
    return diffs


def _norm(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    return "" if s in {"none", "null", "n/a", ""} else s


# -- Report -------------------------------------------------------------

async def build_report(args: argparse.Namespace) -> int:
    backlinks = load_backlinks(args.db)
    cache = load_llm_cache(args.cache)
    review_pmids = sorted(backlinks.keys())
    logger.info("Source reviews with back-links: %s", review_pmids)

    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT},
    ) as client:
        # Resolve PMID→PMCID for each review
        review_info: dict[str, dict] = {}
        for rpmid in review_pmids:
            pmcid, title, year = await resolve_review_pmid_to_pmcid(client, rpmid)
            review_info[rpmid] = {"pmcid": pmcid, "title": title, "year": year}
            await asyncio.sleep(0.35)

        # Fetch PubMed metadata for all back-linked trial PMIDs in one batch
        all_trial_pmids = sorted({p["pmid"] for paps in backlinks.values() for p in paps})
        meta = await fetch_paper_meta(client, all_trial_pmids)

    # Build the report
    lines: list[str] = []
    lines.append("# Phase 1 — Cochrane mapping fidelity audit\n")
    lines.append(f"Generated from {args.db.name} and {args.cache.name}.")
    lines.append(f"Total back-linked papers audited: {sum(len(v) for v in backlinks.values())}\n")

    summary: dict[str, int] = {
        "backlinks_total": 0,
        "pmcid_resolved": 0,
        "cache_present": 0,
        "cache_empty": 0,
        "matched": 0,
        "orphan_no_match": 0,
        "exact_rating_match": 0,
        "partial_rating_match": 0,
        "rating_mismatch": 0,
    }

    for rpmid in review_pmids:
        info = review_info.get(rpmid, {})
        pmcid = info.get("pmcid", "")
        cache_entries = cache.get(pmcid, [])
        papers = backlinks[rpmid]
        summary["backlinks_total"] += len(papers)
        if pmcid:
            summary["pmcid_resolved"] += 1
        if cache_entries:
            summary["cache_present"] += 1
        elif pmcid and pmcid in cache:
            summary["cache_empty"] += 1

        lines.append(f"## Review PMID {rpmid}  (PMCID: {pmcid or '—'})")
        lines.append(f"**Title:** {info.get('title','(unresolved)')}")
        lines.append(f"**Year:** {info.get('year','?')}   "
                     f"**Cached RoB entries:** {len(cache_entries)}   "
                     f"**Back-linked papers in DB:** {len(papers)}\n")

        if not cache_entries:
            lines.append("_No cached extraction available for this review — "
                         "re-extraction required for fidelity check._\n")
            continue

        lines.append("### Cached RoB extractions (what Stage A read from this review)")
        lines.append("| # | study_id | overall | rand | dev | miss | meas | rep |")
        lines.append("|---|----------|---------|------|-----|------|------|-----|")
        for i, ent in enumerate(cache_entries, 1):
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                i,
                ent.get("study_id", "?"),
                ent.get("overall_rob", "?"),
                ent.get("randomization_bias", "-"),
                ent.get("deviation_bias", "-"),
                ent.get("missing_outcome_bias", "-"),
                ent.get("measurement_bias", "-"),
                ent.get("reporting_bias", "-"),
            ))
        lines.append("")

        lines.append("### Back-linked DB papers and their matching extraction")
        lines.append("| PMID | first_author year | title (truncated) | match? | DB rating | extraction rating | diff |")
        lines.append("|------|-------------------|-------------------|--------|-----------|-------------------|------|")
        for p in papers:
            pmid = p["pmid"]
            pm = meta.get(pmid, {})
            author = pm.get("first_author", "?")
            year = pm.get("year", "?")
            title = (pm.get("title") or p.get("title") or "")[:70]
            match, same_year = find_matching_entry(pm, cache_entries)
            if match:
                summary["matched"] += 1
                diffs = rating_match(match, p)
                if not diffs:
                    summary["exact_rating_match"] += 1
                    mstatus, diff_txt = "exact", "—"
                else:
                    if len(diffs) <= 2 and "overall_rob" not in diffs:
                        summary["partial_rating_match"] += 1
                        mstatus = "per-domain only"
                    else:
                        summary["rating_mismatch"] += 1
                        mstatus = "mismatch"
                    diff_txt = "; ".join(f"{k}: ext={v[0]!r} db={v[1]!r}"
                                          for k, v in diffs.items())
                ext_rating = (f"{match.get('overall_rob','?')}  "
                              f"[{match.get('randomization_bias','-')}/"
                              f"{match.get('deviation_bias','-')}/"
                              f"{match.get('missing_outcome_bias','-')}/"
                              f"{match.get('measurement_bias','-')}/"
                              f"{match.get('reporting_bias','-')}]")
                match_note = f"YES: row matching '{match.get('study_id','?')}'"
            else:
                summary["orphan_no_match"] += 1
                mstatus = "ORPHAN"
                ext_rating = "—"
                diff_txt = (f"no study_id matches '{author} {year}'; "
                            f"same-year rows: "
                            f"[{'; '.join(e.get('study_id','?') for e in same_year)[:100] or 'none'}]")
                match_note = "NO"
            db_rating = (f"{p['overall_rob']}  "
                         f"[{p['randomization_bias']}/"
                         f"{p['deviation_bias']}/"
                         f"{p['missing_outcome_bias']}/"
                         f"{p['measurement_bias']}/"
                         f"{p['reporting_bias']}]")
            lines.append(f"| {pmid} | {author} {year} | {title} | "
                         f"**{mstatus}** {match_note} | {db_rating} | "
                         f"{ext_rating} | {diff_txt} |")
        lines.append("")

    # Summary
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", args.output)

    print()
    print("Fidelity summary:")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")
    print(f"\nFull report: {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(build_report(args))


if __name__ == "__main__":
    raise SystemExit(main())
