"""Audit PubMed PublicationType for every Cochrane-tagged paper in the DB.

Purpose: after the Stage E discovery that 3 non-trial papers had leaked
into the cohort as Stage A LLM extraction artifacts, this script
systematically checks every remaining Cochrane-tagged paper against
PubMed's `<PublicationType>` metadata to find any more non-trial
artifacts. Output is a classification per PMID: trial | non_trial |
ambiguous | unknown, with the PubMedtypes list preserved for review.

Writes: dataset/cochrane_pubtype_audit.csv
Prints: summary counts and a breakdown by Cochrane overall_rob rating.

Usage:
    uv run python scripts/audit_publication_types.py
    uv run python scripts/audit_publication_types.py --include-excluded  # also audit soft-deleted PMIDs
    uv run python scripts/audit_publication_types.py --apply-exclusions  # mark found non-trials as excluded
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biasbuster.utils.retry import fetch_with_retry  # noqa: E402

logger = logging.getLogger("audit_publication_types")

PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_DB = Path("dataset/biasbuster.db")
DEFAULT_OUTPUT = Path("dataset/cochrane_pubtype_audit.csv")

BATCH_SIZE = 200  # PubMed efetch accepts up to 200 IDs per request
HTTP_TIMEOUT_SECONDS = 60.0
USER_AGENT = "biasbuster/pubtype-audit"

# PubMed PublicationType strings that indicate a real randomised/controlled trial.
TRIAL_TYPES = frozenset({
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

# PublicationTypes that imply the paper is NOT a research trial.
# Presence of any of these (and absence of any trial type) → non_trial.
NON_TRIAL_TYPES = frozenset({
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
    "Case Reports",  # not RCTs; may still be assessable but not by Cochrane RoB 2
    "Systematic Review",
    "Meta-Analysis",
    "Review",
})

CSV_FIELDS = [
    "pmid",
    "overall_rob",
    "excluded_before",
    "publication_types",
    "classification",  # trial | non_trial | ambiguous | unknown
    "title",
]


def cochrane_pmids(db_path: Path, include_excluded: bool) -> list[tuple[str, str, int, str]]:
    """Return (pmid, overall_rob, excluded, title) for Cochrane-tagged papers."""
    with sqlite3.connect(str(db_path)) as conn:
        where = "source LIKE 'cochrane%'"
        if not include_excluded:
            where += " AND COALESCE(excluded,0)=0"
        rows = conn.execute(
            f"SELECT pmid, COALESCE(overall_rob,''), COALESCE(excluded,0), "
            f"COALESCE(title,'') FROM papers WHERE {where} ORDER BY pmid"
        ).fetchall()
    return rows


async def fetch_pubmed_batch(
    client: httpx.AsyncClient, pmids: list[str]
) -> dict[str, list[str]]:
    """Fetch a batch of PubMed records and return {pmid: [publication_types]}.

    Papers that PubMed doesn't return will be absent from the result dict;
    callers should treat them as 'unknown'.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = await fetch_with_retry(
        client, "GET", PUBMED_EFETCH,
        params=params, max_retries=3, base_delay=2.0,
    )
    resp.raise_for_status()
    return parse_pubtypes(resp.text)


def parse_pubtypes(xml_text: str) -> dict[str, list[str]]:
    """Extract {pmid: [publication_types]} from a PubMed efetch XML response."""
    out: dict[str, list[str]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("XML parse error: %s", exc)
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


def classify(types: list[str]) -> str:
    """Classify a paper by its PublicationType list."""
    if not types:
        return "unknown"
    tset = set(types)
    has_trial = bool(tset & TRIAL_TYPES)
    has_nontrial = bool(tset & NON_TRIAL_TYPES)
    if has_trial and not has_nontrial:
        return "trial"
    if has_trial and has_nontrial:
        # e.g. "Randomized Controlled Trial" + "Review" — keep as ambiguous
        return "ambiguous"
    if has_nontrial:
        return "non_trial"
    # Only "Journal Article" or funding-type tags — ambiguous
    return "ambiguous"


async def run_audit(
    pmids: list[str], batch_size: int = BATCH_SIZE
) -> dict[str, list[str]]:
    """Fetch PublicationType metadata for every PMID. Returns {pmid: types}."""
    types_by_pmid: dict[str, list[str]] = {}
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            logger.info("Fetching PubMed batch %d–%d of %d", i + 1, i + len(batch), len(pmids))
            try:
                got = await fetch_pubmed_batch(client, batch)
                types_by_pmid.update(got)
                if len(got) < len(batch):
                    missing = set(batch) - set(got)
                    logger.warning(
                        "PubMed returned %d/%d records (missing: %s)",
                        len(got), len(batch),
                        ", ".join(sorted(missing)[:5]) + ("..." if len(missing) > 5 else ""),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch fetch failed: %s", exc)
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.5)
    return types_by_pmid


def write_csv(
    rows: list[tuple[str, str, int, str]],
    types_by_pmid: dict[str, list[str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pmid, rob, excluded, title in rows:
            types = types_by_pmid.get(pmid, [])
            writer.writerow({
                "pmid": pmid,
                "overall_rob": rob,
                "excluded_before": excluded,
                "publication_types": "; ".join(types),
                "classification": classify(types),
                "title": title[:120],
            })


def summarize(
    rows: list[tuple[str, str, int, str]],
    types_by_pmid: dict[str, list[str]],
) -> dict:
    class_counts: Counter = Counter()
    class_by_rob: dict[str, Counter] = {}
    non_trial_types: Counter = Counter()
    non_trial_pmids: list[tuple[str, str, str, str]] = []
    for pmid, rob, _excluded, title in rows:
        types = types_by_pmid.get(pmid, [])
        cls = classify(types)
        class_counts[cls] += 1
        class_by_rob.setdefault(rob or "(empty)", Counter())[cls] += 1
        if cls == "non_trial":
            for t in types:
                if t in NON_TRIAL_TYPES:
                    non_trial_types[t] += 1
            non_trial_pmids.append((pmid, rob, "; ".join(types), title[:70]))
    return {
        "class_counts": class_counts,
        "class_by_rob": class_by_rob,
        "non_trial_types": non_trial_types,
        "non_trial_pmids": non_trial_pmids,
    }


def apply_exclusions(db_path: Path, pmids: list[str], reason: str) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        placeholders = ",".join("?" for _ in pmids)
        conn.execute(
            f"UPDATE papers SET excluded=1, excluded_reason=? "
            f"WHERE pmid IN ({placeholders}) AND COALESCE(excluded,0)=0",
            (reason, *pmids),
        )
        conn.commit()
        return conn.total_changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument(
        "--apply-exclusions",
        action="store_true",
        help="Mark all detected non-trial papers as excluded in the DB.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rows = cochrane_pmids(args.db, include_excluded=args.include_excluded)
    pmids = [r[0] for r in rows]
    logger.info("Cochrane papers to audit: %d (include_excluded=%s)",
                len(pmids), args.include_excluded)

    types_by_pmid = asyncio.run(run_audit(pmids))
    write_csv(rows, types_by_pmid, args.output)

    report = summarize(rows, types_by_pmid)
    cc = report["class_counts"]
    total = sum(cc.values())
    print()
    print("Classification summary:")
    for cls in ("trial", "non_trial", "ambiguous", "unknown"):
        n = cc.get(cls, 0)
        pct = 100.0 * n / max(1, total)
        print(f"  {cls:<12} {n:>4}  ({pct:.1f}%)")

    print("\nBreakdown by Cochrane overall_rob:")
    print(f"  {'rob':<14} {'trial':>6} {'non_trial':>10} {'ambiguous':>10} {'unknown':>8}")
    for rob, counts in sorted(report["class_by_rob"].items()):
        print(f"  {rob:<14} {counts.get('trial',0):>6} "
              f"{counts.get('non_trial',0):>10} "
              f"{counts.get('ambiguous',0):>10} "
              f"{counts.get('unknown',0):>8}")

    if report["non_trial_types"]:
        print("\nNon-trial PublicationType frequency:")
        for t, n in report["non_trial_types"].most_common():
            print(f"  {t:<35} {n:>3}")

    if report["non_trial_pmids"]:
        print(f"\nDetected {len(report['non_trial_pmids'])} non-trial PMIDs (first 15):")
        for pmid, rob, types, title in report["non_trial_pmids"][:15]:
            print(f"  {pmid} rob={rob:<14} types=[{types}]")
            print(f"           title: {title}")

    print(f"\nFull audit written to: {args.output}")

    if args.apply_exclusions and report["non_trial_pmids"]:
        to_exclude = [p for p, *_ in report["non_trial_pmids"]]
        reason = "non-trial: PubMed PublicationType audit 2026-04-17"
        changed = apply_exclusions(args.db, to_exclude, reason)
        logger.info("Applied excluded=1 to %d papers", changed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
