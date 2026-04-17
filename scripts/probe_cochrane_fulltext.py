"""Probe Europe PMC for full-text reachability of Cochrane-labeled PMIDs.

Queries Europe PMC's REST search for each Cochrane-tagged PMID in the
biasbuster DB that does not already have JATS XML cached locally, and
records whether a PMC ID exists (i.e. whether full text can be fetched
via the same code path the CLI already uses).

Output: a CSV with one row per probed PMID, written incrementally so
the run is fully resumable. Rerunning skips PMIDs already in the CSV.

Usage:
    uv run python scripts/probe_cochrane_fulltext.py
    uv run python scripts/probe_cochrane_fulltext.py --output dataset/cochrane_fulltext_probe.csv
    uv run python scripts/probe_cochrane_fulltext.py --limit 50  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx

# Project root on sys.path so we can import biasbuster.utils.retry when
# invoked as `uv run python scripts/probe_cochrane_fulltext.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biasbuster.utils.retry import fetch_with_retry  # noqa: E402

logger = logging.getLogger("probe_cochrane_fulltext")

EUROPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_DB = Path("dataset/biasbuster.db")
DEFAULT_OUTPUT = Path("dataset/cochrane_fulltext_probe.csv")
DEFAULT_CACHE_DIR = Path.home() / ".biasbuster" / "downloads" / "pmid"

# Polite rate limit. Europe PMC documents no hard cap but recommends
# ~3 req/s for unauthenticated traffic.
REQUESTS_PER_SECOND = 3
REQUEST_DELAY_SECONDS = 1.0 / REQUESTS_PER_SECOND
HTTP_TIMEOUT_SECONDS = 30.0
USER_AGENT = "biasbuster/cochrane-fulltext-probe (https://github.com/hherb/biasbuster)"

CSV_FIELDS = [
    "pmid",
    "pmcid",          # PMC ID if present, else ""
    "in_pmc",         # bool — paper indexed in PMC
    "is_open_access", # bool — Europe PMC open-access flag
    "has_pdf",        # bool — PDF reachable via Europe PMC
    "source",         # Europe PMC source (MED, PMC, etc.)
    "journal",
    "year",
    "status",         # "reachable" | "no_pmcid" | "no_hit" | "error"
    "error",          # populated only when status == "error"
]


def cochrane_pmids_from_db(db_path: Path) -> list[str]:
    """Return all Cochrane-tagged PMIDs from the biasbuster DB."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            "SELECT pmid FROM papers WHERE source LIKE 'cochrane%' ORDER BY pmid"
        )
        return [row[0] for row in cur.fetchall() if row[0]]


def cached_jats_pmids(cache_dir: Path) -> set[str]:
    """Return the set of PMIDs that already have JATS XML cached locally."""
    if not cache_dir.exists():
        return set()
    return {p.stem.split(".")[0] for p in cache_dir.glob("*.jats.xml")}


def already_probed_pmids(csv_path: Path) -> set[str]:
    """Return PMIDs already present in the output CSV (for resume support)."""
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["pmid"] for row in reader if row.get("pmid")}


def open_csv_for_append(csv_path: Path) -> tuple[csv.DictWriter, "object"]:
    """Open the CSV for append, writing the header if the file is new.

    Returns the writer and the underlying file handle so the caller can
    flush and close.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    f = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
        f.flush()
    return writer, f


async def probe_one(client: httpx.AsyncClient, pmid: str) -> dict:
    """Query Europe PMC for one PMID and return a CSV row dict."""
    params = {
        "query": f"EXT_ID:{pmid} AND SRC:MED",
        "format": "json",
        "resultType": "lite",
        "pageSize": "1",
    }
    try:
        resp = await fetch_with_retry(
            client,
            "GET",
            EUROPMC_SEARCH,
            params=params,
            max_retries=3,
            base_delay=2.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — surface the raw error text
        logger.warning("PMID %s: probe failed: %s", pmid, exc)
        return _row(pmid, status="error", error=str(exc)[:200])

    hits = (data.get("resultList") or {}).get("result") or []
    if not hits:
        return _row(pmid, status="no_hit")

    hit = hits[0]
    pmcid = (hit.get("pmcid") or "").strip()
    in_pmc = (hit.get("inPMC") or "").upper() == "Y"
    is_oa = (hit.get("isOpenAccess") or "").upper() == "Y"
    has_pdf = (hit.get("hasPDF") or "").upper() == "Y"
    status = "reachable" if pmcid else "no_pmcid"

    return _row(
        pmid,
        pmcid=pmcid,
        in_pmc=in_pmc,
        is_open_access=is_oa,
        has_pdf=has_pdf,
        source=hit.get("source", ""),
        journal=hit.get("journalTitle", ""),
        year=hit.get("pubYear", ""),
        status=status,
    )


def _row(
    pmid: str,
    *,
    pmcid: str = "",
    in_pmc: bool = False,
    is_open_access: bool = False,
    has_pdf: bool = False,
    source: str = "",
    journal: str = "",
    year: str = "",
    status: str = "",
    error: str = "",
) -> dict:
    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "in_pmc": str(in_pmc).lower(),
        "is_open_access": str(is_open_access).lower(),
        "has_pdf": str(has_pdf).lower(),
        "source": source,
        "journal": journal,
        "year": year,
        "status": status,
        "error": error,
    }


async def run_probe(
    pmids: Iterable[str],
    csv_path: Path,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """Probe each PMID and append results to ``csv_path``. Returns counts."""
    counts: dict[str, int] = {"reachable": 0, "no_pmcid": 0, "no_hit": 0, "error": 0}
    writer, fh = open_csv_for_append(csv_path)
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for i, pmid in enumerate(pmids, start=1):
                row = await probe_one(client, pmid)
                writer.writerow(row)
                fh.flush()
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                if i % 25 == 0:
                    logger.info(
                        "probed %d — reachable=%d no_pmcid=%d no_hit=%d error=%d",
                        i,
                        counts["reachable"],
                        counts["no_pmcid"],
                        counts["no_hit"],
                        counts["error"],
                    )
                await asyncio.sleep(request_delay)
    finally:
        fh.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Probe at most this many PMIDs (smoke test).",
    )
    parser.add_argument(
        "--include-cached",
        action="store_true",
        help="Don't skip PMIDs that already have JATS cached.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_cochrane = cochrane_pmids_from_db(args.db)
    cached = set() if args.include_cached else cached_jats_pmids(args.cache_dir)
    already_done = already_probed_pmids(args.output)

    candidates = [p for p in all_cochrane if p not in cached and p not in already_done]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    logger.info(
        "Cochrane PMIDs in DB: %d | cached JATS: %d | already probed: %d | to probe now: %d",
        len(all_cochrane),
        len(cached),
        len(already_done),
        len(candidates),
    )
    if not candidates:
        logger.info("Nothing to probe. Exiting.")
        return 0

    started = time.monotonic()
    counts = asyncio.run(run_probe(candidates, args.output))
    elapsed = time.monotonic() - started

    logger.info(
        "Done in %.0fs. New rows: reachable=%d no_pmcid=%d no_hit=%d error=%d",
        elapsed,
        counts.get("reachable", 0),
        counts.get("no_pmcid", 0),
        counts.get("no_hit", 0),
        counts.get("error", 0),
    )
    logger.info("Output: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
