"""Fetch JATS full-text XML for every PMID in ``expert_methodology_ratings``.

QUADAS-2 / Cochrane RoB 2 methodologies are full-text only — the
faithfulness harness can't run against abstracts. This script probes
Europe PMC to see which of the rated papers are available in PMC Open
Access, and caches the JATS XML to the same on-disk cache that the
single-paper annotator reads from
(``~/.biasbuster/downloads/pmid/<PMID>.jats.xml``).

Two-phase:

1. **Resolve** — query Europe PMC search for each PMID; record the
   PMCID (if ``inEPMC=Y``) or mark the paper as not-in-PMC.
2. **Fetch** — download ``{pmcid}/fullTextXML`` for each reachable
   paper, skipping files already on disk.

Usage::

    uv run python scripts/fetch_fulltext_for_expert_ratings.py \\
        --db dataset/biasbuster_recovered.db \\
        --methodology quadas_2

Reports per-PMID status so you can quickly see OA coverage before
deciding whether to chase down the missing ones via other routes
(author contact, interlibrary loan, the paper DOI's journal page).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx

from biasbuster.database import Database
from biasbuster.utils.retry import fetch_with_retry

logger = logging.getLogger(__name__)

EUROPMC_SEARCH = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
EUROPMC_FULLTEXT = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)

DEFAULT_CACHE_DIR = Path.home() / ".biasbuster" / "downloads" / "pmid"

#: Polite rate limit. Europe PMC docs recommend ~3 req/s unauthenticated;
#: full-text is heavier than search so we go a bit slower to stay below.
REQUESTS_PER_SECOND = 2
REQUEST_DELAY_SECONDS = 1.0 / REQUESTS_PER_SECOND
HTTP_TIMEOUT_SECONDS = 60.0
USER_AGENT = (
    "biasbuster/fulltext-fetch "
    "(https://github.com/hherb/biasbuster)"
)

#: Anything smaller than this is almost certainly a stub/404 page rather
#: than real JATS; the same threshold ``fetch_cochrane_jats.py`` uses.
MIN_JATS_BYTES = 1024


def _cache_path(cache_dir: Path, pmid: str) -> Path:
    return cache_dir / f"{pmid}.jats.xml"


def _already_cached(cache_dir: Path, pmid: str) -> bool:
    """A non-empty JATS file exists and passes the stub-size threshold."""
    p = _cache_path(cache_dir, pmid)
    return p.exists() and p.stat().st_size >= MIN_JATS_BYTES


def _rated_pmids(
    db_path: Path, methodology: Optional[str],
) -> list[str]:
    db = Database(db_path)
    try:
        db.initialize()
        q = (
            "SELECT DISTINCT pmid FROM expert_methodology_ratings "
            "WHERE pmid IS NOT NULL"
        )
        params: list = []
        if methodology is not None:
            q += " AND methodology = ?"
            params.append(methodology)
        rows = db.conn.execute(q, params).fetchall()
    finally:
        db.close()
    return sorted({r["pmid"] for r in rows})


async def resolve_pmcid(
    client: httpx.AsyncClient, pmid: str,
) -> Optional[str]:
    """Look up a PMID's PMCID via Europe PMC search.

    Returns the PMCID (with ``PMC`` prefix) when ``inEPMC=Y``, else None.
    None means the paper isn't in Europe PMC Open Access and the JATS
    full-text endpoint won't have it either.
    """
    query = f"EXT_ID:{pmid}"
    url = (
        f"{EUROPMC_SEARCH}?query={quote(query, safe=':')}"
        f"&format=json&resultType=core&pageSize=1"
    )
    try:
        resp = await fetch_with_retry(
            client, "GET", url, max_retries=3, base_delay=2.0,
            headers={"Accept": "application/json"},
        )
    except Exception as exc:  # noqa: BLE001 — surface the raw error
        logger.warning("PMID %s: resolve error: %s", pmid, exc)
        return None
    data = resp.json()
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return None
    first = results[0]
    if first.get("inEPMC") != "Y":
        return None
    pmcid = first.get("pmcid")
    if not pmcid:
        return None
    return pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"


async def fetch_jats(
    client: httpx.AsyncClient, pmid: str, pmcid: str, cache_dir: Path,
) -> tuple[str, int]:
    """Fetch JATS for one ``(pmid, pmcid)``, writing to the cache.

    Returns ``(status, bytes_written)``. Status is ``ok`` / ``too_small`` /
    ``http_error`` / ``fetch_error``. Bytes is 0 for failures.
    """
    url = EUROPMC_FULLTEXT.format(pmcid=pmcid)
    try:
        resp = await fetch_with_retry(
            client, "GET", url, max_retries=3, base_delay=2.0,
            headers={"Accept": "application/xml"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PMID %s (%s): fetch error: %s", pmid, pmcid, exc)
        return "fetch_error", 0
    body = resp.content
    if len(body) < MIN_JATS_BYTES:
        # Europe PMC sometimes serves a tiny stub page with 200 OK when
        # the full text isn't actually available. Treat that as absent
        # so we don't pollute the cache with un-parseable files.
        logger.info(
            "PMID %s (%s): stub response (%d bytes) — treating as absent",
            pmid, pmcid, len(body),
        )
        return "too_small", 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, pmid)
    path.write_bytes(body)
    return "ok", len(body)


async def probe_and_fetch(
    pmids: list[str],
    cache_dir: Path,
    *,
    request_delay: float = REQUEST_DELAY_SECONDS,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Resolve + fetch every PMID; return a per-PMID status record list.

    Each record has keys ``pmid``, ``pmcid``, ``status``, ``bytes``.
    Statuses: ``cached``, ``ok``, ``not_in_pmc``, ``too_small``,
    ``http_error``, ``fetch_error``, ``dry_run``.
    """
    records: list[dict[str, str]] = []
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for pmid in pmids:
            if _already_cached(cache_dir, pmid):
                p = _cache_path(cache_dir, pmid)
                records.append({
                    "pmid": pmid, "pmcid": "",
                    "status": "cached", "bytes": str(p.stat().st_size),
                })
                logger.info("PMID %s: cached (%d bytes)", pmid, p.stat().st_size)
                continue

            pmcid = await resolve_pmcid(client, pmid)
            await asyncio.sleep(request_delay)

            if not pmcid:
                records.append({
                    "pmid": pmid, "pmcid": "",
                    "status": "not_in_pmc", "bytes": "0",
                })
                logger.info("PMID %s: not in Europe PMC OA", pmid)
                continue

            if dry_run:
                records.append({
                    "pmid": pmid, "pmcid": pmcid,
                    "status": "dry_run", "bytes": "0",
                })
                logger.info(
                    "PMID %s (%s): would fetch JATS (dry-run)",
                    pmid, pmcid,
                )
                continue

            status, n_bytes = await fetch_jats(
                client, pmid, pmcid, cache_dir,
            )
            records.append({
                "pmid": pmid, "pmcid": pmcid,
                "status": status, "bytes": str(n_bytes),
            })
            if status == "ok":
                logger.info(
                    "PMID %s (%s): fetched %d bytes", pmid, pmcid, n_bytes,
                )
            await asyncio.sleep(request_delay)
    return records


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, required=True,
                   help="DB with expert_methodology_ratings.")
    p.add_argument(
        "--methodology", default=None,
        help="Restrict to this methodology (e.g. 'quadas_2'). Default: "
             "every PMID in the expert-ratings table.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="JATS cache directory. Defaults to the same path the "
             "single-paper annotator uses.",
    )
    p.add_argument(
        "--report", type=Path, default=None,
        help="Optional CSV path to write the per-PMID status report.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve PMCIDs but don't download full text.")
    return p.parse_args(argv)


def _summarise(records: list[dict[str, str]]) -> dict[str, int]:
    """Count records by status for the CLI summary line."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def _write_report(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pmid", "pmcid", "status", "bytes"],
        )
        writer.writeheader()
        writer.writerows(records)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    pmids = _rated_pmids(args.db, args.methodology)
    if not pmids:
        logger.error("No rated PMIDs found in %s", args.db)
        return 1
    logger.info("Resolving %d PMIDs via Europe PMC", len(pmids))
    records = asyncio.run(probe_and_fetch(
        pmids, args.cache_dir, dry_run=args.dry_run,
    ))
    counts = _summarise(records)
    print(f"Total PMIDs:       {len(pmids)}")
    for status in (
        "cached", "ok", "dry_run", "not_in_pmc",
        "too_small", "http_error", "fetch_error",
    ):
        if counts.get(status):
            print(f"  {status:15s} {counts[status]}")
    if args.report:
        _write_report(args.report, records)
        logger.info("Report written to %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
