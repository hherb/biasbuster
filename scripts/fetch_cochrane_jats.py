"""Fetch JATS full-text XML for Cochrane papers flagged 'reachable' in the probe.

Reads dataset/cochrane_fulltext_probe.csv and, for every row with
status=='reachable', downloads JATS XML from Europe PMC's
{pmcid}/fullTextXML endpoint into the same disk cache the CLI already
uses (~/.biasbuster/downloads/pmid/{pmid}.jats.xml).

Resumable: skips PMIDs that already have a non-empty cached file.

Usage:
    uv run python scripts/fetch_cochrane_jats.py
    uv run python scripts/fetch_cochrane_jats.py --limit 5  # smoke test
    uv run python scripts/fetch_cochrane_jats.py --probe path/to/probe.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biasbuster.utils.retry import fetch_with_retry  # noqa: E402

logger = logging.getLogger("fetch_cochrane_jats")

EUROPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

DEFAULT_PROBE_CSV = Path("dataset/cochrane_fulltext_probe.csv")
DEFAULT_CACHE_DIR = Path.home() / ".biasbuster" / "downloads" / "pmid"

# Polite rate limit. Europe PMC documents no hard cap but recommends
# ~3 req/s for unauthenticated traffic. Full-text is heavier than search
# so we go a bit slower.
REQUESTS_PER_SECOND = 2
REQUEST_DELAY_SECONDS = 1.0 / REQUESTS_PER_SECOND
HTTP_TIMEOUT_SECONDS = 60.0
USER_AGENT = "biasbuster/cochrane-jats-fetch (https://github.com/hherb/biasbuster)"

# Skip suspiciously tiny responses — these are typically 404 stub pages
# or "not available" placeholders rather than real JATS.
MIN_JATS_BYTES = 1024


def reachable_rows(probe_csv: Path) -> list[tuple[str, str]]:
    """Return [(pmid, pmcid), ...] for every reachable row in the probe CSV."""
    if not probe_csv.exists():
        raise FileNotFoundError(f"Probe CSV not found: {probe_csv}")
    out: list[tuple[str, str]] = []
    with probe_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "reachable" and row.get("pmcid"):
                out.append((row["pmid"], row["pmcid"]))
    return out


def already_cached(cache_dir: Path, pmid: str) -> bool:
    """A non-empty JATS file already exists for this PMID."""
    p = cache_dir / f"{pmid}.jats.xml"
    return p.exists() and p.stat().st_size >= MIN_JATS_BYTES


async def fetch_one(
    client: httpx.AsyncClient, pmid: str, pmcid: str, cache_dir: Path
) -> tuple[str, str, int]:
    """Download JATS for one paper. Returns (pmid, status, bytes_written)."""
    url = EUROPMC_FULLTEXT.format(pmcid=pmcid)
    try:
        resp = await fetch_with_retry(
            client, "GET", url, max_retries=3, base_delay=2.0
        )
    except Exception as exc:  # noqa: BLE001 — surface raw error
        logger.warning("PMID %s (%s): fetch error: %s", pmid, pmcid, exc)
        return pmid, "error", 0

    if resp.status_code == 404:
        return pmid, "not_found", 0
    if resp.status_code != 200:
        logger.warning("PMID %s (%s): HTTP %d", pmid, pmcid, resp.status_code)
        return pmid, f"http_{resp.status_code}", 0

    body = resp.content
    if len(body) < MIN_JATS_BYTES:
        logger.warning(
            "PMID %s (%s): too-small response (%d bytes), treating as missing",
            pmid, pmcid, len(body),
        )
        return pmid, "too_small", len(body)

    out_path = cache_dir / f"{pmid}.jats.xml"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return pmid, "ok", len(body)


async def run_fetch(
    pairs: list[tuple[str, str]],
    cache_dir: Path,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """Fetch JATS for each (pmid, pmcid) pair. Returns status counts."""
    counts: dict[str, int] = {}
    bytes_total = 0
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for i, (pmid, pmcid) in enumerate(pairs, start=1):
            pmid, status, n_bytes = await fetch_one(client, pmid, pmcid, cache_dir)
            counts[status] = counts.get(status, 0) + 1
            bytes_total += n_bytes
            if i % 25 == 0:
                logger.info(
                    "fetched %d/%d — ok=%d not_found=%d error=%d (%.1f MB)",
                    i,
                    len(pairs),
                    counts.get("ok", 0),
                    counts.get("not_found", 0),
                    counts.get("error", 0),
                    bytes_total / (1024 * 1024),
                )
            await asyncio.sleep(request_delay)
    counts["_bytes_total"] = bytes_total
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch at most this many PMIDs (smoke test).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if a cached JATS file already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pairs = reachable_rows(args.probe)
    if not args.force:
        pairs = [(pmid, pmcid) for pmid, pmcid in pairs if not already_cached(args.cache_dir, pmid)]
    if args.limit is not None:
        pairs = pairs[: args.limit]

    logger.info(
        "Reachable PMIDs in probe: %d | to fetch now (after skipping cached): %d",
        len(reachable_rows(args.probe)),
        len(pairs),
    )
    if not pairs:
        logger.info("Nothing to fetch. Exiting.")
        return 0

    started = time.monotonic()
    counts = asyncio.run(run_fetch(pairs, args.cache_dir))
    elapsed = time.monotonic() - started

    logger.info(
        "Done in %.0fs. ok=%d not_found=%d error=%d total=%.1f MB",
        elapsed,
        counts.get("ok", 0),
        counts.get("not_found", 0),
        counts.get("error", 0),
        counts.get("_bytes_total", 0) / (1024 * 1024),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
