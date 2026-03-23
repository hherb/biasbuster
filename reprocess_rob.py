#!/usr/bin/env python
"""Re-resolve Cochrane RoB studies that failed PMID resolution.

Reads a list of PMCIDs (from CLI args or a log file), re-processes each
review with the improved resolution code, and saves newly resolved papers
to the database.  Uses the LLM extraction cache to avoid re-spending
tokens — only reviews not in the cache require LLM calls.

Usage:
    # Re-process specific PMCIDs:
    uv run python reprocess_rob.py PMC12987355 PMC11152306 PMC12942296

    # Parse failed PMCIDs from a log file (reviews with >0 extracted, <100% resolved):
    uv run python reprocess_rob.py --from-log log.txt

    # Both (explicit PMCIDs take priority, log adds more):
    uv run python reprocess_rob.py --from-log log.txt PMC12126597
"""

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import asdict

import httpx

from collectors.cochrane_rob import CochraneRoBCollector, RoBAssessment
from config import Config
from database import Database
from utils.retry import fetch_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_failed_pmcids_from_log(log_path: str) -> list[str]:
    """Extract PMCIDs of reviews where extracted > resolved from a log file.

    Parses lines like:
        Review PMC12987355: 9 extracted, 0 with PMID. Running total: 47
    Returns PMCIDs where extracted > pmid_count.
    """
    pattern = re.compile(
        r"Review (PMC\d+): (\d+) extracted, (\d+) with PMID"
    )
    failed: list[str] = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                pmcid = m.group(1)
                extracted = int(m.group(2))
                resolved = int(m.group(3))
                if extracted > resolved:
                    failed.append(pmcid)
    return failed


async def reprocess_reviews(
    pmcids: list[str],
    config: Config,
    db: Database,
) -> None:
    """Re-process specific reviews with improved resolution code.

    For each PMCID:
    1. Fetch full-text XML from Europe PMC (free)
    2. Extract RoB assessments (from cache if available, else LLM)
    3. Extract references from XML
    4. Run multi-layer PMID resolution with the improved code
    5. Save newly resolved papers to the database
    """
    # Pre-load existing PMIDs to avoid duplicates
    existing_pmids: set[str] = set()
    rows = db.conn.execute(
        "SELECT pmid FROM papers WHERE source = 'cochrane_rob'"
    ).fetchall()
    for (pmid,) in rows:
        existing_pmids.add(pmid)
    logger.info(f"Existing Cochrane RoB papers: {len(existing_pmids)}")

    inserted_count = 0

    def save_assessment(a: RoBAssessment) -> None:
        nonlocal inserted_count
        if a.pmid in existing_pmids:
            return
        paper_dict = asdict(a)
        paper_dict["source"] = "cochrane_rob"
        paper_dict["title"] = paper_dict.pop("study_title", "")
        paper_dict["abstract"] = ""
        if db.insert_paper(paper_dict):
            inserted_count += 1
            existing_pmids.add(a.pmid)
            logger.info(
                f"  Saved PMID {a.pmid} (RoB={a.overall_rob}) — "
                f"total inserted: {inserted_count}"
            )

    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
        llm_api_key=config.deepseek_api_key,
        llm_api_base=config.deepseek_api_base,
        llm_model=config.deepseek_model,
        llm_max_tokens=config.deepseek_max_tokens,
    ) as collector:
        for pmcid in pmcids:
            logger.info(f"--- Re-processing {pmcid} ---")

            # Fetch full-text XML
            try:
                ft_resp = await fetch_with_retry(
                    collector.client, "GET",
                    f"{collector.EUROPMC_BASE}/{pmcid}/fullTextXML",
                )
                if ft_resp.status_code != 200:
                    logger.warning(f"Full text not available for {pmcid}: HTTP {ft_resp.status_code}")
                    continue
                full_text = ft_resp.text
            except Exception as e:
                logger.warning(f"Failed to fetch {pmcid}: {e}")
                continue

            # Extract RoB assessments (uses cache if available)
            assessments = await collector.extract_rob_via_llm(pmcid, full_text)
            if not assessments:
                logger.info(f"  No assessments extracted for {pmcid}")
                continue

            # Extract references for PMID matching
            refs = await collector.extract_included_study_refs(pmcid)

            # We don't have review metadata from search — use empty defaults
            for a in assessments:
                a.cochrane_review_pmid = ""
                a.cochrane_review_doi = ""
                a.cochrane_review_title = ""

            # Multi-layer PMID resolution
            await collector._resolve_pmids_from_refs(assessments, refs)
            await collector._resolve_pmids_via_doi(assessments)
            unresolved = [a for a in assessments if not a.pmid]
            if unresolved:
                await collector.resolve_study_pmids(assessments)

            resolved = sum(1 for a in assessments if a.pmid)
            logger.info(
                f"  {pmcid}: {len(assessments)} extracted, "
                f"{resolved} resolved to PMID"
            )

            # Save newly resolved papers
            for a in assessments:
                if a.pmid:
                    save_assessment(a)

            await asyncio.sleep(0.5)

    logger.info(f"Re-processing complete. Inserted {inserted_count} new papers.")

    # Fetch abstracts for newly inserted papers
    if inserted_count > 0:
        logger.info("Fetching abstracts for new papers...")
        from seed_database import fetch_missing_abstracts
        await fetch_missing_abstracts(config, db)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-resolve Cochrane RoB studies with improved PMID resolution"
    )
    parser.add_argument(
        "pmcids", nargs="*",
        help="Specific PMCIDs to re-process (e.g. PMC12987355)",
    )
    parser.add_argument(
        "--from-log", metavar="LOG_FILE",
        help="Parse log file for reviews with failed resolution",
    )
    args = parser.parse_args()

    pmcids: list[str] = list(args.pmcids or [])
    if args.from_log:
        from_log = parse_failed_pmcids_from_log(args.from_log)
        logger.info(f"Found {len(from_log)} reviews with incomplete resolution in {args.from_log}")
        # Merge, preserving order, no duplicates
        seen = set(pmcids)
        for p in from_log:
            if p not in seen:
                pmcids.append(p)
                seen.add(p)

    if not pmcids:
        print("No PMCIDs specified. Use positional args or --from-log.", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Will re-process {len(pmcids)} reviews: {pmcids}")

    config = Config()
    db = Database(config.db_path)
    with db:
        asyncio.run(reprocess_reviews(pmcids, config, db))


if __name__ == "__main__":
    main()
