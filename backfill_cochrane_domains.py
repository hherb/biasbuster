#!/usr/bin/env python
"""Backfill per-domain Cochrane RoB 2 ratings for existing papers.

Clears the LLM extraction cache and re-runs the Cochrane collector with
the updated prompt that requests per-domain ratings (D1-D5). For papers
that already exist in the DB, UPDATEs the domain columns instead of
skipping via INSERT OR IGNORE.

Usage:
    uv run python backfill_cochrane_domains.py

This is a one-time migration script. After running, the 5 domain columns
(randomization_bias, deviation_bias, missing_outcome_bias, measurement_bias,
reporting_bias) will be populated for Cochrane RoB papers, enabling
domain-level alignment analysis in expert_rob_alignment_of_annotations.py.
"""

import asyncio
import logging
from dataclasses import asdict
from pathlib import Path

from config import Config
from database import Database
from collectors.cochrane_rob import CochraneRoBCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Clear cache, re-extract with domain-aware prompt, update DB."""
    config = Config()
    db = Database(config.db_path)

    # Check current state
    total = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob'"
    ).fetchone()[0]
    has_domains = db.conn.execute("""
        SELECT COUNT(*) FROM papers
        WHERE source = 'cochrane_rob'
          AND randomization_bias != '' AND randomization_bias IS NOT NULL
    """).fetchone()[0]
    logger.info(
        f"Cochrane papers: {total} total, {has_domains} with domain data"
    )

    # Rename LLM cache to force re-extraction with updated prompt
    cache_path = Path(CochraneRoBCollector.DEFAULT_CACHE_PATH)
    backup = cache_path.with_suffix(".pre-backfill.bak")
    if cache_path.exists():
        if not backup.exists():
            # First run — preserve the original cache
            cache_path.rename(backup)
            logger.info(f"Moved LLM cache to {backup} (will re-extract with domain-aware prompt)")
        else:
            # Re-run after crash — delete partial new cache, original backup is safe
            cache_path.unlink()
            logger.info("Removed partial cache from previous run (original backup preserved)")
    else:
        logger.info("No LLM cache found — will extract fresh")

    # Track updates
    updated = 0
    inserted = 0

    def on_result(a) -> None:
        """UPDATE existing papers with domain data, INSERT new ones."""
        nonlocal updated, inserted

        pmid = a.pmid
        if not pmid:
            return

        # Check if paper already exists
        existing = db.conn.execute(
            "SELECT pmid FROM papers WHERE pmid = ?", (pmid,)
        ).fetchone()

        if existing:
            # UPDATE domain columns only
            db.conn.execute("""
                UPDATE papers SET
                    randomization_bias = ?,
                    deviation_bias = ?,
                    missing_outcome_bias = ?,
                    measurement_bias = ?,
                    reporting_bias = ?,
                    overall_rob = ?
                WHERE pmid = ?
            """, (
                a.randomization_bias,
                a.deviation_bias,
                a.missing_outcome_bias,
                a.measurement_bias,
                a.reporting_bias,
                a.overall_rob,
                pmid,
            ))
            db.conn.commit()
            updated += 1
            domain_str = ", ".join(
                f"{d}={getattr(a, d) or '?'}"
                for d in (
                    "randomization_bias", "deviation_bias",
                    "missing_outcome_bias", "measurement_bias",
                    "reporting_bias",
                )
                if getattr(a, d)
            )
            logger.info(
                f"  Updated PMID {pmid} (RoB={a.overall_rob}): {domain_str or 'no domain data'}"
            )
        else:
            # New paper — insert normally
            paper_dict = asdict(a)
            paper_dict["source"] = "cochrane_rob"
            paper_dict["title"] = paper_dict.pop("study_title", "")
            paper_dict["abstract"] = ""
            if db.insert_paper(paper_dict):
                inserted += 1
                logger.info(
                    f"  Inserted new PMID {pmid} (RoB={a.overall_rob})"
                )

    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
        llm_api_key=config.deepseek_api_key,
        llm_api_base=config.deepseek_api_base,
        llm_model=config.deepseek_model,
    ) as collector:
        await collector.collect_rob_dataset(
            domains=config.focus_domains[:5],
            max_reviews=config.cochrane_max_reviews,
            max_studies=config.cochrane_rob_max,
            on_result=on_result,
        )

    # Fetch abstracts for any new papers
    if inserted > 0:
        logger.info(f"Fetching abstracts for {inserted} new papers...")
        from seed_database import fetch_missing_abstracts
        await fetch_missing_abstracts(config, db)

    # Summary
    has_domains_after = db.conn.execute("""
        SELECT COUNT(*) FROM papers
        WHERE source = 'cochrane_rob'
          AND randomization_bias != '' AND randomization_bias IS NOT NULL
    """).fetchone()[0]
    total_after = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob'"
    ).fetchone()[0]

    logger.info(
        f"Backfill complete: {updated} updated, {inserted} new. "
        f"Papers with domain data: {has_domains_after}/{total_after}"
    )

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
