#!/usr/bin/env python
"""Backfill per-domain Cochrane RoB 2 ratings for existing papers.

Re-runs the Cochrane collector, leveraging the existing LLM cache where
possible.  Papers that already have domain data are skipped entirely
(both in the callback and at the collector level via skip_pmids).

Uses the shared ``rob_assessment_to_paper_dict`` +
``upsert_cochrane_paper`` path so domain ratings and review metadata are
updated without overwriting PubMed-fetched titles/abstracts.

Supports checkpoint/resume: on restart, papers already backfilled are
skipped, and Cochrane reviews fully processed in a previous run are
skipped via a PMCID checkpoint file.

Usage:
    uv run python backfill_cochrane_domains.py
"""

import asyncio
import json
import logging
from pathlib import Path

from config import Config
from biasbuster.database import Database
from biasbuster.collectors.cochrane_rob import (
    CochraneRoBCollector, rob_assessment_to_paper_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Checkpoint file: tracks PMCIDs fully processed in previous runs
CHECKPOINT_PATH = Path("dataset/backfill_cochrane_checkpoint.json")


def load_checkpoint() -> set[str]:
    """Load set of fully-processed PMCIDs from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text())
            pmcids = set(data.get("processed_pmcids", []))
            logger.info(f"Loaded checkpoint: {len(pmcids)} reviews already processed")
            return pmcids
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt checkpoint file — starting fresh")
    return set()


def save_checkpoint(pmcids: set[str]) -> None:
    """Persist set of fully-processed PMCIDs."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({
        "processed_pmcids": sorted(pmcids),
    }, indent=1))


async def main() -> None:
    """Re-run collector and update papers missing domain-level RoB data."""
    config = Config()
    db = Database(config.db_path)
    db.initialize()  # ensure new columns exist

    # Build set of PMIDs that already have domain data — skip these
    already_done: set[str] = {
        row[0]
        for row in db.conn.execute("""
            SELECT pmid FROM papers
            WHERE source = 'cochrane_rob'
              AND randomization_bias != '' AND randomization_bias IS NOT NULL
        """).fetchall()
    }

    total = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob'"
    ).fetchone()[0]
    needs_backfill = total - len(already_done)
    logger.info(
        f"Cochrane papers: {total} total, {len(already_done)} with domain data, "
        f"{needs_backfill} need backfill"
    )

    if needs_backfill == 0:
        logger.info("All Cochrane papers already have domain data — nothing to do.")
        db.close()
        return

    # Load checkpoint of fully-processed reviews
    processed_pmcids = load_checkpoint()

    # Track progress
    saved = 0
    skipped = 0

    def on_result(a) -> None:
        """Upsert paper with domain + review data, skip if already done."""
        nonlocal saved, skipped

        if not a.pmid:
            return

        # Skip papers that already have domain data (belt-and-suspenders
        # with skip_pmids passed to collector)
        if a.pmid in already_done:
            skipped += 1
            return

        paper_dict = rob_assessment_to_paper_dict(a)
        if db.upsert_cochrane_paper(paper_dict):
            saved += 1
            already_done.add(a.pmid)
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
                f"  Saved PMID {a.pmid} (RoB={a.overall_rob}, "
                f"review={a.cochrane_review_pmid}): {domain_str or 'no domain data'}"
            )

    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
        llm_api_key=config.deepseek_api_key,
        llm_api_base=config.deepseek_api_base,
        llm_model=config.deepseek_model,
        llm_max_tokens=config.deepseek_max_tokens,
        max_retries=config.max_retries,
    ) as collector:
        await collector.collect_rob_dataset(
            domains=config.focus_domains[:5],
            max_reviews=config.cochrane_max_reviews,
            max_studies=config.cochrane_rob_max,
            on_result=on_result,
            skip_pmids=already_done,
            skip_pmcids=processed_pmcids,
        )

        # Save checkpoint: all reviews in the LLM cache (which persists
        # across runs) plus any previously checkpointed PMCIDs.
        all_pmcids = collector.cached_pmcids | processed_pmcids
        save_checkpoint(all_pmcids)

    # Fetch abstracts for any papers missing them
    if saved > 0:
        logger.info("Fetching missing abstracts...")
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
        f"Backfill complete: {saved} saved, {skipped} skipped. "
        f"Papers with domain data: {has_domains_after}/{total_after}"
    )

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
