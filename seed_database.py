"""
Database Seed & Clean Script

Reproducible post-collection cleanup and enrichment steps that bring
a freshly-collected database into a state ready for annotation.

Run after ``pipeline.py --stage collect`` (or as part of ``--stage all``):

    uv run python seed_database.py                    # all steps
    uv run python seed_database.py --step enrich-rw   # just retraction reasons
    uv run python seed_database.py --step fetch-abs   # just missing abstracts
    uv run python seed_database.py --step clean       # just retraction notice filter

Steps (idempotent, safe to re-run):

1. **enrich-rw** — Download the Retraction Watch CSV from Crossref Labs and
   enrich ``retraction_reasons`` in the papers table with structured reason
   codes from the controlled RW vocabulary (~111 categories).

2. **fetch-abs** — Fetch missing abstracts from PubMed for Cochrane RoB
   papers (and any other paper with an empty abstract but a valid PMID).

3. **clean** — Flag/remove bare retraction notices from the papers table
   (papers whose abstract is just "This article has been retracted…").

See docs/MISTAKES_ROUND_1_AND_FIXES.md for why these steps matter.
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import httpx

from biasbuster.annotators import is_retraction_notice
from config import Config
from biasbuster.database import Database
from biasbuster.utils.retry import fetch_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Enrich retraction reasons from Retraction Watch CSV
# ---------------------------------------------------------------------------

async def enrich_retraction_reasons(config: Config, db: Database) -> int:
    """Download the RW CSV and update retraction_reasons for matching papers.

    The Crossref REST API only provides the generic "Retraction" label.
    The CSV from Crossref Labs contains the Retraction Watch controlled
    vocabulary with ~111 structured reason categories.

    Matching is done by PMID (``OriginalPaperPubMedID`` in the CSV).
    Falls back to DOI matching for papers without a PMID in the CSV.

    Returns:
        Number of papers whose retraction_reasons were updated.
    """
    logger.info("=== Enriching retraction reasons from Retraction Watch CSV ===")

    # Cache the CSV locally to avoid re-downloading every run (~30s + bandwidth).
    # Re-download if the cache is older than 7 days.
    cache_path = Path(config.output_dir) / "retraction_watch_cache.csv"
    cache_max_age_days = 7

    need_download = True
    if cache_path.exists():
        import time
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < cache_max_age_days:
            logger.info(
                f"Using cached RW CSV ({age_days:.1f} days old): {cache_path}"
            )
            need_download = False

    if need_download:
        url = f"{config.retraction_watch_csv_url}?{config.crossref_mailto}"
        logger.info(f"Downloading RW CSV from {config.retraction_watch_csv_url} ...")

        async with httpx.AsyncClient(timeout=httpx.Timeout(config.http_timeout_long)) as client:
            resp = await fetch_with_retry(
                client, "GET", url,
                timeout=httpx.Timeout(config.http_timeout_long),
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay,
            )

        if resp.status_code != 200:
            logger.error(f"Failed to download RW CSV: HTTP {resp.status_code}")
            return 0

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        logger.info(f"Cached RW CSV to {cache_path}")
        csv_text = resp.text
    else:
        csv_text = cache_path.read_text(encoding="utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    logger.info(f"Loaded {len(rows)} entries from Retraction Watch CSV")

    # Build lookup indexes: PMID → reasons, DOI → reasons
    pmid_to_reasons: dict[str, list[str]] = {}
    doi_to_reasons: dict[str, list[str]] = {}

    for row in rows:
        reason_str = row.get("Reason", "").strip()
        if not reason_str:
            continue
        reasons = [r.strip() for r in reason_str.split(";") if r.strip()]

        # PMID index
        pmid = (row.get("OriginalPaperPubMedID") or "").strip()
        if pmid and pmid != "0":
            pmid_to_reasons[pmid] = reasons

        # DOI index (normalized to lowercase)
        doi = (row.get("OriginalPaperDOI") or "").strip().lower()
        if doi:
            doi_to_reasons[doi] = reasons

    logger.info(
        f"RW CSV index: {len(pmid_to_reasons)} by PMID, "
        f"{len(doi_to_reasons)} by DOI"
    )

    # Update papers in the database
    conn = db.conn
    retracted = conn.execute(
        "SELECT pmid, doi, retraction_reasons FROM papers "
        "WHERE source = 'retraction_watch'"
    ).fetchall()

    updated = 0
    for pmid, doi, current_reasons_json in retracted:
        # Skip if already has structured reasons (not just ["Retraction"])
        current = json.loads(current_reasons_json) if current_reasons_json else []
        if current and not all(r.lower() == "retraction" for r in current):
            continue  # Already enriched

        # Look up by PMID first, then DOI
        reasons = pmid_to_reasons.get(pmid)
        if not reasons and doi:
            reasons = doi_to_reasons.get(doi.lower())

        if reasons:
            conn.execute(
                "UPDATE papers SET retraction_reasons = ? WHERE pmid = ?",
                (json.dumps(reasons), pmid),
            )
            updated += 1

    conn.commit()
    logger.info(f"Updated retraction_reasons for {updated}/{len(retracted)} papers")

    # Log reason distribution
    if updated > 0:
        from collections import Counter
        reason_counts: Counter[str] = Counter()
        rows_after = conn.execute(
            "SELECT retraction_reasons FROM papers WHERE source = 'retraction_watch'"
        ).fetchall()
        for (r,) in rows_after:
            parsed = json.loads(r) if r else []
            for reason in parsed:
                if reason.lower() != "retraction":
                    reason_counts[reason] += 1
        logger.info("Top 15 structured retraction reasons:")
        for reason, count in reason_counts.most_common(15):
            logger.info(f"  {count:4d}  {reason}")

    return updated


# ---------------------------------------------------------------------------
# Step 2: Fetch missing abstracts from PubMed
# ---------------------------------------------------------------------------

async def fetch_missing_abstracts(config: Config, db: Database) -> int:
    """Fetch abstracts from PubMed for papers with empty/missing abstracts.

    Primarily targets Cochrane RoB papers (collected without abstracts)
    but also fills in any other paper that has a PMID but no abstract.

    Returns:
        Number of abstracts fetched.
    """
    logger.info("=== Fetching missing abstracts from PubMed ===")

    conn = db.conn
    rows = conn.execute(
        "SELECT pmid, source FROM papers "
        "WHERE (abstract IS NULL OR length(abstract) < 100) "
        "AND pmid IS NOT NULL AND pmid != ''"
    ).fetchall()

    if not rows:
        logger.info("No papers with missing abstracts found")
        return 0

    pmids = [r[0] for r in rows]
    sources = {r[0]: r[1] for r in rows}
    logger.info(
        f"Found {len(pmids)} papers with missing/short abstracts "
        f"(sources: {', '.join(set(sources.values()))})"
    )

    # Use the existing PubMed batch fetch infrastructure
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector

    fetched = 0
    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        results = await collector.fetch_pubmed_abstracts_batch(
            pmids, batch_size=config.pubmed_fetch_batch,
        )

    for pmid, data in results.items():
        abstract = data.get("abstract", "")
        if not abstract or len(abstract) < 50:
            continue

        # Update abstract and any other metadata we got
        updates = {"abstract": abstract}
        if data.get("title"):
            updates["title"] = data["title"]
        if data.get("journal"):
            updates["journal"] = data["journal"]
        if data.get("year"):
            updates["year"] = data["year"]
        if data.get("authors"):
            updates["authors"] = json.dumps(data["authors"])
        if data.get("grants"):
            updates["grants"] = json.dumps(data["grants"])
        if data.get("mesh_terms"):
            updates["mesh_terms"] = json.dumps(data["mesh_terms"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [pmid]
        conn.execute(
            f"UPDATE papers SET {set_clause} WHERE pmid = ?",
            values,
        )
        fetched += 1

    conn.commit()
    logger.info(f"Fetched and updated {fetched}/{len(pmids)} abstracts from PubMed")

    # Report remaining gaps
    still_missing = conn.execute(
        "SELECT COUNT(*) FROM papers "
        "WHERE (abstract IS NULL OR length(abstract) < 100) "
        "AND pmid IS NOT NULL AND pmid != ''"
    ).fetchone()[0]
    if still_missing:
        logger.warning(
            f"{still_missing} papers still have missing abstracts "
            "(PubMed may not have them)"
        )

    return fetched


# ---------------------------------------------------------------------------
# Step 3: Clean retraction notices
# ---------------------------------------------------------------------------

def clean_retraction_notices(db: Database) -> int:
    """Soft-delete bare retraction notices in the papers table.

    These are papers whose abstract is just administrative text like
    "This article has been retracted…" with no assessable research content.

    Uses the same ``is_retraction_notice()`` function that the annotation
    stage uses for filtering.  Sets ``excluded=1`` rather than deleting,
    so the data is preserved but skipped by downstream stages.

    Idempotent: already-excluded papers are not re-checked.

    Returns:
        Number of papers newly excluded.
    """
    logger.info("=== Cleaning bare retraction notices ===")

    conn = db.conn
    rows = conn.execute(
        "SELECT pmid, title, abstract FROM papers "
        "WHERE source = 'retraction_watch' AND excluded = 0"
    ).fetchall()

    to_exclude: list[str] = []
    for pmid, title, abstract in rows:
        if is_retraction_notice(title or "", abstract or ""):
            to_exclude.append(pmid)

    if to_exclude:
        placeholders = ", ".join(["?"] * len(to_exclude))
        conn.execute(
            f"UPDATE papers SET excluded = 1, excluded_reason = 'retraction_notice' "
            f"WHERE pmid IN ({placeholders})",
            to_exclude,
        )
        conn.commit()
        logger.info(
            f"Excluded {len(to_exclude)} bare retraction notices (soft-delete)"
        )
    else:
        logger.info("No new bare retraction notices found to exclude")

    # Report
    total_excluded = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE excluded = 1"
    ).fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'retraction_watch' AND excluded = 0"
    ).fetchone()[0]
    logger.info(
        f"Retraction Watch: {active} active, {total_excluded} excluded"
    )

    return len(to_exclude)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(db: Database) -> None:
    """Print a summary of the database state after seeding."""
    conn = db.conn

    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM papers WHERE excluded = 0").fetchone()[0]
    excluded = total - active
    print(f"\nTotal papers: {total} ({active} active, {excluded} excluded)")

    # By source (active only)
    sources = conn.execute(
        "SELECT source, COUNT(*) FROM papers WHERE excluded = 0 "
        "GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    for src, count in sources:
        print(f"  {src}: {count}")

    # Abstracts
    with_abs = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND length(abstract) > 100"
    ).fetchone()[0]
    print(f"\nWith abstract: {with_abs} ({with_abs*100//total}%)")
    print(f"Without abstract: {total - with_abs}")

    # Retraction reasons quality
    rw_total = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'retraction_watch'"
    ).fetchone()[0]
    if rw_total > 0:
        rw_structured = conn.execute(
            "SELECT COUNT(*) FROM papers "
            "WHERE source = 'retraction_watch' "
            "AND retraction_reasons IS NOT NULL "
            "AND retraction_reasons != '[]' "
            "AND retraction_reasons NOT LIKE '%\"Retraction\"%'"
            "AND retraction_reasons LIKE '%/%'"  # structured reasons have "/" in them
        ).fetchone()[0]
        rw_generic = rw_total - rw_structured
        print(f"\nRetraction reasons: {rw_structured} structured, {rw_generic} generic")

    # Cochrane RoB
    cochrane = conn.execute(
        "SELECT overall_rob, COUNT(*) FROM papers "
        "WHERE source = 'cochrane_rob' GROUP BY overall_rob"
    ).fetchall()
    if cochrane:
        print("\nCochrane RoB distribution:")
        for rob, count in cochrane:
            print(f"  {rob}: {count}")
        cochrane_total = sum(c for _, c in cochrane)
        with_domains = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob' "
            "AND randomization_bias != '' AND randomization_bias IS NOT NULL"
        ).fetchone()[0]
        print(f"  Per-domain ratings: {with_domains}/{cochrane_total}")

    # Downstream tables (informational — these are populated by later pipeline stages)
    downstream = []
    for table in ("enrichments", "annotations", "human_reviews", "eval_outputs"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count > 0:
            downstream.append(f"{table}: {count}")
    if downstream:
        print(f"\nDownstream data: {', '.join(downstream)}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = ["enrich-rw", "fetch-abs", "clean"]


async def run_steps(steps: list[str], config: Config, db: Database) -> None:
    """Run the specified seed steps in order."""
    for step in steps:
        if step == "enrich-rw":
            await enrich_retraction_reasons(config, db)
        elif step == "fetch-abs":
            await fetch_missing_abstracts(config, db)
        elif step == "clean":
            clean_retraction_notices(db)
        else:
            logger.error(f"Unknown step: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed & clean the BiasBuster database after collection.",
        epilog="Run after 'pipeline.py --stage collect' or as part of '--stage all'.",
    )
    parser.add_argument(
        "--step",
        choices=ALL_STEPS + ["all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to database (default: from config.py)",
    )
    args = parser.parse_args()

    config = Config()
    db_path = args.db or config.db_path
    db = Database(db_path)
    db.initialize()

    steps = ALL_STEPS if args.step == "all" else [args.step]
    logger.info(f"Running seed steps: {', '.join(steps)} on {db_path}")

    asyncio.run(run_steps(steps, config, db))
    print_summary(db)
    db.close()


if __name__ == "__main__":
    main()
