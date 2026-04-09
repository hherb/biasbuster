#!/usr/bin/env python3
"""Import and annotate a single paper by PMID or DOI.

If the paper already exists in the database, skips straight to annotation.
If not, fetches it from PubMed, stores it, runs enrichment, then annotates.

Usage:
    uv run python annotate_single_paper.py --pmid 41271640
    uv run python annotate_single_paper.py --pmid 41271640 --model anthropic
    uv run python annotate_single_paper.py --pmid 41271640 --force
    uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001
    uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001 --source cochrane_rob
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from biasbuster.annotators import is_retraction_notice
from config import Config
from biasbuster.database import Database
from biasbuster.pipeline import create_annotator

logger = logging.getLogger(__name__)


async def resolve_pmid(
    doi: str, config: Config
) -> Optional[str]:
    """Convert a DOI to a PMID via the NCBI ID Converter API.

    Args:
        doi: The DOI to resolve.
        config: Application configuration (provides mailto, API key).

    Returns:
        PMID string, or None if the DOI could not be resolved.
    """
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector

    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        mapping = await collector.doi_to_pmid([doi])
    pmid = mapping.get(doi)
    if not pmid:
        logger.error(f"Could not resolve DOI {doi} to a PMID")
    return pmid


async def fetch_paper(
    pmid: str, config: Config
) -> Optional[dict]:
    """Fetch a paper from PubMed by PMID.

    Args:
        pmid: The PubMed ID to fetch.
        config: Application configuration (provides mailto, API key).

    Returns:
        Paper dict with title, abstract, authors, etc., or None on failure.
    """
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector

    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        paper = await collector.fetch_pubmed_abstract(pmid)
    if not paper:
        logger.error(f"Could not fetch PMID {pmid} from PubMed")
    return paper


def enrich_paper(
    pmid: str, title: str, abstract: str, db: Database, config: Config
) -> None:
    """Run heuristic enrichment (effect size audit) and store the result.

    Args:
        pmid: Paper PMID.
        title: Paper title.
        abstract: Paper abstract text.
        db: Database instance.
        config: Application configuration (provides suspicion thresholds).
    """
    from biasbuster.enrichers.effect_size_auditor import ReportingPattern, audit_abstract

    audit = audit_abstract(pmid=pmid, title=title, abstract=abstract)
    score = audit.reporting_bias_score

    if score >= config.high_suspicion_threshold:
        suspicion_level = "high"
    elif score < config.low_suspicion_threshold:
        suspicion_level = "low"
    else:
        suspicion_level = "medium"

    db.upsert_enrichment(pmid, {
        "suspicion_level": suspicion_level,
        "reporting_bias_score": score,
        "effect_size_audit": {
            "pattern": audit.pattern.value,
            "reporting_bias_score": score,
            "relative_only": audit.pattern == ReportingPattern.RELATIVE_ONLY,
            "flags": audit.flags,
            "relative_measures": audit.relative_measures_found[:5],
            "absolute_measures": audit.absolute_measures_found[:5],
        },
    })
    logger.info(
        f"Enrichment: suspicion={suspicion_level}, "
        f"reporting_bias_score={score:.3f}, flags={audit.flags}"
    )


async def annotate_paper(
    pmid: str,
    paper: dict,
    db: Database,
    config: Config,
    model_name: str,
    force: bool = False,
    two_call: bool = True,
) -> bool:
    """Annotate a single paper and store the result.

    Args:
        pmid: Paper PMID.
        paper: Full paper dict from the database.
        db: Database instance.
        config: Application configuration.
        model_name: Annotator backend ("anthropic" or "deepseek").
        force: If True, delete any existing annotation and re-annotate.
        two_call: If True (default), use v3 two-call pipeline.

    Returns:
        True if annotation succeeded, False otherwise.
    """
    annotator = create_annotator(config, model_name)
    if annotator is None:
        return False

    # Pipeline uses the backend name (e.g. "deepseek") as the DB key,
    # not the specific model string (e.g. "deepseek-reasoner").
    db_model_name = model_name

    if db.has_annotation(pmid, db_model_name):
        if force:
            db.delete_annotation(pmid, db_model_name)
            logger.info(
                f"Deleted existing annotation for PMID {pmid}/{db_model_name}"
            )
        else:
            logger.info(
                f"PMID {pmid} already annotated by {db_model_name}, skipping. "
                f"Use --force to re-annotate."
            )
            return True

    annotate_fn = (
        annotator.annotate_abstract_two_call if two_call
        else annotator.annotate_abstract
    )
    mode_label = "two-call v3" if two_call else "single-call v1"
    logger.info(f"Using {mode_label} annotation mode")

    async with annotator:
        result = await annotate_fn(
            pmid=pmid,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            metadata=paper,
        )

    if result is None:
        logger.error(f"Annotation failed for PMID {pmid}")
        return False

    # Store — mirror the pipeline's save_annotation logic
    result["abstract_text"] = paper.get("abstract", "")
    result["source"] = paper.get("source", "manual_import")
    db.insert_annotation(pmid, db_model_name, result)
    logger.info(
        f"Annotation saved: severity={result.get('overall_severity')}, "
        f"bias_prob={result.get('overall_bias_probability')}, "
        f"confidence={result.get('confidence')}"
    )
    return True


async def main() -> int:
    """Entry point: resolve identifier, fetch/enrich/annotate the paper.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Import and annotate a single paper by PMID or DOI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pmid", type=str, help="PubMed ID of the paper")
    group.add_argument("--doi", type=str, help="DOI of the paper")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek",
        choices=["anthropic", "deepseek"],
        help="Annotator backend (default: deepseek)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="manual_import",
        help="Source label for newly imported papers (default: manual_import)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if an annotation already exists",
    )
    parser.add_argument(
        "--single-call",
        action="store_true",
        help="Use single-call annotation (v1) instead of two-call (v3). "
             "Default is two-call: Stage 1 extracts facts, Stage 2 assesses bias.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config()
    db = Database(config.db_path)
    db.initialize()

    try:
        # --- Resolve PMID ---
        pmid: Optional[str] = args.pmid
        if args.doi:
            logger.info(f"Resolving DOI {args.doi} to PMID...")
            pmid = await resolve_pmid(args.doi, config)
            if pmid is None:
                return 1
            logger.info(f"Resolved DOI {args.doi} -> PMID {pmid}")

        assert pmid is not None  # guaranteed by argparse

        # --- Fetch / verify paper in DB ---
        paper = db.get_paper(pmid)
        if paper:
            logger.info(
                f"Paper already in database: "
                f"source={paper.get('source')}, "
                f"title={paper.get('title', '')[:80]}"
            )
        else:
            logger.info(f"Paper not in database, fetching from PubMed...")
            paper = await fetch_paper(pmid, config)
            if paper is None:
                return 1
            paper["source"] = args.source
            db.insert_paper(paper)
            logger.info(
                f"Imported: title={paper.get('title', '')[:80]}"
            )

        # --- Validate abstract ---
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        if not abstract or not abstract.strip():
            logger.error(
                f"PMID {pmid} has no abstract — cannot annotate"
            )
            return 1

        if is_retraction_notice(title, abstract, paper):
            logger.error(
                f"PMID {pmid} is a bare retraction/withdrawal notice — "
                f"no assessable content"
            )
            return 1

        # --- Enrich ---
        enrich_paper(pmid, title, abstract, db, config)

        # Reload paper to include enrichment data in metadata
        paper = db.get_paper(pmid)
        assert paper is not None

        # --- Annotate ---
        success = await annotate_paper(
            pmid, paper, db, config, args.model,
            force=args.force,
            two_call=not args.single_call,
        )
        return 0 if success else 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
