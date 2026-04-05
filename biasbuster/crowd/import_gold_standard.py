"""Import crowd consensus annotations as gold standard into production DB.

Computes consensus from crowd annotations (using the revised phase) and
writes them into the production human_reviews table.

Usage:
    # Dry run (preview only)
    uv run python -m crowd.import_gold_standard \\
        --crowd-db dataset/crowd_annotations.db \\
        --prod-db dataset/biasbuster.db \\
        --model deepseek \\
        --dry-run

    # Commit to production
    uv run python -m crowd.import_gold_standard \\
        --crowd-db dataset/crowd_annotations.db \\
        --prod-db dataset/biasbuster.db \\
        --model deepseek \\
        --commit
"""

import argparse
import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from biasbuster.crowd.analysis import (
    compute_annotator_quality,
    analyze_crowd_annotations,
    compute_consensus,
)
from biasbuster.crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)


def _backup_production_db(prod_db_path: str) -> str:
    """Create a timestamped backup of the production database.

    Returns the backup path.
    """
    src = Path(prod_db_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = src.parent / f"{src.stem}_backup_{timestamp}{src.suffix}"
    shutil.copy2(src, backup_path)
    logger.info("Production DB backed up to %s", backup_path)
    return str(backup_path)


def _connect_prod(prod_db_path: str) -> sqlite3.Connection:
    """Open production DB for writing."""
    conn = sqlite3.connect(prod_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def import_gold_standard(
    crowd_db_path: str,
    prod_db_path: str,
    model_name: str,
    min_raters: int = 3,
    min_agreement: float = 0.0,
    dry_run: bool = True,
) -> dict:
    """Import crowd consensus into production human_reviews.

    Args:
        crowd_db_path: Path to crowd annotation database.
        prod_db_path: Path to production database.
        model_name: AI model name (used as human_reviews.model_name).
        min_raters: Minimum crowd annotations per paper.
        min_agreement: Minimum agreement threshold (not yet implemented
            for per-paper alpha; reserved for future use).
        dry_run: If True, preview only without writing.

    Returns:
        Summary dict with counts.
    """
    crowd_db = CrowdDatabase(crowd_db_path)
    crowd_db.initialize()

    try:
        # Compute annotator quality weights
        analysis = analyze_crowd_annotations(crowd_db, model_name)
        quality = analysis.get("annotator_quality", {})
        weights = {
            int(uid): scores["agreement_rate"]
            for uid, scores in quality.items()
        } if quality else None

        # Compute consensus
        consensus_list = compute_consensus(
            crowd_db,
            min_raters=min_raters,
            annotator_weights=weights,
        )

        if not consensus_list:
            logger.warning("No papers with sufficient raters for consensus.")
            return {"imported": 0, "skipped": 0, "total_candidates": 0}

        logger.info(
            "Computed consensus for %d papers (min_raters=%d)",
            len(consensus_list), min_raters,
        )

        if dry_run:
            logger.info("DRY RUN — no changes will be written.")
            for item in consensus_list:
                ann = item["consensus_annotation"]
                logger.info(
                    "  %s: %s (n=%d)",
                    item["pmid"],
                    ann.get("overall_severity", "?"),
                    item["n_raters"],
                )
            return {
                "imported": 0,
                "skipped": 0,
                "total_candidates": len(consensus_list),
                "dry_run": True,
            }

        # Backup before writing
        _backup_production_db(prod_db_path)

        # Write to production
        prod_conn = _connect_prod(prod_db_path)
        imported = 0
        skipped = 0

        try:
            for item in consensus_list:
                pmid = item["pmid"]
                consensus = item["consensus_annotation"]
                n_raters = item["n_raters"]

                # Check for existing human review
                existing = prod_conn.execute(
                    """SELECT notes FROM human_reviews
                       WHERE pmid = ? AND model_name = ?""",
                    (pmid, model_name),
                ).fetchone()

                if existing:
                    notes = existing[0] or ""
                    if not notes.startswith("crowd_consensus"):
                        # Expert review exists — do not overwrite
                        logger.info(
                            "Skipping %s: expert review exists", pmid
                        )
                        skipped += 1
                        continue

                # Determine if consensus agrees with AI
                ai_severity = consensus.get("overall_severity", "none")
                # We need to get the AI's original severity from production
                ai_row = prod_conn.execute(
                    """SELECT overall_severity FROM annotations
                       WHERE pmid = ? AND model_name = ?""",
                    (pmid, model_name),
                ).fetchone()

                validated = 0
                override_severity = None
                if ai_row:
                    ai_original = ai_row[0]
                    validated = int(ai_severity == ai_original)
                    if ai_severity != ai_original:
                        override_severity = ai_severity

                notes = f"crowd_consensus N={n_raters}"

                prod_conn.execute(
                    """INSERT INTO human_reviews
                           (pmid, model_name, validated, override_severity,
                            annotation, notes, reviewed_at)
                       VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                       ON CONFLICT(pmid, model_name) DO UPDATE SET
                           validated = excluded.validated,
                           override_severity = excluded.override_severity,
                           annotation = excluded.annotation,
                           notes = excluded.notes,
                           reviewed_at = excluded.reviewed_at""",
                    (
                        pmid,
                        model_name,
                        validated,
                        override_severity,
                        json.dumps(consensus),
                        notes,
                    ),
                )
                imported += 1

            prod_conn.commit()
            logger.info(
                "Imported %d consensus annotations, skipped %d",
                imported, skipped,
            )

        finally:
            prod_conn.close()

        return {
            "imported": imported,
            "skipped": skipped,
            "total_candidates": len(consensus_list),
        }

    finally:
        crowd_db.close()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Import crowd consensus as gold standard into production DB"
    )
    parser.add_argument(
        "--crowd-db", default="dataset/crowd_annotations.db",
        help="Path to crowd database",
    )
    parser.add_argument(
        "--prod-db", default="dataset/biasbuster.db",
        help="Path to production database",
    )
    parser.add_argument(
        "--model", default="deepseek",
        help="AI model name (default: deepseek)",
    )
    parser.add_argument(
        "--min-raters", type=int, default=3,
        help="Minimum raters per paper (default: 3)",
    )
    parser.add_argument(
        "--min-agreement", type=float, default=0.0,
        help="Minimum agreement threshold (default: 0.0)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run", action="store_true",
        help="Preview only, do not write to production DB",
    )
    group.add_argument(
        "--commit", action="store_true",
        help="Write consensus to production DB",
    )

    args = parser.parse_args()

    result = import_gold_standard(
        crowd_db_path=args.crowd_db,
        prod_db_path=args.prod_db,
        model_name=args.model,
        min_raters=args.min_raters,
        min_agreement=args.min_agreement,
        dry_run=args.dry_run,
    )

    print(f"\nResults: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
