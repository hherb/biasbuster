#!/usr/bin/env python3
"""Flag papers with significant inter-model severity disagreements for human review.

Compares DeepSeek and Anthropic (Sonnet) annotations and inserts
human_reviews rows for papers where the two models disagree by 2+ severity
levels (priority) or 1 level at the moderate/high boundary (secondary).

Papers are flagged with validated=0 and a descriptive note. The existing
review_gui reads from human_reviews and shows unvalidated items.

Usage:
    uv run python flag_review_candidates.py              # flag, show summary
    uv run python flag_review_candidates.py --dry-run    # preview only
    uv run python flag_review_candidates.py --gap 1      # also flag 1-level gaps
"""

import argparse
import json
import logging

from database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
SEVERITY_NAMES = {v: k for k, v in SEVERITY_ORDER.items()}


def severity_gap(sev_a: str, sev_b: str) -> int:
    """Return the absolute ordinal distance between two severity levels."""
    return abs(SEVERITY_ORDER.get(sev_a, 0) - SEVERITY_ORDER.get(sev_b, 0))


def main() -> None:
    """Flag papers with inter-model severity disagreements for human review."""
    parser = argparse.ArgumentParser(
        description="Flag review candidates based on inter-model disagreement",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing to DB",
    )
    parser.add_argument(
        "--gap", type=int, default=2,
        help="Minimum severity gap to flag (default: 2)",
    )
    parser.add_argument(
        "--model-a", default="deepseek",
        help="First annotation model (default: deepseek)",
    )
    parser.add_argument(
        "--model-b", default="anthropic",
        help="Second annotation model (default: anthropic)",
    )
    parser.add_argument(
        "--boundary-only", action="store_true",
        help="With --gap 1, only flag moderate/high boundary disagreements",
    )
    args = parser.parse_args()

    db = Database()
    try:
        # Find all overlapping annotations
        rows = db.conn.execute("""
            SELECT
                a.pmid,
                a.overall_severity AS sev_a,
                b.overall_severity AS sev_b,
                json_extract(a.annotation, '$.overall_bias_probability') AS prob_a,
                json_extract(b.annotation, '$.overall_bias_probability') AS prob_b
            FROM annotations a
            JOIN annotations b ON a.pmid = b.pmid
            WHERE a.model_name = ? AND b.model_name = ?
        """, (args.model_a, args.model_b)).fetchall()

        logger.info(
            "Found %d overlapping annotations between %s and %s",
            len(rows), args.model_a, args.model_b,
        )

        # Classify disagreements
        flagged = []
        for row in rows:
            pmid = row["pmid"]
            sev_a = row["sev_a"] or "none"
            sev_b = row["sev_b"] or "none"
            gap = severity_gap(sev_a, sev_b)

            if gap < args.gap:
                continue

            # With --boundary-only, only flag gap-1 at moderate/high boundary
            if args.boundary_only and gap == 1:
                boundary = {sev_a, sev_b}
                if not (boundary == {"moderate", "high"}):
                    continue

            prob_a = row["prob_a"] or 0.0
            prob_b = row["prob_b"] or 0.0

            if gap >= 2:
                priority = "HIGH"
                reason = (
                    f"Severity gap {gap}: {args.model_a}={sev_a} vs "
                    f"{args.model_b}={sev_b} (prob {prob_a:.2f} vs {prob_b:.2f})"
                )
            else:
                priority = "MODERATE"
                reason = (
                    f"Boundary disagreement: {args.model_a}={sev_a} vs "
                    f"{args.model_b}={sev_b} (prob {prob_a:.2f} vs {prob_b:.2f})"
                )

            flagged.append({
                "pmid": pmid,
                "gap": gap,
                "priority": priority,
                "sev_a": sev_a,
                "sev_b": sev_b,
                "note": reason,
            })

        # Sort: biggest gaps first, then by PMID
        flagged.sort(key=lambda x: (-x["gap"], x["pmid"]))

        # Summary
        high_ct = sum(1 for f in flagged if f["priority"] == "HIGH")
        mod_ct = sum(1 for f in flagged if f["priority"] == "MODERATE")
        logger.info(
            "Flagged %d papers: %d HIGH priority (gap >= 2), %d MODERATE (gap = 1)",
            len(flagged), high_ct, mod_ct,
        )

        # Show flagged papers
        for f in flagged:
            logger.info(
                "  [%s] PMID %s: %s=%s vs %s=%s (gap %d)",
                f["priority"], f["pmid"],
                args.model_a, f["sev_a"],
                args.model_b, f["sev_b"],
                f["gap"],
            )

        if not flagged:
            logger.info("No papers to flag at gap >= %d", args.gap)
            return

        # Insert human_reviews rows
        if not args.dry_run:
            inserted = 0
            for f in flagged:
                db.upsert_review(
                    pmid=f["pmid"],
                    model_name=args.model_a,
                    validated=False,
                    notes=f"[AUTO-FLAGGED] {f['note']}",
                    flagged=True,
                )
                inserted += 1
            logger.info(
                "Inserted %d human_review rows (validated=0) for model '%s'",
                inserted, args.model_a,
            )
        else:
            logger.info("Dry run — no rows written")

        # Gap distribution summary
        from collections import Counter
        gap_dist = Counter(f["gap"] for f in flagged)
        for g in sorted(gap_dist):
            logger.info("  Gap %d: %d papers", g, gap_dist[g])

    finally:
        db.close()


if __name__ == "__main__":
    main()
