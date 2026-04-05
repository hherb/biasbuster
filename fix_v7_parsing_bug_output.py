#!/usr/bin/env python3
"""Re-parse eval outputs using the fixed scorer.

The V7 evaluation ran with an older scorer that had two bugs:
1. Failed to extract valid JSON when the model appended trailing prose
2. Text fallback parsed the <think> block, matching wrong severity values

This script re-parses all stored raw_output using the fixed scorer and
updates the derived columns (parsed_annotation, overall_severity,
overall_bias_probability) in place. No inference is re-run.

Usage:
    uv run python fix_v7_parsing_bug_output.py
    uv run python fix_v7_parsing_bug_output.py --dry-run
    uv run python fix_v7_parsing_bug_output.py --model gpt-oss-20b-biasbusterV8
"""

import argparse
import json
import logging

from biasbuster.database import Database
from biasbuster.evaluation.scorer import parse_model_output
from biasbuster.evaluation.run import _assessment_to_annotation

DEFAULT_MODEL_ID = "gpt-oss-20b-biasbusterV7"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Re-parse eval outputs and update derived columns in the DB."""
    parser = argparse.ArgumentParser(
        description="Re-parse eval outputs with fixed scorer",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to DB",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Model ID to re-parse (default: {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()

    db = Database()
    try:
        rows = db.conn.execute(
            "SELECT pmid, mode, raw_output, overall_severity, overall_bias_probability "
            "FROM eval_outputs WHERE model_id = ?",
            (args.model,),
        ).fetchall()

        if not rows:
            logger.info("No eval outputs found for %s — nothing to do.", args.model)
            return

        logger.info("Re-parsing %d outputs for %s ...", len(rows), args.model)

        changed = 0
        unchanged = 0

        for row in rows:
            pmid = row["pmid"]
            mode = row["mode"]
            old_severity = row["overall_severity"]
            old_probability = row["overall_bias_probability"] or 0.0

            parsed = parse_model_output(row["raw_output"], pmid=pmid, model_id=args.model)
            annotation = _assessment_to_annotation(parsed)

            new_severity = annotation["overall_severity"]
            new_probability = annotation["overall_bias_probability"]

            if new_severity != old_severity or abs(new_probability - old_probability) > 0.001:
                logger.info(
                    "PMID %s: severity %s -> %s, probability %.2f -> %.2f",
                    pmid, old_severity, new_severity,
                    old_probability, new_probability,
                )
                changed += 1

                if not args.dry_run:
                    db.conn.execute(
                        """UPDATE eval_outputs
                           SET parsed_annotation = json(?),
                               overall_severity = ?,
                               overall_bias_probability = ?
                           WHERE pmid = ? AND model_id = ? AND mode = ?""",
                        (json.dumps(annotation), new_severity, new_probability,
                         pmid, args.model, mode),
                    )
            else:
                unchanged += 1

        if not args.dry_run:
            db.conn.commit()

        logger.info(
            "Done: %d/%d outputs changed%s, %d unchanged.",
            changed, len(rows),
            " (dry run — not written)" if args.dry_run else "",
            unchanged,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
