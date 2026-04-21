"""Backfill Cochrane RoB 2 ground truth into ``expert_methodology_ratings``.

The generic faithfulness harness reads expert ratings from a single
table regardless of methodology. Historically Cochrane RoB 2 ratings
lived as hard-coded columns on ``papers`` (``randomization_bias``,
``deviation_bias``, ``missing_outcome_bias``, ``measurement_bias``,
``reporting_bias``, ``overall_rob``). This script copies those into
``expert_methodology_ratings`` under ``methodology='cochrane_rob2'``
so the harness can treat RoB 2 and QUADAS-2 symmetrically.

The ``papers.*_bias`` columns remain on the table; this is an
additive backfill, not a migration. The hard-coded columns stay
because the Cochrane collector + provenance-invariant check
(``upsert_cochrane_paper_v2``) still writes to them.

Usage::

    uv run python scripts/backfill_rob2_expert_ratings.py \\
        --db dataset/biasbuster_recovered.db \\
        --added-by 'rob2-backfill-2026-04-21'

Reading and writing happen in the same DB because the faithfulness
harness needs both ``papers`` rows (for title / abstract context) and
``expert_methodology_ratings`` rows (for ground truth) in one place.
Run :mod:`scripts.copy_papers_for_expert_ratings` first if you need
the paper rows in the target DB.

Idempotent: re-running against the same DB updates the
``domain_ratings`` and ``overall_rating`` but leaves curator-managed
fields (``verified``, ``notes``) untouched per the usual upsert rules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from biasbuster.database import Database

logger = logging.getLogger(__name__)

METHODOLOGY: str = "cochrane_rob2"
METHODOLOGY_VERSION: str = "rob2-2019"

#: Mapping from the canonical RoB 2 domain slug to the ``papers`` column
#: that holds the expert rating. Slugs match
#: :func:`biasbuster.methodologies.cochrane_rob2.evaluation_mapping_to_ground_truth`
#: so the generic harness can join the backfilled rows to the stored
#: predictions without translation.
_DOMAIN_SLUG_TO_COLUMN: dict[str, str] = {
    "randomization": "randomization_bias",
    "deviations_from_interventions": "deviation_bias",
    "missing_outcome_data": "missing_outcome_bias",
    "outcome_measurement": "measurement_bias",
    "selection_of_reported_result": "reporting_bias",
}

#: Valid RoB 2 ratings. Rows with any out-of-vocabulary value are
#: skipped with a warning — feeding garbage through the harness would
#: silently inflate kappa on unrecognised cells.
_VALID_RATINGS: frozenset[str] = frozenset({"low", "some_concerns", "high"})


def _rating_source_for(paper: dict) -> Optional[str]:
    """Derive the ``rating_source`` token for a backfilled row.

    Keyed on the Cochrane review that assessed the paper, so two
    reviews rating the same paper produce two distinct expert rows
    (the PK includes ``rating_source``). Returns ``None`` when we
    can't identify the source review — a legacy row without review
    provenance isn't trustworthy enough to back into ground truth.
    """
    review_pmid = (paper.get("cochrane_review_pmid") or "").strip()
    if review_pmid:
        return f"cochrane_review_pmid:{review_pmid}"
    review_doi = (paper.get("cochrane_review_doi") or "").strip()
    if review_doi:
        return f"cochrane_review_doi:{review_doi}"
    return None


def _extract_domain_ratings(
    paper: dict,
) -> Optional[dict[str, dict[str, str]]]:
    """Pull the 5 per-domain ratings into the ``{slug: {"bias": r}}`` shape.

    Returns ``None`` when any required domain rating is missing or
    off-vocabulary. Matches the QUADAS-2 storage shape so the harness
    doesn't branch on methodology when reading ``domain_ratings``.
    """
    result: dict[str, dict[str, str]] = {}
    for slug, column in _DOMAIN_SLUG_TO_COLUMN.items():
        raw = paper.get(column)
        if not isinstance(raw, str):
            return None
        value = raw.strip()
        if value not in _VALID_RATINGS:
            return None
        result[slug] = {"bias": value}
    return result


def _eligible_papers(db: Database) -> list[dict]:
    """Return every paper with a usable RoB 2 rating in the legacy columns.

    Filtering happens in Python so we can apply the same
    ``_extract_domain_ratings`` vocabulary check that the upsert will.
    SQL-only filtering (``WHERE overall_rob IS NOT NULL``) would let
    malformed rows through.
    """
    rows = db.conn.execute(
        "SELECT * FROM papers "
        "WHERE overall_rob IS NOT NULL AND overall_rob != '' "
        "  AND randomization_bias IS NOT NULL "
        "  AND deviation_bias IS NOT NULL "
        "  AND missing_outcome_bias IS NOT NULL "
        "  AND measurement_bias IS NOT NULL "
        "  AND reporting_bias IS NOT NULL "
        "  AND excluded = 0"
    ).fetchall()
    return [dict(r) for r in rows]


def backfill(
    db_path: Path,
    *,
    added_by: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Read Cochrane expert columns and upsert them into the new table.

    Counts: ``eligible``, ``skipped_invalid_rating``,
    ``skipped_no_source``, ``inserted``, ``updated``.
    """
    counts = {
        "eligible": 0,
        "skipped_invalid_rating": 0,
        "skipped_no_source": 0,
        "inserted": 0,
        "updated": 0,
    }
    db = Database(db_path)
    try:
        db.initialize()
        papers = _eligible_papers(db)
        counts["eligible"] = len(papers)

        for paper in papers:
            pmid = paper.get("pmid", "")
            overall = (paper.get("overall_rob") or "").strip()
            if overall not in _VALID_RATINGS:
                counts["skipped_invalid_rating"] += 1
                logger.warning(
                    "SKIP %s: overall_rob=%r not in %s",
                    pmid, overall, sorted(_VALID_RATINGS),
                )
                continue
            domain_ratings = _extract_domain_ratings(paper)
            if domain_ratings is None:
                counts["skipped_invalid_rating"] += 1
                logger.warning(
                    "SKIP %s: a domain rating is missing or off-vocabulary",
                    pmid,
                )
                continue
            rating_source = _rating_source_for(paper)
            if rating_source is None:
                counts["skipped_no_source"] += 1
                logger.warning(
                    "SKIP %s: no cochrane_review_pmid/doi — cannot "
                    "attribute rating source", pmid,
                )
                continue

            if dry_run:
                logger.info(
                    "DRY-RUN would backfill %s (overall=%s, source=%s)",
                    pmid, overall, rating_source,
                )
                counts["inserted"] += 1
                continue

            status = db.upsert_expert_rating(
                methodology=METHODOLOGY,
                rating_source=rating_source,
                study_label=pmid,
                domain_ratings=domain_ratings,
                overall_rating=overall,
                pmid=pmid,
                doi=paper.get("doi") or None,
                methodology_version=METHODOLOGY_VERSION,
                source_review_pmid=paper.get("cochrane_review_pmid") or None,
                source_review_doi=paper.get("cochrane_review_doi") or None,
                source_reference=paper.get("cochrane_review_title") or None,
                added_by=added_by,
                verified=False,
                commit=False,
            )
            counts[status] += 1
        if not dry_run:
            db.commit()
    finally:
        db.close()
    return counts


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, required=True,
                   help="DB with Cochrane RoB 2 ratings on papers.* columns.")
    p.add_argument("--added-by", default=None,
                   help="Curator tag stamped on each backfilled row.")
    p.add_argument("--dry-run", action="store_true",
                   help="Classify eligible papers without writing rows.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    counts = backfill(args.db, added_by=args.added_by, dry_run=args.dry_run)
    verb = "Would backfill" if args.dry_run else "Inserted"
    print(f"Eligible papers:        {counts['eligible']}")
    print(f"{verb:23s} {counts['inserted']}")
    if not args.dry_run:
        print(f"Updated (existing):     {counts['updated']}")
    print(f"Skipped (bad rating):   {counts['skipped_invalid_rating']}")
    print(f"Skipped (no source):    {counts['skipped_no_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
