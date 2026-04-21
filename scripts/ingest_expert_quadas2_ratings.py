"""Ingest QUADAS-2 expert ratings from a JATS review into the new DB.

Takes the Cochrane-style systematic review JATS XML, runs the Table 2
extractor, cross-references each study against a legacy biasbuster DB
to discard obviously-corrupt metadata, and writes the salvageable rows
to the ``expert_methodology_ratings`` table in the destination DB.

The companion script ``extract_quadas2_ground_truth.py`` does the same
parsing but emits a standalone JSON sidecar; this script piggybacks on
it and persists the result instead.

Usage::

    uv run python scripts/ingest_expert_quadas2_ratings.py \\
        --jats tests/fixtures/cochrane_reviews/jcm-15-01829.xml \\
        --legacy-db dataset/biasbuster_recovered.db \\
        --target-db dataset/biasbuster.db \\
        --rating-source jcm-15-01829 \\
        --added-by curator

Rows where the legacy DB's stored title doesn't plausibly match the
JATS bibliography title (Jaccard < ``_TITLE_MATCH_THRESHOLD``) are
skipped, with a reason logged for each. Rows not present at all in
the legacy DB are skipped unless ``--include-unmatched`` is passed —
in that case the JATS-declared PMID/DOI/title are used as-is.

The ingested rows are written with ``verified=0``; a human curator
should review the table and set ``verified=1`` on rows they trust as
ground truth for the faithfulness harness.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from biasbuster.database import Database

# Reuse the parser + cross-reference logic from the ground-truth extractor
# so this script stays a thin persistence layer and any parse fix lands
# in both places.
from scripts.extract_quadas2_ground_truth import (
    StudyRow,
    cross_reference_legacy_db,
    parse_quadas2_table,
    parse_review_metadata,
)

logger = logging.getLogger(__name__)

#: Methodology slug + version to stamp on ingested rows.
METHODOLOGY: str = "quadas_2"
METHODOLOGY_VERSION: str = "quadas2-2011"


def _to_domain_ratings(study: StudyRow) -> dict[str, dict[str, str]]:
    """Wrap each bias rating in a ``{"bias": ...}`` dict.

    This shape is forward-compatible with future reviews that publish
    applicability ratings too — they'll add an ``"applicability"`` key
    alongside ``"bias"`` per domain without schema churn. Domains not
    recognised by the extractor are simply absent.
    """
    return {
        domain: {"bias": rating}
        for domain, rating in study.bias_ratings.items()
    }


def _classify(study: StudyRow) -> str:
    """Return one of ``plausible``, ``corrupt``, ``not_in_legacy``."""
    if study.legacy_db.get("present") is not True:
        return "not_in_legacy"
    return (
        "plausible" if study.legacy_db.get("title_plausible") is True
        else "corrupt"
    )


def _resolve_identifiers(
    study: StudyRow, status: str,
) -> tuple[Optional[str], Optional[str]]:
    """Return the ``(pmid, doi)`` to stamp on the ingested row.

    Trusts legacy-DB identifiers when the classification is ``plausible``
    (the DB-stored title matches the JATS bibliography), otherwise uses
    the JATS-declared values. Callers are responsible for having filtered
    out ``corrupt`` rows before reaching this function; invariant-checked
    with an assertion rather than silent fall-through.
    """
    assert status != "corrupt", (
        f"corrupt row should have been skipped before id resolution: "
        f"{study.label}"
    )
    if status == "plausible":
        legacy = study.legacy_db
        return (legacy.get("pmid_in_db") or study.pmid,
                legacy.get("doi_in_db") or study.doi)
    return (study.pmid, study.doi)


def _empty_counts() -> dict[str, int]:
    """Initial counts dict used by ``process_studies``."""
    return {
        "total": 0,
        "inserted": 0,
        "updated": 0,
        "skipped_corrupt": 0,
        "skipped_not_in_legacy": 0,
        "skipped_incomplete": 0,
    }


def process_studies(
    studies: list[StudyRow],
    db: Database,
    *,
    rating_source: str,
    review_meta: dict[str, Optional[str]],
    added_by: Optional[str] = None,
    include_unmatched: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Classify, filter, and upsert each study row into ``db``.

    Returns a counts summary. Pure with respect to parsing — accepts
    already-extracted ``StudyRow`` objects so unit tests can exercise
    the classification and persistence logic without crafting JATS
    fixtures.

    ``insert`` vs. ``updated`` counts are driven by the status returned
    by :meth:`Database.upsert_expert_rating`, so a re-run of the same
    source reports ``inserted=0`` / ``updated=N`` instead of the
    previous misleading ``ingested=N``.
    """
    counts = _empty_counts()
    counts["total"] = len(studies)
    for study in studies:
        status = _classify(study)
        if status == "corrupt":
            counts["skipped_corrupt"] += 1
            logger.info(
                "SKIP %s: legacy title mismatch (similarity=%s)",
                study.label,
                study.legacy_db.get("title_jaccard_similarity"),
            )
            continue
        if status == "not_in_legacy" and not include_unmatched:
            counts["skipped_not_in_legacy"] += 1
            logger.info(
                "SKIP %s: not in legacy DB (--include-unmatched "
                "to override)", study.label,
            )
            continue
        if not study.bias_ratings or study.overall is None:
            counts["skipped_incomplete"] += 1
            logger.warning(
                "SKIP %s: incomplete ratings (bias=%s, overall=%s)",
                study.label, study.bias_ratings, study.overall,
            )
            continue

        pmid, doi = _resolve_identifiers(study, status)

        if dry_run:
            logger.info(
                "DRY-RUN would write %s (pmid=%s doi=%s overall=%s)",
                study.label, pmid, doi, study.overall,
            )
            # Dry-run assumes every would-be write is a fresh insert;
            # the alternative (running the existence probe without
            # writing) would leak DB state into a read-only operation.
            counts["inserted"] += 1
            continue
        status_written = db.upsert_expert_rating(
            methodology=METHODOLOGY,
            rating_source=rating_source,
            study_label=study.label,
            domain_ratings=_to_domain_ratings(study),
            overall_rating=study.overall,
            pmid=pmid,
            doi=doi,
            methodology_version=METHODOLOGY_VERSION,
            source_review_pmid=review_meta.get("pmid"),
            source_review_doi=review_meta.get("doi"),
            source_reference=review_meta.get("title"),
            added_by=added_by,
            verified=False,
            commit=False,
        )
        counts[status_written] += 1
    return counts


def ingest(
    jats_path: Path,
    legacy_db_path: Path,
    target_db_path: Path,
    rating_source: str,
    *,
    table_label: str = "Table 2",
    added_by: Optional[str] = None,
    include_unmatched: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Parse the JATS review, cross-reference, and persist salvageable rows.

    Thin wrapper around :func:`process_studies` that handles parsing and
    DB lifecycle. Returns the counts summary from ``process_studies``.
    """
    studies, _ = parse_quadas2_table(jats_path, table_label=table_label)
    cross_reference_legacy_db(studies, legacy_db_path)
    review_meta = parse_review_metadata(jats_path)

    db = Database(target_db_path)
    try:
        db.initialize()
        counts = process_studies(
            studies, db,
            rating_source=rating_source,
            review_meta=review_meta,
            added_by=added_by,
            include_unmatched=include_unmatched,
            dry_run=dry_run,
        )
        if not dry_run:
            db.commit()
    finally:
        db.close()
    return counts


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jats", type=Path, required=True,
                   help="Path to the JATS XML of the systematic review.")
    p.add_argument("--legacy-db", type=Path, required=True,
                   help="Legacy biasbuster DB, used to validate titles.")
    p.add_argument("--target-db", type=Path, required=True,
                   help="New methodology-aware DB to write ratings into.")
    p.add_argument(
        "--rating-source", required=True,
        help="Short token identifying the source of these ratings "
             "(e.g. 'jcm-15-01829'). Part of the table's primary key "
             "together with methodology + study_label.",
    )
    p.add_argument(
        "--table-label", default="Table 2",
        help="JATS <table-wrap><label> text for the QUADAS-2 table. "
             "Default 'Table 2'.",
    )
    p.add_argument("--added-by", default=None,
                   help="Curator identifier stamped on ingested rows.")
    p.add_argument(
        "--include-unmatched", action="store_true",
        help="Also ingest rows not present in the legacy DB, using the "
             "JATS-declared PMID/DOI as-is. Off by default.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Parse and classify but write nothing to the target DB.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    counts = ingest(
        jats_path=args.jats,
        legacy_db_path=args.legacy_db,
        target_db_path=args.target_db,
        rating_source=args.rating_source,
        table_label=args.table_label,
        added_by=args.added_by,
        include_unmatched=args.include_unmatched,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Rows parsed:             {counts['total']}")
        print(f"Would ingest:            {counts['inserted']}")
    else:
        print(f"Rows parsed:             {counts['total']}")
        print(f"Inserted:                {counts['inserted']}")
        print(f"Updated (existing):      {counts['updated']}")
    print(f"Skipped (corrupt):       {counts['skipped_corrupt']}")
    print(f"Skipped (not in legacy): {counts['skipped_not_in_legacy']}")
    print(f"Skipped (incomplete):    {counts['skipped_incomplete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
