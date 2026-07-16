"""
Migration Script: JSONL → SQLite

Imports existing JSONL data and review CSVs into the new SQLite database.
Idempotent — safe to run multiple times (uses INSERT OR IGNORE).

Usage:
    uv run python migrate_jsonl_to_sqlite.py
    uv run python migrate_jsonl_to_sqlite.py --data-dir dataset
    uv run python migrate_jsonl_to_sqlite.py --db-path dataset/biasbuster.db
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from biasbuster.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def migrate_raw_papers(db: Database, raw_dir: Path) -> None:
    """Import raw papers from dataset/raw/*.jsonl."""
    if not raw_dir.exists():
        logger.info(f"No raw directory found at {raw_dir}, skipping")
        return

    # Retracted papers
    retracted_path = raw_dir / "retracted_papers.jsonl"
    records = load_jsonl(retracted_path)
    if records:
        for r in records:
            r["source"] = "retraction_watch"
        count = db.insert_papers(records)
        logger.info(
            f"Retracted papers: {count} new / {len(records)} total "
            f"from {retracted_path}"
        )

    # RCT abstracts
    rct_path = raw_dir / "rct_abstracts.jsonl"
    records = load_jsonl(rct_path)
    if records:
        for r in records:
            r["source"] = "pubmed_rct"
        count = db.insert_papers(records)
        logger.info(
            f"RCT abstracts: {count} new / {len(records)} total "
            f"from {rct_path}"
        )

    # Cochrane RoB
    cochrane_path = raw_dir / "cochrane_rob.jsonl"
    records = load_jsonl(cochrane_path)
    if records:
        for r in records:
            r["source"] = "cochrane_rob"
        count = db.insert_papers(records)
        logger.info(
            f"Cochrane RoB: {count} new / {len(records)} total "
            f"from {cochrane_path}"
        )


def migrate_enrichments(db: Database, enriched_dir: Path) -> None:
    """Import enrichment data from dataset/enriched/*.jsonl."""
    if not enriched_dir.exists():
        logger.info(f"No enriched directory found at {enriched_dir}, skipping")
        return

    for suspicion_level, filename in [
        ("high", "high_suspicion.jsonl"),
        ("low", "low_suspicion.jsonl"),
    ]:
        path = enriched_dir / filename
        records = load_jsonl(path)
        if not records:
            continue

        count = 0
        for r in records:
            pmid = r.get("pmid", "")
            if not pmid:
                continue

            # Ensure paper exists (enriched files may have data not in raw)
            r["source"] = "pubmed_rct"
            db.insert_paper(r)

            enrichment = {
                "suspicion_level": suspicion_level,
                "reporting_bias_score": (
                    r.get("effect_size_audit", {}).get("reporting_bias_score")
                ),
                "effect_size_audit": r.get("effect_size_audit"),
                "outcome_switching": r.get("outcome_switching"),
            }
            db.upsert_enrichment(pmid, enrichment)
            count += 1

        logger.info(
            f"Enrichments ({suspicion_level}): {count} records from {path}"
        )

    # Retracted papers and Cochrane in enriched dir (copies of raw)
    for filename, source in [
        ("retracted_papers.jsonl", "retraction_watch"),
        ("cochrane_rob.jsonl", "cochrane_rob"),
    ]:
        path = enriched_dir / filename
        records = load_jsonl(path)
        if records:
            for r in records:
                r["source"] = source
            db.insert_papers(records)
            logger.info(
                f"Enriched {filename}: ensured {len(records)} papers exist"
            )


def migrate_annotations(db: Database, labelled_dir: Path) -> None:
    """Import annotations from dataset/labelled/{model_name}/*_annotated.jsonl."""
    if not labelled_dir.exists():
        logger.info(
            f"No labelled directory found at {labelled_dir}, skipping"
        )
        return

    model_dirs = [
        d for d in labelled_dir.iterdir()
        if d.is_dir()
    ]

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        total = 0
        inserted = 0

        for jsonl_path in sorted(model_dir.glob("*_annotated.jsonl")):
            records = load_jsonl(jsonl_path)
            for r in records:
                pmid = r.get("pmid", "")
                if not pmid:
                    continue

                # Ensure paper exists in papers table
                paper_data = {
                    "pmid": pmid,
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract_text", r.get("abstract", "")),
                    "source": r.get("source", "unknown"),
                }
                db.insert_paper(paper_data)

                # Determine source key for the annotation
                source_key = r.get("source", "")
                # Build the annotation dict (everything except paper-level fields)
                annotation = {
                    k: v for k, v in r.items()
                    if k not in (
                        "pmid", "title", "abstract_text", "abstract",
                        "source", "_annotation_model",
                    )
                }
                # Preserve source and model info inside annotation JSON
                annotation["source"] = source_key
                annotation["_annotation_model"] = r.get(
                    "_annotation_model", model_name
                )

                if db.insert_annotation(pmid, model_name, annotation):
                    inserted += 1
                total += 1

        logger.info(
            f"Annotations ({model_name}): {inserted} new / {total} total"
        )


def migrate_reviews(db: Database, labelled_dir: Path) -> None:
    """Import human review data from *_review.csv files."""
    if not labelled_dir.exists():
        return

    for csv_path in labelled_dir.rglob("*_review.csv"):
        model_name = csv_path.parent.name
        count = 0

        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pmid = row.get("pmid", "").strip()
                    validated_str = row.get("HUMAN_VALIDATED", "").strip()
                    override = row.get(
                        "HUMAN_OVERRIDE_SEVERITY", ""
                    ).strip() or None
                    notes = row.get("HUMAN_NOTES", "").strip() or None

                    if not pmid:
                        continue
                    if not validated_str and not override and not notes:
                        continue

                    validated = validated_str.lower() == "true"
                    db.upsert_review(
                        pmid, model_name, validated, override, notes
                    )
                    count += 1
        except Exception as e:
            logger.warning(f"Failed to read {csv_path}: {e}")

        if count:
            logger.info(
                f"Reviews ({model_name}): {count} records from {csv_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate JSONL data to SQLite database"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Base data directory (default: dataset)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path (default: {data-dir}/biasbuster.db)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = args.db_path or str(data_dir / "biasbuster.db")

    logger.info(f"Migrating data from {data_dir} to {db_path}")

    db = Database(db_path)
    db.initialize()

    try:
        migrate_raw_papers(db, data_dir / "raw")
        migrate_enrichments(db, data_dir / "enriched")
        migrate_annotations(db, data_dir / "labelled")
        migrate_reviews(db, data_dir / "labelled")

        stats = db.get_stats()
        logger.info("=" * 60)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total papers: {stats['total_papers']}")
        for source, count in stats.get("papers_by_source", {}).items():
            logger.info(f"  {source}: {count}")
        logger.info(f"Total enrichments: {stats['total_enrichments']}")
        for level, count in stats.get("enrichments_by_level", {}).items():
            logger.info(f"  {level}: {count}")
        logger.info(f"Total annotations: {stats['total_annotations']}")
        for model, count in stats.get("annotations_by_model", {}).items():
            logger.info(f"  {model}: {count}")
        logger.info(
            f"Total reviews: {stats['total_reviews']} "
            f"({stats['validated_reviews']} validated)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
