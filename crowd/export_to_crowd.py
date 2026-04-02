"""Export a subset of papers from production DB to the crowd annotation DB.

Reads the production biasbuster.db (read-only), selects papers with valid
AI annotations, strips retraction/Cochrane metadata to prevent annotator bias,
and populates the crowd DB.

Usage:
    uv run python -m crowd.export_to_crowd \\
        --prod-db dataset/biasbuster.db \\
        --crowd-db dataset/crowd_annotations.db \\
        --model deepseek \\
        --limit 200 \\
        --strategy balanced
"""

import argparse
import json
import logging
import random
import sqlite3
from pathlib import Path

from crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)

# Sources to include in balanced sampling
BALANCED_SOURCES = [
    "retracted_papers",
    "cochrane_rob",
    "high_suspicion",
    "low_suspicion",
]


def _connect_prod(db_path: str) -> sqlite3.Connection:
    """Open production DB in read-only mode."""
    if not Path(db_path).exists():
        raise SystemExit(f"Production database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_annotated_papers(
    prod_conn: sqlite3.Connection,
    model_name: str,
) -> list[dict]:
    """Fetch all papers with valid AI annotations for the given model."""
    rows = prod_conn.execute(
        """SELECT p.pmid, p.doi, p.title, p.abstract, p.journal, p.year,
                  p.authors, p.grants, p.mesh_terms, p.source,
                  a.annotation, a.overall_severity,
                  a.overall_bias_probability, a.confidence
           FROM papers p
           JOIN annotations a ON a.pmid = p.pmid
           WHERE a.model_name = ?
             AND p.excluded = 0
             AND p.abstract IS NOT NULL
             AND p.abstract != ''
             AND a.annotation IS NOT NULL""",
        (model_name,),
    ).fetchall()
    return [dict(r) for r in rows]


def _genericize_source(source: str) -> str:
    """Strip revealing source labels to prevent annotator bias."""
    # Map all sources to generic labels
    source_map = {
        "retracted_papers": "pubmed_rct",
        "retraction_watch": "pubmed_rct",
        "cochrane_rob": "pubmed_rct",
        "high_suspicion": "pubmed_rct",
        "low_suspicion": "pubmed_rct",
        "pubmed_rct": "pubmed_rct",
        "manual_import": "pubmed_rct",
    }
    return source_map.get(source, "pubmed_rct")


def _select_balanced(
    papers: list[dict], limit: int, seed: int = 42
) -> list[dict]:
    """Select a balanced sample across source categories."""
    by_source: dict[str, list[dict]] = {}
    for p in papers:
        src = p.get("source", "unknown")
        by_source.setdefault(src, []).append(p)

    rng = random.Random(seed)
    per_source = max(1, limit // len(by_source)) if by_source else 0
    selected: list[dict] = []

    for src, src_papers in by_source.items():
        rng.shuffle(src_papers)
        selected.extend(src_papers[:per_source])

    # Fill remaining slots from any source
    already = {p["pmid"] for p in selected}
    remaining = [p for p in papers if p["pmid"] not in already]
    rng.shuffle(remaining)
    selected.extend(remaining[: limit - len(selected)])

    return selected[:limit]


def _select_random(
    papers: list[dict], limit: int, seed: int = 42
) -> list[dict]:
    """Select a random sample."""
    rng = random.Random(seed)
    rng.shuffle(papers)
    return papers[:limit]


def _select_specific(
    papers: list[dict], pmid_file: str
) -> list[dict]:
    """Select specific PMIDs from a file (one per line)."""
    pmids = set()
    with open(pmid_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pmids.add(line)

    selected = [p for p in papers if p["pmid"] in pmids]
    found = {p["pmid"] for p in selected}
    missing = pmids - found
    if missing:
        logger.warning("PMIDs not found in production DB: %s", missing)
    return selected


def export(
    prod_db_path: str,
    crowd_db_path: str,
    model_name: str,
    limit: int = 200,
    strategy: str = "balanced",
    pmid_file: str = "",
    target_annotations: int = 3,
    seed: int = 42,
) -> int:
    """Export papers from production to crowd DB.

    Returns the number of papers exported.
    """
    prod_conn = _connect_prod(prod_db_path)
    crowd_db = CrowdDatabase(crowd_db_path)
    crowd_db.initialize()

    try:
        # Fetch all annotated papers
        all_papers = _fetch_annotated_papers(prod_conn, model_name)
        logger.info(
            "Found %d annotated papers for model '%s'",
            len(all_papers), model_name,
        )

        if not all_papers:
            logger.warning("No annotated papers found. Nothing to export.")
            return 0

        # Select subset
        if strategy == "balanced":
            selected = _select_balanced(all_papers, limit, seed)
        elif strategy == "random":
            selected = _select_random(all_papers, limit, seed)
        elif strategy == "specific":
            if not pmid_file:
                raise SystemExit("--pmid-file required with --strategy specific")
            selected = _select_specific(all_papers, pmid_file)
        else:
            raise SystemExit(f"Unknown strategy: {strategy}")

        logger.info("Selected %d papers for export", len(selected))

        # Collect original sources for logging before we modify dicts
        original_sources: dict[str, str] = {
            p["pmid"]: p.get("source", "unknown") for p in selected
        }

        # Export
        exported = 0
        for paper in selected:
            # Parse annotation JSON
            ann_raw = paper.pop("annotation")
            ann = json.loads(ann_raw) if isinstance(ann_raw, str) else ann_raw
            paper.pop("overall_severity", None)
            paper.pop("overall_bias_probability", None)
            paper.pop("confidence", None)

            # Genericize source
            paper["source"] = _genericize_source(paper.get("source", "unknown"))
            paper["target_annotations"] = target_annotations

            # Parse JSON columns from production
            for col in ("authors", "grants", "mesh_terms"):
                val = paper.get(col)
                if isinstance(val, str):
                    try:
                        paper[col] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Insert paper (skip if already exists)
            if crowd_db.insert_paper(paper):
                # Insert AI annotation
                crowd_db.insert_ai_annotation(paper["pmid"], model_name, ann)
                exported += 1

        logger.info("Exported %d new papers to crowd DB", exported)

        # Summary by original source (before genericization)
        source_counts: dict[str, int] = {}
        for pmid_key, src in original_sources.items():
            source_counts[src] = source_counts.get(src, 0) + 1
        for src, count in sorted(source_counts.items()):
            logger.info("  %s: %d papers", src, count)

        return exported

    finally:
        prod_conn.close()
        crowd_db.close()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Export papers from production DB to crowd annotation DB"
    )
    parser.add_argument(
        "--prod-db", default="dataset/biasbuster.db",
        help="Path to production database (default: dataset/biasbuster.db)",
    )
    parser.add_argument(
        "--crowd-db", default="dataset/crowd_annotations.db",
        help="Path to crowd database (default: dataset/crowd_annotations.db)",
    )
    parser.add_argument(
        "--model", default="deepseek",
        help="AI model name to export annotations for (default: deepseek)",
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Maximum papers to export (default: 200)",
    )
    parser.add_argument(
        "--strategy", choices=["balanced", "random", "specific"],
        default="balanced",
        help="Selection strategy (default: balanced)",
    )
    parser.add_argument(
        "--pmid-file",
        help="File with PMIDs (one per line), required with --strategy specific",
    )
    parser.add_argument(
        "--target-annotations", type=int, default=3,
        help="Target annotations per paper (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection (default: 42)",
    )

    args = parser.parse_args()
    count = export(
        prod_db_path=args.prod_db,
        crowd_db_path=args.crowd_db,
        model_name=args.model,
        limit=args.limit,
        strategy=args.strategy,
        pmid_file=args.pmid_file or "",
        target_annotations=args.target_annotations,
        seed=args.seed,
    )
    print(f"Exported {count} papers to {args.crowd_db}")


if __name__ == "__main__":
    main()
