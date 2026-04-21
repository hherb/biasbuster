"""Copy papers referenced by ``expert_methodology_ratings`` between DBs.

The ground-truth ingest populates ``expert_methodology_ratings`` in the
new methodology-aware DB, but the actual paper rows (title, abstract,
authors) still live only in the legacy DB. Before the faithfulness
harness can evaluate a methodology, the paper metadata needs to be in
the same database as the annotations that will be produced.

This script copies exactly those papers — no more. It's deliberately
narrow so it can't accidentally drag in thousands of unrelated legacy
rows along with the ~15 we care about.

Usage::

    uv run python scripts/copy_papers_for_expert_ratings.py \\
        --source-db dataset/biasbuster.db \\
        --target-db dataset/biasbuster_recovered.db

Copied papers keep their legacy ``source`` tag. Already-present papers
in the target are left untouched (no field overwrite) — callers who
want to refresh a copied paper should delete and re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from biasbuster.database import Database

logger = logging.getLogger(__name__)


def _rated_pmids(
    target_db_path: Path, methodology: Optional[str],
) -> list[str]:
    """Distinct PMIDs referenced by ``expert_methodology_ratings``.

    Filters by ``methodology`` when given, otherwise returns every
    rated PMID. Rows with ``pmid IS NULL`` are dropped — a rating
    without an identified paper can't be joined against ``papers``.
    """
    db = Database(target_db_path)
    try:
        db.initialize()
        q = (
            "SELECT DISTINCT pmid FROM expert_methodology_ratings "
            "WHERE pmid IS NOT NULL"
        )
        params: list = []
        if methodology is not None:
            q += " AND methodology = ?"
            params.append(methodology)
        rows = db.conn.execute(q, params).fetchall()
    finally:
        db.close()
    return [r["pmid"] for r in rows]


def _fetch_paper_rows(
    source_db_path: Path, pmids: list[str],
) -> dict[str, dict]:
    """Pull the raw ``papers`` rows for ``pmids`` from the source DB.

    Returns a ``{pmid: row_dict}`` mapping. PMIDs not present in the
    source are absent from the map — the caller reports them as missing.
    Uses a parameterised ``IN`` clause so we never interpolate PMIDs
    into the query string.
    """
    if not pmids:
        return {}
    conn = sqlite3.connect(str(source_db_path))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(pmids))
        rows = conn.execute(
            f"SELECT * FROM papers WHERE pmid IN ({placeholders})",
            pmids,
        ).fetchall()
    finally:
        conn.close()
    return {r["pmid"]: dict(r) for r in rows}


#: Columns that legacy DBs may store as JSON strings. ``Database.insert_paper``
#: re-serialises via ``_json_col``; if the legacy DB already stored JSON text
#: we decode it back to a Python object so re-encoding is idempotent.
_JSON_COLUMNS: tuple[str, ...] = (
    "authors", "grants", "mesh_terms", "subjects", "retraction_reasons",
)


def _prepare_for_insert(row: dict) -> dict:
    """Decode JSON columns so ``insert_paper`` re-serialises cleanly."""
    prepared = dict(row)
    for col in _JSON_COLUMNS:
        val = prepared.get(col)
        if isinstance(val, str):
            try:
                prepared[col] = json.loads(val)
            except (TypeError, ValueError):
                prepared[col] = None
    return prepared


def copy_papers(
    source_db_path: Path,
    target_db_path: Path,
    methodology: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy rated papers and return a counts summary.

    Counts: ``rated``, ``found_in_source``, ``missing_in_source``,
    ``newly_copied``, ``already_present``.
    """
    pmids = _rated_pmids(target_db_path, methodology)
    source_rows = _fetch_paper_rows(source_db_path, pmids)

    counts = {
        "rated": len(pmids),
        "found_in_source": len(source_rows),
        "missing_in_source": len(pmids) - len(source_rows),
        "newly_copied": 0,
        "already_present": 0,
    }

    missing = sorted(set(pmids) - set(source_rows))
    for pmid in missing:
        logger.warning("MISSING in source DB: %s", pmid)

    if dry_run:
        for pmid, row in sorted(source_rows.items()):
            logger.info(
                "DRY-RUN would copy %s (title=%r)",
                pmid, (row.get("title") or "")[:60],
            )
        return counts

    db = Database(target_db_path)
    try:
        db.initialize()
        existing = db.get_paper_pmids()
        for pmid, row in source_rows.items():
            if pmid in existing:
                counts["already_present"] += 1
                logger.info("SKIP %s: already present in target", pmid)
                continue
            inserted = db.insert_paper(_prepare_for_insert(row))
            if inserted:
                counts["newly_copied"] += 1
                logger.info(
                    "COPIED %s (title=%r)",
                    pmid, (row.get("title") or "")[:60],
                )
    finally:
        db.close()
    return counts


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source-db", type=Path, required=True,
                   help="Legacy DB with the paper rows to copy from.")
    p.add_argument("--target-db", type=Path, required=True,
                   help="Methodology-aware DB to copy papers into.")
    p.add_argument(
        "--methodology", default=None,
        help="Restrict to PMIDs rated under this methodology (e.g. "
             "'quadas_2'). Default: copy for every rated methodology.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned copies without touching the target DB.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    counts = copy_papers(
        args.source_db, args.target_db, args.methodology,
        dry_run=args.dry_run,
    )
    verb = "Would copy" if args.dry_run else "Newly copied"
    print(f"Rated PMIDs:         {counts['rated']}")
    print(f"Found in source:     {counts['found_in_source']}")
    print(f"Missing from source: {counts['missing_in_source']}")
    print(f"{verb:20s} {counts['newly_copied']}")
    print(f"Already in target:   {counts['already_present']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
