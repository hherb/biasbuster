"""Quarantine a legacy biasbuster database and start fresh with methodology support.

The pre-methodology schema keyed annotations by ``(pmid, model_name)``.
The new schema extends that to ``(pmid, model_name, methodology)`` so a
single paper/model pair can carry multiple annotations under different
risk-of-bias tools (biasbuster, cochrane_rob2, quadas_2, ...).

Because legacy annotations may contain incorrect data we want to
quarantine (not silently auto-forward) them, this script:

1. Archives a *copy* of the legacy DB alongside it as
   ``<db>.legacy_<timestamp>.db`` (if ``--archive-legacy``). The original
   legacy DB file is left in place untouched — nothing is renamed or
   deleted. The archived copy is a dated snapshot for disaster recovery.
2. Creates a fresh DB at the chosen ``--to`` path with the new schema.
3. Optionally copies the ``papers`` and ``enrichments`` tables from the
   legacy DB (``--copy-papers`` / ``--copy-enrichments``). These tables
   are methodology-independent; their rows are safe to carry over.
4. Optionally copies legacy annotation rows with ``methodology='biasbuster'``
   and ``methodology_version='legacy'`` (``--copy-annotations``). Off by
   default because legacy annotations are the data we want to re-run.

Usage::

    uv run python scripts/migrations/add_methodology_support.py \
        --from dataset/biasbuster.db --to dataset/biasbuster_v2.db \
        --archive-legacy --copy-papers --copy-enrichments

All operations are transactional; if any step fails the new DB is
deleted so the legacy DB remains the single source of truth.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

# Tables copied verbatim (schema-compatible between legacy and new).
METHODOLOGY_INDEPENDENT_TABLES: tuple[str, ...] = ("papers", "enrichments")

logger = logging.getLogger(__name__)


def _timestamp_suffix() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _assert_legacy_schema(legacy_db: Path) -> None:
    """Sanity-check the source DB actually has the pre-methodology schema."""
    with sqlite3.connect(str(legacy_db)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='annotations'"
        ).fetchone()
        if row is None:
            raise SystemExit(
                f"{legacy_db} has no 'annotations' table — nothing to migrate."
            )
        cols = {c[1] for c in conn.execute("PRAGMA table_info(annotations)").fetchall()}
        if "methodology" in cols:
            raise SystemExit(
                f"{legacy_db} already has a 'methodology' column on "
                "annotations. This DB is already migrated; nothing to do."
            )


def _create_fresh_db(new_db: Path) -> None:
    """Create a fresh DB using the current Database.initialize() schema."""
    from biasbuster.database import Database

    new_db.parent.mkdir(parents=True, exist_ok=True)
    db = Database(new_db)
    try:
        db.initialize()
    finally:
        db.close()


def _copy_table(
    src: sqlite3.Connection, dst: sqlite3.Connection, table: str
) -> int:
    """Copy every row of ``table`` from src to dst. Returns rowcount."""
    src_cols = [c[1] for c in src.execute(f"PRAGMA table_info({table})").fetchall()]
    dst_cols = [c[1] for c in dst.execute(f"PRAGMA table_info({table})").fetchall()]
    shared = [c for c in src_cols if c in dst_cols]
    if not shared:
        logger.warning("Table %s: no shared columns; skipping.", table)
        return 0
    col_list = ",".join(shared)
    placeholders = ",".join("?" * len(shared))
    rows = src.execute(f"SELECT {col_list} FROM {table}").fetchall()
    dst.executemany(
        f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
        rows,
    )
    dst.commit()
    return len(rows)


def _copy_legacy_annotations(
    src: sqlite3.Connection, dst: sqlite3.Connection
) -> int:
    """Copy legacy annotations tagged as 'biasbuster'/'legacy' methodology."""
    rows = src.execute(
        "SELECT pmid, model_name, annotation, overall_severity, "
        "overall_bias_probability, confidence, annotated_at FROM annotations"
    ).fetchall()
    dst.executemany(
        """INSERT OR IGNORE INTO annotations
           (pmid, model_name, methodology, methodology_version,
            annotation, overall_severity, overall_bias_probability,
            confidence, annotated_at)
           VALUES (?, ?, 'biasbuster', 'legacy', ?, ?, ?, ?, ?)""",
        rows,
    )
    dst.commit()
    return len(rows)


def _copy_legacy_reviews(
    src: sqlite3.Connection, dst: sqlite3.Connection
) -> int:
    """Copy legacy human_reviews tagged as 'biasbuster' methodology."""
    rows = src.execute(
        "SELECT pmid, model_name, validated, override_severity, "
        "annotation, flagged, notes, reviewed_at FROM human_reviews"
    ).fetchall()
    dst.executemany(
        """INSERT OR IGNORE INTO human_reviews
           (pmid, model_name, methodology, validated, override_severity,
            annotation, flagged, notes, reviewed_at)
           VALUES (?, ?, 'biasbuster', ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    dst.commit()
    return len(rows)


def migrate(
    legacy_db: Path,
    new_db: Path,
    *,
    archive_legacy: bool,
    copy_papers: bool,
    copy_enrichments: bool,
    copy_annotations: bool,
    copy_reviews: bool,
    force: bool,
) -> dict[str, int]:
    """Run the migration. Returns per-table rowcount summary.

    The legacy DB file is never modified; if ``archive_legacy`` is True a
    timestamped snapshot copy is placed alongside it for disaster recovery.
    """
    if legacy_db == new_db:
        raise SystemExit("--from and --to must differ.")
    if not legacy_db.exists():
        raise SystemExit(f"Legacy DB not found: {legacy_db}")
    if new_db.exists() and not force:
        raise SystemExit(
            f"Target DB already exists: {new_db}. Pass --force to overwrite."
        )
    # Validate FK-dependency flags up-front so we never create a fresh DB
    # only to discover the copy plan is malformed. (These raise SystemExit,
    # which is NOT an Exception subclass and so would bypass the in-migration
    # rollback block below.)
    if copy_annotations and not copy_papers:
        raise SystemExit(
            "--copy-annotations requires --copy-papers "
            "(annotations reference papers via FK)."
        )
    if copy_reviews and not copy_annotations:
        raise SystemExit(
            "--copy-reviews requires --copy-annotations "
            "(reviews reference annotations via FK)."
        )

    _assert_legacy_schema(legacy_db)

    if new_db.exists():
        new_db.unlink()

    _create_fresh_db(new_db)

    summary: dict[str, int] = {}
    src = sqlite3.connect(str(legacy_db))
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(str(new_db))
    dst.execute("PRAGMA foreign_keys=ON")
    try:
        if copy_papers:
            summary["papers"] = _copy_table(src, dst, "papers")
        if copy_enrichments:
            summary["enrichments"] = _copy_table(src, dst, "enrichments")
        if copy_annotations:
            summary["annotations"] = _copy_legacy_annotations(src, dst)
        if copy_reviews:
            summary["human_reviews"] = _copy_legacy_reviews(src, dst)
    except Exception:
        # Roll back: remove the partially populated new DB so the legacy
        # DB remains the single source of truth on failure.
        dst.close()
        src.close()
        new_db.unlink(missing_ok=True)
        raise
    finally:
        src.close()
        dst.close()

    # Archive last — only drop a timestamped snapshot copy if the
    # migration itself succeeded. Doing the archive earlier would leave
    # a stale dated backup next to the legacy DB whenever the main work
    # raises (disk full, fresh-DB init error, partial copy), confusing
    # future recovery attempts.
    if archive_legacy:
        archived = legacy_db.with_name(
            f"{legacy_db.stem}.legacy_{_timestamp_suffix()}{legacy_db.suffix}"
        )
        shutil.copy2(legacy_db, archived)
        logger.info("Archived legacy DB snapshot to %s", archived)

    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from", dest="legacy", type=Path, required=True,
                   help="Path to the legacy biasbuster.db. The file is never "
                        "modified; pass --archive-legacy to drop a "
                        "timestamped copy alongside it for disaster recovery.")
    p.add_argument("--to", dest="new", type=Path, required=True,
                   help="Path for the fresh methodology-aware DB.")
    p.add_argument("--archive-legacy", action="store_true",
                   help="Write a timestamped copy of the legacy DB alongside "
                        "it (legacy file is never modified). Useful as a "
                        "dated disaster-recovery snapshot.")
    p.add_argument("--copy-papers", action="store_true",
                   help="Copy the papers table from legacy to new (safe: "
                        "schema is methodology-independent).")
    p.add_argument("--copy-enrichments", action="store_true",
                   help="Copy the enrichments table from legacy to new.")
    p.add_argument("--copy-annotations", action="store_true",
                   help="Also copy legacy annotations tagged as methodology="
                        "'biasbuster', methodology_version='legacy'. Off by "
                        "default — you probably want to re-run annotations "
                        "under the new setup.")
    p.add_argument("--copy-reviews", action="store_true",
                   help="Also copy legacy human_reviews (requires "
                        "--copy-annotations).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --to if it already exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    summary = migrate(
        args.legacy,
        args.new,
        archive_legacy=args.archive_legacy,
        copy_papers=args.copy_papers,
        copy_enrichments=args.copy_enrichments,
        copy_annotations=args.copy_annotations,
        copy_reviews=args.copy_reviews,
        force=args.force,
    )
    print(f"Migration complete. Rows copied: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
