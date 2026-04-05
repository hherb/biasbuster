#!/usr/bin/env python
"""Export and import clean seed data (papers + enrichments) as JSONL.

Creates a versionable, git-friendly snapshot of the pipeline seed data
before annotation.  Use this to recover from database corruption without
having to re-collect from external APIs.

Export writes to ``dataset/cleanseed/``:
    papers.jsonl      — all papers (one JSON object per line)
    enrichments.jsonl — heuristic enrichment data
    manifest.json     — row counts, export timestamp, schema version

Import reads from ``dataset/cleanseed/`` and populates a fresh database.

Usage:
    # Export current seed data
    uv run python seed_export.py export

    # Export to a custom directory
    uv run python seed_export.py export --dir dataset/cleanseed_v2

    # Import into a fresh database (backs up existing DB first)
    uv run python seed_export.py import

    # Import from a custom directory into a specific DB
    uv run python seed_export.py import --dir dataset/cleanseed_v2 --db dataset/fresh.db

    # Export full annotated dataset (papers + enrichments + annotations)
    uv run python seed_export.py export-annotated

    # Import annotated dataset
    uv run python seed_export.py import-annotated
"""

import argparse
import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SEED_DIR = Path("dataset/cleanseed")
DEFAULT_DB_PATH = Path("dataset/biasbuster.db")
DEFAULT_ANNOTATED_DIR = Path("dataset/cleanseed/annotated")

# Tables to export (columns are auto-discovered from the DB schema)
SEED_TABLES = ["papers", "enrichments"]
ANNOTATED_TABLES = ["papers", "enrichments", "annotations", "human_reviews"]

# Commit batch size for imports
IMPORT_COMMIT_INTERVAL = 500

# Columns that store JSON strings in SQLite
JSON_COLUMNS = {
    "authors", "grants", "mesh_terms", "subjects",
    "retraction_reasons", "effect_size_audit", "outcome_switching",
    "annotation",
}


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Discover column names from the DB schema via PRAGMA."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _decode_json_fields(row: dict) -> dict:
    """Parse JSON string columns into native Python objects."""
    for col in JSON_COLUMNS:
        val = row.get(col)
        if isinstance(val, str) and val:
            try:
                row[col] = json.loads(val)
            except json.JSONDecodeError:
                pass  # keep as string
    return row


def _encode_json_fields(row: dict) -> dict:
    """Serialize Python objects back to JSON strings for SQLite."""
    for col in JSON_COLUMNS:
        val = row.get(col)
        if val is not None and not isinstance(val, str):
            row[col] = json.dumps(val)
    return row


def _export_tables(
    tables: list[str],
    db_path: Path,
    seed_dir: Path,
    snapshot_type: str = "seed",
) -> None:
    """Export the given tables from the DB to JSONL files.

    Args:
        tables: Ordered list of table names to export.
        db_path: Path to the SQLite database.
        seed_dir: Output directory for JSONL files + manifest.
        snapshot_type: Label for the manifest (e.g. "seed", "annotated").
    """
    seed_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path),
        "snapshot_type": snapshot_type,
        "tables": {},
    }

    for table in tables:
        columns = _get_table_columns(conn, table)
        if not columns:
            logger.warning(f"Table {table} has no columns, skipping")
            continue
        out_path = seed_dir / f"{table}.jsonl"
        col_list = ", ".join(columns)
        cursor = conn.execute(f"SELECT {col_list} FROM {table} ORDER BY pmid")

        count = 0
        with open(out_path, "w") as f:
            for row in cursor:
                record = _decode_json_fields(dict(row))
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1

        manifest["tables"][table] = count
        logger.info(f"Exported {count} rows from {table} → {out_path}")

    manifest_path = seed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(f"Manifest written to {manifest_path}")

    conn.close()

    total_size = sum(f.stat().st_size for f in seed_dir.iterdir())
    logger.info(
        f"Export complete: {total_size / 1024:.0f} KB in {seed_dir}/ "
        f"({manifest['tables']})"
    )


def _import_tables(
    tables: list[str],
    seed_dir: Path,
    db_path: Path,
) -> None:
    """Import JSONL files into the given tables.

    Backs up existing DB before import. Tables are imported in the order
    given (respects foreign key dependencies).

    Args:
        tables: Ordered list of table names to import.
        seed_dir: Directory containing JSONL files + manifest.
        db_path: Path to the target SQLite database.
    """
    manifest_path = seed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {seed_dir}")

    manifest = json.loads(manifest_path.read_text())
    logger.info(
        f"Importing {manifest.get('snapshot_type', 'seed')} data "
        f"exported at {manifest['exported_at']}"
    )

    if db_path.exists():
        backup = db_path.with_suffix(
            f".pre-import-{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        )
        shutil.copy2(db_path, backup)
        logger.info(f"Backed up existing DB to {backup}")

    from biasbuster.database import Database
    db = Database(str(db_path))
    db.initialize()

    for table in tables:
        jsonl_path = seed_dir / f"{table}.jsonl"
        if not jsonl_path.exists():
            logger.info(f"No {jsonl_path}, skipping")
            continue

        with open(jsonl_path) as f:
            first_line = f.readline().strip()
        if not first_line:
            logger.info(f"Empty {jsonl_path}, skipping")
            continue
        columns = list(json.loads(first_line).keys())

        placeholders = ", ".join("?" for _ in columns)
        col_list = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"

        count = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = _encode_json_fields(json.loads(line))
                values = [record.get(col) for col in columns]
                try:
                    db.conn.execute(sql, values)
                    count += 1
                except sqlite3.Error as e:
                    logger.warning(f"Failed to import {table} row: {e}")

                if count % IMPORT_COMMIT_INTERVAL == 0:
                    db.conn.commit()

        db.conn.commit()
        expected = manifest["tables"].get(table, "?")
        logger.info(f"Imported {count}/{expected} rows into {table}")

    db.close()
    logger.info(f"Import complete → {db_path}")


def export_seed(db_path: Path, seed_dir: Path) -> None:
    """Export papers and enrichments to JSONL files."""
    _export_tables(SEED_TABLES, db_path, seed_dir, snapshot_type="seed")


def import_seed(seed_dir: Path, db_path: Path) -> None:
    """Import papers and enrichments from JSONL into a database."""
    _import_tables(SEED_TABLES, seed_dir, db_path)


def export_annotated(db_path: Path, seed_dir: Path) -> None:
    """Export papers, enrichments, annotations, and human reviews to JSONL."""
    _export_tables(ANNOTATED_TABLES, db_path, seed_dir, snapshot_type="annotated")


def import_annotated(seed_dir: Path, db_path: Path) -> None:
    """Import annotated snapshot from JSONL into a database."""
    _import_tables(ANNOTATED_TABLES, seed_dir, db_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export/import seed and annotated data as JSONL"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export seed data (papers + enrichments)")
    exp.add_argument("--dir", type=Path, default=DEFAULT_SEED_DIR)
    exp.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    imp = sub.add_parser("import", help="Import seed data from JSONL")
    imp.add_argument("--dir", type=Path, default=DEFAULT_SEED_DIR)
    imp.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    exp_ann = sub.add_parser(
        "export-annotated",
        help="Export full annotated dataset (papers + enrichments + annotations)",
    )
    exp_ann.add_argument("--dir", type=Path, default=DEFAULT_ANNOTATED_DIR)
    exp_ann.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    imp_ann = sub.add_parser(
        "import-annotated",
        help="Import annotated dataset from JSONL",
    )
    imp_ann.add_argument("--dir", type=Path, default=DEFAULT_ANNOTATED_DIR)
    imp_ann.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    args = parser.parse_args()

    if args.command == "export":
        export_seed(args.db, args.dir)
    elif args.command == "import":
        import_seed(args.dir, args.db)
    elif args.command == "export-annotated":
        export_annotated(args.db, args.dir)
    elif args.command == "import-annotated":
        import_annotated(args.dir, args.db)


if __name__ == "__main__":
    main()
