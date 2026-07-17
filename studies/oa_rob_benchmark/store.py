"""Isolated SQLite store for the OA-first RoB benchmark (spec §8).

A dedicated database (`dataset/oa_rob_benchmark.db`), never touching
`biasbuster.db` or `eisele_metzger_benchmark.db`. Schema is created with
``CREATE TABLE IF NOT EXISTS`` and is append/upsert only — it never DROPs,
mirroring the HANDOVER rule about destructive rebuilds. Every insert must
satisfy the four-part litmus test (spec §4) or raise ``LitmusError``; the
caller catches it, calls ``log_reject``, and continues.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from studies.oa_rob_benchmark.rob2_tuple import CANONICAL_LEVELS

logger = logging.getLogger(__name__)

BENCHMARK_VERSION = "oa-stage-a-2026-07"
REJECTS_PATH = "dataset/oa_rob_benchmark_rejects.jsonl"

_TUPLE_FIELDS = ("rob2_overall", "rob2_d1", "rob2_d2", "rob2_d3",
                 "rob2_d4", "rob2_d5")
_PROVENANCE_FIELDS = ("source_review_pmid", "resolution_method",
                      "extraction_method", "fulltext_path")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_item (
    trial_pmid TEXT PRIMARY KEY,
    trial_pmcid TEXT, trial_doi TEXT, trial_title TEXT,
    trial_license TEXT, license_redistributable INTEGER NOT NULL,
    non_commercial INTEGER NOT NULL, no_derivatives INTEGER NOT NULL,
    fulltext_path TEXT NOT NULL,
    rob2_overall TEXT NOT NULL, rob2_d1 TEXT NOT NULL, rob2_d2 TEXT NOT NULL,
    rob2_d3 TEXT NOT NULL, rob2_d4 TEXT NOT NULL, rob2_d5 TEXT NOT NULL,
    per_outcome_variant INTEGER NOT NULL DEFAULT 0,
    label_source TEXT NOT NULL, source_review_pmid TEXT NOT NULL,
    source_review_pmcid TEXT, table_index INTEGER, row_index INTEGER,
    resolution_method TEXT NOT NULL, similarity_score REAL,
    pubtype_check TEXT NOT NULL, extraction_method TEXT NOT NULL,
    manual_verified INTEGER NOT NULL DEFAULT 0,
    benchmark_version TEXT NOT NULL
);
"""


class LitmusError(ValueError):
    """Raised when an item fails the four-part inclusion litmus test."""


def litmus_violations(item: dict) -> list[str]:
    """Return every four-part inclusion litmus test violation (spec §4).

    Pure function: no I/O, no side effects. Collects every violation
    (rather than short-circuiting on the first) so callers can report the
    complete picture for logging/triage. An empty list means ``item``
    passes the litmus test.

    Shared by ``BenchmarkStore`` (raises ``LitmusError`` on insert) and
    ``scripts/audit_oa_rob_benchmark.py`` (re-checks persisted rows) so the
    four rules have a single source of truth.
    """
    v: list[str] = []
    if not item.get("license_redistributable"):
        v.append("license not redistributable (litmus §4.1)")
    for f in _TUPLE_FIELDS:
        if item.get(f) not in CANONICAL_LEVELS:
            v.append(f"{f}={item.get(f)!r} not a canonical level (§4.2)")
    if item.get("pubtype_check") != "trial":
        v.append(f"pubtype_check={item.get('pubtype_check')!r} not 'trial' (§4.3)")
    for f in _PROVENANCE_FIELDS + ("trial_pmid",):
        if not str(item.get(f, "")).strip():
            v.append(f"{f} empty (§4.4)")
    return v


class BenchmarkStore:
    """Append/upsert store enforcing the OA-benchmark litmus invariant."""

    def __init__(self, db_path: str) -> None:
        """Open (creating if absent) the benchmark DB and ensure schema."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _check_litmus(self, item: dict) -> None:
        """Validate the four-part inclusion litmus test (spec §4).

        Delegates to the module-level ``litmus_violations`` (single source
        of truth shared with the audit script) and raises ``LitmusError``
        joining every violation found.
        """
        v = litmus_violations(item)
        if v:
            raise LitmusError("; ".join(v))

    def upsert_item(self, item: dict) -> bool:
        """Validate against the litmus test and upsert; raise on violation.

        Only keys that are real ``benchmark_item`` schema columns are
        written; any extra (non-schema) keys in ``item`` — e.g. helper
        fields the ingest tasks carry for their own bookkeeping — are
        silently ignored rather than causing an ``sqlite3.OperationalError``
        for an unknown column.
        """
        self._check_litmus(item)
        schema_cols = set(_row_columns())
        filtered = {k: v for k, v in item.items() if k in schema_cols}
        filtered["benchmark_version"] = BENCHMARK_VERSION
        keys = list(filtered.keys())
        placeholders = ",".join("?" * len(keys))
        updates = ",".join(f"{k}=excluded.{k}" for k in keys if k != "trial_pmid")
        sql = (f"INSERT INTO benchmark_item ({','.join(keys)}) "
               f"VALUES ({placeholders}) "
               f"ON CONFLICT(trial_pmid) DO UPDATE SET {updates}")
        vals = [int(x) if isinstance(x, bool) else x for x in filtered.values()]
        self._conn.execute(sql, vals)
        self._conn.commit()
        return True

    def log_reject(self, candidate: dict, rule: str, detail: str) -> None:
        """Append a rejection record to the rejects JSONL (never silent)."""
        Path(REJECTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(REJECTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"rule": rule, "detail": detail,
                                 "candidate": candidate}) + "\n")
        logger.info("rejected candidate (%s): %s", rule, detail)

    def count(self) -> int:
        """Return the number of rows currently in ``benchmark_item``."""
        return self._conn.execute("SELECT COUNT(*) FROM benchmark_item").fetchone()[0]

    def all_items(self) -> list[dict]:
        """Return every row in ``benchmark_item`` as a list of dicts."""
        return [dict(r) for r in
                self._conn.execute("SELECT * FROM benchmark_item").fetchall()]


def _row_columns() -> list[str]:
    """Column names declared in the schema (for filtering unknown keys).

    Parses the column-definition body between the outer parentheses and
    splits on commas — the schema has no nested parens (no CHECK/FK
    clauses) so a flat split is safe. A naive per-line regex would miss
    every column after the first on a multi-column line (e.g.
    ``trial_pmcid TEXT, trial_doi TEXT, trial_title TEXT,``), silently
    dropping required NOT NULL columns from the insert.
    """
    body = _SCHEMA.split("(", 1)[1].rsplit(")", 1)[0]
    return [part.strip().split()[0] for part in body.split(",") if part.strip()]
