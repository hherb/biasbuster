"""
SQLite Database Backend

Single source of truth for all pipeline data: papers, enrichments,
annotations, and human reviews. Replaces the previous JSONL file-based
storage with schema-enforced uniqueness, atomic updates, and flexible
SQL queries.

Usage:
    from database import Database

    db = Database("dataset/biasbuster.db")
    db.initialize()
    db.insert_paper({"pmid": "123", "title": "...", ...})
    db.close()
"""

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Core paper table: single source of truth for all papers
CREATE TABLE IF NOT EXISTS papers (
    pmid TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    journal TEXT,
    year INTEGER,
    authors JSON,
    grants JSON,
    mesh_terms JSON,
    subjects JSON,
    source TEXT NOT NULL,
    collected_at TEXT DEFAULT (datetime('now')),

    -- Retraction-specific
    retraction_doi TEXT,
    retraction_reasons JSON,
    retraction_source TEXT,

    -- Cochrane RoB-specific
    cochrane_review_pmid TEXT,
    cochrane_review_doi TEXT,
    cochrane_review_title TEXT,
    overall_rob TEXT,
    randomization_bias TEXT,
    deviation_bias TEXT,
    missing_outcome_bias TEXT,
    measurement_bias TEXT,
    reporting_bias TEXT,
    domain TEXT,

    -- Soft-delete: papers flagged as excluded (e.g. bare retraction notices)
    -- are kept in the DB but skipped by annotation/export queries.
    excluded INTEGER NOT NULL DEFAULT 0,
    excluded_reason TEXT
);

-- Enrichment data (one row per paper, added during enrich stage)
CREATE TABLE IF NOT EXISTS enrichments (
    pmid TEXT PRIMARY KEY REFERENCES papers(pmid),
    suspicion_level TEXT,
    reporting_bias_score REAL,
    effect_size_audit JSON,
    outcome_switching JSON,
    enriched_at TEXT DEFAULT (datetime('now'))
);

-- LLM annotations (one row per paper per model)
CREATE TABLE IF NOT EXISTS annotations (
    pmid TEXT NOT NULL REFERENCES papers(pmid),
    model_name TEXT NOT NULL,
    annotation JSON NOT NULL,
    overall_severity TEXT,
    overall_bias_probability REAL,
    confidence TEXT,
    annotated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (pmid, model_name)
);

-- Human review (one row per paper per model's annotation)
CREATE TABLE IF NOT EXISTS human_reviews (
    pmid TEXT NOT NULL,
    model_name TEXT NOT NULL,
    validated INTEGER DEFAULT 0,
    override_severity TEXT,
    annotation JSON,
    flagged INTEGER DEFAULT 0,
    notes TEXT,
    reviewed_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (pmid, model_name),
    FOREIGN KEY (pmid, model_name) REFERENCES annotations(pmid, model_name)
);

-- Evaluation outputs (separate from annotation pipeline)
CREATE TABLE IF NOT EXISTS eval_outputs (
    pmid TEXT NOT NULL,
    model_id TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'zero-shot',
    raw_output TEXT,
    parsed_annotation JSON,
    overall_severity TEXT,
    overall_bias_probability REAL,
    latency_seconds REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error TEXT,
    evaluated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (pmid, model_id, mode)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_enrichments_suspicion ON enrichments(suspicion_level);
CREATE INDEX IF NOT EXISTS idx_annotations_model ON annotations(model_name);
CREATE INDEX IF NOT EXISTS idx_annotations_severity ON annotations(overall_severity);
CREATE INDEX IF NOT EXISTS idx_human_reviews_validated ON human_reviews(validated);
CREATE INDEX IF NOT EXISTS idx_eval_outputs_model ON eval_outputs(model_id);
"""


def _json_col(value) -> Optional[str]:
    """Serialize a value to JSON for storage, or None if empty.

    If value is already a string, validates it's valid JSON before storing.
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Validate it's actually JSON, not an arbitrary string
        try:
            json.loads(value)
            return value
        except (json.JSONDecodeError, TypeError):
            # Wrap plain strings as JSON
            return json.dumps(value)
    return json.dumps(value)


def _json_load(value) -> Optional[list | dict]:
    """Deserialize a JSON column, returning None if empty."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


class Database:
    """SQLite database backend for the bias detection pipeline."""

    def __init__(self, db_path: str | Path = "dataset/biasbuster.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        """Open a new SQLite connection with standard pragmas."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_connected(self) -> None:
        """Reconnect if the database connection was closed."""
        try:
            self.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            self.conn = self._open_connection()

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ---- Schema management ----

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        self.conn.executescript(SCHEMA_SQL)
        # Migrate: add columns if missing (non-destructive for existing databases)
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if "excluded" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0"
            )
        if "excluded_reason" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN excluded_reason TEXT"
            )
        if "cochrane_review_doi" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN cochrane_review_doi TEXT"
            )
        if "cochrane_review_title" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN cochrane_review_title TEXT"
            )
        # Migrate human_reviews: add annotation JSON, flagged columns if missing
        hr_cols = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(human_reviews)"
            ).fetchall()
        }
        if "annotation" not in hr_cols:
            self.conn.execute(
                "ALTER TABLE human_reviews ADD COLUMN annotation JSON"
            )
        if "flagged" not in hr_cols:
            self.conn.execute(
                "ALTER TABLE human_reviews ADD COLUMN flagged INTEGER DEFAULT 0"
            )
            # Backfill: mark existing auto-flagged rows
            self.conn.execute(
                "UPDATE human_reviews SET flagged = 1 "
                "WHERE notes LIKE '[AUTO-FLAGGED]%'"
            )
        # Index on flagged (created here, not in SCHEMA_SQL, because the
        # column may have just been added by the migration above)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_human_reviews_flagged "
            "ON human_reviews(flagged)"
        )
        self.conn.commit()

    # ---- Papers ----

    def insert_paper(self, paper: dict) -> bool:
        """Insert a paper. Returns True if newly inserted (not a duplicate)."""
        pmid = str(paper.get("pmid", ""))
        if not pmid:
            return False
        try:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO papers
                   (pmid, doi, title, abstract, journal, year,
                    authors, grants, mesh_terms, subjects, source,
                    retraction_doi, retraction_reasons, retraction_source,
                    cochrane_review_pmid, cochrane_review_doi,
                    cochrane_review_title, overall_rob,
                    randomization_bias, deviation_bias,
                    missing_outcome_bias, measurement_bias,
                    reporting_bias, domain)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pmid,
                    paper.get("doi"),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("journal"),
                    paper.get("year"),
                    _json_col(paper.get("authors")),
                    _json_col(paper.get("grants")),
                    _json_col(paper.get("mesh_terms")),
                    _json_col(paper.get("subjects")),
                    paper.get("source", "unknown"),
                    paper.get("retraction_doi"),
                    _json_col(paper.get("retraction_reasons")),
                    paper.get("retraction_source"),
                    paper.get("cochrane_review_pmid"),
                    paper.get("cochrane_review_doi"),
                    paper.get("cochrane_review_title"),
                    paper.get("overall_rob"),
                    paper.get("randomization_bias"),
                    paper.get("deviation_bias"),
                    paper.get("missing_outcome_bias"),
                    paper.get("measurement_bias"),
                    paper.get("reporting_bias"),
                    paper.get("domain"),
                ),
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to insert paper {pmid}: {e}")
            return False

    def insert_papers(self, papers: list[dict]) -> int:
        """Bulk insert papers. Returns count of newly inserted rows."""
        rows = []
        for paper in papers:
            pmid = str(paper.get("pmid", ""))
            if not pmid:
                continue
            rows.append((
                pmid,
                paper.get("doi"),
                paper.get("title", ""),
                paper.get("abstract", ""),
                paper.get("journal"),
                paper.get("year"),
                _json_col(paper.get("authors")),
                _json_col(paper.get("grants")),
                _json_col(paper.get("mesh_terms")),
                _json_col(paper.get("subjects")),
                paper.get("source", "unknown"),
                paper.get("retraction_doi"),
                _json_col(paper.get("retraction_reasons")),
                paper.get("retraction_source"),
                paper.get("cochrane_review_pmid"),
                paper.get("cochrane_review_doi"),
                paper.get("cochrane_review_title"),
                paper.get("overall_rob"),
                paper.get("randomization_bias"),
                paper.get("deviation_bias"),
                paper.get("missing_outcome_bias"),
                paper.get("measurement_bias"),
                paper.get("reporting_bias"),
                paper.get("domain"),
            ))
        if not rows:
            return 0
        changes_before = self.conn.total_changes
        self.conn.executemany(
            """INSERT OR IGNORE INTO papers
               (pmid, doi, title, abstract, journal, year,
                authors, grants, mesh_terms, subjects, source,
                retraction_doi, retraction_reasons, retraction_source,
                cochrane_review_pmid, cochrane_review_doi,
                cochrane_review_title, overall_rob,
                randomization_bias, deviation_bias,
                missing_outcome_bias, measurement_bias,
                reporting_bias, domain)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        return self.conn.total_changes - changes_before

    def upsert_cochrane_paper(
        self, paper: dict, *, commit: bool = True
    ) -> bool:
        """Insert or update a Cochrane RoB paper, preserving expensive data.

        On conflict (paper already exists):
        - **Always updates** (Cochrane-authoritative): RoB domain ratings,
          overall_rob, cochrane review metadata, domain, doi.
        - **Preserves if non-empty** (PubMed-authoritative): title,
          abstract, journal, year, authors, grants, mesh_terms, subjects.
        - Review metadata uses COALESCE so an empty string from a re-run
          cannot blank out a previously stored value.

        Use ``collectors.cochrane_rob.rob_assessment_to_paper_dict()`` to
        build the *paper* dict from a ``RoBAssessment`` dataclass.

        Returns True if a row was inserted or updated.
        """
        pmid = str(paper.get("pmid", ""))
        if not pmid:
            return False
        try:
            cursor = self.conn.execute(
                """INSERT INTO papers
                   (pmid, doi, title, abstract, journal, year,
                    authors, grants, mesh_terms, subjects, source,
                    retraction_doi, retraction_reasons, retraction_source,
                    cochrane_review_pmid, cochrane_review_doi,
                    cochrane_review_title, overall_rob,
                    randomization_bias, deviation_bias,
                    missing_outcome_bias, measurement_bias,
                    reporting_bias, domain)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pmid) DO UPDATE SET
                       -- Always update Cochrane-authoritative fields
                       randomization_bias = excluded.randomization_bias,
                       deviation_bias = excluded.deviation_bias,
                       missing_outcome_bias = excluded.missing_outcome_bias,
                       measurement_bias = excluded.measurement_bias,
                       reporting_bias = excluded.reporting_bias,
                       overall_rob = excluded.overall_rob,
                       domain = COALESCE(NULLIF(excluded.domain, ''), papers.domain),
                       doi = COALESCE(NULLIF(excluded.doi, ''), papers.doi),
                       -- Update review metadata only if new value is non-empty
                       cochrane_review_pmid = COALESCE(
                           NULLIF(excluded.cochrane_review_pmid, ''),
                           papers.cochrane_review_pmid),
                       cochrane_review_doi = COALESCE(
                           NULLIF(excluded.cochrane_review_doi, ''),
                           papers.cochrane_review_doi),
                       cochrane_review_title = COALESCE(
                           NULLIF(excluded.cochrane_review_title, ''),
                           papers.cochrane_review_title),
                       -- Preserve PubMed-fetched data: only fill if currently empty
                       title = CASE WHEN papers.title IS NULL OR papers.title = ''
                                    THEN excluded.title ELSE papers.title END,
                       abstract = CASE WHEN papers.abstract IS NULL
                                           OR papers.abstract = ''
                                       THEN excluded.abstract
                                       ELSE papers.abstract END""",
                (
                    pmid,
                    paper.get("doi"),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("journal"),
                    paper.get("year"),
                    _json_col(paper.get("authors")),
                    _json_col(paper.get("grants")),
                    _json_col(paper.get("mesh_terms")),
                    _json_col(paper.get("subjects")),
                    paper.get("source", "cochrane_rob"),
                    paper.get("retraction_doi"),
                    _json_col(paper.get("retraction_reasons")),
                    paper.get("retraction_source"),
                    paper.get("cochrane_review_pmid"),
                    paper.get("cochrane_review_doi"),
                    paper.get("cochrane_review_title"),
                    paper.get("overall_rob"),
                    paper.get("randomization_bias"),
                    paper.get("deviation_bias"),
                    paper.get("missing_outcome_bias"),
                    paper.get("measurement_bias"),
                    paper.get("reporting_bias"),
                    paper.get("domain"),
                ),
            )
            if commit:
                self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to upsert Cochrane paper {pmid}: {e}")
            return False

    def _row_to_paper(self, row: sqlite3.Row) -> dict:
        """Convert a papers table row to a dict with deserialized JSON columns."""
        d = dict(row)
        for col in ("authors", "grants", "mesh_terms", "subjects",
                     "retraction_reasons"):
            d[col] = _json_load(d.get(col))
        return d

    def get_paper(self, pmid: str) -> Optional[dict]:
        """Get a single paper by PMID."""
        row = self.conn.execute(
            "SELECT * FROM papers WHERE pmid = ?", (pmid,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def get_papers(
        self,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        include_excluded: bool = False,
    ) -> list[dict]:
        """Get papers, optionally filtered by source.

        By default, excludes soft-deleted papers (excluded=1).
        Pass ``include_excluded=True`` to get everything.
        """
        query = "SELECT * FROM papers"
        params: list = []
        conditions: list[str] = []
        if source:
            conditions.append("source = ?")
            params.append(source)
        if not include_excluded:
            conditions.append("excluded = 0")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pmid"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_paper_pmids(self, source: Optional[str] = None) -> set[str]:
        """Get all PMIDs, optionally filtered by source."""
        if source:
            rows = self.conn.execute(
                "SELECT pmid FROM papers WHERE source = ?", (source,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT pmid FROM papers").fetchall()
        return {r["pmid"] for r in rows}

    # ---- Enrichments ----

    def upsert_enrichment(
        self, pmid: str, enrichment: dict, *, commit: bool = True
    ) -> None:
        """Insert or update enrichment data for a paper.

        Args:
            pmid: Paper PMID.
            enrichment: Enrichment data dict.
            commit: If False, caller is responsible for committing
                    (use for batch operations).
        """
        self.conn.execute(
            """INSERT INTO enrichments
               (pmid, suspicion_level, reporting_bias_score,
                effect_size_audit, outcome_switching)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(pmid) DO UPDATE SET
                   suspicion_level = excluded.suspicion_level,
                   reporting_bias_score = excluded.reporting_bias_score,
                   effect_size_audit = excluded.effect_size_audit,
                   outcome_switching = excluded.outcome_switching,
                   enriched_at = datetime('now')""",
            (
                pmid,
                enrichment.get("suspicion_level"),
                enrichment.get("reporting_bias_score"),
                _json_col(enrichment.get("effect_size_audit")),
                _json_col(enrichment.get("outcome_switching")),
            ),
        )
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        """Explicitly commit the current transaction."""
        self.conn.commit()

    def get_enriched_papers(
        self,
        suspicion_level: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Get papers joined with their enrichment data."""
        query = """
            SELECT p.*, e.suspicion_level, e.reporting_bias_score,
                   e.effect_size_audit, e.outcome_switching
            FROM papers p
            JOIN enrichments e ON p.pmid = e.pmid
        """
        params: list = []
        if suspicion_level:
            query += " WHERE e.suspicion_level = ?"
            params.append(suspicion_level)
        query += " ORDER BY p.pmid"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = self._row_to_paper(r)
            d["suspicion_level"] = r["suspicion_level"]
            d["reporting_bias_score"] = r["reporting_bias_score"]
            d["effect_size_audit"] = _json_load(r["effect_size_audit"])
            d["outcome_switching"] = _json_load(r["outcome_switching"])
            results.append(d)
        return results

    def get_papers_by_source_for_annotation(
        self, source: str, limit: Optional[int] = None
    ) -> list[dict]:
        """Get papers from a specific source with enrichment data if available.

        For 'high_suspicion' and 'low_suspicion', queries via enrichments table.
        For 'retracted_papers' and 'cochrane_rob', queries papers by source.
        """
        source_map = {
            "high_suspicion": ("suspicion", "high"),
            "low_suspicion": ("suspicion", "low"),
            "retracted_papers": ("source", "retraction_watch"),
            "cochrane_rob": ("source", "cochrane_rob"),
        }
        query_type, value = source_map.get(source, ("source", source))

        if query_type == "suspicion":
            return self.get_enriched_papers(
                suspicion_level=value, limit=limit
            )
        else:
            return self.get_papers(source=value, limit=limit)

    # ---- Annotations ----

    def insert_annotation(
        self, pmid: str, model_name: str, annotation: dict
    ) -> bool:
        """Insert an annotation. Returns True if newly inserted."""
        try:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO annotations
                   (pmid, model_name, annotation,
                    overall_severity, overall_bias_probability, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    pmid,
                    model_name,
                    json.dumps(annotation),
                    annotation.get("overall_severity"),
                    annotation.get("overall_bias_probability"),
                    annotation.get("confidence"),
                ),
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(
                f"Failed to insert annotation {pmid}/{model_name}: {e}"
            )
            return False

    def upsert_annotation(
        self, pmid: str, model_name: str, annotation: dict
    ) -> bool:
        """Insert or update an annotation. Returns True if row was written."""
        try:
            cursor = self.conn.execute(
                """INSERT INTO annotations
                   (pmid, model_name, annotation,
                    overall_severity, overall_bias_probability, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(pmid, model_name) DO UPDATE SET
                       annotation = excluded.annotation,
                       overall_severity = excluded.overall_severity,
                       overall_bias_probability = excluded.overall_bias_probability,
                       confidence = excluded.confidence,
                       annotated_at = datetime('now')""",
                (
                    pmid,
                    model_name,
                    json.dumps(annotation),
                    annotation.get("overall_severity"),
                    annotation.get("overall_bias_probability"),
                    annotation.get("confidence"),
                ),
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(
                f"Failed to upsert annotation {pmid}/{model_name}: {e}"
            )
            return False

    def has_annotation(self, pmid: str, model_name: str) -> bool:
        """Check whether an annotation exists for a given PMID and model."""
        row = self.conn.execute(
            "SELECT 1 FROM annotations WHERE pmid = ? AND model_name = ?",
            (pmid, model_name),
        ).fetchone()
        return row is not None

    def delete_annotation(self, pmid: str, model_name: str) -> bool:
        """Delete an annotation. Returns True if a row was deleted."""
        cursor = self.conn.execute(
            "DELETE FROM annotations WHERE pmid = ? AND model_name = ?",
            (pmid, model_name),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_annotations(
        self,
        model_name: Optional[str] = None,
        pmid: Optional[str] = None,
    ) -> list[dict]:
        """Get annotations, optionally filtered by model and/or PMID."""
        query = "SELECT * FROM annotations"
        conditions = []
        params: list = []
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if pmid:
            conditions.append("pmid = ?")
            params.append(pmid)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pmid"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["overall_severity"] = r["overall_severity"]
            ann["overall_bias_probability"] = r["overall_bias_probability"]
            ann["confidence"] = r["confidence"]
            ann["annotated_at"] = r["annotated_at"]
            results.append(ann)
        return results

    def get_annotated_pmids(self, model_name: str) -> set[str]:
        """Get all PMIDs that have been annotated by a specific model."""
        rows = self.conn.execute(
            "SELECT pmid FROM annotations WHERE model_name = ?",
            (model_name,),
        ).fetchall()
        return {r["pmid"] for r in rows}

    def get_model_names(self) -> list[str]:
        """Get all distinct model names that have annotations."""
        rows = self.conn.execute(
            "SELECT DISTINCT model_name FROM annotations ORDER BY model_name"
        ).fetchall()
        return [r["model_name"] for r in rows]

    # ---- Evaluation outputs ----

    def upsert_eval_output(
        self, pmid: str, model_id: str, mode: str, output: dict
    ) -> bool:
        """Insert or update an evaluation output. Returns True if row was written."""
        try:
            cursor = self.conn.execute(
                """INSERT INTO eval_outputs
                   (pmid, model_id, mode, raw_output, parsed_annotation,
                    overall_severity, overall_bias_probability,
                    latency_seconds, input_tokens, output_tokens, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(pmid, model_id, mode) DO UPDATE SET
                       raw_output = excluded.raw_output,
                       parsed_annotation = excluded.parsed_annotation,
                       overall_severity = excluded.overall_severity,
                       overall_bias_probability = excluded.overall_bias_probability,
                       latency_seconds = excluded.latency_seconds,
                       input_tokens = excluded.input_tokens,
                       output_tokens = excluded.output_tokens,
                       error = excluded.error,
                       evaluated_at = datetime('now')""",
                (
                    pmid,
                    model_id,
                    mode,
                    output.get("raw_output"),
                    json.dumps(output["parsed_annotation"]) if output.get("parsed_annotation") else None,
                    output.get("overall_severity"),
                    output.get("overall_bias_probability"),
                    output.get("latency_seconds"),
                    output.get("input_tokens"),
                    output.get("output_tokens"),
                    output.get("error"),
                ),
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to upsert eval output {pmid}/{model_id}: {e}")
            return False

    def get_evaluated_pmids(self, model_id: str, mode: str) -> set[str]:
        """Get PMIDs already evaluated by a model+mode (successful only)."""
        rows = self.conn.execute(
            "SELECT pmid FROM eval_outputs WHERE model_id = ? AND mode = ? AND error IS NULL",
            (model_id, mode),
        ).fetchall()
        return {r["pmid"] for r in rows}

    # ---- Human reviews ----

    def upsert_review(
        self,
        pmid: str,
        model_name: str,
        validated: bool,
        override_severity: Optional[str] = None,
        notes: Optional[str] = None,
        annotation: Optional[dict] = None,
        flagged: Optional[bool] = None,
    ) -> None:
        """Insert or update a human review.

        Args:
            annotation: Full structured annotation JSON (same schema as LLM output).
            flagged: If True, mark this paper as flagged for review.
                     If None, preserve existing flagged state on update.
        """
        self._ensure_connected()
        annotation_json = _json_col(annotation) if annotation else None
        flagged_int = int(flagged) if flagged is not None else None
        self.conn.execute(
            """INSERT INTO human_reviews
               (pmid, model_name, validated, override_severity, notes,
                annotation, flagged)
               VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 0))
               ON CONFLICT(pmid, model_name) DO UPDATE SET
                   validated = excluded.validated,
                   override_severity = excluded.override_severity,
                   notes = excluded.notes,
                   annotation = excluded.annotation,
                   flagged = COALESCE(excluded.flagged, human_reviews.flagged),
                   reviewed_at = datetime('now')""",
            (pmid, model_name, int(validated), override_severity, notes,
             annotation_json, flagged_int),
        )
        self.conn.commit()

    def get_reviews(
        self, model_name: Optional[str] = None
    ) -> list[dict]:
        """Get human reviews, optionally filtered by model."""
        self._ensure_connected()
        if model_name:
            rows = self.conn.execute(
                "SELECT * FROM human_reviews WHERE model_name = ? ORDER BY pmid",
                (model_name,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM human_reviews ORDER BY pmid"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_annotations_with_paper_data(
        self, model_name: str
    ) -> list[dict]:
        """Get annotations joined with paper metadata for the review UI.

        Returns dicts with the full annotation JSON plus paper fields
        (title, abstract, authors, grants, mesh_terms, journal,
        retraction_reasons, Cochrane RoB fields, enrichment data).
        """
        self._ensure_connected()
        rows = self.conn.execute("""
            SELECT a.pmid, a.model_name, a.annotation,
                   a.overall_severity, a.overall_bias_probability, a.confidence,
                   p.title, p.abstract, p.authors, p.grants, p.mesh_terms,
                   p.journal, p.retraction_reasons, p.source,
                   p.overall_rob, p.randomization_bias, p.deviation_bias,
                   p.missing_outcome_bias, p.measurement_bias, p.reporting_bias,
                   e.effect_size_audit
            FROM annotations a
            JOIN papers p ON a.pmid = p.pmid
            LEFT JOIN enrichments e ON a.pmid = e.pmid
            WHERE a.model_name = ? AND p.excluded = 0
            ORDER BY a.pmid
        """, (model_name,)).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["overall_severity"] = r["overall_severity"]
            ann["overall_bias_probability"] = r["overall_bias_probability"]
            ann["confidence"] = r["confidence"]
            ann["title"] = r["title"] or ""
            ann["abstract_text"] = r["abstract"] or ""
            # Paper metadata for build_user_message
            ann["_paper_metadata"] = {
                "authors": _json_load(r["authors"]),
                "grants": _json_load(r["grants"]),
                "mesh_terms": _json_load(r["mesh_terms"]),
                "journal": r["journal"],
                "retraction_reasons": _json_load(r["retraction_reasons"]),
                "overall_rob": r["overall_rob"],
                "randomization_bias": r["randomization_bias"],
                "deviation_bias": r["deviation_bias"],
                "missing_outcome_bias": r["missing_outcome_bias"],
                "measurement_bias": r["measurement_bias"],
                "reporting_bias": r["reporting_bias"],
                "effect_size_audit": _json_load(r["effect_size_audit"]),
            }
            results.append(ann)
        return results

    # ---- Cross-model queries ----

    def get_disagreements(
        self,
        model_a: str,
        model_b: str,
        field: str = "overall_severity",
    ) -> list[dict]:
        """Find papers where two models disagree on a field."""
        if field == "overall_severity":
            query = """
                SELECT a.pmid, a.overall_severity AS severity_a,
                       b.overall_severity AS severity_b,
                       p.title
                FROM annotations a
                JOIN annotations b ON a.pmid = b.pmid
                JOIN papers p ON a.pmid = p.pmid
                WHERE a.model_name = ? AND b.model_name = ?
                  AND a.overall_severity != b.overall_severity
                ORDER BY a.pmid
            """
        elif field == "overall_bias_probability":
            query = """
                SELECT a.pmid,
                       a.overall_bias_probability AS prob_a,
                       b.overall_bias_probability AS prob_b,
                       ABS(a.overall_bias_probability - b.overall_bias_probability) AS diff,
                       p.title
                FROM annotations a
                JOIN annotations b ON a.pmid = b.pmid
                JOIN papers p ON a.pmid = p.pmid
                WHERE a.model_name = ? AND b.model_name = ?
                ORDER BY diff DESC
            """
        else:
            return []
        rows = self.conn.execute(query, (model_a, model_b)).fetchall()
        return [dict(r) for r in rows]

    def get_annotation_comparison(self, models: list[str]) -> list[dict]:
        """Get all annotations for papers annotated by all specified models."""
        if not models:
            return []
        placeholders = ",".join("?" * len(models))
        query = f"""
            SELECT a.pmid, a.model_name, a.annotation,
                   a.overall_severity, a.overall_bias_probability
            FROM annotations a
            WHERE a.pmid IN (
                SELECT pmid FROM annotations
                WHERE model_name IN ({placeholders})
                GROUP BY pmid
                HAVING COUNT(DISTINCT model_name) = ?
            )
            ORDER BY a.pmid, a.model_name
        """
        rows = self.conn.execute(
            query, (*models, len(models))
        ).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["overall_severity"] = r["overall_severity"]
            ann["overall_bias_probability"] = r["overall_bias_probability"]
            results.append(ann)
        return results

    # ---- Export helpers ----

    def get_all_annotations_for_export(self) -> list[dict]:
        """Get all annotations with paper data merged in, for export.

        Excludes soft-deleted papers (excluded=1).
        """
        rows = self.conn.execute("""
            SELECT a.pmid, a.model_name, a.annotation,
                   p.title, p.abstract
            FROM annotations a
            JOIN papers p ON a.pmid = p.pmid
            WHERE p.excluded = 0
            ORDER BY a.pmid
        """).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["title"] = r["title"]
            ann["abstract_text"] = r["abstract"]
            ann["_annotation_model"] = r["model_name"]
            results.append(ann)
        return results

    def export_review_csv(self, model_name: str, output_path: Path) -> None:
        """Export annotations + human reviews as a CSV for human review."""
        from annotators import REVIEW_CSV_COLUMNS

        rows = self.conn.execute("""
            SELECT a.pmid, a.annotation, a.overall_severity,
                   a.overall_bias_probability, a.confidence,
                   p.title,
                   h.validated AS HUMAN_VALIDATED,
                   h.override_severity AS HUMAN_OVERRIDE_SEVERITY,
                   h.notes AS HUMAN_NOTES
            FROM annotations a
            JOIN papers p ON a.pmid = p.pmid
            LEFT JOIN human_reviews h
                ON a.pmid = h.pmid AND a.model_name = h.model_name
            WHERE a.model_name = ?
            ORDER BY a.pmid
        """, (model_name,)).fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REVIEW_CSV_COLUMNS)
            for r in rows:
                ann = _json_load(r["annotation"]) or {}
                stat = ann.get("statistical_reporting", {})
                spin = ann.get("spin", {})
                coi = ann.get("conflict_of_interest", {})
                reasoning = ann.get("reasoning", "")

                writer.writerow([
                    r["pmid"],
                    (r["title"] or "")[:100],
                    r["overall_severity"] or "",
                    r["overall_bias_probability"] or "",
                    stat.get("severity", ""),
                    stat.get("relative_only", ""),
                    spin.get("spin_level", ""),
                    coi.get("funding_type", ""),
                    r["confidence"] or "",
                    reasoning[:200],
                    "True" if r["HUMAN_VALIDATED"] else "",
                    r["HUMAN_OVERRIDE_SEVERITY"] or "",
                    r["HUMAN_NOTES"] or "",
                ])
        logger.info(f"Exported review CSV to {output_path}")

    # ---- Stats ----

    def get_stats(self) -> dict:
        """Get summary statistics for the database."""
        stats: dict = {}

        # Paper counts by source
        rows = self.conn.execute(
            "SELECT source, COUNT(*) as cnt FROM papers GROUP BY source"
        ).fetchall()
        stats["papers_by_source"] = {r["source"]: r["cnt"] for r in rows}
        stats["total_papers"] = sum(stats["papers_by_source"].values())

        # Enrichment counts
        rows = self.conn.execute(
            "SELECT suspicion_level, COUNT(*) as cnt "
            "FROM enrichments GROUP BY suspicion_level"
        ).fetchall()
        stats["enrichments_by_level"] = {
            r["suspicion_level"]: r["cnt"] for r in rows
        }
        stats["total_enrichments"] = sum(
            stats["enrichments_by_level"].values()
        )

        # Annotation counts by model
        rows = self.conn.execute(
            "SELECT model_name, COUNT(*) as cnt "
            "FROM annotations GROUP BY model_name"
        ).fetchall()
        stats["annotations_by_model"] = {
            r["model_name"]: r["cnt"] for r in rows
        }
        stats["total_annotations"] = sum(
            stats["annotations_by_model"].values()
        )

        # Review counts
        row = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN validated = 1 THEN 1 ELSE 0 END) as validated "
            "FROM human_reviews"
        ).fetchone()
        stats["total_reviews"] = row["total"]
        stats["validated_reviews"] = row["validated"] or 0

        return stats
