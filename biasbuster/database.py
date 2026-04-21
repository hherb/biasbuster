"""
SQLite Database Backend

Single source of truth for all pipeline data: papers, enrichments,
annotations, and human reviews. Replaces the previous JSONL file-based
storage with schema-enforced uniqueness, atomic updates, and flexible
SQL queries.

Usage:
    from biasbuster.database import Database

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

-- LLM annotations (one row per paper per model per methodology).
-- `methodology` identifies which risk-of-bias tool was applied
-- (biasbuster, cochrane_rob2, quadas_2, ...). The PK lets the same
-- model annotate the same paper under different methodologies.
-- `overall_severity` holds the methodology-agnostic overall rating
-- in the methodology's rollup vocabulary (e.g. "high" for biasbuster,
-- "some_concerns" for cochrane_rob2). Column name kept for
-- back-compat; treat it as "overall rating" for non-biasbuster rows.
CREATE TABLE IF NOT EXISTS annotations (
    pmid TEXT NOT NULL REFERENCES papers(pmid),
    model_name TEXT NOT NULL,
    methodology TEXT NOT NULL DEFAULT 'biasbuster',
    methodology_version TEXT,
    annotation JSON NOT NULL,
    overall_severity TEXT,
    overall_bias_probability REAL,
    confidence TEXT,
    annotated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (pmid, model_name, methodology)
);

-- Human review (one row per paper per model per methodology).
CREATE TABLE IF NOT EXISTS human_reviews (
    pmid TEXT NOT NULL,
    model_name TEXT NOT NULL,
    methodology TEXT NOT NULL DEFAULT 'biasbuster',
    validated INTEGER DEFAULT 0,
    override_severity TEXT,
    annotation JSON,
    flagged INTEGER DEFAULT 0,
    notes TEXT,
    reviewed_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (pmid, model_name, methodology),
    FOREIGN KEY (pmid, model_name, methodology)
        REFERENCES annotations(pmid, model_name, methodology)
);

-- Expert ground-truth ratings from published systematic reviews,
-- Cochrane RoB assessments, or other authoritative sources. Serves as
-- the evaluation target for the methodology-faithfulness harness.
--
-- Identity is (methodology, rating_source, study_label): the same study
-- can carry multiple rating rows (under different methodologies, or from
-- different reviews), and a single review can assess multiple strata of
-- the same study (e.g. "Egboh (2022)-Male" vs "-Female"). No FK on
-- ``pmid`` because a rating may be recorded before its paper has been
-- ingested into ``papers``.
--
-- ``domain_ratings`` is a methodology-specific JSON blob. For QUADAS-2:
--     {"patient_selection": {"bias": "low"}, ...}
-- For RoB 2:
--     {"randomization": "low", "deviations": "some_concerns", ...}
-- ``verified=1`` marks rows a human has inspected and trusts for use as
-- ground truth (vs. machine-extracted with a plausibility heuristic).
CREATE TABLE IF NOT EXISTS expert_methodology_ratings (
    methodology TEXT NOT NULL,
    rating_source TEXT NOT NULL,
    study_label TEXT NOT NULL,
    pmid TEXT,
    doi TEXT,
    methodology_version TEXT,
    source_review_pmid TEXT,
    source_review_doi TEXT,
    source_reference TEXT,
    domain_ratings JSON NOT NULL,
    overall_rating TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    added_by TEXT,
    notes TEXT,
    added_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (methodology, rating_source, study_label)
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

CREATE TABLE IF NOT EXISTS manually_verified (
    pmid TEXT NOT NULL REFERENCES papers(pmid),
    verification_set TEXT NOT NULL,
    trial_name TEXT,
    source_review TEXT,
    fulltext_path TEXT,
    fulltext_ok INTEGER NOT NULL DEFAULT 0,
    added_at TEXT DEFAULT (datetime('now')),
    notes TEXT,
    PRIMARY KEY (pmid, verification_set)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_enrichments_suspicion ON enrichments(suspicion_level);
CREATE INDEX IF NOT EXISTS idx_annotations_model ON annotations(model_name);
CREATE INDEX IF NOT EXISTS idx_annotations_methodology ON annotations(methodology);
CREATE INDEX IF NOT EXISTS idx_annotations_meth_model ON annotations(methodology, model_name);
CREATE INDEX IF NOT EXISTS idx_annotations_severity ON annotations(overall_severity);
CREATE INDEX IF NOT EXISTS idx_human_reviews_validated ON human_reviews(validated);
CREATE INDEX IF NOT EXISTS idx_eval_outputs_model ON eval_outputs(model_id);
CREATE INDEX IF NOT EXISTS idx_expert_ratings_pmid
    ON expert_methodology_ratings(pmid);
CREATE INDEX IF NOT EXISTS idx_expert_ratings_methodology
    ON expert_methodology_ratings(methodology);
CREATE INDEX IF NOT EXISTS idx_expert_ratings_source
    ON expert_methodology_ratings(rating_source);
-- The faithfulness harness joins (methodology, pmid) onto annotations,
-- so this composite index is the hot path. Single-column indexes above
-- still help ad-hoc queries filtered on only one dimension.
CREATE INDEX IF NOT EXISTS idx_expert_ratings_meth_pmid
    ON expert_methodology_ratings(methodology, pmid);
CREATE INDEX IF NOT EXISTS idx_manually_verified_set
    ON manually_verified(verification_set);
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


def json_load(value) -> Optional[list | dict]:
    """Deserialize a JSON column, returning None if empty."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


# Keep backward-compatible alias for internal uses
_json_load = json_load

# -- Cochrane rebuild (REBUILD_DESIGN.md §7) invariant constants ----------

ROB_REQUIRED_FIELDS: tuple[str, ...] = (
    "overall_rob",
    "randomization_bias",
    "deviation_bias",
    "missing_outcome_bias",
    "measurement_bias",
    "reporting_bias",
)

VALID_ROB_LEVELS: frozenset[str] = frozenset({"low", "some_concerns", "high"})

CURRENT_ROB_SOURCE_VERSION: str = "rebuild-2026-04"


class RoBInvariantError(ValueError):
    """Raised when a paper fails the RoB provenance invariant (§7.2)."""

    def __init__(self, pmid: str, violations: list[str]) -> None:
        self.pmid = pmid
        self.violations = violations
        super().__init__(
            f"Paper {pmid} violates RoB provenance invariant: "
            + "; ".join(violations)
        )


class LegacySchemaError(RuntimeError):
    """Raised when opening a database with the pre-methodology schema.

    The caller must run the migration script to quarantine the legacy
    data and create a fresh methodology-aware database.
    """


# -- expert_methodology_ratings upsert SQL --------------------------------
#
# Two upsert paths sharing the same INSERT column list. The ON CONFLICT
# clauses are split into named constants so the SQL is readable at a
# glance and unit tests can pin the preservation behaviour.

_EXPERT_RATING_INSERT_SQL: str = """
    INSERT INTO expert_methodology_ratings
        (methodology, rating_source, study_label,
         pmid, doi, methodology_version,
         source_review_pmid, source_review_doi, source_reference,
         domain_ratings, overall_rating,
         verified, added_by, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

#: On conflict, refresh the machine-produced fields and bump ``updated_at``
#: but keep every curator-managed field (``verified``, ``added_by``,
#: ``notes``) intact. ``pmid`` / ``doi`` / provenance fields are
#: COALESCE-guarded so a re-ingest with ``None`` cannot blank out a
#: previously-resolved value.
_EXPERT_RATING_CONFLICT_PRESERVE_CURATION: str = """
    ON CONFLICT(methodology, rating_source, study_label) DO UPDATE SET
        domain_ratings = excluded.domain_ratings,
        overall_rating = excluded.overall_rating,
        pmid = COALESCE(excluded.pmid, expert_methodology_ratings.pmid),
        doi = COALESCE(excluded.doi, expert_methodology_ratings.doi),
        methodology_version = COALESCE(
            excluded.methodology_version,
            expert_methodology_ratings.methodology_version),
        source_review_pmid = COALESCE(
            excluded.source_review_pmid,
            expert_methodology_ratings.source_review_pmid),
        source_review_doi = COALESCE(
            excluded.source_review_doi,
            expert_methodology_ratings.source_review_doi),
        source_reference = COALESCE(
            excluded.source_reference,
            expert_methodology_ratings.source_reference),
        updated_at = datetime('now')
"""

#: Force-overwrite path: every field (including curator-managed ones) is
#: replaced with the incoming values. ``added_at`` is preserved because
#: it's the provenance timestamp of the original insertion.
_EXPERT_RATING_CONFLICT_FORCE_OVERWRITE: str = """
    ON CONFLICT(methodology, rating_source, study_label) DO UPDATE SET
        domain_ratings = excluded.domain_ratings,
        overall_rating = excluded.overall_rating,
        pmid = excluded.pmid,
        doi = excluded.doi,
        methodology_version = excluded.methodology_version,
        source_review_pmid = excluded.source_review_pmid,
        source_review_doi = excluded.source_review_doi,
        source_reference = excluded.source_reference,
        verified = excluded.verified,
        added_by = excluded.added_by,
        notes = excluded.notes,
        updated_at = datetime('now')
"""


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

    def _check_legacy_annotations_schema(self) -> None:
        """Refuse to initialize on top of a pre-methodology annotations table.

        The pre-methodology schema had PK (pmid, model_name) and no
        `methodology` column. If we detect that shape, raise rather than
        auto-migrate — the user must explicitly run the migration script
        so legacy (potentially incorrect) annotations don't silently
        carry over into the new DB.
        """
        row = self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='annotations'"
        ).fetchone()
        if row is None:
            return  # fresh DB; SCHEMA_SQL will create the new shape
        cols = {
            c[1]
            for c in self.conn.execute("PRAGMA table_info(annotations)").fetchall()
        }
        if "methodology" not in cols:
            raise LegacySchemaError(
                f"Database at {self.db_path} has the pre-methodology "
                "annotations schema. Run "
                "`uv run python scripts/migrations/add_methodology_support.py "
                f"--from {self.db_path} --to <new_db_path>` "
                "to preserve the legacy DB and start fresh, or point this "
                "Database at a new path."
            )

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist.

        Raises:
            LegacySchemaError: If the database exists but has the pre-methodology
                annotations schema (PK was (pmid, model_name)). The caller must
                run the migration script
                ``scripts/migrations/add_methodology_support.py`` or point at a
                fresh DB path. We refuse to silently auto-migrate because the
                legacy database may contain incorrect annotations that should be
                quarantined before fresh runs, per the multi-methodology rebuild
                plan.
        """
        self._check_legacy_annotations_schema()
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
        # Cochrane rebuild (REBUILD_DESIGN.md §7.1): provenance invariants.
        # rob_provenance holds the full extraction-traceability record
        # (review PMID/PMCID, section, table_index, row_index, study_id_text,
        # resolution_method, extraction_method, etc.). rob_source_version
        # tags which harvest run populated the row, letting rebuilt rows
        # co-exist with archived legacy rows without silent conflation.
        if "rob_provenance" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN rob_provenance JSON"
            )
        if "rob_source_version" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN rob_source_version TEXT"
            )
        # Multi-methodology support: which bias assessment tool was used
        # (rob2, quadas2, robins_i, etc.) and a generic JSON column for
        # tool-specific per-domain ratings that don't fit the hardcoded
        # RoB 2 columns. The hardcoded columns stay for RoB 2 backward
        # compat; new tools use bias_domain_ratings exclusively.
        if "methodology" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN methodology TEXT"
            )
        if "bias_domain_ratings" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN bias_domain_ratings JSON"
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
        # expert_methodology_ratings: backfill updated_at for DBs that
        # had the table without the column.
        expert_cols = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(expert_methodology_ratings)"
            ).fetchall()
        }
        if expert_cols and "updated_at" not in expert_cols:
            self.conn.execute(
                "ALTER TABLE expert_methodology_ratings "
                "ADD COLUMN updated_at TEXT"
            )
            self.conn.execute(
                "UPDATE expert_methodology_ratings "
                "SET updated_at = added_at WHERE updated_at IS NULL"
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

    def upsert_cochrane_paper_v2(
        self, paper: dict, *, commit: bool = True
    ) -> bool:
        """Insert or update a Cochrane paper with the v2 provenance invariant.

        Enforces REBUILD_DESIGN.md §7.2: every row must carry all five
        domain ratings + overall, a valid cochrane_review_pmid, a non-null
        rob_provenance JSON blob, and ``rob_source_version`` matching the
        current rebuild version.

        Raises:
            RoBInvariantError: If any invariant field is missing, empty,
                or has an unexpected value. The caller should catch this,
                log the rejection, and move on.

        Returns:
            True if a row was inserted or updated.
        """
        pmid = str(paper.get("pmid", "")).strip()
        violations: list[str] = []

        if not pmid:
            violations.append("pmid is empty")

        for field in ROB_REQUIRED_FIELDS:
            val = paper.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                violations.append(f"{field} is empty")
            elif val not in VALID_ROB_LEVELS:
                violations.append(
                    f"{field}={val!r} not in {sorted(VALID_ROB_LEVELS)}"
                )

        if not paper.get("cochrane_review_pmid"):
            violations.append("cochrane_review_pmid is empty")

        provenance = paper.get("rob_provenance")
        if not provenance:
            violations.append("rob_provenance is empty")

        src_ver = paper.get("rob_source_version")
        if src_ver != CURRENT_ROB_SOURCE_VERSION:
            violations.append(
                f"rob_source_version={src_ver!r} "
                f"(expected {CURRENT_ROB_SOURCE_VERSION!r})"
            )

        if violations:
            raise RoBInvariantError(pmid, violations)

        try:
            cursor = self.conn.execute(
                """INSERT INTO papers
                   (pmid, doi, title, abstract, journal, year,
                    authors, grants, mesh_terms, subjects, source,
                    cochrane_review_pmid, cochrane_review_doi,
                    cochrane_review_title, overall_rob,
                    randomization_bias, deviation_bias,
                    missing_outcome_bias, measurement_bias,
                    reporting_bias, domain,
                    rob_provenance, rob_source_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pmid) DO UPDATE SET
                       overall_rob = excluded.overall_rob,
                       randomization_bias = excluded.randomization_bias,
                       deviation_bias = excluded.deviation_bias,
                       missing_outcome_bias = excluded.missing_outcome_bias,
                       measurement_bias = excluded.measurement_bias,
                       reporting_bias = excluded.reporting_bias,
                       cochrane_review_pmid = excluded.cochrane_review_pmid,
                       cochrane_review_doi = excluded.cochrane_review_doi,
                       cochrane_review_title = excluded.cochrane_review_title,
                       domain = excluded.domain,
                       rob_provenance = excluded.rob_provenance,
                       rob_source_version = excluded.rob_source_version,
                       excluded = 0,
                       excluded_reason = NULL,
                       -- Preserve PubMed-fetched data if already present
                       title = CASE WHEN papers.title IS NULL
                                        OR papers.title = ''
                                    THEN excluded.title
                                    ELSE papers.title END,
                       abstract = CASE WHEN papers.abstract IS NULL
                                           OR papers.abstract = ''
                                       THEN excluded.abstract
                                       ELSE papers.abstract END,
                       doi = COALESCE(NULLIF(excluded.doi, ''), papers.doi)
                """,
                (
                    pmid,
                    paper.get("doi", ""),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("journal"),
                    paper.get("year"),
                    _json_col(paper.get("authors")),
                    _json_col(paper.get("grants")),
                    _json_col(paper.get("mesh_terms")),
                    _json_col(paper.get("subjects")),
                    paper.get("source", "cochrane_rob"),
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
                    _json_col(provenance),
                    src_ver,
                ),
            )
            if commit:
                self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to upsert v2 Cochrane paper {pmid}: {e}")
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
        self,
        pmid: str,
        model_name: str,
        annotation: dict,
        methodology: str = "biasbuster",
        methodology_version: Optional[str] = None,
    ) -> bool:
        """Insert an annotation. Returns True if newly inserted.

        Args:
            pmid: Paper PMID.
            model_name: LLM backend identifier.
            annotation: Structured annotation dict (methodology-specific shape).
            methodology: Which risk-of-bias tool was applied. Defaults to
                "biasbuster" for back-compat with pre-methodology callers.
            methodology_version: Optional prompt/schema version tag
                (e.g. "v5a" for biasbuster, "rob2-2019" for Cochrane RoB 2).
        """
        try:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO annotations
                   (pmid, model_name, methodology, methodology_version,
                    annotation, overall_severity,
                    overall_bias_probability, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pmid,
                    model_name,
                    methodology,
                    methodology_version,
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
                f"Failed to insert annotation {pmid}/{model_name}"
                f"/{methodology}: {e}"
            )
            return False

    def upsert_annotation(
        self,
        pmid: str,
        model_name: str,
        annotation: dict,
        methodology: str = "biasbuster",
        methodology_version: Optional[str] = None,
    ) -> bool:
        """Insert or update an annotation. Returns True if row was written.

        See :meth:`insert_annotation` for argument semantics. On conflict
        the annotation JSON and summary fields are replaced, but the
        (pmid, model_name, methodology) identity is preserved.
        """
        try:
            cursor = self.conn.execute(
                """INSERT INTO annotations
                   (pmid, model_name, methodology, methodology_version,
                    annotation, overall_severity,
                    overall_bias_probability, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(pmid, model_name, methodology) DO UPDATE SET
                       methodology_version = excluded.methodology_version,
                       annotation = excluded.annotation,
                       overall_severity = excluded.overall_severity,
                       overall_bias_probability = excluded.overall_bias_probability,
                       confidence = excluded.confidence,
                       annotated_at = datetime('now')""",
                (
                    pmid,
                    model_name,
                    methodology,
                    methodology_version,
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
                f"Failed to upsert annotation {pmid}/{model_name}"
                f"/{methodology}: {e}"
            )
            return False

    def has_annotation(
        self,
        pmid: str,
        model_name: str,
        methodology: str = "biasbuster",
    ) -> bool:
        """Check whether an annotation exists for (PMID, model, methodology)."""
        row = self.conn.execute(
            "SELECT 1 FROM annotations "
            "WHERE pmid = ? AND model_name = ? AND methodology = ?",
            (pmid, model_name, methodology),
        ).fetchone()
        return row is not None

    def delete_annotation(
        self,
        pmid: str,
        model_name: str,
        methodology: str = "biasbuster",
    ) -> bool:
        """Delete a single annotation. Returns True if a row was deleted."""
        # Delete dependent human_reviews first (FK constraint)
        self.conn.execute(
            "DELETE FROM human_reviews "
            "WHERE pmid = ? AND model_name = ? AND methodology = ?",
            (pmid, model_name, methodology),
        )
        cursor = self.conn.execute(
            "DELETE FROM annotations "
            "WHERE pmid = ? AND model_name = ? AND methodology = ?",
            (pmid, model_name, methodology),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_retracted_paper_pmids_with_reasons(self) -> list[dict]:
        """Get all non-excluded retracted papers with their retraction reasons.

        Returns list of dicts with 'pmid', 'title', 'retraction_reasons'.
        """
        rows = self.conn.execute("""
            SELECT pmid, title, retraction_reasons
            FROM papers
            WHERE retraction_reasons IS NOT NULL
              AND retraction_reasons != '[]'
              AND excluded = 0
        """).fetchall()
        return [dict(r) for r in rows]

    def delete_annotations_for_pmids(
        self,
        pmids: list[str],
        model_name: str | None = None,
        methodology: str | None = None,
    ) -> int:
        """Delete annotations for a list of PMIDs.

        Args:
            pmids: List of PMIDs to delete annotations for.
            model_name: If provided, only delete for this model.
                If None, delete all models' annotations.
            methodology: If provided, only delete annotations for this
                methodology. If None, delete rows for every methodology.

        Returns:
            Number of annotations deleted.
        """
        if not pmids:
            return 0
        placeholders = ",".join("?" * len(pmids))
        where_parts = [f"pmid IN ({placeholders})"]
        params: list = list(pmids)
        if model_name is not None:
            where_parts.append("model_name = ?")
            params.append(model_name)
        if methodology is not None:
            where_parts.append("methodology = ?")
            params.append(methodology)
        where_sql = " AND ".join(where_parts)
        # Delete dependent human_reviews first (FK constraint)
        self.conn.execute(
            f"DELETE FROM human_reviews WHERE {where_sql}", params
        )
        cursor = self.conn.execute(
            f"DELETE FROM annotations WHERE {where_sql}", params
        )
        self.conn.commit()
        return cursor.rowcount

    def get_annotations(
        self,
        model_name: Optional[str] = None,
        pmid: Optional[str] = None,
        methodology: Optional[str] = None,
    ) -> list[dict]:
        """Get annotations, optionally filtered by model, PMID, and methodology.

        Args:
            model_name: Filter to a specific LLM backend if given.
            pmid: Filter to a single paper if given.
            methodology: Filter to a specific methodology (e.g. "biasbuster",
                "cochrane_rob2"). If None, returns rows for every methodology.
        """
        query = "SELECT * FROM annotations"
        conditions = []
        params: list = []
        if model_name is not None:
            conditions.append("model_name = ?")
            params.append(model_name)
        if pmid is not None:
            conditions.append("pmid = ?")
            params.append(pmid)
        if methodology is not None:
            conditions.append("methodology = ?")
            params.append(methodology)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pmid"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["methodology"] = r["methodology"]
            ann["methodology_version"] = r["methodology_version"]
            ann["overall_severity"] = r["overall_severity"]
            ann["overall_bias_probability"] = r["overall_bias_probability"]
            ann["confidence"] = r["confidence"]
            ann["annotated_at"] = r["annotated_at"]
            results.append(ann)
        return results

    def get_annotated_pmids(
        self,
        model_name: str,
        methodology: Optional[str] = None,
    ) -> set[str]:
        """Get all PMIDs annotated by a specific model.

        Args:
            model_name: LLM backend identifier.
            methodology: If provided, restrict to papers annotated under
                this methodology. If None, returns PMIDs annotated by this
                model under any methodology.
        """
        if methodology is not None:
            rows = self.conn.execute(
                "SELECT pmid FROM annotations "
                "WHERE model_name = ? AND methodology = ?",
                (model_name, methodology),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT pmid FROM annotations WHERE model_name = ?",
                (model_name,),
            ).fetchall()
        return {r["pmid"] for r in rows}

    def get_model_names(
        self, methodology: Optional[str] = None,
    ) -> list[str]:
        """Get all distinct model names that have annotations.

        Args:
            methodology: If provided, only models with annotations produced
                under this methodology. If None, all models across any
                methodology.
        """
        if methodology is not None:
            rows = self.conn.execute(
                "SELECT DISTINCT model_name FROM annotations "
                "WHERE methodology = ? ORDER BY model_name",
                (methodology,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT model_name FROM annotations "
                "ORDER BY model_name"
            ).fetchall()
        return [r["model_name"] for r in rows]

    def get_methodology_names(self) -> list[str]:
        """Get all distinct methodology slugs that have annotations."""
        rows = self.conn.execute(
            "SELECT DISTINCT methodology FROM annotations ORDER BY methodology"
        ).fetchall()
        return [r["methodology"] for r in rows]

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
        methodology: str = "biasbuster",
    ) -> None:
        """Insert or update a human review.

        Args:
            annotation: Full structured annotation JSON (same schema as LLM output).
            flagged: If True, mark this paper as flagged for review.
                     If None, preserve existing flagged state on update.
            methodology: Which methodology's annotation this review refers to.
                Defaults to "biasbuster" for back-compat.
        """
        self._ensure_connected()
        annotation_json = _json_col(annotation) if annotation else None
        flagged_int = int(flagged) if flagged is not None else None
        self.conn.execute(
            """INSERT INTO human_reviews
               (pmid, model_name, methodology, validated, override_severity,
                notes, annotation, flagged)
               VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0))
               ON CONFLICT(pmid, model_name, methodology) DO UPDATE SET
                   validated = excluded.validated,
                   override_severity = excluded.override_severity,
                   notes = excluded.notes,
                   annotation = excluded.annotation,
                   flagged = COALESCE(excluded.flagged, human_reviews.flagged),
                   reviewed_at = datetime('now')""",
            (pmid, model_name, methodology, int(validated), override_severity,
             notes, annotation_json, flagged_int),
        )
        self.conn.commit()

    def get_reviews(
        self,
        model_name: Optional[str] = None,
        methodology: Optional[str] = None,
    ) -> list[dict]:
        """Get human reviews, optionally filtered by model and/or methodology."""
        self._ensure_connected()
        query = "SELECT * FROM human_reviews"
        conditions: list[str] = []
        params: list = []
        if model_name is not None:
            conditions.append("model_name = ?")
            params.append(model_name)
        if methodology is not None:
            conditions.append("methodology = ?")
            params.append(methodology)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pmid"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_annotations_with_paper_data(
        self,
        model_name: str,
        methodology: Optional[str] = None,
    ) -> list[dict]:
        """Get annotations joined with paper metadata for the review UI.

        Returns dicts with the full annotation JSON plus paper fields
        (title, abstract, authors, grants, mesh_terms, journal,
        retraction_reasons, Cochrane RoB fields, enrichment data).

        Args:
            model_name: LLM backend identifier.
            methodology: If given, restrict to annotations produced under
                this methodology. If None, returns rows for every methodology
                (callers relying on a single per-paper row should pass this).
        """
        self._ensure_connected()
        query = """
            SELECT a.pmid, a.model_name, a.methodology, a.methodology_version,
                   a.annotation,
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
        """
        params: list = [model_name]
        if methodology is not None:
            query += " AND a.methodology = ?"
            params.append(methodology)
        query += " ORDER BY a.pmid"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["methodology"] = r["methodology"]
            ann["methodology_version"] = r["methodology_version"]
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
        methodology: str = "biasbuster",
    ) -> list[dict]:
        """Find papers where two models disagree on a field.

        The comparison is scoped to a single methodology — disagreement
        between a biasbuster annotation and a cochrane_rob2 annotation is
        not meaningful (the rollup vocabularies differ).
        """
        if field == "overall_severity":
            query = """
                SELECT a.pmid, a.overall_severity AS severity_a,
                       b.overall_severity AS severity_b,
                       p.title
                FROM annotations a
                JOIN annotations b ON a.pmid = b.pmid
                    AND a.methodology = b.methodology
                JOIN papers p ON a.pmid = p.pmid
                WHERE a.model_name = ? AND b.model_name = ?
                  AND a.methodology = ?
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
                    AND a.methodology = b.methodology
                JOIN papers p ON a.pmid = p.pmid
                WHERE a.model_name = ? AND b.model_name = ?
                  AND a.methodology = ?
                ORDER BY diff DESC
            """
        else:
            return []
        rows = self.conn.execute(
            query, (model_a, model_b, methodology)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_annotation_comparison(
        self,
        models: list[str],
        methodology: str = "biasbuster",
    ) -> list[dict]:
        """Get annotations for papers annotated by all specified models.

        Scoped to a single methodology so the compared rows share a
        rollup vocabulary.
        """
        if not models:
            return []
        placeholders = ",".join("?" * len(models))
        query = f"""
            SELECT a.pmid, a.model_name, a.methodology, a.annotation,
                   a.overall_severity, a.overall_bias_probability
            FROM annotations a
            WHERE a.methodology = ?
              AND a.pmid IN (
                SELECT pmid FROM annotations
                WHERE model_name IN ({placeholders})
                  AND methodology = ?
                GROUP BY pmid
                HAVING COUNT(DISTINCT model_name) = ?
            )
            ORDER BY a.pmid, a.model_name
        """
        rows = self.conn.execute(
            query, (methodology, *models, methodology, len(models))
        ).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["model_name"] = r["model_name"]
            ann["methodology"] = r["methodology"]
            ann["overall_severity"] = r["overall_severity"]
            ann["overall_bias_probability"] = r["overall_bias_probability"]
            results.append(ann)
        return results

    # ---- Export helpers ----

    def get_all_annotations_for_export(
        self,
        model_name: str | None = None,
        methodology: str = "biasbuster",
    ) -> list[dict]:
        """Get all annotations with paper data merged in, for export.

        Args:
            model_name: If provided, only export annotations from this model.
                If None, exports all models' annotations.
            methodology: Export only annotations produced under this
                methodology. Defaults to "biasbuster" because the training
                export pipeline currently only knows how to render
                biasbuster-shaped annotations; other methodologies are
                evaluation-only in v1.

        Excludes soft-deleted papers (excluded=1).
        """
        query = """
            SELECT a.pmid, a.model_name, a.methodology, a.annotation,
                   p.title, p.abstract, p.retraction_reasons
            FROM annotations a
            JOIN papers p ON a.pmid = p.pmid
            WHERE p.excluded = 0 AND a.methodology = ?
        """
        params: list = [methodology]
        if model_name:
            query += " AND a.model_name = ?"
            params.append(model_name)
        query += " ORDER BY a.pmid"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            ann = _json_load(r["annotation"]) or {}
            ann["pmid"] = r["pmid"]
            ann["title"] = r["title"]
            ann["abstract_text"] = r["abstract"]
            ann["_annotation_model"] = r["model_name"]
            ann["methodology"] = r["methodology"]
            ann["retraction_reasons"] = r["retraction_reasons"]
            results.append(ann)
        return results

    def export_review_csv(
        self,
        model_name: str,
        output_path: Path,
        methodology: str = "biasbuster",
    ) -> None:
        """Export annotations + human reviews as a CSV for human review.

        The CSV layout is biasbuster-specific (it surfaces the 5 biasbuster
        domains). Scope to a single methodology so the columns match; this
        defaults to biasbuster.
        """
        from biasbuster.annotators import REVIEW_CSV_COLUMNS

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
                ON a.pmid = h.pmid
                AND a.model_name = h.model_name
                AND a.methodology = h.methodology
            WHERE a.model_name = ? AND a.methodology = ?
            ORDER BY a.pmid
        """, (model_name, methodology)).fetchall()

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

    # ---- Expert methodology ratings ----

    def upsert_expert_rating(
        self,
        *,
        methodology: str,
        rating_source: str,
        study_label: str,
        domain_ratings: dict,
        overall_rating: Optional[str] = None,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        methodology_version: Optional[str] = None,
        source_review_pmid: Optional[str] = None,
        source_review_doi: Optional[str] = None,
        source_reference: Optional[str] = None,
        added_by: Optional[str] = None,
        verified: bool = False,
        notes: Optional[str] = None,
        commit: bool = True,
        preserve_curation: bool = True,
    ) -> str:
        """Insert or update an expert ground-truth rating.

        Identity is ``(methodology, rating_source, study_label)``.

        Returns ``"inserted"`` when the row did not previously exist and
        ``"updated"`` when it did. Callers (e.g. the ingest summary) use
        this to distinguish the two cases — SQLite's ``rowcount`` is 1
        for both on ``INSERT ... ON CONFLICT DO UPDATE``.

        When ``preserve_curation=True`` (default), re-ingesting from a
        machine source does not clobber the curator-managed fields
        (``verified``, ``added_by``, ``notes``) once they have been set.
        ``pmid`` / ``doi`` / provenance fields are COALESCE-guarded so a
        ``None`` in the new row cannot blank out a previously resolved
        value.

        Note: ``verified`` is **not** auto-cleared when ``domain_ratings``
        or ``overall_rating`` change on re-ingest. A curator who wants
        re-verification after a parser fix should pass
        ``preserve_curation=False`` (or reset ``verified`` manually).
        The default assumes machine re-runs over the same source should
        be idempotent — divergent results indicate a parser bug, not a
        reason to invalidate prior human sign-off on the mapping.

        Pass ``preserve_curation=False`` for an explicit full overwrite.
        """
        # SQLite's cursor.rowcount is 1 for both INSERT and UPDATE under
        # ON CONFLICT DO UPDATE, so probe up-front to distinguish the two.
        exists = self.conn.execute(
            "SELECT 1 FROM expert_methodology_ratings "
            "WHERE methodology = ? AND rating_source = ? AND study_label = ?",
            (methodology, rating_source, study_label),
        ).fetchone() is not None
        conflict_clause = (
            _EXPERT_RATING_CONFLICT_PRESERVE_CURATION
            if preserve_curation
            else _EXPERT_RATING_CONFLICT_FORCE_OVERWRITE
        )
        self.conn.execute(
            _EXPERT_RATING_INSERT_SQL + conflict_clause,
            (
                methodology,
                rating_source,
                study_label,
                pmid,
                doi,
                methodology_version,
                source_review_pmid,
                source_review_doi,
                source_reference,
                _json_col(domain_ratings),
                overall_rating,
                int(bool(verified)),
                added_by,
                notes,
            ),
        )
        if commit:
            self.conn.commit()
        return "updated" if exists else "inserted"

    def get_expert_ratings(
        self,
        methodology: Optional[str] = None,
        rating_source: Optional[str] = None,
        pmid: Optional[str] = None,
        verified_only: bool = False,
    ) -> list[dict]:
        """Return expert ratings, optionally filtered.

        ``domain_ratings`` is returned deserialised. Pass
        ``verified_only=True`` to restrict to human-curated rows.
        """
        query = "SELECT * FROM expert_methodology_ratings"
        conditions: list[str] = []
        params: list = []
        if methodology is not None:
            conditions.append("methodology = ?")
            params.append(methodology)
        if rating_source is not None:
            conditions.append("rating_source = ?")
            params.append(rating_source)
        if pmid is not None:
            conditions.append("pmid = ?")
            params.append(pmid)
        if verified_only:
            conditions.append("verified = 1")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY methodology, rating_source, study_label"
        rows = self.conn.execute(query, params).fetchall()
        results: list[dict] = []
        for r in rows:
            d = dict(r)
            d["domain_ratings"] = _json_load(d.get("domain_ratings")) or {}
            results.append(d)
        return results

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

        # Annotation counts by methodology
        rows = self.conn.execute(
            "SELECT methodology, COUNT(*) as cnt "
            "FROM annotations GROUP BY methodology"
        ).fetchall()
        stats["annotations_by_methodology"] = {
            r["methodology"]: r["cnt"] for r in rows
        }

        # Review counts
        row = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN validated = 1 THEN 1 ELSE 0 END) as validated "
            "FROM human_reviews"
        ).fetchone()
        stats["total_reviews"] = row["total"]
        stats["validated_reviews"] = row["validated"] or 0

        return stats
