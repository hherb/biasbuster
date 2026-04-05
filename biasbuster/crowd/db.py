"""SQLite database for the crowd annotation platform.

Separate from the production biasbuster.db — stores only the data
needed for crowd evaluation: users, a paper subset, AI annotations,
and crowd-submitted annotations (blind + revised phases).
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
-- Users with secure password storage
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    expertise_level TEXT NOT NULL DEFAULT 'unknown',
    created_at TEXT DEFAULT (datetime('now')),
    last_login_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    is_admin INTEGER NOT NULL DEFAULT 0,
    annotation_count INTEGER NOT NULL DEFAULT 0
);

-- Papers subset exported from production (stripped of retraction data)
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
    source TEXT NOT NULL,
    target_annotations INTEGER NOT NULL DEFAULT 3,
    current_annotations INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    imported_at TEXT DEFAULT (datetime('now'))
);

-- AI annotations imported from production (one per paper per model)
CREATE TABLE IF NOT EXISTS ai_annotations (
    pmid TEXT NOT NULL REFERENCES papers(pmid),
    model_name TEXT NOT NULL,
    annotation JSON NOT NULL,
    overall_severity TEXT,
    overall_bias_probability REAL,
    confidence TEXT,
    PRIMARY KEY (pmid, model_name)
);

-- Crowd annotations: blind + optional revised phase per user per paper
CREATE TABLE IF NOT EXISTS crowd_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    pmid TEXT NOT NULL REFERENCES papers(pmid),
    blind_annotation JSON NOT NULL,
    blind_submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    blind_time_seconds REAL,
    revised_annotation JSON,
    revised_submitted_at TEXT,
    revised_time_seconds REAL,
    revision_notes TEXT,
    blind_agrees_with_ai INTEGER,
    revised_agrees_with_ai INTEGER,
    phase TEXT NOT NULL DEFAULT 'blind',
    UNIQUE(user_id, pmid)
);

-- Rate limiting for login/registration
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    username TEXT,
    action TEXT NOT NULL DEFAULT 'login',
    success INTEGER NOT NULL DEFAULT 0,
    attempted_at TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_crowd_user ON crowd_annotations(user_id);
CREATE INDEX IF NOT EXISTS idx_crowd_pmid ON crowd_annotations(pmid);
CREATE INDEX IF NOT EXISTS idx_crowd_phase ON crowd_annotations(phase);
CREATE INDEX IF NOT EXISTS idx_login_attempts_ip
    ON login_attempts(ip_address, attempted_at);
CREATE INDEX IF NOT EXISTS idx_papers_priority
    ON papers(priority DESC, current_annotations ASC);
"""


def _json_load(val: Any) -> Any:
    """Parse a JSON column value, returning the original if not a string."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


class CrowdDatabase:
    """SQLite backend for the crowd annotation platform."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _open_connection(self) -> sqlite3.Connection:
        """Open a new database connection with WAL mode."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_connected(self) -> sqlite3.Connection:
        """Return the current connection, reconnecting if closed."""
        if self._conn is None:
            self._conn = self._open_connection()
        try:
            self._conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            self._conn = self._open_connection()
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Active database connection."""
        return self._ensure_connected()

    def initialize(self) -> None:
        """Create tables and indexes."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        logger.info("Crowd database initialized at %s", self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def commit(self) -> None:
        """Explicit commit."""
        self.conn.commit()

    def __enter__(self) -> "CrowdDatabase":
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── User management ──────────────────────────────────────────────

    def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        display_name: str = "",
        expertise_level: str = "unknown",
    ) -> int:
        """Create a new user. Returns the user ID.

        Raises:
            sqlite3.IntegrityError: If username or email already exists.
        """
        cur = self.conn.execute(
            """INSERT INTO users (username, email, password_hash,
                                  display_name, expertise_level)
               VALUES (?, ?, ?, ?, ?)""",
            (username, email, password_hash, display_name, expertise_level),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Fetch a user by username."""
        row = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Fetch a user by ID."""
        row = self.conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_last_login(self, user_id: int) -> None:
        """Update last login timestamp."""
        self.conn.execute(
            "UPDATE users SET last_login_at = datetime('now') WHERE id = ?",
            (user_id,),
        )
        self.conn.commit()

    def update_password_hash(self, user_id: int, new_hash: str) -> None:
        """Update password hash (for rehashing when parameters change)."""
        self.conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        self.conn.commit()

    # ── Rate limiting ────────────────────────────────────────────────

    def record_attempt(
        self,
        ip_address: str,
        action: str = "login",
        username: str = "",
        success: bool = False,
    ) -> None:
        """Record a login or registration attempt."""
        self.conn.execute(
            """INSERT INTO login_attempts (ip_address, username, action, success)
               VALUES (?, ?, ?, ?)""",
            (ip_address, username, action, int(success)),
        )
        self.conn.commit()

    def count_recent_attempts(
        self,
        ip_address: str,
        action: str = "login",
        minutes: int = 15,
        success_only: bool = False,
    ) -> int:
        """Count recent attempts from an IP for rate limiting."""
        query = """
            SELECT COUNT(*) FROM login_attempts
            WHERE ip_address = ?
              AND action = ?
              AND attempted_at > datetime('now', ?)
        """
        params: list[Any] = [ip_address, action, f"-{minutes} minutes"]
        if not success_only:
            query += " AND success = 0"
        row = self.conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def count_user_annotations_recent(
        self, user_id: int, minutes: int = 60
    ) -> int:
        """Count annotations completed by a user in the last N minutes."""
        row = self.conn.execute(
            """SELECT COUNT(*) FROM crowd_annotations
               WHERE user_id = ?
                 AND phase = 'completed'
                 AND revised_submitted_at > datetime('now', ?)""",
            (user_id, f"-{minutes} minutes"),
        ).fetchone()
        return row[0] if row else 0

    def cleanup_old_login_attempts(self, days: int = 7) -> int:
        """Delete login attempts older than N days. Returns rows deleted."""
        cursor = self.conn.execute(
            "DELETE FROM login_attempts WHERE attempted_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Paper management ─────────────────────────────────────────────

    def insert_paper(self, paper: dict) -> bool:
        """Insert a paper. Returns True if new, False if already exists."""
        try:
            self.conn.execute(
                """INSERT INTO papers (pmid, doi, title, abstract, journal,
                                       year, authors, grants, mesh_terms,
                                       source, target_annotations, priority)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper["pmid"],
                    paper.get("doi"),
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("journal"),
                    paper.get("year"),
                    json.dumps(paper.get("authors")) if paper.get("authors") else None,
                    json.dumps(paper.get("grants")) if paper.get("grants") else None,
                    json.dumps(paper.get("mesh_terms")) if paper.get("mesh_terms") else None,
                    paper.get("source", "pubmed_rct"),
                    paper.get("target_annotations", 3),
                    paper.get("priority", 0),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_paper(self, pmid: str) -> Optional[dict]:
        """Fetch a paper by PMID, parsing JSON columns."""
        row = self.conn.execute(
            "SELECT * FROM papers WHERE pmid = ?", (pmid,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for col in ("authors", "grants", "mesh_terms"):
            d[col] = _json_load(d.get(col))
        return d

    def get_paper_count(self) -> int:
        """Count total papers in the crowd DB."""
        row = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return row[0] if row else 0

    # ── AI annotations ───────────────────────────────────────────────

    def insert_ai_annotation(
        self,
        pmid: str,
        model_name: str,
        annotation: dict,
    ) -> bool:
        """Insert an AI annotation. Returns True if new."""
        try:
            self.conn.execute(
                """INSERT INTO ai_annotations
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
            return True
        except sqlite3.IntegrityError:
            return False

    def get_ai_annotation(
        self, pmid: str, model_name: Optional[str] = None
    ) -> Optional[dict]:
        """Fetch the AI annotation for a paper.

        If model_name is None, returns the first available.
        """
        if model_name:
            row = self.conn.execute(
                "SELECT * FROM ai_annotations WHERE pmid = ? AND model_name = ?",
                (pmid, model_name),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM ai_annotations WHERE pmid = ? LIMIT 1",
                (pmid,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["annotation"] = _json_load(d.get("annotation"))
        return d

    def get_ai_model_names(self) -> list[str]:
        """Get distinct AI model names in the crowd DB."""
        rows = self.conn.execute(
            "SELECT DISTINCT model_name FROM ai_annotations ORDER BY model_name"
        ).fetchall()
        return [r[0] for r in rows]

    # ── Paper assignment ─────────────────────────────────────────────

    def get_next_paper(self, user_id: int) -> Optional[str]:
        """Get the next unassigned paper PMID for a user.

        Prioritizes under-annotated papers and high-priority papers.
        Returns None if no papers are available.
        """
        row = self.conn.execute(
            """SELECT p.pmid
               FROM papers p
               WHERE p.current_annotations < p.target_annotations
                 AND p.pmid NOT IN (
                     SELECT ca.pmid FROM crowd_annotations ca
                     WHERE ca.user_id = ?
                 )
               ORDER BY p.priority DESC,
                        p.current_annotations ASC,
                        RANDOM()
               LIMIT 1""",
            (user_id,),
        ).fetchone()
        return row[0] if row else None

    def get_in_progress_annotations(self, user_id: int) -> list[dict]:
        """Get annotations where the user started but hasn't completed revision."""
        rows = self.conn.execute(
            """SELECT ca.pmid, ca.phase, ca.blind_submitted_at,
                      p.title
               FROM crowd_annotations ca
               JOIN papers p ON p.pmid = ca.pmid
               WHERE ca.user_id = ? AND ca.phase = 'revealed'
               ORDER BY ca.blind_submitted_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Crowd annotations ────────────────────────────────────────────

    def has_annotation(self, user_id: int, pmid: str) -> bool:
        """Check if a user has already annotated this paper."""
        row = self.conn.execute(
            "SELECT 1 FROM crowd_annotations WHERE user_id = ? AND pmid = ?",
            (user_id, pmid),
        ).fetchone()
        return row is not None

    def get_annotation_phase(self, user_id: int, pmid: str) -> Optional[str]:
        """Get the current phase for a user's annotation of a paper."""
        row = self.conn.execute(
            "SELECT phase FROM crowd_annotations WHERE user_id = ? AND pmid = ?",
            (user_id, pmid),
        ).fetchone()
        return row[0] if row else None

    def save_blind_annotation(
        self,
        user_id: int,
        pmid: str,
        annotation: dict,
        time_seconds: float,
        ai_annotation: Optional[dict] = None,
    ) -> None:
        """Save a blind (Phase 1) annotation and transition to 'revealed'.

        Args:
            user_id: The annotator's user ID.
            pmid: Paper PMID.
            annotation: The blind annotation JSON.
            time_seconds: Time spent on the annotation.
            ai_annotation: If provided, computes blind_agrees_with_ai.
        """
        agrees = None
        if ai_annotation is not None:
            agrees = int(
                annotation.get("overall_severity")
                == ai_annotation.get("overall_severity")
            )

        self.conn.execute(
            """INSERT INTO crowd_annotations
                   (user_id, pmid, blind_annotation, blind_time_seconds,
                    blind_agrees_with_ai, phase)
               VALUES (?, ?, ?, ?, ?, 'revealed')""",
            (
                user_id,
                pmid,
                json.dumps(annotation),
                time_seconds,
                agrees,
            ),
        )
        self.conn.commit()

    def save_revised_annotation(
        self,
        user_id: int,
        pmid: str,
        annotation: dict,
        time_seconds: float,
        revision_notes: str = "",
        ai_annotation: Optional[dict] = None,
    ) -> None:
        """Save a revised (Phase 2) annotation and mark as 'completed'.

        Also increments the paper's current_annotations count and
        the user's annotation_count.
        """
        agrees = None
        if ai_annotation is not None:
            agrees = int(
                annotation.get("overall_severity")
                == ai_annotation.get("overall_severity")
            )

        # Use explicit transaction for multi-table atomicity
        with self.conn:
            self.conn.execute(
                """UPDATE crowd_annotations
                   SET revised_annotation = ?,
                       revised_submitted_at = datetime('now'),
                       revised_time_seconds = ?,
                       revision_notes = ?,
                       revised_agrees_with_ai = ?,
                       phase = 'completed'
                   WHERE user_id = ? AND pmid = ?""",
                (
                    json.dumps(annotation),
                    time_seconds,
                    revision_notes,
                    agrees,
                    user_id,
                    pmid,
                ),
            )
            self.conn.execute(
                """UPDATE papers
                   SET current_annotations = current_annotations + 1
                   WHERE pmid = ?""",
                (pmid,),
            )
            self.conn.execute(
                """UPDATE users
                   SET annotation_count = annotation_count + 1
                   WHERE id = ?""",
                (user_id,),
            )

    def get_crowd_annotation(self, user_id: int, pmid: str) -> Optional[dict]:
        """Fetch a specific crowd annotation."""
        row = self.conn.execute(
            "SELECT * FROM crowd_annotations WHERE user_id = ? AND pmid = ?",
            (user_id, pmid),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["blind_annotation"] = _json_load(d.get("blind_annotation"))
        d["revised_annotation"] = _json_load(d.get("revised_annotation"))
        return d

    def get_all_completed_annotations(self, pmid: Optional[str] = None) -> list[dict]:
        """Fetch all completed crowd annotations, optionally for a specific paper."""
        query = "SELECT * FROM crowd_annotations WHERE phase = 'completed'"
        params: list[Any] = []
        if pmid:
            query += " AND pmid = ?"
            params.append(pmid)
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["blind_annotation"] = _json_load(d.get("blind_annotation"))
            d["revised_annotation"] = _json_load(d.get("revised_annotation"))
            results.append(d)
        return results

    # ── Statistics ───────────────────────────────────────────────────

    def get_user_progress(self, user_id: int) -> dict:
        """Get annotation progress for a user."""
        total_papers = self.conn.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]

        completed = self.conn.execute(
            """SELECT COUNT(*) FROM crowd_annotations
               WHERE user_id = ? AND phase = 'completed'""",
            (user_id,),
        ).fetchone()[0]

        in_progress = self.conn.execute(
            """SELECT COUNT(*) FROM crowd_annotations
               WHERE user_id = ? AND phase = 'revealed'""",
            (user_id,),
        ).fetchone()[0]

        available = self.conn.execute(
            """SELECT COUNT(*) FROM papers p
               WHERE p.current_annotations < p.target_annotations
                 AND p.pmid NOT IN (
                     SELECT ca.pmid FROM crowd_annotations ca
                     WHERE ca.user_id = ?
                 )""",
            (user_id,),
        ).fetchone()[0]

        return {
            "total_papers": total_papers,
            "completed": completed,
            "in_progress": in_progress,
            "available": available,
        }

    def get_global_stats(self) -> dict:
        """Get overall platform statistics."""
        total_papers = self.conn.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]
        total_users = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE is_active = 1"
        ).fetchone()[0]
        total_completed = self.conn.execute(
            "SELECT COUNT(*) FROM crowd_annotations WHERE phase = 'completed'"
        ).fetchone()[0]
        fully_annotated = self.conn.execute(
            """SELECT COUNT(*) FROM papers
               WHERE current_annotations >= target_annotations"""
        ).fetchone()[0]

        return {
            "total_papers": total_papers,
            "total_users": total_users,
            "total_completed": total_completed,
            "fully_annotated": fully_annotated,
        }
