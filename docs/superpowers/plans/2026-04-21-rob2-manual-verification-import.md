# RoB 2 Manual Verification Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import a 10-trial curated RoB 2 verification set into the biasbuster SQLite database, tag the rows under a dedicated `manually_verified` table, and cache each trial's Europe PMC JATS full text on disk for later methodology runs.

**Architecture:** Source of truth is a git-tracked CSV at `dataset/manual_verification_sets/rob2_manual_verify_20260421.csv`. A one-shot async script reads the CSV, upserts each row into `papers` (via the existing `upsert_cochrane_paper` path), fetches PubMed metadata and Europe PMC full text, writes the XML to `dataset/rob2_verification_fulltexts/<PMID>.xml`, and records a tag row in a new `manually_verified` FK table keyed on `(pmid, verification_set)`. Idempotent: re-runs refresh without duplicating.

**Tech Stack:** Python 3.12, SQLite (via `biasbuster.database.Database`), `httpx` async HTTP (via existing `RetractionWatchCollector` and the Europe PMC fetch pattern from `scripts/fetch_cochrane_jats.py`), `pytest`, `uv` for env/dep management.

**Spec:** [docs/superpowers/specs/2026-04-21-rob2-manual-verification-import-design.md](../specs/2026-04-21-rob2-manual-verification-import-design.md)

---

## File Structure

| File | Purpose |
|---|---|
| `dataset/manual_verification_sets/rob2_manual_verify_20260421.csv` (create) | Git-tracked seed data — 10 curated trials with RoB 2 ratings. Source of truth. |
| `biasbuster/database.py` (modify) | Add `manually_verified` CREATE TABLE + index to `SCHEMA_SQL`; add `Database.upsert_manually_verified()` method. |
| `scripts/import_rob2_verification_set.py` (create) | Async import script. Reads CSV, upserts papers, fetches abstracts + full text, tags rows. |
| `tests/test_manually_verified.py` (create) | Unit tests for the new table and DB helper. |
| `.gitignore` (modify) | Exclude `dataset/rob2_verification_fulltexts/`. |

---

### Task 1: Commit the source CSV

**Files:**
- Create: `dataset/manual_verification_sets/rob2_manual_verify_20260421.csv`

This lives in git so the set can be rebuilt from source if the DB is ever lost again (as the user explicitly requested). No Python hardcoding.

- [ ] **Step 1: Create the CSV**

Write the file at `dataset/manual_verification_sets/rob2_manual_verify_20260421.csv` with exactly this content (column order and values as provided by the user):

```csv
study_id,trial_name,pmid,pmc_url,source_review,randomization,deviations,missing_data,measurement,reporting,overall
1,RECOVERY dexamethasone,32678530,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7383595/,AHRQ/Cochrane-style,low,some_concerns,low,low,low,low
2,ACTT-1 remdesivir,32445440,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7262788/,Cochrane COVID review,low,low,low,low,low,low
3,SOLIDARITY trial,33264556,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729089/,WHO meta-analysis,low,some_concerns,low,low,low,low
4,SPARCL atorvastatin,16899775,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1852416/,Cochrane stroke review,low,low,low,low,low,low
5,RE-LY dabigatran,19717844,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717924/,Cochrane anticoagulation,low,high,low,low,low,some_concerns
6,ALLHAT antihypertensive,12479763,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC149558/,Cochrane HTN review,low,low,low,low,low,low
7,UKPDS metformin,9742977,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC28637/,Cochrane diabetes review,low,high,low,low,low,some_concerns
8,MAGELLAN rivaroxaban,21970978,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3164000/,Cochrane VTE review,low,some_concerns,low,low,low,low
9,PROWESS sepsis APC,12495323,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1236530/,Cochrane sepsis review,low,low,some_concerns,low,low,low
10,CRASH-2 tranexamic acid,20554319,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2898595/,Cochrane trauma review,low,low,low,low,low,low
```

- [ ] **Step 2: Verify file**

Run: `wc -l dataset/manual_verification_sets/rob2_manual_verify_20260421.csv`
Expected: `11 dataset/manual_verification_sets/rob2_manual_verify_20260421.csv` (header + 10 rows)

- [ ] **Step 3: Commit**

```bash
git add dataset/manual_verification_sets/rob2_manual_verify_20260421.csv
git commit -m "data(rob2): curated 10-trial manual verification set

Plaintext seed for the rob2_manual_verify_20260421 verification_set tag.
Source of truth in git so the DB can be rebuilt by re-running the import
script if the SQLite database is ever lost or corrupted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `manually_verified` table to the schema

**Files:**
- Modify: `biasbuster/database.py` (`SCHEMA_SQL` constant, lines 151-186 region)
- Test: `tests/test_manually_verified.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_manually_verified.py`:

```python
"""Tests for the manually_verified FK table and helper."""

import sqlite3
from pathlib import Path

import pytest

from biasbuster.database import Database


@pytest.fixture
def db_with_paper(tmp_path: Path) -> Database:
    """Fresh DB with one paper inserted."""
    db = Database(tmp_path / "t.db")
    db.initialize()
    db.insert_paper(
        {"pmid": "P1", "title": "T", "abstract": "A", "source": "cochrane_rob"}
    )
    return db


def test_manually_verified_table_exists(db_with_paper: Database) -> None:
    row = db_with_paper.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='manually_verified'"
    ).fetchone()
    assert row is not None


def test_manually_verified_composite_pk(db_with_paper: Database) -> None:
    cols = db_with_paper.conn.execute(
        "PRAGMA table_info(manually_verified)"
    ).fetchall()
    pk_cols = sorted(c[1] for c in cols if c[5] > 0)  # pk column index is 5
    assert pk_cols == ["pmid", "verification_set"]


def test_manually_verified_foreign_key(db_with_paper: Database) -> None:
    # Inserting with an unknown pmid should raise (foreign_keys=ON).
    with pytest.raises(sqlite3.IntegrityError):
        db_with_paper.conn.execute(
            "INSERT INTO manually_verified (pmid, verification_set) "
            "VALUES (?, ?)",
            ("UNKNOWN_PMID", "rob2_manual_verify_20260421"),
        )
        db_with_paper.conn.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manually_verified.py -v`
Expected: 3 tests FAIL — `manually_verified` table does not yet exist.

- [ ] **Step 3: Add the table to `SCHEMA_SQL`**

In `biasbuster/database.py`, after the `eval_outputs` CREATE TABLE block (around line 165) and before the `-- Indexes for common queries` comment (around line 167), insert:

```sql
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
```

Then, alongside the other `CREATE INDEX` lines at the bottom of `SCHEMA_SQL`, add:

```sql
CREATE INDEX IF NOT EXISTS idx_manually_verified_set
    ON manually_verified(verification_set);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manually_verified.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add biasbuster/database.py tests/test_manually_verified.py
git commit -m "feat(database): add manually_verified FK tag table

Composite-PK table on (pmid, verification_set) for tagging papers as
members of a human-curated verification set. Foreign key into papers
ensures the tag always points at a real paper row.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `Database.upsert_manually_verified()` helper

**Files:**
- Modify: `biasbuster/database.py` (add new method alongside other `upsert_*` methods)
- Modify: `tests/test_manually_verified.py` (extend)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_manually_verified.py`:

```python
class TestUpsertManuallyVerified:
    def test_insert_then_update(self, db_with_paper: Database) -> None:
        db_with_paper.upsert_manually_verified(
            pmid="P1",
            verification_set="rob2_manual_verify_20260421",
            trial_name="Test trial",
            source_review="Test review",
            fulltext_path="dataset/rob2_verification_fulltexts/P1.xml",
            fulltext_ok=True,
        )
        row = db_with_paper.conn.execute(
            "SELECT pmid, verification_set, trial_name, source_review, "
            "fulltext_path, fulltext_ok FROM manually_verified"
        ).fetchone()
        assert row["trial_name"] == "Test trial"
        assert row["fulltext_ok"] == 1

        # Re-run updates, does not duplicate.
        db_with_paper.upsert_manually_verified(
            pmid="P1",
            verification_set="rob2_manual_verify_20260421",
            trial_name="Updated trial name",
            source_review="Test review",
            fulltext_path="dataset/rob2_verification_fulltexts/P1.xml",
            fulltext_ok=False,
            notes="fulltext went away",
        )
        rows = db_with_paper.conn.execute(
            "SELECT trial_name, fulltext_ok, notes FROM manually_verified"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["trial_name"] == "Updated trial name"
        assert rows[0]["fulltext_ok"] == 0
        assert rows[0]["notes"] == "fulltext went away"

    def test_distinct_verification_sets_coexist(
        self, db_with_paper: Database
    ) -> None:
        db_with_paper.upsert_manually_verified(
            pmid="P1", verification_set="set_a"
        )
        db_with_paper.upsert_manually_verified(
            pmid="P1", verification_set="set_b"
        )
        count = db_with_paper.conn.execute(
            "SELECT COUNT(*) AS n FROM manually_verified WHERE pmid='P1'"
        ).fetchone()["n"]
        assert count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manually_verified.py::TestUpsertManuallyVerified -v`
Expected: FAIL with `AttributeError: 'Database' object has no attribute 'upsert_manually_verified'`.

- [ ] **Step 3: Add the method**

In `biasbuster/database.py`, add this method inside the `Database` class (place it near the other `upsert_*` methods, e.g. right after `upsert_cochrane_paper_v2`):

```python
    def upsert_manually_verified(
        self,
        pmid: str,
        verification_set: str,
        *,
        trial_name: Optional[str] = None,
        source_review: Optional[str] = None,
        fulltext_path: Optional[str] = None,
        fulltext_ok: bool = False,
        notes: Optional[str] = None,
    ) -> bool:
        """Insert or update a manually_verified tag row.

        Tags a paper as a member of a human-curated verification set.
        On conflict on (pmid, verification_set), every optional field is
        overwritten by the new values. Returns True if the row was
        inserted or updated.

        Args:
            pmid: PMID of the paper. Must already exist in ``papers``.
            verification_set: Tag identifying the curated set (e.g.
                ``'rob2_manual_verify_20260421'``).
            trial_name: Human-readable trial name from the source CSV.
            source_review: Name of the review that produced the ratings.
            fulltext_path: Repo-root-relative path to the cached JATS XML,
                or ``None`` if no full text was fetched.
            fulltext_ok: ``True`` iff the full text was fetched and is
                at least ``MIN_JATS_BYTES`` bytes on disk.
            notes: Free-form notes (e.g. error messages from the fetch).
        """
        self._ensure_connected()
        cursor = self.conn.execute(
            """INSERT INTO manually_verified
                   (pmid, verification_set, trial_name, source_review,
                    fulltext_path, fulltext_ok, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pmid, verification_set) DO UPDATE SET
                   trial_name = excluded.trial_name,
                   source_review = excluded.source_review,
                   fulltext_path = excluded.fulltext_path,
                   fulltext_ok = excluded.fulltext_ok,
                   notes = excluded.notes""",
            (
                pmid,
                verification_set,
                trial_name,
                source_review,
                fulltext_path,
                1 if fulltext_ok else 0,
                notes,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manually_verified.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add biasbuster/database.py tests/test_manually_verified.py
git commit -m "feat(database): upsert_manually_verified helper

Minimal insert-or-update method for the manually_verified tag table,
with composite-PK conflict handling so re-imports of a verification
set refresh the row rather than duplicating it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `.gitignore` — exclude the full-text cache directory

**Files:**
- Modify: `.gitignore` (append one line)

The cached JATS XML blobs are ~1-5 MB each and are reproducible from the committed CSV's PMCIDs. They belong on disk only.

- [ ] **Step 1: Append the ignore line**

Find the `dataset/*.db-wal` line in `.gitignore` (around line 34) and after the `dataset/cleanseed/annotated/` line (around line 38), add:

```
dataset/rob2_verification_fulltexts/
dataset/manual_verification_sets/*.jats.xml
```

(The second line is a belt-and-suspenders guard in case anyone ever places per-set fulltext caches alongside the CSV — the CSV itself stays tracked.)

- [ ] **Step 2: Verify the CSV is still tracked**

Run: `git check-ignore -v dataset/manual_verification_sets/rob2_manual_verify_20260421.csv`
Expected: no output (file is NOT ignored). Exit code 1 is correct — `check-ignore` returns 1 when the file would be tracked.

Run: `git check-ignore -v dataset/rob2_verification_fulltexts/example.xml`
Expected: exit code 0 and output naming the ignore pattern.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): exclude rob2 fulltext cache directory

Reproducible from committed CSV PMCIDs; each XML is ~1-5 MB.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Write the import script

**Files:**
- Create: `scripts/import_rob2_verification_set.py`

Single-purpose async CLI. Takes `--csv` and `--verification-set` so the same script can import future curated sets. Reuses `RetractionWatchCollector.fetch_pubmed_abstracts_batch()` for PubMed and the fetch pattern from `scripts/fetch_cochrane_jats.py` for Europe PMC.

- [ ] **Step 1: Create the script**

Create `scripts/import_rob2_verification_set.py` with this content:

```python
"""Import a curated RoB 2 manual verification CSV into the database.

Reads a plaintext CSV seed at ``dataset/manual_verification_sets/*.csv``
and:

  1. Upserts each row into ``papers`` (source='cochrane_rob', with the
     Cochrane-authoritative RoB 2 ratings populated from the CSV).
  2. Fetches PubMed metadata (title/abstract/journal/year/authors) for
     any paper whose abstract is still empty/short.
  3. Fetches Europe PMC ``fullTextXML`` for each row's PMCID and caches
     it at ``dataset/rob2_verification_fulltexts/<PMID>.xml``.
  4. Tags each paper in the ``manually_verified`` table under the
     requested ``verification_set``.

The import is fully idempotent: re-runs refresh metadata and full text
without duplicating rows.

Usage:
    uv run python -m scripts.import_rob2_verification_set \\
        [--csv PATH] \\
        [--verification-set TAG]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from biasbuster.collectors.cochrane_rob import (
    RoBAssessment,
    rob_assessment_to_paper_dict,
)
from biasbuster.collectors.retraction_watch import RetractionWatchCollector
from biasbuster.database import Database
from biasbuster.utils.retry import fetch_with_retry
from config import Config

logger = logging.getLogger("import_rob2_verification_set")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = (
    REPO_ROOT
    / "dataset"
    / "manual_verification_sets"
    / "rob2_manual_verify_20260421.csv"
)
DEFAULT_VERIFICATION_SET = "rob2_manual_verify_20260421"
FULLTEXT_DIR = REPO_ROOT / "dataset" / "rob2_verification_fulltexts"

EUROPMC_FULLTEXT = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)
# Matches the guard in scripts/fetch_cochrane_jats.py: responses smaller
# than this are almost always 404 stub pages, not real JATS.
MIN_JATS_BYTES = 1024
HTTP_TIMEOUT_SECONDS = 60.0
REQUEST_DELAY_SECONDS = 0.5  # ~2 req/s to Europe PMC, polite default
USER_AGENT = "biasbuster/rob2-manual-verify-import"

PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
ABSTRACT_MIN_CHARS = 100  # refetch if shorter than this


@dataclass
class RowResult:
    pmid: str
    trial_name: str
    overall_rob: str
    abstract_chars: int
    pmcid: str
    fulltext_bytes: int
    fulltext_ok: bool
    status: str  # 'ok' | 'no_abstract' | 'fulltext_missing' | 'fulltext_error'
    notes: str = ""


def parse_csv_rows(csv_path: Path) -> list[dict]:
    """Load and lightly validate the verification-set CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    required = {
        "study_id", "trial_name", "pmid", "pmc_url", "source_review",
        "randomization", "deviations", "missing_data",
        "measurement", "reporting", "overall",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"CSV {csv_path} has no data rows")
    return rows


def row_to_paper_dict(row: dict) -> dict:
    """Convert a CSV row to the ``papers`` insert dict via RoBAssessment."""
    a = RoBAssessment(
        study_id=row["trial_name"],
        pmid=row["pmid"],
        study_title=row["trial_name"],
        randomization_bias=row["randomization"],
        deviation_bias=row["deviations"],
        missing_outcome_bias=row["missing_data"],
        measurement_bias=row["measurement"],
        reporting_bias=row["reporting"],
        overall_rob=row["overall"],
        cochrane_review_title=row["source_review"],
    )
    return rob_assessment_to_paper_dict(a)


def extract_pmcid(pmc_url: str) -> str:
    """Extract a bare PMCID (e.g. 'PMC7383595') from a URL."""
    m = PMCID_RE.search(pmc_url)
    return m.group(1).upper() if m else ""


def fulltext_path_for(pmid: str) -> Path:
    return FULLTEXT_DIR / f"{pmid}.xml"


def fulltext_already_cached(pmid: str) -> bool:
    p = fulltext_path_for(pmid)
    return p.exists() and p.stat().st_size >= MIN_JATS_BYTES


async def fetch_fulltext(
    client: httpx.AsyncClient, pmid: str, pmcid: str
) -> tuple[int, str]:
    """Fetch Europe PMC JATS for one PMCID.

    Returns (bytes_written, status). status values:
      'ok' | 'already_cached' | 'not_found' | 'too_small' | 'error' | 'http_<N>'
    """
    if fulltext_already_cached(pmid):
        return fulltext_path_for(pmid).stat().st_size, "already_cached"
    if not pmcid:
        return 0, "no_pmcid"

    url = EUROPMC_FULLTEXT.format(pmcid=pmcid)
    try:
        resp = await fetch_with_retry(
            client, "GET", url, max_retries=3, base_delay=2.0
        )
    except Exception as exc:  # noqa: BLE001 — surface raw error
        logger.warning("PMID %s (%s): fetch error: %s", pmid, pmcid, exc)
        return 0, "error"

    if resp.status_code == 404:
        return 0, "not_found"
    if resp.status_code != 200:
        logger.warning(
            "PMID %s (%s): HTTP %d", pmid, pmcid, resp.status_code
        )
        return 0, f"http_{resp.status_code}"

    body = resp.content
    if len(body) < MIN_JATS_BYTES:
        logger.warning(
            "PMID %s (%s): response too small (%d bytes)",
            pmid, pmcid, len(body),
        )
        return len(body), "too_small"

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    fulltext_path_for(pmid).write_bytes(body)
    return len(body), "ok"


async def fetch_all_pubmed_metadata(
    config: Config, pmids: list[str]
) -> dict[str, dict]:
    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        return await collector.fetch_pubmed_abstracts_batch(pmids)


def update_paper_from_pubmed(db: Database, pmid: str, pubmed: dict) -> None:
    """Patch title/abstract/journal/year/authors into an existing paper row."""
    existing = db.conn.execute(
        "SELECT abstract, title FROM papers WHERE pmid = ?", (pmid,)
    ).fetchone()
    if existing is None:
        return

    abstract = pubmed.get("abstract") or ""
    title = pubmed.get("title") or ""
    journal = pubmed.get("journal")
    year = pubmed.get("year")
    authors = pubmed.get("authors")

    updates: list[tuple[str, object]] = []
    if abstract and len(abstract) >= ABSTRACT_MIN_CHARS and (
        not existing["abstract"]
        or len(existing["abstract"]) < ABSTRACT_MIN_CHARS
    ):
        updates.append(("abstract", abstract))
    if title and (
        not existing["title"] or len(existing["title"]) < 10
    ):
        updates.append(("title", title))
    if journal:
        updates.append(("journal", journal))
    if year:
        updates.append(("year", year))
    if authors:
        updates.append(("authors", json.dumps(authors)))

    if not updates:
        return
    set_clause = ", ".join(f"{col} = ?" for col, _ in updates)
    params = [val for _, val in updates] + [pmid]
    db.conn.execute(
        f"UPDATE papers SET {set_clause} WHERE pmid = ?", params
    )
    db.conn.commit()


def abstract_chars(db: Database, pmid: str) -> int:
    row = db.conn.execute(
        "SELECT length(abstract) AS n FROM papers WHERE pmid = ?", (pmid,)
    ).fetchone()
    return int(row["n"] or 0) if row else 0


def render_summary(results: list[RowResult]) -> str:
    lines = [
        "| PMID | trial | overall_rob | abs_chars | pmcid | fulltext_bytes | status |",
        "|------|-------|-------------|-----------|-------|----------------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r.pmid} | {r.trial_name} | {r.overall_rob} | "
            f"{r.abstract_chars} | {r.pmcid or '-'} | "
            f"{r.fulltext_bytes} | {r.status} |"
        )
    return "\n".join(lines)


async def run(csv_path: Path, verification_set: str, config: Config) -> int:
    rows = parse_csv_rows(csv_path)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    db = Database(config.db_path)
    db.initialize()

    # Stage 1: upsert papers with RoB 2 ratings (Cochrane-authoritative).
    for row in rows:
        paper = row_to_paper_dict(row)
        db.upsert_cochrane_paper(paper)

    # Stage 2: fetch PubMed metadata in one batch.
    pmids = [row["pmid"] for row in rows]
    logger.info("Fetching PubMed metadata for %d PMIDs", len(pmids))
    pubmed_results = await fetch_all_pubmed_metadata(config, pmids)

    for pmid, data in pubmed_results.items():
        update_paper_from_pubmed(db, pmid, data)

    # Stage 3: fetch Europe PMC JATS full text per row, with polite spacing.
    results: list[RowResult] = []
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for row in rows:
            pmid = row["pmid"]
            pmcid = extract_pmcid(row["pmc_url"])
            bytes_written, ft_status = await fetch_fulltext(
                client, pmid, pmcid
            )
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            ft_ok = ft_status in {"ok", "already_cached"} and bytes_written >= MIN_JATS_BYTES
            ft_path = (
                str(fulltext_path_for(pmid).relative_to(REPO_ROOT))
                if ft_ok
                else None
            )
            abs_chars = abstract_chars(db, pmid)

            status: str
            if abs_chars < ABSTRACT_MIN_CHARS:
                status = "no_abstract"
            elif not ft_ok:
                status = ft_status or "fulltext_error"
            else:
                status = "ok"

            notes: Optional[str] = None
            if status != "ok":
                notes = f"fulltext_status={ft_status}; abs_chars={abs_chars}"

            db.upsert_manually_verified(
                pmid=pmid,
                verification_set=verification_set,
                trial_name=row["trial_name"],
                source_review=row["source_review"],
                fulltext_path=ft_path,
                fulltext_ok=ft_ok,
                notes=notes,
            )

            results.append(
                RowResult(
                    pmid=pmid,
                    trial_name=row["trial_name"],
                    overall_rob=row["overall"],
                    abstract_chars=abs_chars,
                    pmcid=pmcid,
                    fulltext_bytes=bytes_written,
                    fulltext_ok=ft_ok,
                    status=status,
                    notes=notes or "",
                )
            )

    print(render_summary(results))
    failures = [r for r in results if r.status != "ok"]
    if failures:
        logger.warning(
            "%d/%d rows have non-'ok' status; see summary table above",
            len(failures), len(results),
        )
    return 0 if not failures else 1


def _configure_logging() -> None:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir
        / f"import_rob2_verification_set_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging to %s", log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Path to the verification-set CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--verification-set", default=DEFAULT_VERIFICATION_SET,
        help=(
            "Tag value written to manually_verified.verification_set "
            f"(default: {DEFAULT_VERIFICATION_SET})"
        ),
    )
    args = parser.parse_args()

    _configure_logging()

    config = Config()
    return asyncio.run(run(args.csv, args.verification_set, config))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax check**

Run: `uv run python -c "import ast; ast.parse(open('scripts/import_rob2_verification_set.py').read())"`
Expected: no output, exit code 0.

- [ ] **Step 3: Argparse smoke test (no network)**

Run: `uv run python -m scripts.import_rob2_verification_set --help`
Expected: usage message mentioning `--csv` and `--verification-set`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/import_rob2_verification_set.py
git commit -m "feat(scripts): import RoB 2 manual verification CSV

One-shot async importer: upserts papers with Cochrane-authoritative RoB 2
ratings, refreshes PubMed metadata, caches Europe PMC JATS full text at
dataset/rob2_verification_fulltexts/<PMID>.xml, and tags each paper in
the manually_verified table under a configurable verification_set.

Idempotent — safe to re-run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Live run — hand the command to the user

**Files:**
- Read: `dataset/biasbuster.db` (via the import script)
- Write: `dataset/biasbuster.db` (import results)
- Write: `dataset/rob2_verification_fulltexts/*.xml`
- Write: `logs/import_rob2_verification_set_*.log`

Per `CLAUDE.md`, any process that may run >2 minutes and makes live network calls is handed to the user to execute in their own terminal — not run in Claude's shell.

- [ ] **Step 1: Print the run command for the user**

Print this exact block to the conversation:

```
Ready to run the import. Please execute in your terminal:

    uv run python -m scripts.import_rob2_verification_set

Expected runtime: ~60-120 s (10 PubMed fetches + 10 Europe PMC fullText fetches with polite spacing).

When it finishes, paste:
  - The Markdown summary table printed to stdout
  - Any WARNING-level log lines
so we can verify the import.
```

- [ ] **Step 2: After the user returns, verify the import**

When the user reports results back, run these read-only verification queries:

```bash
uv run python -c "
from biasbuster.database import Database
from config import Config
db = Database(Config().db_path)
rows = db.conn.execute(
    'SELECT pmid, verification_set, fulltext_ok, length(fulltext_path) AS p '
    'FROM manually_verified WHERE verification_set=? ORDER BY pmid',
    ('rob2_manual_verify_20260421',)
).fetchall()
for r in rows:
    print(tuple(r))
print('count:', len(rows))
"
```

Expected: 10 rows, `verification_set='rob2_manual_verify_20260421'` for each; most/all with `fulltext_ok=1` and non-null `fulltext_path`.

Also:

```bash
ls -lh dataset/rob2_verification_fulltexts/ 2>/dev/null | head -15
```

Expected: up to 10 XML files, each ≥ 1 KB (most will be a few hundred KB to a few MB).

- [ ] **Step 3: Record findings**

If any row has `fulltext_ok=0`, note its PMID and status in the PR description / follow-up. Do not attempt to force-fetch PDFs or other formats in this pass — that's a separate decision per the spec's non-goals. Missing full text is recorded in the `notes` column for later review.

No commit is needed for this task (runtime artifacts only; DB is in `.gitignore`).

---

## Self-Review

**Spec coverage:**

- §Input data — CSV at `dataset/manual_verification_sets/rob2_manual_verify_20260421.csv` → Task 1.
- §Changes 1 (table) → Task 2.
- §Changes 2 (DB helper) → Task 3.
- §Changes 3 (fulltext directory + .gitignore) → Task 4 (+ Task 5 for the mkdir at runtime).
- §Changes 4 (import script) → Task 5.
- §Changes 5 (tests) → Tasks 2 and 3 (pytest at `tests/test_manually_verified.py`).
- §Risks — handled in script: fulltext 404 → `fulltext_ok=0`; missing abstract → status `no_abstract`; existing paper row → `upsert_cochrane_paper` semantics; idempotency via `INSERT OR IGNORE` / `ON CONFLICT DO UPDATE` / file-size guard.
- §Run instruction (>2-min rule) → Task 6 hands the command to the user.

No gaps.

**Placeholder scan:** No TBDs, TODOs, or "implement later" markers. All code blocks are complete.

**Type consistency:** Method signatures in Task 3 (`upsert_manually_verified`) match the calls in Task 5 (script) and Task 3 test block. PMC URL regex and filename convention (`<PMID>.xml`) consistent across all tasks.

---

## Execution Handoff

Plan complete and saved to [docs/superpowers/plans/2026-04-21-rob2-manual-verification-import.md](2026-04-21-rob2-manual-verification-import.md). Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
