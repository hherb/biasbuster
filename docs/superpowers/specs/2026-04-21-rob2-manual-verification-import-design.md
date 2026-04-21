# RoB 2 Manual Verification Set — Import & Full-Text Cache

**Date:** 2026-04-21
**Scope:** One-shot import of a 10-trial curated RoB 2 dataset, plus a new
`manually_verified` tag table and on-disk full-text cache.
**Verification set tag:** `rob2_manual_verify_20260421`

---

## Purpose

Import a human-curated 10-row RoB 2 dataset (provided inline by the user) so
that the newly built Cochrane RoB 2 methodology (Step 7 scaffold,
[biasbuster/methodologies/cochrane_rob2/](../../biasbuster/methodologies/cochrane_rob2/))
can be exercised against trials with known, published expert RoB 2 judgments.

The set is curated (not auto-extracted from Cochrane reviews), so we need a
way to tag these rows as manually verified and distinguish them from the
bulk `source='cochrane_rob'` collection.

Each row also ships with a PMC URL, and the methodology assessor works best
when it can read the original trial's full text. We therefore front-load the
full-text fetch and store a local cache alongside the DB row.

## Input data

Ten rows, embedded as a Python constant in the import script (the set is
frozen at this pilot — not a CSV we re-read). Columns:

| CSV column | Meaning |
|---|---|
| `study_id` | 1..10 sequence number (not stored) |
| `trial_name` | e.g. "RECOVERY dexamethasone" |
| `pmid` | NCBI PubMed ID for the trial |
| `pmc_url` | PMC URL for the trial full text |
| `source_review` | Name of the review that graded the trial |
| `randomization`, `deviations`, `missing_data`, `measurement`, `reporting`, `overall` | RoB 2 domain ratings |

Rating values are RoB 2 vocabulary: `low`, `some_concerns`, `high`.

## Changes

### 1. New DB table: `manually_verified`

Schema (added to `Database._create_tables()` in
[biasbuster/database.py](../../biasbuster/database.py)):

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
CREATE INDEX IF NOT EXISTS idx_manually_verified_set
    ON manually_verified(verification_set);
```

- Composite PK lets the same paper appear in future verification sets.
- Foreign key ensures the tag always points at a real `papers` row.
- `fulltext_path` is a repository-root-relative path
  (`dataset/rob2_verification_fulltexts/<PMID>.xml`).
- `fulltext_ok` is `1` when the XML was fetched and written at or above
  the minimum-size threshold (`MIN_JATS_BYTES = 1024`, matching
  [scripts/fetch_cochrane_jats.py](../../scripts/fetch_cochrane_jats.py)).

### 2. DB helper: `Database.upsert_manually_verified()`

One method, no abstraction layer:

```python
def upsert_manually_verified(
    self,
    pmid: str,
    verification_set: str,
    *,
    trial_name: str | None = None,
    source_review: str | None = None,
    fulltext_path: str | None = None,
    fulltext_ok: bool = False,
    notes: str | None = None,
) -> bool:
    """Insert or update a manually_verified tag row. Returns True on change."""
```

Implementation: `INSERT ... ON CONFLICT(pmid, verification_set) DO UPDATE`,
updating every field. Commits immediately.

### 3. Full-text storage directory

`dataset/rob2_verification_fulltexts/<PMID>.xml` for every successfully
fetched JATS XML. Directory is created on first write.

The repo's `.gitignore` excludes `dataset/*.db*` but not arbitrary subfolders.
As part of this change we add one line — `dataset/rob2_verification_fulltexts/` —
to `.gitignore` so the ~1–5 MB XML blobs stay out of git. This is a one-time
edit, not something the import script does at runtime.

### 4. Import script: `scripts/import_rob2_verification_set.py`

Single-purpose, idempotent, async. Runs outside the main pipeline.

**Flow per row:**

1. Build a `RoBAssessment` (from [biasbuster/collectors/cochrane_rob.py](../../biasbuster/collectors/cochrane_rob.py))
   and convert via `rob_assessment_to_paper_dict()`. Map CSV columns:
   - `randomization` → `randomization_bias`
   - `deviations` → `deviation_bias`
   - `missing_data` → `missing_outcome_bias`
   - `measurement` → `measurement_bias`
   - `reporting` → `reporting_bias`
   - `overall` → `overall_rob`
   - `source_review` → `cochrane_review_title`
2. `db.upsert_cochrane_paper(paper)` — reuses the validated preserve-on-conflict
   semantics (PubMed-fetched title/abstract survive if already populated).
3. Fetch PubMed metadata for the 10 PMIDs via
   `RetractionWatchCollector.fetch_pubmed_abstracts_batch()`
   (same code path as `seed_database.py --step fetch-abs`).
   Update `papers.abstract`, `title`, `journal`, `year`, `authors` for any
   row whose abstract is still empty/short.
4. Extract the PMCID from `pmc_url` (regex: `PMC\d+`).
5. Fetch Europe PMC fullTextXML at
   `https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML`,
   reusing the fetch/retry/min-size pattern from `scripts/fetch_cochrane_jats.py`
   (`fetch_with_retry`, 60 s timeout, 2 req/s, `MIN_JATS_BYTES = 1024`).
   Write to `dataset/rob2_verification_fulltexts/<PMID>.xml` on success.
6. `db.upsert_manually_verified(pmid, 'rob2_manual_verify_20260421',
   trial_name=..., source_review=..., fulltext_path=...,
   fulltext_ok=...)`.

**Output:** Markdown summary table printed to stdout:

```
| PMID | trial | overall_rob | abs_chars | pmcid | fulltext | status |
|------|-------|-------------|-----------|-------|----------|--------|
| ...  | ...   | ...         | ...       | ...   | 2.1 MB   | ok     |
```

Also logged to `logs/import_rob2_verification_set_YYYYMMDD_HHMMSS.log`.

**Idempotency:**
- DB: `INSERT OR IGNORE` for papers (with Cochrane fields always updated via
  `upsert_cochrane_paper`), `ON CONFLICT DO UPDATE` for `manually_verified`.
- Disk: skip fetch if `dataset/rob2_verification_fulltexts/<PMID>.xml` exists
  and is ≥ 1024 bytes (same guard as `fetch_cochrane_jats.py`).

### 5. Tests

Minimal pytest at [tests/test_manually_verified.py](../../tests/test_manually_verified.py):
- Creates a temp DB, inserts a dummy paper, calls `upsert_manually_verified`
  twice, asserts second call updates (not duplicates) the row.
- Asserts FK is enforced: inserting with an unknown `pmid` raises.

No integration test against live PubMed/Europe PMC — the import script
itself is the smoke test, and the CLAUDE.md standard is module demo blocks
rather than a formal suite.

## Non-goals

- **Not** adding a reusable "manual verification set" framework. One table,
  one script, one method. If a second set is curated later, we copy this
  script or generalise then.
- **Not** modifying the main pipeline (`pipeline.py`) or the RoB 2
  methodology itself. This import is upstream of the assessor.
- **Not** attempting PDF extraction — only JATS XML from Europe PMC.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| A PMID in the set no longer has a PubMed abstract | Paper row still created with Cochrane RoB ratings; abstract stays empty; flagged in summary table with `abs_chars=0`. Not a blocker. |
| Europe PMC `fullTextXML` returns 404 for some PMCIDs | Record `fulltext_ok=0` and empty `fulltext_path`. Logged at WARNING. The assessor already has a code path for missing full text. |
| Existing `papers` row for one of these PMIDs was already imported under a different source | `upsert_cochrane_paper` always updates the Cochrane-authoritative fields; source column is updated to `cochrane_rob`. The `manually_verified` row is additive (tag), so the pre-existing context is preserved in any other tables. |
| User re-runs the script | Fully idempotent — abstracts re-fetched only if short, full-text re-fetched only if missing from disk, DB rows updated in place. |

## Run instruction

Per [CLAUDE.md](../../CLAUDE.md)'s >2-min rule the script is handed back to
the user to execute:

```bash
uv run python -m scripts.import_rob2_verification_set
```

Expected runtime: 60–120 s (10 PubMed fetches + 10 Europe PMC fullText
fetches with 0.5 s spacing).

## Open questions

None at present. Tag name (`rob2_manual_verify_20260421`) confirmed with user.
