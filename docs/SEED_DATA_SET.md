# Seed Data Snapshot

Versionable JSONL snapshot of the clean pipeline seed data (papers +
enrichments, **before** annotation).  Use this to recover from database
corruption without re-collecting from external APIs.

## Contents

```
dataset/cleanseed/
├── papers.jsonl        # 4 184 papers (one JSON object per line)
├── enrichments.jsonl   # 3 238 enrichment records
└── manifest.json       # row counts, export timestamp
```

### Data sources included

| Source | Count | Description |
|--------|-------|-------------|
| `retraction_watch` | 767 | Retracted papers with structured RW reason codes |
| `pubmed_rct` | 3 238 | PubMed RCTs across 7 medical domains |
| `cochrane_rob` | 179 | Expert RoB 2 assessments (48 high, 54 some_concerns, 77 low) |

### What is **not** included

- **Annotations** — the old annotations used a pre-unified prompt and must be
  re-generated.  They are deliberately excluded.
- **Human reviews** — none existed at export time.
- **Evaluation outputs** — separate from seed data.

## Export

```bash
# Export current DB seed data (papers + enrichments)
uv run python seed_export.py export

# Export to a custom directory
uv run python seed_export.py export --dir dataset/cleanseed_v2
```

This reads from the SQLite database and writes compact JSONL (one JSON
object per line, no pretty-printing).  JSON columns (authors, grants,
mesh_terms, etc.) are decoded into native objects for clean diffs.

## Import (Disaster Recovery)

```bash
# Restore seed data into the default DB (backs up existing DB first)
uv run python seed_export.py import

# Restore into a specific DB
uv run python seed_export.py import --db dataset/fresh.db
```

Import automatically:
1. Backs up the existing DB (`.pre-import-YYYYMMDD_HHMMSS.bak`)
2. Initialises the schema
3. Inserts papers first, then enrichments (respects foreign keys)
4. Uses `INSERT OR REPLACE` so it can safely merge into a partially populated DB
5. Commits every 500 rows for safety

After import, continue the pipeline from the annotation stage:

```bash
uv run python pipeline.py --stage annotate
```

## Format

**JSONL** (JSON Lines) — chosen over YAML, CSV, or binary formats because:

- **Git-friendly**: one record per line produces clean line-level diffs
- **Compact**: ~19 MB for 4 184 papers (no pretty-printing overhead)
- **Standard**: already used throughout the project's export pipeline
- **Nested fields**: handles JSON columns (authors, mesh_terms) natively
- **Fast**: streaming read/write, no need to load entire file into memory
