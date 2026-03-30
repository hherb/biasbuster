# Seed Data Snapshot

Versionable JSONL snapshot of the clean pipeline seed data (papers +
enrichments, **before** annotation).  Use this to recover from database
corruption without re-collecting from external APIs.

## Contents

```
dataset/cleanseed/
├── papers.jsonl        # 4 265 papers (one JSON object per line)
├── enrichments.jsonl   # 3 238 enrichment records
└── manifest.json       # row counts, export timestamp
```

### Data sources included

| Source | Count | Description |
|--------|-------|-------------|
| `retraction_watch` | 767 | Retracted papers with structured RW reason codes (~111 categories). Split into abstract-detectable and abstract-undetectable retractions for training — see below. |
| `pubmed_rct` | 3 238 | PubMed RCTs across 7 medical domains |
| `cochrane_rob` | 260 | Expert RoB 2 assessments (76 high, 99 some_concerns, 85 low) with per-domain ratings (D1-D5) |

### Cochrane RoB detail

Each Cochrane paper stores the overall RoB judgment plus five domain-level
ratings (when available from per-domain backfill):

| Column | RoB 2 Domain |
|--------|-------------|
| `randomization_bias` | D1 — bias from the randomization process |
| `deviation_bias` | D2 — deviations from intended interventions |
| `missing_outcome_bias` | D3 — missing outcome data |
| `measurement_bias` | D4 — measurement of the outcome |
| `reporting_bias` | D5 — selection of the reported result |

Cochrane review metadata (`cochrane_review_pmid`, `cochrane_review_doi`,
`cochrane_review_title`) traces each study back to the source systematic
review for audit purposes.

### What is **not** included

- **Annotations** — the old annotations used a pre-unified prompt and must be
  re-generated.  They are deliberately excluded.
- **Human reviews** — none existed at export time.
- **Evaluation outputs** — separate from seed data.

## Enrichment Stage

The enrichment stage (`pipeline.py --stage enrich`) runs heuristic analysis
on PubMed RCT papers and buckets them by suspicion level:

| Enricher | What it does | Output field |
|----------|-------------|-------------|
| `effect_size_auditor` | Scores reporting bias 0-1 (relative-only, missing NNT, baseline risk omission) | `reporting_bias_score`, `effect_size_audit` |
| `clinicaltrials_gov` | Detects outcome switching between registry and publication | `outcome_switching` |

Papers are classified as `high_suspicion`, `medium`, or `low_suspicion`
based on the combined heuristic scores.  The suspicion level determines
annotation priority and sampling weight in export.

Currently 3 238 enrichment records exist (all PubMed RCTs).  Retracted
papers and Cochrane RoB papers are not enriched — they have ground truth
from their source metadata.

### Abstract-detectable vs abstract-undetectable retractions

Retracted papers are classified by whether the retraction reason produces
visible bias signals in the abstract text.  This distinction matters for
training data quality:

- **Abstract-undetectable** (fabrication, fraud, manipulation, etc.):
  The abstract looks clean — fraud is invisible in the text.  These papers
  are annotated WITHOUT retraction context so the LLM rates the abstract
  on its own merits.  They are excellent test cases for the agent harness
  (which should discover the retraction via Retraction Watch / Crossref).

- **Abstract-detectable** (statistical errors, flawed analysis, COI):
  Bias signals may be visible in the text.  These papers are annotated
  WITH retraction context and severity floors.

See `enrichers/retraction_classifier.py` for the full category mapping
and [ANNOTATED_DATA_SET.md](ANNOTATED_DATA_SET.md) for the complete table.

## Annotation Stage

The annotation stage (`pipeline.py --stage annotate`) sends abstracts to
LLMs for structured 5-domain bias assessment.  Key features:

- **Multi-model**: `--models anthropic,deepseek` runs both backends.
  Each annotation is stored per `(pmid, model_name)`.
- **Incremental persistence**: Each annotation is saved to SQLite via
  `on_result` callback immediately after LLM response — survives crashes.
- **Checkpoint/resume**: `get_annotated_pmids(model)` skips already-annotated
  papers on re-run.
- **Retraction notice filter**: `is_retraction_notice()` skips bare notices
  before sending to LLM (saves tokens).
- **Retraction context**: `build_user_message()` includes classified
  retraction reasons and severity floors from `retraction_classifier`.
- **Cochrane RoB context**: Expert domain-level RoB judgments are passed
  to the LLM when available for calibration.
- **JSON validation**: `parse_llm_json()` repairs malformed JSON and
  rejects truncated annotations (all 9 required fields must be present).

### Annotation backends

| Backend | Model | Module | Transport |
|---------|-------|--------|-----------|
| `anthropic` | Claude (configurable) | `annotators/llm_prelabel.py` | `anthropic` async SDK |
| `deepseek` | DeepSeek Reasoner | `annotators/openai_compat.py` | `httpx` (OpenAI-compatible) |

### Current annotation state

Previous DeepSeek annotations (893 records) were cleared because they
used the pre-unified prompt without severity boundaries, retraction
floors, or Cochrane RoB context.  Re-annotation with the unified prompt
is pending.

Annotations cover papers from all sources (retracted, PubMed RCTs,
Cochrane RoB) proportionally, with configurable per-source caps in
`config.py`.

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
- **Compact**: ~19 MB for 4 265 papers (no pretty-printing overhead)
- **Standard**: already used throughout the project's export pipeline
- **Nested fields**: handles JSON columns (authors, mesh_terms) natively
- **Fast**: streaming read/write, no need to load entire file into memory
