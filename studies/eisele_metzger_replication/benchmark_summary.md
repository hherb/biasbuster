# Benchmark Database Build Summary

**Generated:** by `studies/eisele_metzger_replication/build_benchmark_db.py`
**SQLite path:** `dataset/eisele_metzger_benchmark.db` (gitignored)
**Companion:** `benchmark_spotcheck.md` (random 10-RCT manual verification sample)

## RCT-level coverage

- Total RCTs in benchmark: **100**
- With resolved PMID: 91 (91%)
- With abstract text: 91 (91%)
- With full text: 41 (41%)
- With trial registration: 61 (61%)

## Judgment-row counts by source

| Source | Rows |
|---|---:|
| cochrane | 600 |
| em_claude2_c3 | 600 |
| em_claude2_run1 | 600 |
| em_claude2_run2 | 600 |
| em_claude2_run3 | 600 |

Expected: 6 domains × 100 RCTs = 600 rows per source. 5 sources from EM × 600 = 3000 total.
Actual: **3000** rows inserted.

## Validation

✅ No empty cells, no unrecognised labels — clean ingest.

## Label distribution (valid judgments only)

| Source | low | some_concerns | high |
|---|---:|---:|---:|
| cochrane | 406 | 144 | 50 |
| em_claude2_c3 | 252 | 309 | 39 |
| em_claude2_run1 | 416 | 176 | 8 |
| em_claude2_run2 | 387 | 184 | 29 |
| em_claude2_run3 | 478 | 101 | 21 |

## Schema

```sql
DROP TABLE IF EXISTS benchmark_judgment;
DROP TABLE IF EXISTS benchmark_rct;

CREATE TABLE benchmark_rct (
    rct_id TEXT PRIMARY KEY,
    cr_id TEXT NOT NULL,
    pmid TEXT,
    doi TEXT,
    nct_nr TEXT,
    title TEXT,
    authors_text TEXT,
    publication_year INTEGER,
    condition TEXT,
    intervention TEXT,
    outcome_text TEXT,
    has_abstract INTEGER NOT NULL DEFAULT 0,
    has_fulltext INTEGER NOT NULL DEFAULT 0,
    has_registration INTEGER NOT NULL DEFAULT 0,
    fulltext_source TEXT,
    em_rct_ref TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE benchmark_judgment (
    rct_id TEXT NOT NULL,
    source TEXT NOT NULL,
    domain TEXT NOT NULL,
    judgment TEXT,
    rationale TEXT,
    valid INTEGER NOT NULL DEFAULT 1,
    raw_label TEXT,
    PRIMARY KEY (rct_id, source, domain),
    FOREIGN KEY (rct_id) REFERENCES benchmark_rct(rct_id),
    CHECK (domain IN ('d1', 'd2', 'd3', 'd4', 'd5', 'overall')),
    CHECK (judgment IS NULL OR judgment IN ('low', 'some_concerns', 'high'))
);

CREATE INDEX idx_judgment_source ON benchmark_judgment(source);
CREATE INDEX idx_judgment_domain ON benchmark_judgment(domain);
CREATE INDEX idx_rct_has_fulltext ON benchmark_rct(has_fulltext);
```