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
DROP TABLE IF EXISTS evaluation_run;
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

-- Phase 5 evaluation runs: per-call audit trail with timing, token
-- counts, raw responses, retry/parse status, and errors. One row per
-- (rct_id, source, domain) — the same key as benchmark_judgment so a
-- 1-to-1 join recovers the call's full provenance.
CREATE TABLE evaluation_run (
    rct_id TEXT NOT NULL,
    source TEXT NOT NULL,        -- e.g. "gpt_oss_20b_abstract_pass1"
    domain TEXT NOT NULL,        -- d1..d5 or overall (synthesis call)
    model_id TEXT NOT NULL,      -- e.g. "gpt-oss:20b" (Ollama tag) or "claude-sonnet-4-6"
    protocol TEXT NOT NULL,      -- "abstract" or "fulltext"
    pass_n INTEGER NOT NULL,     -- 1, 2, 3
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,   -- Anthropic prompt caching only
    cache_write_tokens INTEGER,  -- Anthropic prompt caching only
    raw_response TEXT,           -- full JSON-or-text response
    parse_status TEXT NOT NULL,  -- 'ok', 'retry_succeeded', 'parse_failure', 'api_error'
    parse_attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT,                  -- non-NULL if the call failed
    PRIMARY KEY (rct_id, source, domain),
    FOREIGN KEY (rct_id) REFERENCES benchmark_rct(rct_id),
    CHECK (domain IN ('d1', 'd2', 'd3', 'd4', 'd5', 'overall')),
    CHECK (protocol IN ('abstract', 'fulltext')),
    CHECK (pass_n BETWEEN 1 AND 3),
    CHECK (parse_status IN ('ok', 'retry_succeeded', 'parse_failure', 'api_error', 'in_flight'))
);

CREATE INDEX idx_judgment_source ON benchmark_judgment(source);
CREATE INDEX idx_judgment_domain ON benchmark_judgment(domain);
CREATE INDEX idx_rct_has_fulltext ON benchmark_rct(has_fulltext);
CREATE INDEX idx_eval_run_model ON evaluation_run(model_id);
CREATE INDEX idx_eval_run_protocol ON evaluation_run(protocol);
CREATE INDEX idx_eval_run_status ON evaluation_run(parse_status);
```