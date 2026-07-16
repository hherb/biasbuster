# Attic — retired material awaiting manual review

Nothing here is deleted; files are moved (via `git mv`, so history follows)
when they appear dead or completed and no live code references them. Review
each item and either delete it for good or restore it with
`git mv attic/<path> <original-path>`.

Retirement rationale is recorded below. If an entry turns out to still be
needed, restoring it is safe — nothing in the live tree imports from `attic/`.

## scripts/ (retired 2026-07-16, repo restructure)

| File | Was at | Why retired |
|---|---|---|
| `scripts/migrate_jsonl_to_sqlite.py` | repo root | One-time migration of legacy JSONL + review CSVs into SQLite. Migration completed; SQLite has been the single source of truth since. |
| `scripts/fix_v7_parsing_bug_output.py` | repo root | One-time re-parse of stored V7 evaluation outputs after a scorer bug fix (see `docs/history/ROUND_4.md`). Fix applied; keeping the script would only invite accidental re-runs. |
| `scripts/flag_review_candidates.py` | repo root | One-off analysis flagging inter-model severity disagreements into `human_reviews`. No references anywhere in code or docs; superseded by the human-review workflow in the crowd/review tooling. |
| `scripts/diagnose_v5a_disagreements.py` | repo root | One-off diagnostic comparing gemma4 vs Sonnet on the 16 V5A validation papers (2026-04). Analysis concluded; results captured in the V5A docs. |

## prompts/V10 (retired 2026-04-02, pre-existing)

Retired prompt version V10; see `prompts/V10/README.md` for its own rationale.

## Explicitly NOT retired

`biasbuster/collectors/rob_table_extractor.py` was a candidate (imported by
nothing), but it is Phase 2 of the still-current medRxiv-V5 Cochrane corpus
rebuild (`docs/papers/drafts/medrxiv_V5/REBUILD_DESIGN.md` §9.1) with test
fixtures in `tests/fixtures/cochrane_reviews/`. Unfinished work, not dead code.
