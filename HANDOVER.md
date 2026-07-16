# HANDOVER — BiasBuster

_Last updated: 2026-07-16 (maintenance scaffolding introduced: this file, ROADMAP.md,
and the `nextsession`/`fixall` skills. Seeded from the 2026-05-01 next-session runbook,
now archived at `docs/history/EISELE_METZGER_RUNBOOK_2026-07.md`. All state below was
re-verified on 2026-07-16 against the live benchmark DB, the test suite, and GitHub.)_

This file briefs the next session on what is done, what is still open, and the
conventions to keep. Update it whenever a session materially changes the plan; delete
sections that are finished and no longer instructive. Per-PR implementation detail
lives in git history and `docs/superpowers/specs/` — do not re-narrate it here.
Session workflow: invoke the `nextsession` skill at session start; coding rules live
in CLAUDE.md.

## State of play

The active work is the **Eisele-Metzger 2025 RoB 2 replication study** (objective 3).
Core context: memory file `project_claude2_rob_paper.md`; the pre-registration is
**LOCKED** at commit `7854a1c`
(`docs/papers/eisele_metzger_replication/preanalysis_plan.md` + `prompt_v1.md`).
The §9 publishability gate was already cleared with gpt-oss:20b and Sonnet 4.6.

- **The Phase 5 evaluation matrix is complete in the DB** (verified 2026-07-16):
  all four models (gpt-oss:20b, Sonnet 4.6, gemma4:26b, qwen3.6:35b) ×
  {abstract, fulltext} × 3 passes, plus ensemble rows, are in
  `dataset/eisele_metzger_benchmark.db` (546 rows = 91 RCTs × 6 domains per pass;
  gpt-oss/qwen fulltext have a handful of invalid rows, e.g. the known RCT030
  wrong-paper acquisition). A gpt-oss **fulltext temperature sweep**
  (T = 0/0.3/0.6/0.8/1.2 × 3 passes on a 10-RCT subset) is also present.
- **The 2026-07-16 code audit is fully fixed**: all 16 findings + regression B0
  (two were paper-critical: untagged live-path FALLBACK ingest, and κ scripts that
  could not exclude FALLBACK rows). PR #20 merged; issues #7–#19 closed. The full
  finding-by-finding record is in the archived runbook.
- **Test suite: 574 passed, 2 failed** (`uv run pytest`). Both failures are
  stale tests asserting pre-fix behaviour — tracked as issues #21 and #22.
- **The reported κ tables are stale**: `phase6_results.{md,csv}` were generated
  2026-05-06, before the FALLBACK/strict-mode fixes landed, and no
  `phase6_results.strict.*` files exist yet. The regeneration gate below is the
  main open work; the manuscript numbers will change.

## Open work (in priority order)

### 1. Fix the two failing tests (issues #21, #22)

- **#22** `tests/test_cochrane_rob2.py::TestDomainResponseParser::test_missing_judgement_with_answers_defaults_to_some_concerns`
  — still asserts the old hardcoded `some_concerns`; the fixed assessor now derives
  the judgement via `algorithms.derive_domain_judgement`. Update the test to assert
  the algorithm-derived label.
- **#21** `tests/test_export.py::TestExportDataset::test_split_proportions` — asserts
  80/10/10 on a fixture that collapses into too few PMID groups after the
  PMID-grouped-split fix. Rebuild the fixture with enough distinct PMIDs.

### 2. Manuscript κ regeneration gate (needs the live DB)

Run against the canonical DB (and any shard) — the reported figures change:

1. `uv run python studies/eisele_metzger_replication/retro_tag_live_fallback.py --dry-run`
   then `--apply` — back-tags rows written by the old untagged live path. Note:
   gemma/qwen sources currently show **0 FALLBACK rows**; the dry run also tells you
   whether that is genuine (no schema drift) or untagged fallback.
2. `uv run python studies/eisele_metzger_replication/recover_parse_failures.py --dry-run`
   (apply if anything new is recoverable).
3. Regenerate in **both** modes — strict is the pre-registered primary:
   ```bash
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py                     # inclusive
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py --exclude-fallback  # strict → *.strict.{md,csv}
   ```
   (The ensemble-overall figure now reflects worst-wins per the #7 fix.)
4. Update **both** preprint drafts (§3 tables, abstract, conclusion) with the strict
   4-model picture; report inclusive numbers as a sensitivity analysis; drop the
   "(pending: gemma4 and qwen3.6)" caveats.

### 3. Papers

- Primary draft: `docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`
  (thesis: harness over model).
- Companion draft: `docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md`
  (thesis: AI follows the algorithm; experts deviate).
- `docs/papers/drafts/medrxiv_V5/` holds the **Cochrane corpus rebuild design**
  (`REBUILD_DESIGN.md`, status: design, awaiting owner sign-off; forensics in
  `FORENSICS.md`). Rebuilds Stage A ground truth only — separate from the EM study.
- Pre-manuscript spot-checks (from the runbook): right-for-the-right-reasons audit of
  Sonnet's `low` judgements; check whether Sonnet/gemma/qwen share gpt-oss's D1
  instability (§3.6); one more Limitations pass with all four models in.
- Stretch (only if numbers are unequivocal): forest-plot figure from
  `phase6_forest_data.csv` (Figure 1); confidence-calibrated ensemble as a
  future-work appendix (primary use would need a pre-reg amendment); OSF mirror of
  the pre-registration.

### 4. Decisions needed from the repo owner

- Personal review of the 5 systematic-failure RCTs (RCT024, RCT025, RCT034, RCT038,
  RCT062) before publication.
- OpenAthens full-text fetch for the ~50 PMID-but-no-fulltext RCTs (would lift native
  full-text from 41/100 to ~85+/100; ~30 min manual work) — optional polish, decide
  once final numbers are in.
- Submission ordering — recommendation: harness-vs-naive first, then companion.
  Confirm before posting either.
- Sign-off on the medrxiv_V5 Cochrane corpus rebuild design.

### 5. Housekeeping (no issue filed)

- **33 Dependabot vulnerabilities** (12 high, 13 moderate, 8 low) flagged on the
  default branch — worth a triage pass.
- `dataset/` contains stray `* copy.db` files — confirm with the owner they are
  disposable before removing.
- **#26**: the skills' `allowed-tools` need converting to the `Bash(cmd:*)` prefix
  form (exact lines in the issue) — owner must apply by hand; the permission
  classifier blocks agents from editing their own permission surface.

## Conventions and gotchas

- Run the suite with `uv run pytest` (`testpaths = ["tests"]` keeps collection away
  from the stray root `worktrees/` checkouts).
- **Do not modify the locked pre-analysis plan or prompt spec** at `7854a1c` — any
  change requires a numbered amendment (§12 of the pre-reg).
- **Never re-run `build_benchmark_db.py`** after evaluation rows exist — it DROPs and
  rebuilds all tables, destroying the model evaluation data. Copy data out first if
  the schema must change.
- **Never commit `DATA/`** — EM 2025 supplementary-data redistribution rights are
  unresolved (`.gitignore`).
- **Do not re-run the full Sonnet batch** — $30–80 of API credits; results are in
  the DB.
- Do not push to branches other than feature branches + PR to `main` without
  confirming with the owner.
- Two near-identical filenames in `biasbuster/methodologies/cochrane_rob2/`:
  `algorithm.py` (consistency checking) vs `algorithms.py` (per-domain truth
  tables). Both carry cross-reference notes — mind which one you touch.
- All κ / ensemble / synthesis numbers are computed in code, never by the model.
- Repo-wide rules (CLAUDE.md): `uv` only; prompts single-sourced in `prompts*.py`;
  processes >2 min are printed for the user to run, never run in-session; anything
  producing results over time saves incrementally with checkpoint/resume; never
  truncate data destined for analysis.

### Key locations

| Artefact | Path |
|---|---|
| Locked pre-analysis plan / prompt | `docs/papers/eisele_metzger_replication/{preanalysis_plan,prompt_v1}.md` |
| Preprint drafts | `docs/papers/drafts/20260501_*.md` (primary), `20260423_*.md` (companion) |
| Cochrane corpus rebuild design | `docs/papers/drafts/medrxiv_V5/` |
| Study scripts (phases 1–6) | `studies/eisele_metzger_replication/` |
| Cross-model κ report | `studies/eisele_metzger_replication/compute_phase6_kappa.py` → `phase6_results*.{md,csv}` |
| Recovery / retro-tagging | `studies/eisele_metzger_replication/{recover_parse_failures,retro_tag_live_fallback}.py` |
| Shard merge (Mac ⇄ DGX) | `studies/eisele_metzger_replication/merge_eval_dbs.py` |
| Per-domain Cochrane algorithms | `biasbuster/methodologies/cochrane_rob2/algorithms.py` |
| Benchmark DB (gitignored) | `dataset/eisele_metzger_benchmark.db` (+ `.spark.db` shard) |
| EM 2025 source data (gitignored, no redistribution) | `DATA/20240318_Data_for_analysis_full/` |
| COI divergence rationale (standing design) | `docs/harness/DESIGN_RATIONALE_COI.md` |

### Quick state check

```bash
sqlite3 dataset/eisele_metzger_benchmark.db <<'SQL'
.mode column
.headers on
SELECT source, COUNT(*) AS n_judgments,
       SUM(CASE WHEN valid=1 THEN 1 ELSE 0 END) AS n_valid,
       SUM(CASE WHEN raw_label='FALLBACK' THEN 1 ELSE 0 END) AS n_fallback
FROM benchmark_judgment
WHERE source != 'cochrane' AND source NOT LIKE 'em_claude2_%'
GROUP BY source ORDER BY source;
SQL
```

## Standing decisions from the repo owner

- **BiasBuster flags *risk* of bias, not proof of misconduct.** The hard-HIGH COI
  trigger (industry funding + sponsor-employed authors) is validated policy; the
  structural disagreement with Cochrane RoB 2 on COI is by design, not a bug
  (`docs/harness/DESIGN_RATIONALE_COI.md`).
- **medRxiv authorship**: human-only authorship + detailed AI-use disclosure in
  Methods (disclosure paragraph at §11 of the pre-reg; policy snapshot at
  `docs/literature/rob_validation/medrxiv_ai_policy_2026-04-30.md`).
- Correctness outranks everything: false positives (besmirching valid work) and
  false negatives (missing fraud) are both harmful.
