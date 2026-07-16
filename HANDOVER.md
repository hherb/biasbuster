# HANDOVER — BiasBuster

_Last updated: 2026-07-17 (stale tests #21/#22 fixed, suite green at 579 passed;
κ tables regenerated and both drafts updated to strict-primary numbers, but a
review of PR #28 found those tables are polluted by RCT030 wrong-paper judgements —
tracked as blocking issue #29, must regenerate before submission. Wrong-paper
recovery guard added. Maintenance scaffolding introduced 2026-07-16: this file,
ROADMAP.md, and the `nextsession`/`fixall` skills, seeded from the 2026-05-01
next-session runbook, now archived at
`docs/history/EISELE_METZGER_RUNBOOK_2026-07.md`.)_

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
- **Test suite: 576 passed, 0 failed** (`uv run pytest`). The two stale tests
  (issues #21, #22) were realigned to the intended post-fix behaviour on
  2026-07-17: distinct PMIDs for the PMID-grouped export split, and the
  algorithm-derived judgement for the RoB 2 missing-judgement fallback.
- **The κ regeneration gate ran (2026-07-17) but the output is BLOCKED by issue
  #29**: retro-tag was a genuine no-op (0 untagged rows in 11,592 scanned);
  `phase6_results.{md,csv}` and the new `phase6_results.strict.{md,csv}` + forest
  CSVs were regenerated; both preprint drafts updated to the strict-primary
  picture. The regenerated tables (and the numbers in both drafts) are **polluted
  by RCT030 wrong-paper judgements** and must be regenerated again once #29 is
  fixed — do not submit on the current numbers. The qualitative findings likely
  survive but every figure will shift. For the record, the current (polluted)
  headline: under the corrected worst-wins ensemble, naive ensembling underperforms
  the best single pass for all four models (former "qwen exception" +0.012 became
  −0.012); Sonnet strict best-pass fulltext κ_quad 0.236 (0.264 inclusive),
  four-model spread 0.021.
- **Recovery guard added (2026-07-17), but only covers parse-failure recovery**:
  `recover_parse_failures.py` recovered 2 qwen fulltext d3 rows that belonged to
  **RCT030 — the wrong-paper acquisition** (its signalling describes the parent
  Cochrane review, not the trial). The rows were reverted from backup,
  `WRONG_PAPER_RCTS = {"RCT030"}` now guards the recovery script (with tests in
  `tests/test_recover_parse_failures.py`), and RCT030's exclusion is documented in
  `benchmark_rct.notes`. **BUT** the guard does NOT touch RCT030's 179 already-valid
  `benchmark_judgment` rows (full cochrane + all-model judgements, both protocols),
  which are still counted in every κ script — see issue #29. So "excluded from
  analysis" is documented and enforced in the *recovery* path only, not in the
  *analysis* path. FALLBACK total stands at 91 (all Sonnet); the 29 RCT030
  parse-failure rows stay out of recovery, but the wrong-paper *valid* rows do not.

## Open work (in priority order)

### 1. Fix the RCT030 κ pollution and regenerate (issue #29) — BLOCKS the manuscript

RCT030 is documented "excluded from analysis" in `benchmark_rct.notes`, but the κ
scripts have no RCT030 filter, so its wrong-paper judgements (179 valid rows) are in
every phase-6 table and both drafts. Fix: centralise `WRONG_PAPER_RCTS` into a shared
study module, enforce it in `compute_phase6_kappa.py` / `interim_analysis.py` /
`temperature_analysis.py` / `sanity_check_kappa.py`, regenerate both κ modes against
the canonical DB, then re-update both drafts (RCT030 is a *wrong-paper* exclusion
distinct from the 9 unrecoverable regional-journal RCTs → n=90, not 91, for affected
sources; §3.1 narrative needs rewording). Owner decision needed on whether RCT030 is
the *only* wrong-paper acquisition (the set was populated reactively). Full evidence
and a proposed patch are in issue #29.

### 2. Papers

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

### 3. Decisions needed from the repo owner

- Personal review of the 5 systematic-failure RCTs (RCT024, RCT025, RCT034, RCT038,
  RCT062) before publication.
- OpenAthens full-text fetch for the ~50 PMID-but-no-fulltext RCTs (would lift native
  full-text from 41/100 to ~85+/100; ~30 min manual work) — optional polish, decide
  once final numbers are in.
- Submission ordering — recommendation: harness-vs-naive first, then companion.
  Confirm before posting either.
- Sign-off on the medrxiv_V5 Cochrane corpus rebuild design.

### 4. Housekeeping (no issue filed)

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
- **RCT030 is the wrong-paper acquisition** — its 29 parse-failure rows must stay
  excluded; `recover_parse_failures.py` guards this via `WRONG_PAPER_RCTS`. Never
  "recover" them.
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
