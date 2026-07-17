# HANDOVER — BiasBuster

_Last updated: 2026-07-17 (issue #29 code fix merged as PR #30 — RCT030 exclusion
now centralised in `exclusions.py` and enforced across all κ loaders. A follow-up
completeness audit then found RCT030 is NOT the only wrong-paper acquisition:
≥4 more Tier-A wrong documents (RCT008/080/088/093) plus a Tier-B "wrong report"
class — see Open work #1. Regeneration is now blocked on an OWNER DECISION about
the full exclusion set, not just on running the script. New reproducible audit
tool + tests committed. Suite green at 596 passed. Maintenance scaffolding
(HANDOVER/ROADMAP + `nextsession`/`fixall` skills) seeded 2026-07-16 from the
runbook archived at `docs/history/EISELE_METZGER_RUNBOOK_2026-07.md`.)_

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
- **Test suite: 596 passed, 0 failed** (`uv run pytest`). Stale tests #21/#22
  were realigned 2026-07-17; +12 from the new wrong-paper audit tests.
- **The current `phase6_results*.{md,csv}` and both drafts are STALE and
  polluted** and must be regenerated once the exclusion set is agreed (Open
  work #1). They carry RCT030 *and* the newly-found wrong-paper judgements. Do
  not submit on the current numbers. Qualitative findings likely survive but
  every figure will shift. For the record, the current (polluted) headline:
  under the corrected worst-wins ensemble, naive ensembling underperforms the
  best single pass for all four models (former "qwen exception" +0.012 became
  −0.012); Sonnet strict best-pass fulltext κ_quad 0.236 (0.264 inclusive),
  four-model spread 0.021.
- **RCT030 wrong-paper exclusion is now enforced in BOTH paths** (PR #30 merged
  2026-07-17): `studies/eisele_metzger_replication/exclusions.py` is the single
  source of truth (`WRONG_PAPER_RCTS` + `wrong_paper_filter()` SQL helper), imported
  by `recover_parse_failures.py` (recovery guard) and by the analysis loaders in
  `compute_phase6_kappa.py` / `interim_analysis.py` / `temperature_analysis.py`.
  Deliberately NOT applied in `sanity_check_kappa.py` (it reproduces EM's published
  κ from EM's own correctly-acquired data). Tests in
  `tests/test_kappa_exclusions.py` + `tests/test_recover_parse_failures.py`.
  FALLBACK total stands at 91 (all Sonnet).

## Open work (in priority order)

### 1. Wrong-paper exclusion set is BIGGER than {RCT030} — OWNER DECISION blocks the manuscript

The 2026-07-17 completeness audit (issue #29 step 5, full evidence in the issue
comment) found **RCT030 is not the only wrong-paper acquisition**. Phase 1
mis-resolved the fetched document for several more RCTs; regenerating the κ
tables with only RCT030 excluded would still ship polluted numbers. **Do not
regenerate or touch the drafts until the owner decides the exclusion set.**

- **Tier A — wrong document entirely, recommend excluding**: RCT008 (systematic
  review, not the Jolly RCT), RCT080 (Scandinavian mortality stats, not the
  kindergarten language RCT), RCT088 (concrete engineering, not the calcifediol
  COVID RCT), RCT093 (RECOVERY *empagliflozin* arm, not the intended *aspirin*
  arm) — alongside RCT030. All model-confirmed via rationales.
- **Tier B — right trial, wrong report (protocol/sub-analysis/different
  follow-up), owner adjudicates**: RCT017, RCT074 (protocols not results),
  RCT100 (pooled 4-trial analysis vs single COVE trial), RCT095 (STOIC
  mechanistic sub-study), RCT040 (insulin antibody 104-wk vs 52-wk efficacy —
  verify), RCT009 (TTM2 oxygen sub-analysis), RCT064 (PRET-PD cognition report),
  RCT019 (fluocinolone 3-yr vs 12-mo results).
- Reproducible tools: `audit_wrong_paper_acquisitions.py` (detection) and
  `recover_wrong_papers.py` (obtainability + surgical recovery), tests in
  `tests/test_wrong_paper_audit.py` + `tests/test_recover_wrong_papers.py`.

**Exclude vs recover — obtainability audit (2026-07-17, `recovery_obtainability.md`):**
most of these are *recoverable*, not just excludable, because the `cochrane` /
`em_claude2_*` ground truth is already for the correct trial — only the fetched
document is wrong. `recover_wrong_papers.py report` re-resolves each intended
trial (validated by title coverage vs `em_rct_ref` — the gate that would have
prevented the bug):
- **8 auto-recoverable** (correct PMID found): RCT008, RCT009, RCT019, RCT040,
  RCT064, RCT074, RCT095, RCT100.
- **2 manual-recoverable** (verified PMID, resolver can't select): RCT088=32871238
  (compound surname), RCT093=34800427 (same-platform arm). `MANUAL_PMIDS` holds these.
- **3 to exclude / manual-lookup**: RCT030, RCT080 (correct paper not PubMed-indexed
  → genuine exclude); RCT017 (results paper exists but resolver locks on the
  near-identical protocol PMID — needs a manual PMID lookup).

**Owner decision:** exclude-all (simplest, conservative) vs recover the ~10-11
obtainable ones (restores n, needs a pre-reg §12 amendment + a targeted model
re-run). `recover_wrong_papers.py apply RCTxxx [RCTyyy=PMID] --apply` does the
surgery (re-fetch correct doc → update `benchmark_rct` → delete stale *model*
rows only, never ground truth; DB backed up first) and prints the
`run_evaluation.py` re-assess commands (its existence check recomputes only the
deleted rows). The model re-run is expensive/owner-gated — NOT run automatically.

Whichever path: add exclude-only IDs to `exclusions.WRONG_PAPER_RCTS` (enforced
across all κ loaders by #30), then regenerate both κ modes, re-derive both drafts,
and rewrite §3.1 to name a **wrong-paper class** distinct from the 9 unrecoverable
regional-journal RCTs. RCT030-only magnitude was small (+0.007–0.009 κ_quad, PR
#30 pre-check); the full set moves figures more.

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
- **Wrong-paper acquisitions are a class, not just RCT030** — RCT030 is the only
  one *enforced* in `exclusions.WRONG_PAPER_RCTS` today, but the audit found
  more (Open work #1). Before any κ regeneration, run
  `audit_wrong_paper_acquisitions.py` and get the owner's sign-off on the set.
  RCT030's 29 parse-failure rows must always stay out of recovery
  (`recover_parse_failures.py` guards via `WRONG_PAPER_RCTS`). Never "recover"
  them.
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
| Wrong-paper exclusion set / audit / recovery | `studies/eisele_metzger_replication/{exclusions,audit_wrong_paper_acquisitions,recover_wrong_papers}.py`; obtainability at `recovery_obtainability.md` |
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
