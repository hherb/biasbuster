# HANDOVER — BiasBuster

_Last updated: 2026-07-19 (Figure 1 forest-plot session). Recent merges: pre-manuscript
spot-checks (PR #42); before that the `--sensitivity` κ flag (PR #40), the wrong-paper
**hybrid** (all 13 excluded from the primary; both drafts regenerated on n=78 — PR #31),
**OA-first RoB benchmark Stage A** (PRs #32, #37). **This session** built and shipped
**Figure 1** — a forest plot of κ_quad vs Cochrane RoB 2 across all four models × both
protocols × three passes + ensembles, with bootstrap 95% CIs (quadratic weighting, 500
resamples) and three literature reference lines (EM Claude 2 κ=0.22; Minozzi 2020 κ=0.16;
Minozzi 2021 κ=0.42) — and wired it into primary draft §3.2 (Open work #A.4, now done).
Along the way found and fixed a **pre-existing reproducibility bug**: `load_pairs` in
`compute_phase6_kappa.py` had no `ORDER BY`, so bootstrap CIs for ensemble rows drifted
between runs; point estimates were never affected (Cohen's κ is order-invariant) and no
published number moved. A final whole-branch review then found the identical bug in the
sibling `load_paired` (`sanity_check_kappa.py`); same `ORDER BY a.rct_id` fix applied,
`sanity_check_report.md` regenerated (byte-identical — this DB's physical row order for
the static `cochrane`/`em_claude2_run1` sources already matched `rct_id` order, so nothing
moved, but the latent bug is now closed for any future DB state). The same review also
found Figure 1's caption claimed the pre-registered **strict** n=78 primary corpus, when
the figure is actually rendered from the **inclusive** (FALLBACK-included) variant — same
n=78 and numbers for gpt-oss/gemma/qwen, but different κ_quad for Sonnet (strict
0.101/0.281/0.246 vs inclusive 0.186/0.309/0.246); caption now discloses this. Suite green
at **736 passed**._

This file briefs the next session on what is done, what is still open, and the
conventions to keep. Update it whenever a session materially changes the plan; delete
sections that are finished and no longer instructive. Per-PR implementation detail
lives in git history and `docs/superpowers/specs/` — do not re-narrate it here.
Session workflow: invoke the `nextsession` skill at session start; coding rules live
in CLAUDE.md.

## State of play

Objective 3 now has **two active workstreams**, both at owner-gated checkpoints:

### A. Eisele-Metzger 2025 RoB 2 replication study (manuscript-finalisation stage)

Core context: memory `project_claude2_rob_paper.md`; the pre-registration is
**LOCKED** at commit `7854a1c`
(`docs/papers/eisele_metzger_replication/preanalysis_plan.md` + `prompt_v1.md`).
The §9 publishability gate was cleared with gpt-oss:20b and Sonnet 4.6.

- **Phase 5 evaluation matrix is complete in the DB**: all four models (gpt-oss:20b,
  Sonnet 4.6, gemma4:26b, qwen3.6:35b) × {abstract, fulltext} × 3 passes, plus
  ensemble rows, in `dataset/eisele_metzger_benchmark.db`. A gpt-oss fulltext
  **temperature sweep** (T = 0/0.3/0.6/0.8/1.2 × 3 passes on a 10-RCT subset) is also
  present.
- **Wrong-paper hybrid DONE** (PR #31): `WRONG_PAPER_RCTS` holds all 13;
  `UNRECOVERABLE_WRONG_PAPER_RCTS = {RCT030, RCT080}` is defined for the sensitivity
  analysis. Both phase-6 κ modes + the algorithm-conformance audit were regenerated on
  **n=78**, and both drafts updated (numbers + reframed finding #1). Enforcement is
  centralised in `studies/eisele_metzger_replication/exclusions.py`, imported by every
  κ loader (deliberately NOT `sanity_check_kappa.py`, which reproduces EM's own κ).
- **Headline (regenerated, n=78):** best-pass κ_quad fulltext (strict) — gpt-oss
  **0.321**, Sonnet **0.281**, qwen **0.273**, gemma **0.259** (spread 0.062; the old
  "tight clustering" finding is gone). Ceiling 0.04–0.10 above EM's 0.22 but still far
  below human-usable. Ensemble-loses-to-best-pass and run-to-run ordering both SURVIVE;
  conformance lenient-asymmetry sharpened to 8.5:1 pooled / 11.7:1 at 4/4 consensus.
- **Pre-manuscript spot-checks DONE** (2026-07-18 session, Open work #A.2): Sonnet `low`
  audit (2/5 match Cochrane but right-for-the-right-reasons — 2 RCTs, the 3 misses all one
  shared-with-gpt-oss D2 case); per-domain instability audit shows §3.6's D1 concentration
  is audit-set-specific, not corpus-wide; §3.5/§3.6/§5 updated. All numbers from
  `premanuscript_spotchecks.py` (read-only over the benchmark DB).
- **Figure 1 DONE** (this session, Open work #A.4): forest plot of κ_quad vs Cochrane —
  `studies/eisele_metzger_replication/figures/{figure1_forest.pdf,.png}`, committed (the
  benchmark DB that regenerates them is gitignored) — wired into primary draft §3.2. The
  caption's final sentence was rewritten from a pre-drafted claim that didn't hold once
  checked against `phase6_forest_data.csv`: fulltext CIs all contain EM's 0.22, but it's
  the point estimates (not the intervals) that stay below Minozzi 2021's 0.42 — several
  fulltext CI upper bounds (e.g. gpt-oss pass 1, 0.49) exceed it. Reproducibility bug found
  + fixed along the way — see header above. `phase6_results.strict.{csv,md}` and
  `phase6_forest_data.strict.csv` now lag the primary schema by one generation (see
  Conventions and gotchas).
- **Test suite: 736 passed, 0 failed** (`uv run pytest`).

### B. OA-first Risk-of-Bias benchmark (Stage A shipped; owner actions pending)

New benchmark started 2026-07-17 to fix the paywalled/wrong-paper sourcing that broke
the earlier corpora — it **inverts sourcing** (enumerate the OA population first, admit
a trial only when full text is in hand AND it carries a verified human-expert RoB 2
label). Context: memory `project_oa_first_rob_benchmark.md`; spec
`docs/superpowers/specs/2026-07-17-oa-first-rob-benchmark-design.md`; plan
`docs/superpowers/plans/2026-07-17-oa-rob-benchmark-stage-a.md`.

- Stage A code shipped (PRs #32, #37): study package `studies/oa_rob_benchmark/`,
  isolated store `dataset/oa_rob_benchmark.db` (append/upsert only; never touches
  `biasbuster.db` or the EM benchmark DB), a four-part litmus test enforced on every
  admitted item. ROBoto2 real data landed and the ingest was rewritten for its actual
  shape (`convert_roboto2_csv.py` → `dataset/roboto2/roboto2.json`; `title_resolver.py`
  for title→PMID via PubMed esearch + similarity ≥0.90).
- **This is a data-construction workstream, not yet a study** — Stage B (the fresh
  Europe-PMC OA-first harvest, spec §7) is designed but unplanned. See Open work #B.

## Open work (in priority order)

### A. EM replication — finish the manuscript

1. **Sensitivity analysis (owner-gated data work — expensive model re-run).** The
   `--sensitivity` flag on `compute_phase6_kappa.py` is **now built + unit-tested**
   (this session): it excludes only `UNRECOVERABLE_WRONG_PAPER_RCTS`, writes to
   `phase6_results.sensitivity.{md,csv}` (composes with `--exclude-fallback` →
   `.sensitivity.strict.*`), and **refuses to run** until every
   `RECOVERABLE_WRONG_PAPER_RCTS` carries the recovery marker in
   `benchmark_rct.notes` — so it cannot silently score stale wrong-document
   judgements. What remains is the owner-gated data pipeline: recover the ~11
   obtainable RCTs, re-assess only the deleted rows, THEN run the flag:
   ```bash
   # 1. recover (surgical: re-fetch correct doc, update benchmark_rct, delete stale
   #    MODEL rows only; backs up the DB first). Dry-run first, then --apply.
   uv run python studies/eisele_metzger_replication/recover_wrong_papers.py apply \
     RCT008 RCT009 RCT019 RCT040 RCT064 RCT074 RCT095 RCT100 \
     RCT017=31968595 RCT088=32871238 RCT093=34800427 --apply
   # 2. re-assess ONLY the deleted rows (existence check skips the rest):
   uv run python studies/eisele_metzger_replication/run_evaluation.py --model gpt_oss_20b --protocol abstract   # ×{abstract,fulltext}×{gpt_oss_20b,gemma4_26b,qwen3_6_35b}
   uv run python studies/eisele_metzger_replication/run_evaluation_anthropic.py --protocol abstract            # Sonnet ×{abstract,fulltext}
   # 3. compute the sensitivity κ (the guard confirms the DB is recovered first):
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py --sensitivity
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py --sensitivity --exclude-fallback  # strict variant
   ```
2. ~~**Pre-manuscript spot-checks**~~ **DONE 2026-07-18** — `premanuscript_spotchecks.py`
   (+ `premanuscript_spotcheck_results.md` / `premanuscript_instability.csv`; tests in
   `tests/test_premanuscript_spotchecks.py`). Sonnet `low` audit and full-corpus per-domain
   instability audit computed in code; primary draft §3.5 (Sonnet audit added), §3.6
   (reframed — D1 concentration does not generalise), and §5 Limitations (four-model pass)
   updated. Only remaining piece is the owner's final read-through — folded into #A.3.
3. **Final prose pass on both drafts** — the n=78 numbers and reframed finding #1 are in
   (PR #31), Figure 1 is now wired into §3.2 (this session), but the owner still wants a
   read-through before submission.
4. ~~**Forest-plot figure**~~ **DONE this session** — Figure 1 (κ_quad vs Cochrane, all
   four models × both protocols × three passes + ensembles, bootstrap 95% CIs at
   quadratic weighting) wired into primary draft §3.2. Remaining stretch items: a
   confidence-calibrated ensemble as a future-work appendix (primary use would need a
   pre-reg amendment); OSF mirror of the pre-reg.

### B. OA-first benchmark — owner actions before Stage B

These are **owner tasks, not code** (network I/O over the full cohort, kept out of CI
per the >2-min rule):
1. **Confirm ROBoto2 license reuse terms** with its authors before *publishing* any
   ROBoto2-derived rows (spec risk R1). Raw CSV + derived JSON are gitignored until then.
2. **Run the two terminal ingests** from your own terminal:
   `uv run python -m studies.oa_rob_benchmark.convert_roboto2_csv` (CSV → JSON) then
   `... ingest_roboto2` and `... ingest_em_candidates`.
3. **Review the 20-row manual-gate manifest** (`manual_gate.py`; ≥19/20 must match)
   before Stage B is planned.
4. **Stage B** — the fresh Europe-PMC OA-first harvest (spec §7) — is a separate
   follow-up, unplanned until the gate passes.

### C. Decisions needed from the repo owner (EM study)

- Personal review of the 5 systematic-failure RCTs (RCT024, RCT025, RCT034, RCT038,
  RCT062) before publication.
- OpenAthens full-text fetch for the ~50 PMID-but-no-fulltext RCTs (would lift native
  full-text from 41/100 to ~85+/100; ~30 min manual work) — optional polish, decide
  once final numbers are in.
- Submission ordering — recommendation: harness-vs-naive first, then companion.
  Confirm before posting either.
- Sign-off on the `docs/papers/drafts/medrxiv_V5/` Cochrane corpus rebuild design
  (separate from both active workstreams; forensics in `FORENSICS.md`).

### D. Housekeeping (no issue filed)

- `dataset/` still contains two stray `* copy.db` files
  (`eisele_metzger_benchmark copy.db`, `eisele_metzger_benchmark.spark copy.db`) —
  confirm with the owner they are disposable before removing.
- Only **one** GitHub issue is open: **#29** (RCT030/wrong-paper κ pollution). The code
  fix + drafts are done (PRs #30, #31); it stays open to track the owner-gated
  sensitivity analysis (Open work #A.1). Close it once the sensitivity κ ships or the
  owner decides to skip it.

## Conventions and gotchas

- Run the suite with `uv run pytest` (`testpaths = ["tests"]` keeps collection away
  from stray root `worktrees/` checkouts). CI now runs it on every push (PR #34); a
  weekly workflow refreshes the lockfile (#36).
- **Do not modify the locked pre-analysis plan or prompt spec** at `7854a1c` — any
  change requires a numbered amendment (§12 of the pre-reg).
- **Never re-run `build_benchmark_db.py`** after evaluation rows exist — it DROPs and
  rebuilds all tables, destroying the model evaluation data. Copy data out first if
  the schema must change.
- **The OA benchmark store is isolated** — `studies/oa_rob_benchmark/store.py` writes
  only `dataset/oa_rob_benchmark.db` (append/upsert, never DROP) and must never touch
  `biasbuster.db` or the EM benchmark DB. `upsert_item` preserves human curation
  (`manual_verified`) on conflict via `_CURATION_FIELDS`.
- **Never commit `DATA/`** (EM 2025 supplementary data — redistribution unresolved) or
  the raw/derived ROBoto2 files (license R1 unconfirmed); all are `.gitignore`d.
- **Do not re-run the full Sonnet batch** — $30–80 of API credits; results are in the DB.
- Do not push to branches other than feature branches + PR to `main` without
  confirming with the owner.
- Two near-identical filenames in `biasbuster/methodologies/cochrane_rob2/`:
  `algorithm.py` (consistency checking) vs `algorithms.py` (per-domain truth tables).
  Both carry cross-reference notes — mind which one you touch.
- **Wrong-paper acquisitions are a class of 13, not just RCT030** — all 13 are now
  enforced in `exclusions.WRONG_PAPER_RCTS` and excluded from every κ loader. Before any
  further κ regeneration, re-run `audit_wrong_paper_acquisitions.py` and get owner
  sign-off if the set changes. Parse-failure rows for wrong papers must always stay out
  of recovery (`recover_parse_failures.py` guards via `WRONG_PAPER_RCTS`).
- **Strict-mode phase-6 outputs lag the primary schema by one generation**:
  `phase6_results.strict.{csv,md}` and `phase6_forest_data.strict.csv` are git-tracked but
  predate the quadratic-CI addition — they lack `ci_quad_lo`/`ci_quad_hi` because only the
  primary (non-strict) outputs were regenerated this session. Regenerate deliberately with
  `compute_phase6_kappa.py --exclude-fallback` before relying on strict-mode CIs, so the
  diff isn't a surprise mix of schema catch-up and real content changes.
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
| EM study scripts (phases 1–6) | `studies/eisele_metzger_replication/` |
| Cross-model κ report | `studies/eisele_metzger_replication/compute_phase6_kappa.py` → `phase6_results*.{md,csv}` |
| Wrong-paper exclusion set / audit / recovery | `studies/eisele_metzger_replication/{exclusions,audit_wrong_paper_acquisitions,recover_wrong_papers}.py`; obtainability at `recovery_obtainability.md` |
| Shard merge (Mac ⇄ DGX) | `studies/eisele_metzger_replication/merge_eval_dbs.py` |
| Per-domain Cochrane algorithms | `biasbuster/methodologies/cochrane_rob2/algorithms.py` |
| EM benchmark DB (gitignored) | `dataset/eisele_metzger_benchmark.db` (+ `.spark.db` shard) |
| OA benchmark study package | `studies/oa_rob_benchmark/` (spec + plan under `docs/superpowers/{specs,plans}/2026-07-17-oa-*`) |
| OA benchmark store (gitignored DB) | `dataset/oa_rob_benchmark.db`; ROBoto2 raw/derived under `dataset/roboto2*` (gitignored) |
| EM 2025 source data (gitignored, no redistribution) | `DATA/20240318_Data_for_analysis_full/` |
| COI divergence rationale (standing design) | `docs/harness/DESIGN_RATIONALE_COI.md` |

### Quick state check (EM benchmark)

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
