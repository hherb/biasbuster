# BiasBuster — next session runbook

**Written:** 2026-05-01 end-of-session
**Author:** Claude (Opus 4.7) at the request of Horst Herb
**Purpose:** hand-off to the next Claude session so progress isn't lost. The Eisele-Metzger 2025 replication study is mid-Phase-5, with two of four models complete and two still running. This document is the runbook to pick up cleanly.

---

## TL;DR for the next-session Claude

You are continuing the **Eisele-Metzger 2025 RoB 2 replication study**. Read these *first*, in order:

1. **Memory file:** `~/.claude/projects/-Users-hherb-src-biasbuster/memory/project_claude2_rob_paper.md` — the core context, citation, replication plan, and current strategy.
2. **Pre-registration (LOCKED):** [`docs/papers/eisele_metzger_replication/preanalysis_plan.md`](docs/papers/eisele_metzger_replication/preanalysis_plan.md) at commit `7854a1c` (2026-04-30). Methodology is locked; changes require a numbered amendment.
3. **Locked prompt:** [`docs/papers/eisele_metzger_replication/prompt_v1.md`](docs/papers/eisele_metzger_replication/prompt_v1.md) — same lock hash.
4. **Primary preprint draft:** [`docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`](docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md) — core thesis: harness over model.
5. **Companion preprint draft:** [`docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md`](docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md) — separable thesis: AI follows algorithm; experts deviate.
6. **Latest results:** [`studies/eisele_metzger_replication/phase6_results.md`](studies/eisele_metzger_replication/phase6_results.md).

**Pre-registration publishability gate (§9 of pre-reg) is already cleared** for both gpt-oss:20b and Sonnet 4.6. The study has its empirical anchor; gemma + qwen will round it out, not change the conclusion.

---

## ⚠️ KNOWN BUGS — found in 2026-07-16 code review (READ BEFORE FINALIZING MANUSCRIPT)

A logical/flow-and-bug review on 2026-07-16 (four parallel reviewers + spot-verification)
found the following. Nothing here is fixed yet — the review was scoped to *find*, not fix.
The first two are **paper-critical**: they affect the headline κ numbers and the pre-registered
"model-emitted vs algorithm-derived" distinction the whole study rests on. Address these before
posting either preprint.

### Paper-critical (evaluation / statistics)

1. **Live-path algorithmic fallback is stored UNTAGGED as a genuine model judgement.**
   [`eval_ollama.py:315-327`](studies/eisele_metzger_replication/eval_ollama.py:315) derives a
   domain judgement via `derive_domain_judgement()` when the model omits the `judgement` field,
   and [`run_evaluation.py:145-154`](studies/eisele_metzger_replication/run_evaluation.py:145)
   ingests it with `valid=1` and `raw_label = result.judgment` (the derived value) — i.e. it is
   indistinguishable from a model-emitted label. Same for Sonnet at
   [`run_evaluation_anthropic.py:431-434 → ingest ~467`](studies/eisele_metzger_replication/run_evaluation_anthropic.py:431).
   This directly contradicts `recover_parse_failures.py`, which tags the *identical* derivation as
   `raw_label='FALLBACK'` precisely so it can be excluded. **Consequence:** any *new* run (the
   gemma/qwen passes that were still running) silently contaminates the pre-registered
   "model-emitted judgement" primary arm with algorithm-derived labels. Fix: tag inline-derived
   judgements as `FALLBACK` at ingest, matching the post-hoc recovery script.

2. **The κ scripts never exclude `FALLBACK` rows**, so every reported κ mixes model-emitted and
   algorithm-derived labels — and there is no code path to reproduce the pre-registered primary
   metric. [`compute_phase6_kappa.py:75-97`](studies/eisele_metzger_replication/compute_phase6_kappa.py:75)
   (`load_pairs`/`load_judgments`) and
   [`interim_analysis.py`](studies/eisele_metzger_replication/interim_analysis.py) filter only on
   `valid = 1`. `recover_parse_failures.py`'s own docstring says downstream analysis should filter
   `raw_label != 'FALLBACK'` "to reproduce the original primary metric" — but no script offers that
   filter. Fix: add a `--strict/--exclude-fallback` option (and report both strict and inclusive κ).

3. **Ensemble "overall" κ is a direct majority vote, but the report claims worst-wins synthesis.**
   [`compute_phase6_kappa.py:132-142`](studies/eisele_metzger_replication/compute_phase6_kappa.py:132)
   majority-votes the `overall` label directly (DOMAINS includes `"overall"`), yet the Section 3
   markdown at line 464 says "Per-domain majority vote … then worst-wins synthesis." The printed
   number does not match its own description. Decide which you want and make text + code agree.

4. **The §8 parse-failure halt aborts only the current pass; later passes still run.**
   [`run_evaluation.py:286-289`](studies/eisele_metzger_replication/run_evaluation.py:286) returns
   from `run_one_pass` on halt, but `main` (lines 357-363) loops over passes without checking the
   halt — passes 2 and 3 run at the same broken failure rate. Defeats the "halt and revise prompts"
   rule and wastes a full local-model run.

### Production BiasBuster code (not the paper, but wrong RoB output)

5. **`some_concerns` is hardcoded when the LLM omits `judgement` but signalling answers are present.**
   [`assessor.py:213-221`](biasbuster/methodologies/cochrane_rob2/assessor.py:213) stamps
   `some_concerns` unconditionally, citing "Cochrane's ambiguity rule" — but that rule only applies
   when answers match *neither* the low nor high pattern. With answers present, the domain truth
   table should decide. E.g. D1 `1.3=Y/PY` (baseline imbalance) or D4 `4.1=Y` → algorithm says
   **high**, but this records **some_concerns**, and worst-wins can then downgrade the whole paper.
   The correct function (`algorithms.derive_domain_judgement`) already exists but is never called here.

6. **The per-domain consistency check is a no-op**, so the LLM's `judgement` is trusted verbatim.
   [`algorithm.py:107-118`](biasbuster/methodologies/cochrane_rob2/algorithm.py:107)
   `domain_judgement_is_consistent` always returns True; [`schema.py:80-83`](biasbuster/methodologies/cochrane_rob2/schema.py:80)
   says the per-domain judgement "must be reproducible from the signalling inputs alone." The
   truth-table functions in `algorithms.py` (plural) are **dead code** — nothing imports them except
   the study's fallback path. Two near-identically-named files (`algorithm.py` vs `algorithms.py`) is
   itself a trap: a future dev can wire in the wrong one silently. (Also: `algorithm.py:5` docstring
   references a function `aggregate_domain` that does not exist.)

7. **`upsert_cochrane_paper` overwrites all five domain ratings unconditionally** while the review-
   metadata fields right below are COALESCE/NULLIF-guarded.
   [`database.py:713-719`](biasbuster/database.py:713). `collect-rob` is designed to be re-run; if a
   later run yields an empty/None domain (extraction dropped it), it blanks a previously-stored expert
   rating. Matches the documented "domains are always authoritative" intent, so *verify intent* — but
   a `COALESCE(NULLIF(excluded.x,''), papers.x)` guard would prevent empty-value blanking with no
   downside.

8. **Multi-model export can leak the same abstract across train/test.**
   [`export.py:874-910`](export.py:874) `_stratified_split` splits per converted example with no
   PMID-level grouping; [`pipeline.py:637`](biasbuster/pipeline.py:637) exports all models by default
   (`export_model=None`). A PMID annotated by both anthropic and deepseek yields two examples with
   identical abstract text that can land in different splits. Single-model export is unaffected. Fix:
   group-by-PMID before splitting, or dedup abstract text across splits.

### Collectors / annotators

9. **Retracted papers are dropped even when `require_abstract=False`.**
   [`retraction_watch.py:383-392`](biasbuster/collectors/retraction_watch.py:383) appends a paper only
   inside `if paper.pmid in abstract_data:`, so papers with no PMID or not returned by PubMed are
   discarded regardless of the flag. **Latent** today — the only caller
   ([`pipeline.py:68`](biasbuster/pipeline.py:68)) uses the default `True` — but the flag is dead in
   the `False` case.

10. **`is_retraction_notice` misses long notices with an original-looking title.**
    [`annotators/__init__.py:88-116`](biasbuster/annotators/__init__.py:88): if the title doesn't match
    a retraction pattern, the abstract is only checked when it is `< 200` chars. A >200-char abstract
    beginning "This article has been retracted at the request of…" with a normal title slips through and
    gets sent for bias annotation.

11. **Reversed JSON-array search can pick a trailing `[5]` citation → zero studies for the chunk.**
    [`cochrane_rob.py:777-793`](biasbuster/collectors/cochrane_rob.py:777): iterating array matches in
    reverse tries a trailing bracketed citation first; `json.loads("[5]")` succeeds, the real studies
    array is discarded, and the chunk silently contributes zero RoB assessments.

### Low severity / latent

12. `seed_database.py:367` — `ZeroDivisionError` in `print_summary` when the papers table is empty.
13. `merge_eval_dbs.py:63-78` — `INSERT OR IGNORE` gives no warning if two shards wrote the same
    `(rct_id, source, domain)` with *different* data (silent divergence).
14. `compute_phase6_kappa.py:172-203` — `mcnemar_test` is dead code and, if wired up, zips two
    independently-ordered query results assuming matching row order (SQLite gives no such guarantee).
15. `database.py:1289` / `pipeline.py:863-867` — unbatched `pmid IN (?, …)` can exceed SQLite's
    parameter limit on very large undetectable-paper sets (only bites on SQLite < 3.32).

### Documentation

16. **`CLAUDE.md`'s Architecture section is stale.** It describes a flat root layout (`collectors/`,
    `annotators/`, `database.py`, `pipeline.py`) that no longer exists — all code is under the
    `biasbuster/` package now. This actively misdirects (it cost this review a detour). Worth a refresh.

### Manuscript regeneration gate — do NOT gate on fixing all 15 issues

We deliberately do **not** need every issue fixed before regenerating the κ tables. Most of the
open findings do not touch the Eisele-Metzger study numbers at all:

- **Flow into the reported κ tables:** the two already-fixed bugs (1, 2), plus
  **#7** (ensemble-overall is a direct majority vote but the report text says "worst-wins" — that
  is a *reported figure*, so fix it before regenerating). **#8** (parse-failure halt) matters only
  if you *re-run* evaluation; the existing gpt-oss/Sonnet runs stand as-is.
- **Independent of the study numbers** (fix on their own timeline, not blocking the paper):
  #9/#10/#11 are the production `assessor.py` RoB path — the study uses `eval_ollama` +
  `algorithms.py`, not that path. #12 is training-data export. #13/#14 are the dataset-builder
  collectors. #16–#19 are low-severity util/scaling issues.

**Regeneration sequence (the actual gate):**

1. `uv run python studies/eisele_metzger_replication/retro_tag_live_fallback.py --dry-run`
   then `--apply` against the canonical DB (and any merged shard) — back-tags rows written by the
   *old* untagged live path before the Bug-1 fix. Without this, gemma/qwen rows already written by
   the old path stay mislabeled as model-emitted even though the code is now fixed.
2. Fix **#7** so the ensemble-overall number matches its description.
3. Regenerate in **both** modes and report the strict (pre-registered primary) numbers:
   ```bash
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py                     # inclusive
   uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py --exclude-fallback  # strict → *.strict.{md,csv}
   ```
4. Update both preprint drafts' §3 tables + abstract with the strict primary κ (and note the
   inclusive numbers as a sensitivity analysis).

### Housekeeping note (unrelated to this work)

- GitHub flagged **33 Dependabot vulnerabilities** on the default branch during the push
  (12 high, 13 moderate, 8 low): https://github.com/hherb/biasbuster/security/dependabot .
  Not caused by this branch — worth a triage pass when convenient.
- Review fixes + issues live on PR [#20](https://github.com/hherb/biasbuster/pull/20)
  (branch `claude/project-review-bugs-aad20e`). Remaining findings tracked as issues
  [#7–#19](https://github.com/hherb/biasbuster/issues) labeled `audit-2026-07`.

---

## State at end of last session

### Phase 5 evaluation matrix

| Model | Abstract × 3 passes | Fulltext × 3 passes | Notes |
|---|:-:|:-:|---|
| **gpt-oss:20b** | ✅ complete | ✅ complete | Best-pass κ_quad = 0.257 (matches EM 0.22). Run-to-run κ_quad = 0.441. |
| **Claude Sonnet 4.6** | ✅ complete | ✅ complete | Best-pass κ_quad = 0.264. Run-to-run κ_quad = 0.768 (1.83× Minozzi 2021). |
| **gemma4:26b-a4b-it-q8_0** | ⏳ running on **Spark DGX** | ⏳ running on **Spark DGX** | Writes to `dataset/eisele_metzger_benchmark.spark.db` on the DGX |
| **qwen3.6:35b-a3b-q8_0** | ⏳ running on **Mac** (~22/91 RCTs done last check) | ⏳ pending Mac | Writes to canonical `dataset/eisele_metzger_benchmark.db` |

### Recovery infrastructure (already applied)

- 48 domain rows recovered via algorithmic fallback (Cochrane per-domain rules applied to model-emitted signalling answers when explicit `judgement` field missing).
- 45 synthesis rows derived post-hoc via worst-wins from recovered domains.
- 93 total rows tagged `raw_label='FALLBACK'` in `benchmark_judgment` for sensitivity-analysis filtering.
- 15 unrecoverable rows = RCT030 from gpt-oss only (wrong-paper acquisition; correctly cannot be recovered).
- **Live-path fallback** is now in `eval_ollama.parse_response` (commit `3f1d78d`), so any **new** runs (gemma, remaining qwen passes) will auto-recover schema-drift cases without needing the post-hoc script.

### Recent commit chain (most recent first)

```
ee0aa76 docs(papers): update companion draft §3.5 with Sonnet 4.6 results
c5986fe docs(papers): update preprint with post-recovery numbers + sensitivity analysis
3f1d78d feat(studies): live-path algorithmic fallback in parse_response
b95584b feat(cochrane_rob2): per-domain decision algorithms + parse-failure recovery
3e772a2 docs(papers): integrate Phase 6 findings into harness-vs-naive draft
8d5286f fix(studies): use hyphen separator in Sonnet batch custom_id
bedbefc feat(studies): Phase 5.8 Anthropic Sonnet runner via Batch API
33bbc59 feat(studies): Phase 6 cross-model comparison + ensemble + forest data
```

---

## Immediate actions when this session resumes (in order)

### 1. Check whether gemma and qwen completed

```bash
# Check qwen progress (Mac canonical DB)
uv run python studies/eisele_metzger_replication/interim_analysis.py \
    --model qwen3_6_35b --protocol abstract
uv run python studies/eisele_metzger_replication/interim_analysis.py \
    --model qwen3_6_35b --protocol fulltext

# If user has a Spark shard ready, ask them to rsync it back:
#   rsync -av spark.local:~/src/biasbuster/dataset/eisele_metzger_benchmark.spark.db \
#       ~/src/biasbuster/dataset/
# Then inspect:
uv run python studies/eisele_metzger_replication/merge_eval_dbs.py \
    --dest dataset/eisele_metzger_benchmark.db \
    --source dataset/eisele_metzger_benchmark.spark.db \
    --show-only
```

### 2. Merge spark shard (if present)

```bash
uv run python studies/eisele_metzger_replication/merge_eval_dbs.py \
    --dest dataset/eisele_metzger_benchmark.db \
    --source dataset/eisele_metzger_benchmark.spark.db
```

Default is `INSERT OR IGNORE`. Mac's qwen rows and Spark's gemma rows have disjoint source labels → no collision.

### 3. Apply recovery to any new schema drift

The live-path fallback handles drift on calls made *after* commit `3f1d78d`, but gemma may have started earlier. Re-run the recovery script to be safe:

```bash
uv run python studies/eisele_metzger_replication/recover_parse_failures.py --dry-run
# If anything new is recoverable:
uv run python studies/eisele_metzger_replication/recover_parse_failures.py
```

### 4. Refresh Phase 6 cross-model table

```bash
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py
```

This regenerates `phase6_results.md`, `phase6_results.csv`, `phase6_forest_data.csv`. Includes the ensemble-of-3 majority-vote computation automatically (gemma and qwen will get their ensemble rows added now that 3 passes are present).

### 5. Update both preprint drafts with final numbers

- **Primary draft**: [`docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`](docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md)
  - §3.1 Coverage table: add gemma + qwen rows
  - §3.2 κ vs Cochrane table: add gemma + qwen × {abstract, fulltext} × {1,2,3} = 12 rows
  - §3.3 Run-to-run κ table: add gemma and qwen rows
  - §3.7 Ensemble table: add gemma and qwen ensemble rows
  - §3.8 Conservatism: check whether gemma and qwen show the same systematic conservatism pattern
  - Abstract: update best-pass numbers if a new model takes the lead (qwen at n=22 was at κ_quad = 0.316 — could change the abstract headline)
  - Conclusion §6: drop the "(pending: gemma4 and qwen3.6)" caveat

- **Companion draft**: [`docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md`](docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md)
  - §3.5 update with the dual-model-now-quad-model picture
  - §4 "Harness over model" paragraph — strengthen with 4-model evidence

### 6. Final spot-checks before declaring "manuscript-ready"

1. **Right-for-the-right-reasons audit on Sonnet's `low` judgments.** We did this for gpt-oss (7/8 correct, §3.5 of the primary draft). Sonnet has even fewer `low` calls — 1–2 per pass — but worth eyeballing the rationales of any RCT where Sonnet said `low` and Cochrane disagreed (or vice versa).
2. **Coverage of §3.6 (D1 instability).** We documented this for gpt-oss. Check whether Sonnet shows the same D1 noise pattern in its run-to-run disagreements.
3. **Limitations section.** User signed off but might want one more pass once all four models are in.

---

## Decision points (need user input)

- **OpenAthens full-text fetch for the 50 PMID-but-no-fulltext RCTs.** Currently abstract-fallback under the FULLTEXT protocol. User's institutional access could push native full-text from 41/100 → ~85+/100. **Cost:** ~25–30 min of manual paywall-clicking. **Benefit:** cleaner subgroup analysis (jats_xml vs abstract_fallback at §6.5 of pre-reg). **Recommend asking the user once final numbers are in** — if the qualitative conclusion is robust, the OpenAthens fetch is optional polish.

- **Submission ordering.** Pre-reg recommendation (in both drafts' §"Open questions"): submit harness-vs-naive (`20260501_*`) first, then assessor-algorithm-conformance (`20260423_*`). Confirm with user before posting either.

- **medRxiv co-author / disclosure language.** We landed on "human-only authorship + detailed AI-use disclosure in Methods" per medRxiv policy (memory file references the verbatim policy text at `docs/literature/rob_validation/medrxiv_ai_policy_2026-04-30.md`). User indicated this works. The disclosure paragraph itself is at §11 of the pre-reg.

- **5 systematic-failure RCTs flagged for personal review** (RCT024, RCT025, RCT034, RCT038, RCT062). User said they'd review personally. If they haven't yet, prompt them. None of the 5 look suspicious about the underlying papers — just schema drift on Sonnet's part — but a quick eyeball before publication is sensible.

---

## Stretch goals (only if results are unequivocal and time permits)

1. **Forest-plot figure** for the manuscript. `phase6_forest_data.csv` already has the exact data shape needed. matplotlib + nothing fancy. Should slot into the primary draft as Figure 1.

2. **Run a confidence-calibrated ensemble** (instead of naive majority vote, weight passes by some reliability proxy). We documented in §3.7 that naive ensemble loses to best single pass; a calibrated approach might do better. Methodologically this would be a "future work" appendix rather than primary results — adding it as a primary metric *would* require a pre-reg amendment.

3. **OpenAthens fetch + subgroup analysis** as described above. Cleanest version of §6.5 of the pre-reg.

4. **Pre-register the analysis plan on OSF** before posting the medRxiv preprint. Currently locked in git history (commit hash `7854a1c`); OSF mirror would be more discoverable for reviewers. Optional but recommended.

---

## Key data and code locations

| Artefact | Path |
|---|---|
| Locked pre-analysis plan | [`docs/papers/eisele_metzger_replication/preanalysis_plan.md`](docs/papers/eisele_metzger_replication/preanalysis_plan.md) |
| Locked prompt spec | [`docs/papers/eisele_metzger_replication/prompt_v1.md`](docs/papers/eisele_metzger_replication/prompt_v1.md) |
| Cost estimate (Sonnet) | [`docs/papers/eisele_metzger_replication/cost_estimate.md`](docs/papers/eisele_metzger_replication/cost_estimate.md) |
| Primary preprint draft | [`docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`](docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md) |
| Companion preprint draft | [`docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md`](docs/papers/drafts/20260423_medrxiv_assessor_algorithm_conformance_v1.md) |
| medRxiv policy snapshot | [`docs/literature/rob_validation/medrxiv_ai_policy_2026-04-30.md`](docs/literature/rob_validation/medrxiv_ai_policy_2026-04-30.md) |
| Literature kappa benchmark table | [`docs/literature/rob_validation/benchmark_kappa_table.md`](docs/literature/rob_validation/benchmark_kappa_table.md) |
| Per-domain Cochrane algorithms | [`biasbuster/methodologies/cochrane_rob2/algorithms.py`](biasbuster/methodologies/cochrane_rob2/algorithms.py) |
| Locked per-domain prompts | [`biasbuster/methodologies/cochrane_rob2/prompts.py`](biasbuster/methodologies/cochrane_rob2/prompts.py) |
| Phase 1 acquisition script | [`studies/eisele_metzger_replication/acquire_fulltext.py`](studies/eisele_metzger_replication/acquire_fulltext.py) |
| Phase 2 contamination check | [`studies/eisele_metzger_replication/contamination_check.py`](studies/eisele_metzger_replication/contamination_check.py) |
| Phase 3 benchmark DB build | [`studies/eisele_metzger_replication/build_benchmark_db.py`](studies/eisele_metzger_replication/build_benchmark_db.py) |
| Phase 4 sanity-check (κ ≈ 0.22) | [`studies/eisele_metzger_replication/sanity_check_kappa.py`](studies/eisele_metzger_replication/sanity_check_kappa.py) |
| Phase 5 Ollama runner | [`studies/eisele_metzger_replication/run_evaluation.py`](studies/eisele_metzger_replication/run_evaluation.py) |
| Phase 5 Anthropic runner | [`studies/eisele_metzger_replication/run_evaluation_anthropic.py`](studies/eisele_metzger_replication/run_evaluation_anthropic.py) |
| Multi-host shard merger | [`studies/eisele_metzger_replication/merge_eval_dbs.py`](studies/eisele_metzger_replication/merge_eval_dbs.py) |
| Parse-failure recovery | [`studies/eisele_metzger_replication/recover_parse_failures.py`](studies/eisele_metzger_replication/recover_parse_failures.py) |
| Phase 6 cross-model comparison | [`studies/eisele_metzger_replication/compute_phase6_kappa.py`](studies/eisele_metzger_replication/compute_phase6_kappa.py) |
| Interim analysis (per-model) | [`studies/eisele_metzger_replication/interim_analysis.py`](studies/eisele_metzger_replication/interim_analysis.py) |
| Latest Phase 6 results | [`studies/eisele_metzger_replication/phase6_results.md`](studies/eisele_metzger_replication/phase6_results.md) |
| Forest-plot data CSV | [`studies/eisele_metzger_replication/phase6_forest_data.csv`](studies/eisele_metzger_replication/phase6_forest_data.csv) |
| Benchmark DB (gitignored) | `dataset/eisele_metzger_benchmark.db` |
| EM 2025 source CSVs (gitignored, redistribution-restricted) | `DATA/20240318_Data_for_analysis_full/` |
| Acquired full text (gitignored) | `DATA/20240318_Data_for_analysis_full/fulltext/{rct_id}/` |

---

## Useful command snippets

### Quick state check

```bash
sqlite3 dataset/eisele_metzger_benchmark.db <<'SQL'
.mode column
.headers on
SELECT source, COUNT(*) AS n_judgments,
       SUM(CASE WHEN valid=1 THEN 1 ELSE 0 END) AS n_valid,
       SUM(CASE WHEN raw_label='FALLBACK' THEN 1 ELSE 0 END) AS n_fallback
FROM benchmark_judgment
WHERE source NOT LIKE 'cochrane' AND source NOT LIKE 'em_claude2_%'
GROUP BY source ORDER BY source;
SQL
```

### Per-model interim analysis

```bash
for model in gpt_oss_20b sonnet_4_6 gemma4_26b qwen3_6_35b; do
  for protocol in abstract fulltext; do
    echo "=== $model × $protocol ==="
    uv run python studies/eisele_metzger_replication/interim_analysis.py \
        --model "$model" --protocol "$protocol" 2>&1 | head -40
  done
done
```

### Sensitivity check (strict-parse only, no FALLBACK)

```bash
sqlite3 dataset/eisele_metzger_benchmark.db <<'SQL'
SELECT source, COUNT(*) AS n_strict
FROM benchmark_judgment
WHERE valid = 1 AND (raw_label IS NULL OR raw_label != 'FALLBACK')
  AND domain = 'overall' AND source LIKE 'sonnet_4_6_%'
GROUP BY source ORDER BY source;
SQL
```

---

## Open todo list (carry forward)

**Bug fixes from 2026-07-16 review (see "⚠️ KNOWN BUGS" section above):**
- [x] **BUG 1 — FIXED (2026-07-16):** inline algorithmic-fallback judgements are now tagged
      `raw_label='FALLBACK'` at ingest in both runners (`eval_ollama.parse_response` returns an
      `is_fallback` flag; `run_evaluation.py` + `run_evaluation_anthropic.py` persist it, matching
      `recover_parse_failures.py`). **Action still needed:** run the new
      `studies/eisele_metzger_replication/retro_tag_live_fallback.py --apply` against the canonical
      DB (and any shard) to back-tag rows written by the *old* untagged live path before this fix.
- [x] **BUG 2 — FIXED (2026-07-16):** `compute_phase6_kappa.py` and `interim_analysis.py` now
      accept `--exclude-fallback`; strict mode writes to `phase6_results.strict.{md,csv}` and the
      report header states which mode produced the numbers. **Action still needed:** regenerate the
      manuscript κ tables in both modes and report the strict (pre-registered primary) numbers.
- [x] **DOC — FIXED (2026-07-16):** CLAUDE.md now has a "Repository layout" note (code is under
      `biasbuster/`; `config.py` stays at repo root; correct `python -m biasbuster.pipeline`
      invocation).
- [ ] BUG 3 ([#7](https://github.com/hherb/biasbuster/issues/7)): reconcile ensemble-overall
      computation (majority vote) with the report text (worst-wins)
- [ ] BUG 4 ([#8](https://github.com/hherb/biasbuster/issues/8)): make the §8 parse-failure halt
      actually stop subsequent passes
- [ ] BUG 5 ([#9](https://github.com/hherb/biasbuster/issues/9)): call `derive_domain_judgement`
      instead of hardcoding `some_concerns` in `assessor.py`
- [ ] BUG 6 ([#10](https://github.com/hherb/biasbuster/issues/10)): implement
      `domain_judgement_is_consistent` / resolve the `algorithm.py` vs `algorithms.py` trap
- [ ] BUG 7 ([#11](https://github.com/hherb/biasbuster/issues/11)): verify intent + NULLIF guard on
      `upsert_cochrane_paper` domain fields
- [ ] BUG 8 ([#12](https://github.com/hherb/biasbuster/issues/12)): group-by-PMID before
      `_stratified_split` to stop multi-model train/test leakage
- [ ] BUG 9 ([#13](https://github.com/hherb/biasbuster/issues/13)): retraction-paper drop with
      `require_abstract=False`
- [ ] BUG 10 ([#14](https://github.com/hherb/biasbuster/issues/14)): `is_retraction_notice`
      long-notice gap
- [ ] BUG 11 ([#15](https://github.com/hherb/biasbuster/issues/15)): cochrane `[5]`-citation JSON pick
- [ ] BUG 12 ([#16](https://github.com/hherb/biasbuster/issues/16)): empty-DB div-by-zero in
      `seed_database.py`
- [ ] BUG 13 ([#17](https://github.com/hherb/biasbuster/issues/17)): `merge_eval_dbs` collision
      warning
- [ ] BUG 14 ([#18](https://github.com/hherb/biasbuster/issues/18)): dead/misaligned `mcnemar_test`
- [ ] BUG 15 ([#19](https://github.com/hherb/biasbuster/issues/19)): unbatched `IN` parameter limit

**Original study carry-forward:**
- [ ] Wait for gemma4 and qwen3.6 evaluations to complete on respective hosts
- [ ] Merge Spark shard back to canonical DB (when DGX run done)
- [ ] Re-run `recover_parse_failures.py` against the post-merge DB (catches anything new)
- [ ] Re-run `compute_phase6_kappa.py` for the final cross-model table
- [ ] Update both preprint drafts (§3 tables, abstract, conclusion) with final 4-model picture
- [ ] User: review the 5 systematic-failure RCTs (RCT024, RCT025, RCT034, RCT038, RCT062)
- [ ] User: decide on OpenAthens fetch for full-text ceiling lift (optional)
- [ ] User: confirm submission ordering (recommend harness-vs-naive first)
- [ ] Stretch: forest-plot figure from `phase6_forest_data.csv`
- [ ] Stretch: confidence-calibrated ensemble as future-work appendix
- [ ] Stretch: OSF pre-registration mirror

---

## Things to *not* do without user permission

- **Do not modify the locked pre-analysis plan or prompt spec** at commit `7854a1c`. Any change requires a numbered amendment with the original preserved in git history (per §12 of the pre-reg).
- **Do not regenerate the benchmark DB** with `build_benchmark_db.py` after Phase 5 evaluation rows have been written — that script DROPs and rebuilds the tables, including the `evaluation_run` rows. The user has the source CSVs gitignored locally; rebuilding would lose all model evaluation data. (If the schema needs to change, copy the existing data out first.)
- **Do not commit the `DATA/` folder** — gitignored at `.gitignore:53` because EM 2025 supplementary data redistribution rights are unresolved.
- **Do not run the full Sonnet batch a second time** — costs $30–$80 of API credits unnecessarily. Existing results are in the DB.
- **Do not push to a branch** other than `main` without confirming with the user.

---

*End of runbook. Pick up at "Immediate actions when this session resumes" and you'll be productive within minutes.*
