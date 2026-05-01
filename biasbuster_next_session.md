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
