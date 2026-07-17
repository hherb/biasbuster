# Roadmap

This document tracks planned and implemented work for BiasBuster, grouped by the
three project objectives (see README / CLAUDE.md). Keep entries to one line each;
current session-level detail lives in HANDOVER.md, historical records in
`docs/history/`.

| Status | Feature | Details |
|--------|---------|---------|
| **Objective 1 — Dataset curation + fine-tuning pipeline** | | |
| ✅ Done | Multi-source collection | Crossref/Retraction Watch retracted papers, PubMed RCTs by MeSH domain, Cochrane RoB expert assessments via Europe PMC (`biasbuster/collectors/`) |
| ✅ Done | Seed/cleanup stage | Retraction-reason enrichment from the RW ~111-category vocabulary, missing-abstract fetch from PubMed, bare-notice removal; idempotent (`seed_database.py`) |
| ✅ Done | Heuristic enrichment | Effect-size auditing, outcome switching vs ClinicalTrials.gov, funding classification, author COI via ORCID/Europe PMC/CMS Open Payments (`biasbuster/enrichers/`) |
| ✅ Done | Retraction-reason classifier | Severity floors + abstract-detectability split — undetectable retractions (fraud, fabrication) re-annotated without retraction context |
| ✅ Done | Multi-model LLM annotation | Anthropic + any OpenAI-compatible backend; shared prompts and message construction; incremental save + checkpoint/resume |
| ✅ Done | Human review tool | NiceGUI web review of annotations between annotate and export stages |
| ✅ Done | Training export | alpaca (with `<think>` verification chains), sharegpt, openai_chat; 80/10/10 splits grouped by PMID so one abstract never straddles splits |
| ✅ Done | Model comparison stage | Per-dimension F1, Cohen's κ, McNemar significance vs human ground truth; Markdown report |
| ✅ Done | LoRA training — DGX Spark backend | TRL `SFTTrainer` in NGC Docker: Qwen3.5-27B, OLMo-3.1-32B, GPT-OSS-20B MoE (MXFP4 dequantize, attention-only LoRA) |
| ✅ Done | LoRA training — Apple Silicon backend | Native MLX QLoRA (Qwen 9B/27B 4/8-bit, GPT-OSS-20B MoE); same `metrics.jsonl` contract |
| ✅ Done | Live training monitor | NiceGUI dashboard over `metrics.jsonl`: loss curves, LR schedule, GPU memory, grad norms |
| ✅ Done | Fine-tuning workbench GUI | 4-tab settings → train → evaluate → export (`biasbuster/gui/`) |
| ✅ Done | End-to-end orchestrator | `train_and_evaluate.sh`: train → merge → Ollama export → evaluate, auto-versioned V{n} |
| ✅ Done | Six fine-tuning rounds (→ V9) | Postmortems and run logs in `docs/history/` |
| ⬜ Planned | Cochrane corpus rebuild (Stage A) | `docs/papers/drafts/medrxiv_V5/REBUILD_DESIGN.md` — rebuild ground truth with DB invariants + validation gate; design awaiting owner sign-off |
| **Objective 2 — Agentic risk-of-bias assessment harness** | | |
| ✅ Done | `biasbuster` CLI | Per-paper analysis by PMID/DOI/file; multi-backend `--model` (ollama:/anthropic:); markdown/JSON output |
| ✅ Done | V5A decomposed methodology | LLM extraction → Python mechanical rules → per-domain LLM override calls; the current recommended analysis path |
| ✅ Done | Cochrane RoB 2 methodology module | Signalling questions, per-domain truth tables (`algorithms.py`), worst-wins synthesis, emitted-judgement consistency check |
| ✅ Done | COI divergence by design | Industry funding + sponsor-employed authors → hard HIGH; deliberate, documented divergence from RoB 2 (`docs/harness/DESIGN_RATIONALE_COI.md`) |
| ✅ Done | Cochrane-agreement report | `scripts/compare_vs_cochrane.py` for V5A annotations vs expert ratings |
| **Objective 3 — Evaluations & papers** | | |
| ✅ Done | EM replication Phases 1–4 | Full-text acquisition, contamination check, benchmark DB build, sanity-check κ ≈ 0.22 reproduction |
| ✅ Done | Phase 5 evaluation matrix | 4 models (gpt-oss:20b, Sonnet 4.6, gemma4:26b, qwen3.6:35b) × {abstract, fulltext} × 3 passes; multi-host sharding (Mac + DGX Spark) with merge tool |
| ✅ Done | Parse-failure recovery | Live-path + post-hoc algorithmic fallback, tagged `FALLBACK`; strict (pre-registered primary) vs inclusive κ modes |
| ✅ Done | Phase 6 cross-model comparison | Per-domain κ, run-to-run reliability, ensemble via majority-vote domains + worst-wins overall, forest-plot data |
| ✅ Done | gpt-oss temperature sweep | T = 0–1.2 fulltext × 3 passes on a 10-RCT subset (in benchmark DB) |
| 🔶 In progress | κ regeneration + manuscript update | Retro-tag (no-op) → strict+inclusive tables regenerated → both drafts on strict-primary numbers; ensemble now loses for all 4 models (2026-07-17). **Blocked: tables polluted by RCT030 wrong-paper judgements — must regenerate after issue #29** |
| ✅ Done | Wrong-paper recovery guard | `WRONG_PAPER_RCTS` in `recover_parse_failures.py` + tests; RCT030 rows reverted and documented in `benchmark_rct.notes` (2026-07-17). Recovery path only — analysis-path exclusion tracked in #29 |
| ✅ Done | Enforce RCT030 exclusion in κ scripts (#29 code fix) | `exclusions.py` single source of truth, enforced in `compute_phase6_kappa.py`/`interim_analysis.py`/`temperature_analysis.py`; deliberately not in `sanity_check_kappa.py` (EM reproduction) — PR #30, 2026-07-17 |
| ✅ Done | Wrong-paper completeness audit (#29 step 5) | `audit_wrong_paper_acquisitions.py` + tests; found ≥4 more Tier-A wrong docs (RCT008/080/088/093) + a Tier-B "wrong report" class — RCT030 is not unique (2026-07-17) |
| ✅ Done | Obtainability audit + recovery tool | `recover_wrong_papers.py` (validated re-resolver, `report`/`apply`, surgical DB update, `MANUAL_PMIDS`) + tests; `recovery_obtainability.md`: 8 auto- + 2 manual-recoverable, 2 genuine excludes, RCT017 needs manual PMID (2026-07-17) |
| ⛔ Blocked | Exclude-vs-recover decision (owner) | Exclude-all (simple) vs recover the ~10-11 obtainable (restores n; needs pre-reg §12 amendment + targeted model re-run). Sets denominator + §3.1 narrative; gates κ regeneration |
| 🔶 In progress | Regenerate κ tables + drafts post-#29 | **Blocked** on the exclusion-set decision above; then regenerate both κ modes + re-derive both drafts (wrong-paper exclusion class distinct from the 9 regional-journal RCTs) |
| ⬜ Planned | Forest-plot figure | Figure 1 for the primary draft, from `phase6_forest_data.csv` |
| ⬜ Planned | OSF pre-registration mirror | Pre-reg currently locked in git history only (commit `7854a1c`) |
| ⬜ Planned | Confidence-calibrated ensemble | Future-work appendix; use as a primary metric would require a pre-reg amendment |
| ⬜ Decision | OpenAthens full-text ceiling lift | Manual fetch for ~50 PMID-only RCTs (41/100 → ~85+/100); optional polish, owner decides |
| ⬜ Planned | Preprint submission | Recommended order: harness-vs-naive first, then assessor-algorithm-conformance; owner to confirm |
| **Shared infrastructure** | | |
| ✅ Done | SQLite single source of truth | `dataset/biasbuster.db`; schema-enforced PMID uniqueness, atomic upserts, WAL, FK constraints |
| ✅ Done | Single-sourced prompts | Severity boundaries, domain criteria, verification-database guidance in `prompts*.py`, shared by annotators and export |
| ✅ Done | Single-paper import & annotation | `annotate_single_paper.py` — PMID/DOI resolve, fetch, enrich, validate, annotate, `--force` re-annotation |
| ✅ Done | Session maintenance scaffolding | HANDOVER.md, ROADMAP.md, `nextsession`/`fixall` skills (2026-07-16) |
| ⬜ Planned | Dependabot triage | 33 alerts (12 high) on the default branch |
| ✅ Done | pytest collection scoping | `testpaths = ["tests"]` so bare `uv run pytest` no longer collects stray `worktrees/` checkouts (2026-07-16) |
| ✅ Done | Fix stale tests | #21 (export split fixture vs PMID grouping), #22 (RoB 2 missing-judgement fallback expectation) — realigned 2026-07-17 |
