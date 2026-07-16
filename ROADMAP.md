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
| 🔶 In progress | κ regeneration + manuscript update | Retro-tag → recover → strict+inclusive tables → update both drafts; see HANDOVER.md §2 |
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
| ⬜ Planned | Fix stale tests | #21 (export split fixture vs PMID grouping), #22 (RoB 2 missing-judgement fallback expectation) |
