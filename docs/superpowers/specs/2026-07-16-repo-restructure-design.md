# BiasBuster Repository Restructure — Design

**Date:** 2026-07-16
**Status:** Approved by Horst (interactive review, this session)
**Branch:** `claude/biasbuster-restructure-e7bfd1`

## Problem

BiasBuster began as a model-training pipeline and grew organically into three
distinct-but-related efforts. The repo root carries 14 loose Python scripts and
8 shell scripts of mixed vitality; `docs/` mixes current methodology docs with
superseded designs, six rounds of fine-tuning logs, and manual chapters whose
commands no longer resolve. New developers cannot tell which objective a file
serves or whether it is still alive.

## The three objectives

1. **Dataset curation + fine-tuning pipeline** — build curated bias-annotation
   training data and fine-tune local models.
   Code: `biasbuster/{collectors,enrichers,annotators,pipeline,training,gui,crowd}`,
   root `export.py`, training orchestrator shell scripts.
2. **Agentic risk-of-bias assessment harness** — the `biasbuster` CLI applying
   the V5A decomposed methodology (mechanical rules engine + per-domain LLM
   overrides) to a single paper.
   Code: `biasbuster/{cli,agent,assessment,methodologies}`,
   `assessment_decomposed.py`, `prompts_v5a.py`.
3. **Evaluations and papers** — studies quantifying performance, currently the
   Eisele-Metzger 2025 replication and two medRxiv preprint drafts.
   Code: `biasbuster/evaluation/`, `studies/eisele_metzger_replication/`,
   `docs/papers/`, `docs/literature/`.

Shared core: `biasbuster/{database.py,annotators/,schemas/,utils/}` and the
prompt modules (`prompts.py`, `prompts_v3.py`, `prompts_v4.py`, `prompts_v5a.py`).

## Constraints (non-negotiable for this phase)

- **No Python import-path changes inside `biasbuster/`.** The Eisele-Metzger
  study is mid-Phase-5/6 with evaluation runs in flight; internal renames risk
  breaking a live study. A package-internal split into per-objective
  subpackages is explicitly deferred to a later phase.
- **`export.py` stays at the repo root.** Despite looking like a script, it is
  imported as a module by `biasbuster/agent/model_client.py` and
  `biasbuster/evaluation/harness.py`.
- **`biasbuster_next_session.md` stays at the repo root** while the
  Eisele-Metzger study is running (owner's decision). _(Superseded by PR #25 on
  2026-07-16: archived to `docs/history/EISELE_METZGER_RUNBOOK_2026-07.md`,
  replaced by the living `HANDOVER.md`.)_
- **Locked study artifacts do not move**: `docs/papers/eisele_metzger_replication/`
  (pre-registered analysis plan), active preprint drafts, `docs/literature/`.
- All moves use `git mv` so history follows. Nothing is deleted; suspected-dead
  material goes to `attic/` for manual review by the owner.

## Design

### 1. Repo root (after)

Stays: `main.py`, `annotate_single_paper.py`, `seed_database.py`, `export.py`,
`config.example.py`, `Modelfile`, `README.md`, `CLAUDE.md`,
`biasbuster_next_session.md`, `pyproject.toml`, `uv.lock`, `LICENSE`, and the
six active training orchestrators (`run_training.sh`, `run_training_mlx.sh`,
`run_merge.sh`, `run_merge_mlx.sh`, `lora2ollama.sh`, `train_and_evaluate.sh` —
the GUI's `process_runner` invokes these by path).

Moves to `scripts/` (occasional tools, verified not module-imported):
`seed_export.py`, `backfill_cochrane_domains.py`, `reprocess_rob.py`,
`expert_rob_alignment_of_annotations.py`, `compare_vs_cochrane.py`,
`run_v5a_eval.sh`, `run_v5a_validation.sh`. Internal relative paths in the
shell scripts are updated; scripts remain invocable from the repo root
(`uv run python scripts/<name>.py`).

Moves to `attic/` (dead / completed one-offs, each with a reason in
`attic/README.md`):
- `migrate_jsonl_to_sqlite.py` — completed legacy JSONL→SQLite migration
- `fix_v7_parsing_bug_output.py` — completed V7 scorer re-parse fix
- `flag_review_candidates.py` — orphaned one-off analysis (no references)
- `diagnose_v5a_disagreements.py` — orphaned one-off analysis (no references)

**Amendment (re-verification result):** `biasbuster/collectors/rob_table_extractor.py`
was slated for attic as unreferenced, but re-verification showed it is Phase 2
of the still-current medRxiv-V5 Cochrane corpus rebuild (commit d464155,
2026-04-17; planned in `docs/papers/drafts/medrxiv_V5/REBUILD_DESIGN.md` §9.1,
with test fixtures in `tests/fixtures/cochrane_reviews/`). It is unfinished
work-in-progress, not dead code — it stays in the package.

### 2. docs/ reorganized by objective

- `docs/pipeline/` — objective 1: `ANNOTATED_DATA_SET.md`, `SEED_DATA_SET.md`,
  `ANNOTATION_JSON_SPEC.md`, `EXTRACTING_EXPERT_RATINGS.md`, `TRAINING.md`,
  `MLX_TRAINING.md`, `TRAINING_INTERPRETATION.md`, `DGXSPARK.md`,
  `MODEL_CARD.md`, `OLLAMA_MODELCARD_README.md`.
- `docs/harness/` — objective 2: `ASSESSING_RISK_OF_BIAS.md`,
  `BIASBUSTER_CLI.md`, `three_step_approach/` (current V5A methodology docs),
  and `DESIGN_RATIONALE_COI.md` (extracted from the superseded two-step folder
  because its rationale is still current).
- `docs/papers/`, `docs/literature/` — objective 3, unchanged locations; only
  clearly historical items within them move to history (see below).
- `docs/manual/` — stays; stale invocations corrected (see §3).
- `docs/history/` — with `README.md` index: `MISTAKES_ROUND_1_AND_FIXES.md`,
  `MISTAKES_TO_ROUND_3.md`, `ROUND_2_PREPARATIONS.md`, `ROUND_3.md`,
  `PREPARING_ROUND_4.md`, `ROUND_4.md`, `OPTIMISING_FOR_ROUND_5.md`,
  `ROUND_5.md`, `HOWTO/`, `two_step_approach/` (minus DESIGN_RATIONALE_COI),
  `papers/FIRST_RUN.md` … `SIXTH_RUN.md`,
  `papers/drafts/biasbuster_medrxiv_draft.md`, `papers/ESSAY.md`,
  `papers/JOURNEY_CREATING_BIASBUSTER_DATASET.md`.
- All cross-references (README, CLAUDE.md, doc-to-doc links) updated.

Distinction maintained: `docs/history/` = valuable record of decisions kept in
the visible documentation; `attic/` = suspected-dead material awaiting the
owner's manual review.

### 3. Documentation content updates

- Six `docs/manual/` chapters (02, 03, 04, 06, 08b, 10) plus `TRAINING.md`,
  `MLX_TRAINING.md`, `ANNOTATED_DATA_SET.md`, `SEED_DATA_SET.md` still use the
  defunct root `python pipeline.py` invocation → corrected to
  `python -m biasbuster.pipeline`. History docs are left as written.
- **README.md**: new top section presenting the three objectives with a
  "which entry point do I want?" table and a repo map for new developers.
- **CLAUDE.md**: repository-layout section rewritten to describe the new
  structure directly (dropping the accumulated historical-migration caveats);
  command examples updated for moved scripts.

### 4. Verification

Full `pytest` run; `biasbuster --help`; `python -m biasbuster.pipeline --help`;
repo-wide grep for each moved filename to catch dangling references; confirm
`gui/process_runner` shell-script paths unaffected. Work lands as one commit
per phase (spec / attic / scripts / docs moves / doc content / README+CLAUDE /
verification fixes) on this worktree branch for review before merge.

## Explicitly out of scope

- Splitting `biasbuster/` into per-objective subpackages (deferred until the
  Eisele-Metzger study completes; would use compat shims).
- Deleting anything.
- Touching `dataset/`, `dataset_V2/`, `training_output/`, `eval_results/`,
  `ollamazip/` (data and vendored tool directories).
- New features or behavioral code changes.
