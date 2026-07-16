# Project History

Documents in this directory are **records, not current documentation**. They
capture how BiasBuster evolved — what was tried, what failed, and why the
current design looks the way it does. Commands and file paths inside them are
left exactly as written at the time and may no longer resolve.

For current documentation see `docs/pipeline/` (dataset curation +
fine-tuning), `docs/harness/` (the agentic risk-of-bias assessor), and
`docs/papers/` (active studies and drafts).

## Fine-tuning rounds (chronological)

The first phase of the project: six rounds of LoRA fine-tuning experiments,
each with its postmortem.

| Document | What it records |
|---|---|
| `MISTAKES_ROUND_1_AND_FIXES.md` | Round 1 postmortem — including the prompt-mismatch lesson that led to `prompts.py` as single source of truth |
| `ROUND_2_PREPARATIONS.md` | Round 2 data preparation (Cochrane RoB backfill, seed cleanup) |
| `MISTAKES_TO_ROUND_3.md` | Evaluation-infrastructure blind spots discovered going into Round 3 |
| `ROUND_3.md` | Round 3 hyperparameter optimization |
| `PREPARING_ROUND_4.md` | Round 4 preparation — GPT-OSS-20B MoE handling |
| `ROUND_4.md` | Round 4 results — V7 evaluation and the prompt-format gap |
| `OPTIMISING_FOR_ROUND_5.md` | Severity-calibration fixes |
| `ROUND_5.md` | Round 5 — JSON training data and V9 |
| `training_runs/FIRST_RUN.md` … `SIXTH_RUN.md` | Raw run logs for the six training runs |
| `HOWTO/` | Early fine-tuning tutorials (GPT-OSS, single-paper annotation) written during rounds 1–4 |

## Superseded assessment architectures

The per-paper analyzer went through V3 (two-call) and V4 (tool-calling agent)
before arriving at the current V5A decomposed methodology
(`docs/harness/three_step_approach/`).

| Document | What it records |
|---|---|
| `two_step_approach/architecture_guide.md` | V3 two-call architecture |
| `two_step_approach/INITIAL_FINDINGS_V3.md` | V3 evaluation findings |
| `two_step_approach/V4_AGENT_DESIGN.md` | V4 tool-calling agent design |
| `two_step_approach/MERGE_STRATEGY.md` | V3/V4 merge strategy |
| `two_step_approach/CONTEXT_FOR_CLAUDE_CODE.md` | Working context notes from the V3/V4 era |
| `two_step_approach/prompts_v3_two_call.py` | V3 prompt code snapshot |

Note: `DESIGN_RATIONALE_COI.md` (why COI assessment deliberately diverges from
Cochrane RoB 2) was written in this era but is still-current rationale — it
lives in `docs/harness/`, not here.

## Early paper drafts and essays

| Document | What it records |
|---|---|
| `biasbuster_medrxiv_draft.md` | Early dataset-paper draft (superseded by the drafts in `docs/papers/drafts/`) |
| `ESSAY.md` | Narrative essay on the project |
| `JOURNEY_CREATING_BIASBUSTER_DATASET.md` | Narrative account of building the dataset |

## Session runbooks

| Document | What it records |
|---|---|
| `EISELE_METZGER_RUNBOOK_2026-07.md` | The 2026-05→07 next-session runbook for the Eisele-Metzger replication: Phase-5 state mid-run, the recovery infrastructure, and the 2026-07-16 sixteen-finding code-audit record with per-fix summary. Superseded by the living `HANDOVER.md` at the repo root |
