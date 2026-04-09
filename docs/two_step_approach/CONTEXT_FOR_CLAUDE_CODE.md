# BiasBuster Prompt Architecture: Why We're Here

## Context for Claude Code

This document explains the reasoning behind the v3 two-call prompt architecture in `prompts_v3_two_call.py`. Read this before implementing changes to the BiasBuster pipeline.

## What happened

We ran BiasBuster's bias assessment on the same paper (a Seed Health synbiotic RCT, DOI 10.3390/antibiotics15020138) using two models: Claude Sonnet 4.6 and a local gpt-oss:120b. Both used the same prompts from `prompts.py` and processed the full-text JATS. Both reached the same overall verdict (HIGH), but the local model missed critical findings that Claude caught:

1. **Methodology rated NONE** when it should have been HIGH. The paper had 34.4% attrition with differential dropout (43.75% treatment vs 25% placebo), per-protocol-only analysis with no ITT, and no multiplicity correction across dozens of endpoints. The local model didn't flag any of this.

2. **COI rated MODERATE** instead of HIGH. The local model noted industry funding and author affiliations but missed that a sponsor employee performed all data analysis AND led manuscript drafting — a structural conflict beyond mere affiliation.

3. **Post-verification reasoning was shallow.** ClinicalTrials.gov data showed a registered patient-reported GI symptom outcome was absent from the published paper, while the conclusions claimed clinical GI benefit. Claude caught this; the local model didn't connect the dots and just restated its initial assessment with a trivial probability bump (78%→80%).

4. **Subgroup emphasis missed.** A post-hoc butyrate subgroup analysis was presented as a key finding without pre-specification or multiplicity correction. The local model set `subgroup_emphasis = No`.

## Root cause analysis

The failures weren't random — they traced to specific prompt-level gaps:

- **The prompts said "clinical trial abstracts" but the tool was processing full text.** This may have primed the local model to limit its analysis scope.
- **The methodology domain had only 5 boolean flags**, none covering attrition, sample size adequacy, or multiplicity. The model wasn't asked to look for these things, so it didn't.
- **No structured extraction step.** The prompt asked the model to simultaneously find facts AND judge them. Weaker models handle "fill in every field" much better than "notice things spontaneously."
- **No guidance for post-verification reasoning.** The model got verification data but had no instructions for how to compare it against the initial assessment.

## The solution: split extraction from assessment

The key insight is that **extraction** (what does the paper say?) and **assessment** (how concerned should we be?) are fundamentally different cognitive tasks. Extraction is mechanical information retrieval; assessment is analytical reasoning. By splitting them:

- **Stage 1 (Extraction):** Read the paper, fill in a structured JSON of facts. No judgments. Even 7-13B models can do this well because the task is objective and the evaluation is straightforward (did you get the right number?).
- **Stage 2 (Assessment):** Given the extracted facts, apply domain criteria and assign severities. The model works from clean structured input rather than raw text, and the criteria explicitly tell it to compute attrition from extracted sample sizes, check each flag against extracted values, etc.
- **Stage 3 (Verification Integration):** Given the assessment plus external verification results, produce a refined assessment with explicit change notes. Structured comparison: for each verification source, state what it confirms, contradicts, or adds.

## Why this matters for training

The same prompts drive both inference and fine-tuning. The split architecture gives us:

- **Independent evaluation.** If the model gets the wrong severity, we can now tell whether it failed to extract the fact or failed to interpret it. This targets training data investment.
- **Model size flexibility.** Cheap model for extraction, strong model for assessment, API model for verification. Each stage only needs to do one thing well.
- **Rare flag detection.** New methodology flags (differential_attrition, no_multiplicity_correction, etc.) will be TRUE in <5% of papers. We need to oversample these during training — the architecture makes this possible per-stage.
- **No cascading errors in training data.** A hallucinated sample size in Stage 1 gets caught before it corrupts the assessment training signal.

## Files

| File | Purpose |
|------|---------|
| `prompts_v3_two_call.py` | The implementation. Drop-in replacement for `prompts.py`. Contains Stage 1/2/3 prompts, combined single-call prompts for backward compatibility, and convenience accessors. |
| `architecture_guide.md` | Detailed training pipeline guide: per-stage metrics, oversampling strategy, v1→v3 migration, evaluation framework, curriculum training sequence. |
| `CHANGELOG_prompts_v2.md` | Documents every change from v1 with rationale. Written against the intermediate v2 (single-call with expanded flags) but the reasoning carries through to v3. |

## Implementation notes

- The combined single-call prompts (`ANNOTATION_SYSTEM_PROMPT`, `TRAINING_SYSTEM_PROMPT`) still exist for strong models that can handle everything in one pass. They concatenate Stage 1 + Stage 2 instructions.
- The JSON schema adds an `extraction_checklist` / `extraction` object and several new boolean flags. Downstream consumers (report generator, database, training export) need schema updates.
- Stage 3 is probably not worth fine-tuning — keep it on the API model.
- When converting existing v1 training data, use `None`/null for new flags (not `False`) and mask the loss for null-valued fields during training so the model doesn't learn "new flags are always false."
- The `with_thinking` variant of the assessment prompt is for models that benefit from explicit CoT. Strip think blocks from training data for models without native CoT support — the extraction JSON itself serves as the chain-of-thought.
