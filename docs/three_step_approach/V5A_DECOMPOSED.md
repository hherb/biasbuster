# V5A — Decomposed Pipeline Design

## Hypothesis

Small models (gemma4-26B, gpt-oss-20B) *can* reason about a single, narrowly-scoped question. They *cannot* reliably handle "review 5 domains, decide which overrides apply, write your reasoning, then emit a 30-field JSON" in one shot.

Decompose the v4 agentic loop into focused per-domain calls, each with one tiny task and a 3-field output schema.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Stage 1: Per-Section Extraction (LLM, unchanged from v4)   │
│   reuses _extract_full_text_sections()                     │
│   → merged_extraction dict                                 │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Stage 2: Mechanical Assessment (Python, no LLM)            │
│   reuses assess_extraction()                               │
│   → draft_assessment + _provenance                         │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Stage 3: Per-Domain Override Decisions (LLM, N parallel)   │
│   For each domain where:                                   │
│     domain_severities[d] >= "moderate" AND                 │
│     domain_overridable[d] == True                          │
│   make ONE focused LLM call:                               │
│     INPUT:  domain, severity, rule fired, the specific     │
│             extraction fields the rule read, brief paper   │
│             context (title + 1-paragraph summary)          │
│     OUTPUT: {decision, target_severity, reason}            │
│   Calls run in parallel (asyncio.gather)                   │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Stage 4: Optional Verification (only on borderline cases)  │
│   If a Stage-3 decision is "downgrade" + the LLM           │
│   requests verification, dispatch one tool call            │
│   reuses biasbuster/agent/tools.py wrappers                │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Stage 5: Synthesis (Python-only OR one small LLM call)     │
│   - Apply per-domain decisions to draft assessment         │
│   - Run enforce_hard_rules() — guarantees COI HIGH stays   │
│   - Compute overall_severity (max) and                     │
│     overall_bias_probability (calibrated from severities)  │
│   - Optional: one final LLM call to write a 2-3 sentence   │
│     summary `reasoning` field for the report               │
│   - Generate recommended_verification_steps from the       │
│     domain rationales + Stage-4 tool calls                 │
└────────────────────────────────────────────────────────────┘
```

## Why this should work for small models

- Each Stage 3 call has **one** well-defined task with a **3-field** output schema (vs the v4 agent's 30-field final JSON).
- The relevant context is **filtered up front** — the model only sees the firing rule + the specific extraction fields that triggered it, not the full extraction.
- No multi-step reasoning across 5 domains in one head — model handles one at a time.
- Smaller output → less chance of derailment, JSON parse failures, or instruction drift.
- Stage 5 synthesis is mostly Python; the LLM is not asked to assemble a complex final JSON.

## Files to create / modify

### New files

- `biasbuster/assessment_decomposed.py` — `DecomposedAssessor` class implementing Stages 2–5
- `biasbuster/prompts_v5a.py` — focused per-domain override prompts (one template per domain since each domain reads different extraction fields, and each has its own "legitimate override" criteria)

### Modified files

- `biasbuster/annotators/__init__.py` — add `annotate_full_text_decomposed()` method that calls `_extract_full_text_sections()` (Stage 1) then hands off to `DecomposedAssessor`
- `annotate_single_paper.py` — add `--decomposed` flag (mutually exclusive with `--agentic`); db tag suffix `_fulltext_decomposed`

## Per-domain override prompt template (sketch)

```
You are reviewing ONE bias domain on ONE clinical trial.
Decide whether the mechanical rule that fired genuinely applies here.

Paper: {title}
Brief summary: {one_paragraph_summary}

Domain: {domain_name}
Mechanical severity: {severity}
Rule that fired: {rationale}
The specific extracted facts that triggered the rule:
{focused_extraction_subset}

Legitimate reasons to downgrade for THIS domain:
{domain_specific_legitimate_overrides}

Illegitimate reasons (do NOT apply):
{domain_specific_illegitimate_overrides}

Return ONLY this JSON, no commentary:
{
  "decision": "keep" | "downgrade",
  "target_severity": "none|low|moderate|high|critical",
  "reason": "<= 80 words, must cite a paper-specific fact"
}
```

The per-domain "legitimate / illegitimate" lists already exist in `prompts_v4.py` lines 91–135 — they just need to be split per-domain instead of bundled.

## Per-domain extraction subsets

Each mechanical rule inspects a specific, minimal subset of the extraction dict. Stage 3 only needs to pass that subset to the LLM, not the full extraction:

| Domain                 | Extraction fields the rules read |
|------------------------|----------------------------------|
| statistical_reporting  | `outcomes.effect_size_quotes`, `subgroups.subgroups` |
| spin                   | `paper_metadata.title`, `outcomes.primary_outcomes_stated`, `conclusions.clinical_language_in_conclusions`, `conclusions.uncertainty_language_present` |
| outcome_reporting      | `outcomes.primary_outcomes_stated`, `outcomes.composite_components_disaggregated`, `outcomes` registered outcome flags |
| conflict_of_interest   | `conflicts.*`, `outcomes.primary_outcomes_stated` (surrogate check) |
| methodology            | `sample.attrition`, `analysis.*`, `methodology_details.*`, `conclusions.uncertainty_language_present`, computed `total_endpoints` |

## Reused code

| Purpose | Location |
|---------|----------|
| Stage 1 extraction | `_extract_full_text_sections()` in `biasbuster/annotators/__init__.py:880` |
| Stage 2 mechanical | `assess_extraction()` in `biasbuster/assessment/aggregate.py:122` |
| Stage 5 hard-rule enforcement | `enforce_hard_rules()` in `biasbuster/assessment_agent_enforcement.py` |
| JSON repair | `parse_llm_json()` in `biasbuster/annotators/__init__.py` |
| Stage 4 verification tools | wrappers in `biasbuster/agent/tools.py` |
| Anthropic/bmlib transport | patterns in `biasbuster/assessment_agent.py:324-423` |

## Hard-rule invariant

Stage 3 MUST NOT produce decisions for non-overridable domains (COI HIGH from structural triggers a/b/c/d — see [`DESIGN_RATIONALE_COI.md`](../two_step_approach/DESIGN_RATIONALE_COI.md)). Stage 5 runs `enforce_hard_rules()` as a belt-and-braces check; Stage 3 should simply skip non-overridable domains up-front (they're marked `domain_overridable[d] == False` in the provenance).

## Verification

1. Implement V5A
2. Run on the same 5 papers used in current comparison:
   ```bash
   for pmid in 32382720 39691748 39777610 39905419 41750436; do
     for model in anthropic ollama:gemma4:26b-a4b-it-q8_0 ollama:gpt-oss:20b; do
       uv run python annotate_single_paper.py --pmid $pmid --model $model --decomposed --force
     done
   done
   uv run python -m biasbuster.pipeline --stage compare --models \
     anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed,ollama:gpt-oss:20b_fulltext_decomposed
   ```
3. **Success criterion:** gemma4 severity κ improves from -0.154 to ≥ +0.30; gpt-oss from -0.250 to ≥ +0.30.
   - 0.0–0.3 → "tried but didn't work"
   - ≥ 0.30 → meaningful agreement
   - ≥ 0.50 → publishable
4. **Decision point:** if both small models hit ≥ 0.30, scale up to 25-paper validation set. If either misses, pivot to [V5B](./V5B_FINE_TUNING.md).

## Cost / effort

- Implementation: ~1 day of focused work (most pieces already exist; this is orchestration)
- Per-paper inference: 1 extraction call + up to 5 per-domain calls + optional synthesis ≈ same order as the v4 agent (which often needs 2–3 turns). Latency comparable.
- Risk: low — if it fails we learn something useful (that small models can't even handle the focused single-domain task) and pivot to V5B with that signal.
