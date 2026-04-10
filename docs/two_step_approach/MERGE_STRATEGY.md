# Full-Text Extraction: Merge Strategy and Known Limitations

## Context

The full-text annotation path uses map-reduce over semantic chunks
(`biasbuster/cli/chunking.py`). Each chunk goes through Stage 1 extraction
independently, producing a partial extraction JSON. These partials are
then merged into a single extraction object that feeds Stage 2 assessment.

This document explains the merge policy and — importantly — its known
limitations, so we can recognise the failure modes when we see them.

## Merge policy

The merge is **field-type specific**, not "highest severity wins". See
`annotators/_merge_extractions()` for the implementation.

### 1. Lists — union with key-based dedup

Examples: `authors_with_industry_affiliation`, `primary_outcomes_stated`,
`secondary_outcomes_stated`, `subgroups`, `effect_size_quotes`,
`conclusion_quotes`, `limitations_acknowledged`, `attrition_quotes`.

**Rule:** concatenate across chunks, drop duplicates by name/key.

### 2. Presence booleans — OR

Examples: `attrition_stated`, `coi_statement_present`,
`funding_disclosed_in_abstract`, `subgroup_analyses_present`,
`run_in_or_enrichment`, `early_stopping`, `clinical_language_in_conclusions`,
`further_research_recommended`, `uncertainty_language_present`,
`composite_components_disaggregated`.

**Rule:** if any chunk reports True, the paper stated it.

### 3. Singleton facts — authoritative-section priority

Each field family has a "home" section. The value from the most authoritative
chunk wins. If absent, fall back in priority order.

| Field family | Authoritative sections |
|--------------|-------------------------|
| `paper_metadata.*` | abstract, front-matter |
| `sample.*` (n_randomised, n_analysed, n_per_arm_*) | Methods, Results (CONSORT) |
| `analysis.*` (blinding, analysis_population, multiplicity) | Methods |
| `outcomes.primary_outcomes_stated` definitions | Methods |
| `outcomes.primary_outcomes_stated` values | Results |
| `conflicts.*` (funding_type, data_analyst, manuscript_drafter) | COI / Funding / Acknowledgments back-matter |
| `methodology_details.follow_up_duration` | Methods |
| `conclusions.*` | Abstract conclusion, Discussion |

**Rule:** first non-null value from the highest-priority matching chunk.

### 4. Conflict detection

When two *authoritative* chunks disagree on a singleton field, both values
are preserved in a `_merge_conflicts` array on the merged extraction.
Stage 2 assessment sees this and treats it as a reporting-consistency
signal (separate flag in `statistical_reporting.evidence_quotes`).

## Why not "highest severity wins"?

That was considered and rejected for four reasons (see conversation
history for full discussion):

1. **It amplifies hallucinations.** A wrong value from one chunk would
   override correct values from others. There's no coherent severity
   ordering on most numeric fields.
2. **It mis-handles sensitivity analyses.** If Methods says "primary
   analysis: ITT; sensitivity analysis: per-protocol", one chunk might
   correctly extract ITT while another correctly extracts per-protocol.
   "Worst wins" flags the paper as per-protocol-only — factually wrong.
3. **It trains the model to over-flag.** If training data systematically
   picks worst-case values, the assessment stage learns to call everything
   HIGH.
4. **It throws away conflict signal.** When chunks genuinely disagree,
   that's a bias indicator itself — the paper reports different values
   in different places.

Severity-level reasoning already happens correctly at the **assessment**
stage, where multiple HIGH flags combine into HIGH overall severity.

## Known limitation: context loss across sections

Section-level extraction loses whole-paper context for fields that require
seeing multiple sections simultaneously. Specifically:

| Field | Why it needs cross-section context |
|-------|-------------------------------------|
| `outcome_reporting.registered_outcome_not_reported` | Requires comparing registered outcomes (Methods/registration) against reported outcomes (Results). A single chunk sees only one side. |
| `outcome_reporting.composite_not_disaggregated` | Requires knowing the composite was defined in Methods *and* never broken down in Results. |
| `spin.focus_on_secondary_when_primary_ns` | Requires comparing primary outcome result (Results) against what the conclusion (Discussion) emphasises. |
| `spin.conclusion_matches_results` | Same cross-section comparison. |
| `statistical_reporting.selective_p_values` | Requires counting reported vs. omitted p-values across Results — single chunks only see a subset. |
| `methodology.no_multiplicity_correction` vs. `n_primary_endpoints + n_secondary_endpoints > 5` | Requires counting endpoints across all chunks *and* checking multiplicity method. |

For these fields, the section-level extraction will under-report: any
section that alone cannot evidence the flag will set it to null or false.
The merge step cannot synthesise them from partial evidence either.

## If evaluation reveals this limitation

If Stage 2 assessment systematically misses these flags when run on
map-reduced extractions (compared to whole-paper extractions from strong
models), the fix is a **coherence pass**:

### Proposed coherence pass

Add a third LLM call after merge, before assessment:

```
Input: merged extraction JSON
       + list of {section: str, summary: str} for each chunk
Task: fill in cross-cutting fields the per-section extraction could not:
      - registered_outcome_not_reported
      - composite_not_disaggregated
      - focus_on_secondary_when_primary_ns
      - conclusion_matches_results
      - selective_p_values
      - no_multiplicity_correction determination
Output: patched extraction JSON with those fields set
```

This would be a cheap call (input is structured, not raw text) and can use
the same model as Stage 2 assessment. Cost ≈ one extra LLM call per paper.

**Do not build this preemptively.** Build it only when evaluation shows
those specific fields are systematically under-reported. Over-engineering
the pipeline before we have data would be speculative.

## How to detect the need

When running the evaluation framework (Phase 3 in
`docs/two_step_approach/architecture_guide.md`), compare per-flag F1 scores
between:

- **Baseline**: whole-paper extraction (no map-reduce)
- **Candidate**: map-reduced extraction with merge

If the baseline beats the candidate specifically on the fields listed in
the table above, build the coherence pass. If all flags are within
evaluation noise, skip it — the simpler pipeline wins.
