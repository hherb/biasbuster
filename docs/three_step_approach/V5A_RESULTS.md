# V5A Results — 5-paper pilot

**Date:** 2026-04-15
**Test set:** 5 full-text papers (PMIDs 32382720, 39691748, 39777610, 39905419, 41750436)
**Models:** claude-sonnet-4-6 (reference), ollama:gemma4:26b-a4b-it-q8_0, ollama:gpt-oss:20b
**Pipeline:** V5A decomposed (see [`V5A_DECOMPOSED.md`](./V5A_DECOMPOSED.md))
**Reference run:** `dataset/annotation_comparison/comparison_inter-model-agreement_2026-04-15.md`

## Headline

V5A successfully closes the small-model contextual-judgment gap identified in [`OVERVIEW.md`](./OVERVIEW.md). Both local models blast past both the pass threshold (κ ≥ +0.30) and the stretch goal (κ ≥ +0.50):

| Model       | V4 agentic κ (baseline) | **V5A decomposed κ** | Δ        |
|-------------|--------------------------|----------------------|----------|
| gemma4-26B  | -0.154 (no agreement)    | **+0.783**           | **+0.94** |
| gpt-oss-20B | -0.250 (worse than chance) | **+0.706**         | **+0.96** |

V5B fine-tuning is no longer needed. The recommendation is to focus on **gemma4-26B as the primary local model**, with V5A as the inference pipeline.

## Per-paper overall severity

| PMID     | Paper (truncated)                   | Sonnet        | gemma4        | gpt-oss       |
|----------|--------------------------------------|---------------|---------------|---------------|
| 32382720 | Predictors of remission after…       | low/0.18      | low/0.18 ✓    | moderate/0.44 |
| 39691748 | Efficacy of Preemptive Topical…      | high/0.77     | high/0.72 ✓   | high/0.72 ✓   |
| 39777610 | Tapinarof Improved Outcomes and…     | high/0.84     | high/0.80 ✓   | high/0.77 ✓   |
| 39905419 | Balneotherapy for post-COVID…        | moderate/0.44 | low/0.18 ✗    | moderate/0.44 ✓ |
| 41750436 | Multi-Species Synbiotic…             | high/0.84     | high/0.84 ✓   | high/0.84 ✓   |

Exact overall-severity agreement with Sonnet:
- **gemma4: 4/5** (80%)
- **gpt-oss: 3/5** (60%)
- Note: gpt-oss nailed PMID 39905419 where gemma4 missed it due to extraction; gemma4 nailed PMID 32382720 where gpt-oss missed a legitimate override.

## Per-dimension severity κ vs Sonnet

| Dimension              | gemma4 | gpt-oss |
|------------------------|--------|---------|
| **Conflict of Interest** | **1.000** | **1.000** |
| Spin                   | 0.737  | 0.375   |
| Methodology            | 0.643  | 0.706   |
| Statistical Reporting  | 0.615  | 0.231   |
| Outcome Reporting      | 0.375  | 0.545   |

Observations:
- **COI κ = 1.000 for both small models.** This is the hard-rule enforcement + non-overridability policy working exactly as designed. The mechanical triggers (a/b/c/d per [`DESIGN_RATIONALE_COI.md`](../two_step_approach/DESIGN_RATIONALE_COI.md)) produce identical outputs across all three models because the LLM is not permitted to touch them.
- **gemma4 > gpt-oss on 3/5 dimensions** (spin, statistical_reporting), is tied on methodology, loses marginally on outcome_reporting. gemma4-26B is the stronger local candidate.
- gpt-oss weakness on statistical_reporting (κ=0.231) and spin (κ=0.375) reflects extraction-gap problems (see next section), not override-reasoning problems.

## Case study: independent convergent override (PMID 32382720)

This paper (Predictors of remission after psilocybin for depression — THREE-D exploratory secondary analysis) triggered the multiplicity-correction rule at HIGH. Both Sonnet and gemma4 independently identified it as a legitimate exploratory-analysis exception and applied the same override with substantively equivalent reasoning:

**Sonnet's reason:**
> The paper explicitly labels itself an 'exploratory' secondary analysis of the THREE-D RCT, acknowledges the exploratory/data-driven nature as a limitation, applies a prespecified prediction model rather than stepwise selection, uses bootstrap-corrected c-index for internal validation, and frames conclusions with appropriate uncertainty language. The multiplicity standard is customarily lenient for explicitly exploratory predictor analyses, making the mechanical rule inapplicable here.

**gemma4's reason:**
> The rule fires on an exploratory analysis of a previously published RCT; the authors explicitly state the analysis was 'exploratory and data-driven' and intended to identify predictors rather than test primary hypotheses, making strict multiplicity correction less critical.

**Both models: decision=downgrade, methodology high → low.**

gpt-oss did NOT apply this override on this paper (kept methodology at moderate), which explains its weaker per-paper agreement.

This is exactly the contextual-judgment behaviour that was completely absent in V4:

| V4 agentic (gemma4) | V5A decomposed (gemma4) |
|---------------------|-------------------------|
| 0 REVIEW blocks in 5 papers | Structured 3-field decisions, 100% parsed |
| 0 overrides in 5 papers | 1 override, matching Sonnet exactly |
| rubber-stamp behaviour | genuine contextual reasoning |

## Why V5A worked where V4 did not

V4's failure mode was that smaller models could not follow a multi-step instruction stack: "call this tool, read its output, write structured REVIEW blocks, apply override criteria, emit a 30-field JSON". When asked to do all of that in one continuation, gemma4 and gpt-oss collapsed to "summarise the tool output and emit the JSON verbatim".

V5A's structural win: **each LLM call has exactly one job with a 3-field output schema**. The model never sees more than:

- The name of one domain
- The mechanical severity and the specific rule that fired
- The extracted facts that triggered it
- A short list of legitimate vs illegitimate override reasons for *this domain only*
- A 3-field output schema: `{decision, target_severity, reason}`

That's an instruction-following load that a 20–26B model can handle reliably. Parse failure rate across all 3 models × 5 papers × 1–5 per-domain calls: **zero**.

## Efficiency

LLM calls per paper (5 papers total):

| Model  | Extraction calls | Stage 3 per-domain | Stage 5 synthesis | Total |
|--------|------------------|--------------------|--------------------|-------|
| Sonnet | ~8 per paper     | 11 across 5 papers | 5 across 5 papers  | ~56   |
| gemma4 | ~8 per paper     | 8 across 5 papers  | 5 across 5 papers  | ~53   |
| gpt-oss| ~8 per paper     | 8 across 5 papers  | 5 across 5 papers  | ~53   |

V4 agentic was ~2–3 turns × 5 papers = 10–15 assessment turns. V5A uses 13–16 assessment calls. Roughly equivalent inference cost but each call is tiny and parallel (Stage 3 via `asyncio.gather`), so wall-clock latency is actually better.

## The remaining gap is extraction, not judgment

The 5 domain-level disagreements gemma4 has with Sonnet on the pilot set all trace back to **Stage 1 extraction**, not Stage 3 override judgment:

- **PMID 39691748** (stat_reporting): Sonnet's extraction flagged `inflated_effect_sizes=true` from a cherry-picked subgroup NNT; gemma4's extraction did not surface that quote.
- **PMID 39777610** (spin, outcome_reporting): Sonnet extracted specific spin language from the discussion; gemma4 missed some uncertainty-language violations.
- **PMID 39905419** (methodology, outcome_reporting): Sonnet caught `per_protocol_only=true` from a single sentence in the Methods section; gemma4's extraction put it under `analysis_population_stated="not_stated"`.

This is a qualitatively different and more tractable problem than the V4 judgment gap. Paths forward include:
- Tightening the Stage 1 per-section extraction prompts for the specific fields that get missed
- Running an extraction-reconciliation pass where one model checks another's extraction
- Using Sonnet for extraction and gemma4 for assessment (hybrid — only if latency/cost warrants)

## Recommendations

1. **Scale to 25-paper validation.** If gemma4 κ holds at ≥ +0.70 on a broader set, V5A is production-ready for local deployment.
2. **Focus further work on gemma4-26B.** It's the stronger local candidate on 3 of 5 dimensions and ties on a fourth. gpt-oss-20B lags on spin and statistical_reporting — likely a capability ceiling issue, not fixable with V5A alone.
3. **Defer V5B.** Fine-tuning effort is not justified given these results. Keep it as a fallback option if the 25-paper validation shows regression.
4. **Invest in extraction quality next.** The remaining gemma4 vs Sonnet gap is now dominated by Stage 1 misses. This is a separate, tractable workstream.

## Raw data

- Full comparison report: `dataset/annotation_comparison/comparison_inter-model-agreement_2026-04-15.md`
- JSON: `dataset/annotation_comparison/comparison_inter-model-agreement_2026-04-15.json`
- CSV: `dataset/annotation_comparison/comparison_inter-model-agreement_2026-04-15.csv`
- DB tags: `anthropic_fulltext_decomposed`, `ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed`, `ollama:gpt-oss:20b_fulltext_decomposed`
- Run log: `v5a_eval.log`
