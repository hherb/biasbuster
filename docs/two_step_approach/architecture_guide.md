# BiasBuster Two-Call Architecture — Design & Training Guide

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Paper      │     │  Extracted   │     │   Verification   │
│   (text)     │────▶│   Facts      │────▶│   Results        │
│              │  S1 │   (JSON)     │     │   (external)     │
└─────────────┘     └──────┬───────┘     └────────┬─────────┘
                           │                      │
                           ▼                      │
                    ┌─────────────┐               │
                    │  Assessment  │◀──────────────┘
                    │  (JSON)      │           S3
                    │              │  ┌──────────────────┐
                    └──────┬───────┘  │ Refined          │
                           │    S2   │ Assessment        │
                           │         │ (JSON)            │
                           ▼         └──────────────────┘
                    ┌─────────────┐
                    │   Report     │
                    │   (markdown) │
                    └─────────────┘

S1 = Stage 1: Extraction
S2 = Stage 2: Assessment
S3 = Stage 3: Verification Integration
```

## Stage Descriptions

### Stage 1: Extraction
**Input:** Paper text (abstract or full-text JATS/PDF)
**Output:** Structured JSON of extracted facts
**Task:** Pure information retrieval — no judgments, no severity ratings
**Evaluable against:** Ground truth extracted by human reviewers

### Stage 2: Assessment
**Input:** Stage 1 JSON output
**Output:** 5-domain bias assessment JSON
**Task:** Apply domain criteria to extracted facts, compute flags, assign severities
**Evaluable against:** Expert human assessments

### Stage 3: Verification Integration
**Input:** Stage 2 JSON + external verification results (ClinicalTrials.gov, ORCID, etc.)
**Output:** Refined assessment JSON with explicit change notes
**Task:** Compare verification data against assessment, adjust where warranted
**Evaluable against:** Expert review of refinement quality

---

## Why Two Calls > One Call for Training

### 1. Independent evaluation of sub-skills

With a single call, if the model produces the wrong severity, you can't tell
whether it failed to *extract* the relevant fact or failed to *interpret* it
correctly. With two calls:

| Failure mode                | Where it shows up     | How to fix             |
|-----------------------------|-----------------------|------------------------|
| Didn't find the dropout     | Stage 1: n_analysed   | More extraction        |
| numbers                     | = null                | training data          |
| Found dropouts but didn't   | Stage 2: high_        | Refine severity        |
| flag them                   | attrition = false     | boundary prompts       |
| Flagged correctly but       | Stage 2: severity =   | Calibration training   |
| wrong severity              | "low" instead of      |                        |
|                             | "high"                |                        |

### 2. Model size flexibility

| Stage   | Complexity         | Minimum viable model  |
|---------|--------------------|----------------------|
| Stage 1 | Information lookup  | 7-13B (good at       |
|         |                    | structured extraction)|
| Stage 2 | Analytical judgment | 13-70B (needs        |
|         |                    | reasoning capability) |
| Stage 3 | Evidence integration| 30-70B+ (needs       |
|         |                    | cross-source reasoning)|

You can use your strongest model for Stage 2 (the hard part) and a
cheaper/faster model for Stage 1 (the mechanical part). Or fine-tune a
small model specifically for extraction, which is a much simpler task to
learn than end-to-end bias assessment.

### 3. Training data efficiency

Stage 1 extraction training data can be generated semi-automatically:
- Extract the ground truth from the paper programmatically where possible
  (n_randomised from CONSORT tables, ClinicalTrials.gov registration, etc.)
- Human review only needs to verify extracted values, not the full assessment
- This means you can generate 10x more extraction training data than
  assessment training data for the same human reviewer cost

### 4. Error isolation prevents cascading failures

In single-call: if the model hallucinates a sample size, the methodology
assessment is wrong, which skews the overall probability, which corrupts
the training signal for probability calibration.

In two-call: Stage 1 errors are caught before they propagate. You can
validate Stage 1 output against known values before feeding it to Stage 2.

---

## Training Pipeline Changes

### Generating training data

```python
# Old pipeline (single call)
paper_text → strong_model(ANNOTATION_SYSTEM_PROMPT) → full_annotation.json

# New pipeline (two call)
paper_text → strong_model(EXTRACTION_SYSTEM_PROMPT) → extraction.json
extraction.json → strong_model(ASSESSMENT_SYSTEM_PROMPT) → assessment.json
# Optional: + verification_results
assessment.json + verification → strong_model(VERIFICATION_PROMPT) → refined.json

# Training data format
training_example = {
    "stage": "extraction" | "assessment" | "verification",
    "input": <stage-appropriate input>,
    "output": <stage-appropriate output>,
    "schema_version": "3.0"
}
```

### Recommended training sequence

#### Phase 1: Extraction model
- **Data:** 500-1000 papers with human-verified extraction JSONs
- **Task:** paper_text → extraction.json
- **Eval metric:** Per-field accuracy against ground truth
  - Exact match for integers, booleans, enums
  - Semantic similarity for free-text fields
  - Null-vs-present detection rate (did it find the COI statement?)
- **Key difficulty:** Structured output adherence. Consider constrained
  decoding (e.g., outlines, grammar constraints) if your framework supports it.

#### Phase 2: Assessment model
- **Data:** 500-1000 extraction JSONs with human-verified assessments
- **Task:** extraction.json → assessment.json
- **Eval metrics (separate):**
  1. **Flag accuracy:** Per-boolean F1 score (did it set per_protocol_only correctly?)
  2. **Severity accuracy:** Per-domain exact match (did it say HIGH when experts said HIGH?)
  3. **Probability calibration:** Brier score across the test set
  4. **Reasoning quality:** Does the reasoning cite specific extracted values?
- **Key difficulty:** Rare flag detection. See oversampling strategy below.

#### Phase 3: Verification integration model
- **Data:** 200-500 assessment + verification pairs with human-verified refinements
- **Task:** (assessment.json + verification_results) → refined_assessment.json
- **Eval metric:** Did the model correctly identify new information from verification
  and adjust appropriately?
- **Key difficulty:** This is the hardest stage. Consider keeping this on your
  strongest API model rather than fine-tuning.

### Handling rare flags

Several flags will be TRUE in <5% of training examples:

| Flag                     | Expected prevalence | Strategy              |
|--------------------------|--------------------|-----------------------|
| differential_attrition   | ~3-5%              | Oversample            |
| analytical_flexibility   | ~2-3%              | Oversample            |
| premature_stopping       | ~3-5%              | Oversample            |
| inflated_effect_sizes    | ~5-8%              | Oversample            |
| registered_outcome_      | ~5-10%             | Targeted search for   |
| not_reported             |                    | papers with known gaps|

**Oversampling strategy:**
1. Search PubMed/ClinicalTrials.gov for papers with known characteristics:
   - `"per protocol" NOT "intention to treat"` for per_protocol_only
   - Papers registered with patient-reported outcomes that report only surrogates
   - Papers with published CONSORT diagrams showing high/differential dropout
2. Generate extraction + assessment annotations for these papers specifically
3. Include at 2-3x their natural frequency in training batches
4. Use a class-weighted loss that upweights rare TRUE flags

### Backward compatibility with v1 training data

If you have existing v1 annotations, you can partially reuse them:

```python
def convert_v1_to_v3_assessment(v1_annotation):
    """Convert v1 single-call annotation to v3 assessment format."""
    v3 = copy.deepcopy(v1_annotation)

    # Add new methodology flags with conservative defaults
    v3["methodology"]["high_attrition"] = None  # unknown, not false
    v3["methodology"]["differential_attrition"] = None
    v3["methodology"]["inadequate_sample_size"] = None
    v3["methodology"]["no_multiplicity_correction"] = None
    v3["methodology"]["analytical_flexibility"] = None

    # Add new statistical reporting flag
    v3["statistical_reporting"]["inflated_effect_sizes"] = None

    # Add new COI flags
    v3["conflict_of_interest"]["sponsor_controls_analysis"] = None
    v3["conflict_of_interest"]["sponsor_controls_manuscript"] = None

    # Add new outcome reporting flag
    v3["outcome_reporting"]["registered_outcome_not_reported"] = None

    # Mark as migrated
    v3["_schema_version"] = "3.0_migrated_from_v1"
    v3["_migration_note"] = "New flags set to null (unknown). Do not train on null flags."

    return v3
```

**Important:** Use `None`/null for unknown flags, NOT `False`. During training,
mask the loss for null-valued flags so the model doesn't learn "new flags are
always false." When generating training batches, filter:

```python
# Only include examples where new flags have been explicitly annotated
if example["_schema_version"] == "3.0_migrated_from_v1":
    # Use for Stage 2 training ONLY for the original 5 domains
    # Skip for methodology flag training on new flags
    pass
```

---

## Evaluation Framework

### Per-stage metrics

```python
# Stage 1: Extraction accuracy
extraction_metrics = {
    "integer_fields_exact_match": accuracy(["n_randomised", "n_analysed", ...]),
    "boolean_fields_f1": f1(["attrition_stated", "coi_statement_present", ...]),
    "enum_fields_accuracy": accuracy(["study_type", "analysis_population_stated", ...]),
    "null_detection_rate": recall_at_null(all_nullable_fields),
    "quote_relevance": semantic_sim(extracted_quotes, reference_quotes),
}

# Stage 2: Assessment accuracy
assessment_metrics = {
    # Per-flag
    "flag_f1": {flag_name: f1_score for each boolean flag},
    "flag_f1_macro": macro_average(all_flag_f1_scores),

    # Per-domain severity
    "severity_exact_match": {domain: exact_match for each domain},
    "severity_within_one": {domain: within_one_level for each domain},
    "severity_weighted_kappa": {domain: cohens_kappa for each domain},

    # Overall
    "overall_severity_exact": exact_match("overall_severity"),
    "probability_brier": brier_score("overall_bias_probability"),
    "probability_calibration_curve": calibration_bins(10),
}

# Stage 3: Verification integration
verification_metrics = {
    "new_findings_recall": did_model_catch_all_new_findings,
    "probability_adjustment_direction": correct_direction_of_change,
    "probability_adjustment_magnitude": within_5pct_of_reference,
    "false_escalation_rate": escalated_when_shouldnt_have,
}
```

### Aggregate quality score

```python
def quality_score(extraction_metrics, assessment_metrics):
    """Weighted aggregate quality score for a model.

    Weights reflect impact on end-user report quality.
    """
    return (
        0.15 * extraction_metrics["integer_fields_exact_match"]
        + 0.10 * extraction_metrics["null_detection_rate"]
        + 0.25 * assessment_metrics["flag_f1_macro"]
        + 0.25 * assessment_metrics["severity_weighted_kappa_macro"]
        + 0.15 * (1.0 - assessment_metrics["probability_brier"])
        + 0.10 * assessment_metrics["reasoning_cites_extracted_values"]
    )
```

---

## Migration Checklist

- [ ] Update JSON schema in database / storage layer for new fields
- [ ] Update report generator to handle both v1 and v3 schemas
- [ ] Generate v3 extraction + assessment annotations for 200+ papers using Claude
- [ ] Human-review a sample of 50 extraction JSONs for accuracy
- [ ] Human-review a sample of 50 assessment JSONs for severity calibration
- [ ] Run both models (Claude, gpt-oss:120b) on the Seed Health test paper with new prompts
- [ ] Run both models on 3-5 known LOW-bias papers to check for inflation
- [ ] Implement two-call pipeline in annotator code
- [ ] Implement per-stage evaluation metrics
- [ ] Oversample rare-flag papers for training set
- [ ] Fine-tune extraction model (Phase 1)
- [ ] Evaluate extraction model before proceeding
- [ ] Fine-tune assessment model (Phase 2)
- [ ] Evaluate assessment model against Claude baseline
- [ ] Decide whether to fine-tune or API-serve Stage 3
