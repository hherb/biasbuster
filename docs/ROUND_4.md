# Round 4: V7 Evaluation Findings and Prompt Format Gap

**Date:** 2026-03-26
**Status:** V7 evaluation in progress (6/123 papers at time of investigation)
**Prerequisite:** `docs/PREPARING_ROUND_4.md`, `docs/MISTAKES_TO_ROUND_3.md`

---

## V7 Early Observations

V7 (Run 10) trained with single-channel format and 2 epochs. The model
produces well-structured output: a `<think>` block with detailed reasoning
followed by a narrative markdown assessment with severity tables. The
thinking quality is substantially improved over V6.

However, two parsing issues surfaced in the first 6 outputs.

---

## Issue 1: `overall_bias_probability` Is Always 0.0

### What happened

All 6 V7 outputs have `overall_bias_probability = 0.0`, including papers
where multiple domains are rated HIGH.

| PMID | Stat | Spin | Outcome | COI | Method | Overall Sev | Probability |
|------|------|------|---------|-----|--------|-------------|-------------|
| 40944125 | moderate | high | none | moderate | low | moderate | **0.0** |
| 40763896 | moderate | high | high | moderate | high | moderate | **0.0** |
| 15647576 | none | none | none | none | none | none | **0.0** |
| 38546803 | moderate | high | high | moderate | high | moderate | **0.0** |
| 18701925 | none | none | none | none | none | none | **0.0** |
| 20207208 | none | none | none | none | none | low | **0.0** |

### Root cause

The model never outputs a numeric `overall_bias_probability` value in its
narrative. The scorer's text fallback parser (`_parse_from_text`) looks for
`probability:\s+(\d+)` but finds nothing, so it defaults to 0.0.

The model's post-`</think>` output is **markdown narrative**, not JSON.
The embedded JSON block inside the narrative (at ~pos 20k in the raw output)
contains `"overall_bias_severity": "HIGH"` but no numeric probability field.
Additionally, this embedded JSON is invalid (contains JS-style `//` comments)
so it wouldn't parse even if extracted.

This is a **prompt gap**: `TRAINING_SYSTEM_PROMPT` (used by the export
pipeline to create training data) never includes the JSON schema from
`_JSON_SCHEMA`. The model was trained on examples that showed the format
implicitly but was never given explicit structural instructions for its output.

### Immediate fix (scorer fallback)

Added a severity-to-probability mapping in `evaluation/scorer.py` so that
when the model produces a valid severity but no numeric probability, the
scorer infers a reasonable probability:

```python
_SEVERITY_TO_PROBABILITY = {
    "none": 0.05,
    "low": 0.25,
    "moderate": 0.55,
    "high": 0.80,
    "critical": 0.95,
}
```

This is applied as a fallback in both `_parse_from_json` and
`_parse_from_text` when `overall_bias_probability` is 0.0 but
`overall_severity` is not "none".

### Permanent fix (prompt + retrain as V8)

`TRAINING_SYSTEM_PROMPT` must include explicit JSON output format
instructions. Added a new `_JSON_OUTPUT_INSTRUCTIONS` block to `prompts.py`
that specifies:
- The exact JSON structure expected after `</think>`
- That `overall_bias_probability` must be a float 0.0–1.0
- That the response must end with a valid JSON object, no trailing text

This requires re-exporting training data and retraining as V8.

---

## Issue 2: Model Outputs Narrative Instead of JSON

### What happened

After `</think>`, the model produces a multi-section markdown assessment
with severity tables, verification checklists, and prose commentary. It
does NOT produce a clean JSON object as the annotation pipeline expects.

Example structure of V7 output:
```
<think>
[~13k chars of detailed reasoning]
</think>

**Biomedical Research Integrity Bias Assessment**
**Study Identified by PMID:** 40944125
**Overall Bias Severity:** HIGH

| Domain | Severity | Key Concerns |
|--------|----------|-------------|
| Statistical Reporting | MODERATE | ... |
| Spin | HIGH | ... |
...

[embedded JSON block with // comments — invalid]

| Domain | Summary | Severity |
...

**Overall:** The study's conclusions are **not reliable**...
```

### Root cause

Same as Issue 1: `TRAINING_SYSTEM_PROMPT` lacks JSON format instructions.
The model learned to produce the reasoning pattern (excellent) but not the
output structure (missing). The training examples implicitly contain JSON
but there's no explicit instruction telling the model to produce JSON after
its reasoning.

### Impact

The scorer falls back to `_parse_from_text()` (regex heuristics) for all
V7 outputs. This works for severity extraction (all 5 domains correctly
parsed) but cannot extract:
- `overall_bias_probability` (no numeric value in narrative)
- Structured flags (e.g., `relative_only`, `surrogate_without_validation`)
- Verification step details (URLs, specific database queries)

The text fallback is adequate for severity-level evaluation but loses the
rich structured data that makes the output actionable.

### Fix

Same as Issue 1: add JSON output instructions to `TRAINING_SYSTEM_PROMPT`
and retrain as V8. The instructions must be explicit about:
1. After `</think>`, output ONLY a JSON object
2. No markdown, no tables, no prose after the JSON
3. The exact schema with all required fields
4. `overall_bias_probability` must be a numeric float

---

## Changes Made

| File | Change |
|------|--------|
| `evaluation/scorer.py` | Added `_SEVERITY_TO_PROBABILITY` fallback mapping for when model omits numeric probability |
| `prompts.py` | Added `_JSON_OUTPUT_INSTRUCTIONS` to `TRAINING_SYSTEM_PROMPT` with explicit JSON format requirements |
| `docs/ROUND_4.md` | This document |

---

## V8 Training Requirements

The prompt change means the current training data must be re-exported and
the model retrained. V8 should:

1. **Re-export training data** with updated `TRAINING_SYSTEM_PROMPT` that
   includes JSON output format instructions
2. **Keep all Round 4 hyperparameters** (2 epochs, single-channel, lr=5e-6)
3. **Validate format compliance** on 10 training examples before full run:
   check that exported examples end with valid JSON after `</think>`
4. **Evaluation**: verify `overall_bias_probability > 0` for biased papers
   and JSON parse success rate >= 95%

---

## Lessons

### 1. Implicit format ≠ learned format

Training data contained JSON output but no explicit instruction to produce
JSON. The model learned the reasoning quality but not the structural format.
MoE attention-only LoRA (0.04% params) needs every bit of signal — implicit
patterns in training data are not enough for format adherence.

### 2. Fallbacks should be documented, not silent

The scorer's text fallback parser silently succeeds with `parse_success=True`
even when it can only extract a subset of fields. The 0.0 probability was
indistinguishable from "model assessed no bias" vs "model didn't output a
number." The severity-to-probability fallback makes this explicit.

### 3. Validate the output contract, not just the metrics

Each training round should include a "format compliance" check: does the
model output match the expected schema? Severity metrics alone don't reveal
structural issues — V7 would have scored well on severity F1 while being
completely non-functional for downstream JSON consumers.
