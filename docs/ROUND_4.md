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

## V8 Evaluation (with 8K token budget)

V8 was trained with explicit JSON output instructions added to
`TRAINING_SYSTEM_PROMPT`. First evaluation ran with the wrong `max_tokens`
default (4000 instead of 8000 — see Issue 3 below), producing misleadingly
bad results (F1=0.636). After fixing the CLI default and re-running:

### V8 Results (2026-03-27, 8K tokens)

| Metric | Baseline | V7 | V8 | Best |
|--------|:---:|:---:|:---:|:---:|
| Binary F1 | 0.929 | 0.949 | **0.966** | V8 |
| Severity kappa | **0.098** | 0.095 | 0.097 | Baseline |
| Calibration Error | 0.802 | **0.306** | 0.323 | V7 |
| Verification Score | 0.460 | **0.517** | 0.513 | V7 |

Per-dimension F1 / kappa:

| Dimension | Baseline | V7 | V8 |
|-----------|:---:|:---:|:---:|
| Stat Reporting | **0.836** / **0.411** | 0.760 / 0.136 | 0.795 / 0.225 |
| Spin | 0.610 / 0.164 | 0.752 / 0.208 | **0.767** / **0.254** |
| Outcome | 0.708 / 0.156 | 0.798 / 0.261 | **0.800** / **0.270** |
| COI | 0.643 / 0.073 | **0.894** / **0.100** | 0.880 / -0.031 |
| Methodology | 0.563 / 0.114 | **0.636** / **0.212** | 0.590 / 0.108 |

### V8 Analysis

V8 is the **best binary detector** (F1=0.966) and has the best
per-dimension kappa for spin (0.254) and outcome reporting (0.270).
The JSON prompt instructions partially helped severity grading for
those dimensions.

However, V8's COI kappa is **-0.031** (worse than random) — it detects
COI presence (F1=0.880) but grades severity inversely. The explicit
JSON format instruction may have interfered with COI calibration.

### Severity distribution mismatch (the persistent problem)

| Severity | Ground Truth | Baseline | V7 | V8 |
|----------|:---:|:---:|:---:|:---:|
| none | 5 | 3 | 8 | — |
| low | 17 | 20 | 1 | — |
| moderate | 73 | 73 | 48 | — |
| high | 24 | 23 | 65 | — |
| critical | 4 | 4 | 1 | — |

The baseline almost perfectly matches the ground truth distribution.
V7 collapses `low` (17 GT → 1 predicted) and massively over-predicts
`high` (24 GT → 65 predicted). This explains why overall kappa is stuck
near 0.095-0.098 despite improving binary detection.

**Root cause:** The training data is 61% moderate, 19.6% high. The model
learns to over-call moderate-to-high. The fine-tuning is *degrading* the
pre-trained model's natural calibration for severity, while improving its
ability to detect bias presence.

---

## Issue 3: CLI `max_tokens` Default Was Not Updated

### What happened

V8's first evaluation produced F1=0.636 with 100% `overall_severity=none`.
Investigation revealed 38/123 outputs hit exactly 4000 tokens — the model
exhausted its budget on `<think>` reasoning before outputting JSON.

### Root cause

The `EvalConfig` dataclass in `harness.py` had the correct default (8000),
but `run.py`'s argparse default was never updated from 4000. Since argparse
always passes a value, it silently overrode the dataclass default.

### Fix

Changed `run.py` line 117: `default=4000` → `default=8000`.

---

## Changes Made

| File | Change |
|------|--------|
| `evaluation/scorer.py` | Severity-to-probability fallback; JSON extraction with trailing text; text fallback excludes `<think>`; last-match for overall severity |
| `evaluation/run.py` | Fixed `--max-tokens` CLI default from 4000 to 8000 |
| `prompts.py` | Added `_JSON_OUTPUT_INSTRUCTIONS` to `TRAINING_SYSTEM_PROMPT` |
| `fix_v7_parsing_bug_output.py` | One-shot re-parse script for eval outputs after scorer fixes |
| `docs/ROUND_4.md` | This document |
| `docs/PREPARING_ROUND_4.md` | Updated with scorer fix documentation |

---

## Key Findings

### 1. Binary detection is solved; severity calibration is not

Both V7 (F1=0.949) and V8 (F1=0.966) exceed the baseline (0.929) on
binary bias detection. But severity kappa has never improved beyond the
baseline's 0.098 across all fine-tuning rounds. The fine-tuning improves
"is there bias?" while degrading "how much bias?"

### 2. Format changes move kappa around but don't improve it

Adding JSON instructions (V7→V8) improved spin kappa (0.208→0.254)
and outcome kappa (0.261→0.270) while destroying COI kappa (0.100→-0.031).
The overall kappa remained unchanged. The bottleneck is the training data
distribution, not the output format.

### 3. The training data format is markdown, not JSON

The `export.py` module produces training data in markdown format:
`## Overall: MODERATE (bias probability: 40%)`. Adding JSON instructions
to the system prompt creates a prompt-data mismatch: the prompt says
"output JSON" but every training example shows markdown. The model either
ignores the instruction (V7) or gets confused (V8 partial).

### 4. Attention-only LoRA cannot learn new output formats

Confirmed across Rounds 3-4: Harmony channels (failed), JSON format
(partially failed), `<think>` convention (partially works because it's
just text). With 0.04% trainable params in attention layers only, the
model adapts reasoning patterns but not structural output changes.

See `docs/OPTIMISING_FOR_ROUND_5.md` for the strategy to address these.
