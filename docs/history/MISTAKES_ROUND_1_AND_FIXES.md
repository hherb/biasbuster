# Round 1 Fine-Tuning: What Went Wrong and How to Fix It

## Summary

After multiple fine-tuning runs (GPT-OSS-20B, Qwen3.5-27B, OLMo-3.1-32B), severity
calibration consistently degraded compared to the baseline model. The fine-tuned
GPT-OSS-20B v2 scored **worse** than the untuned base on every ordinal metric.

This document records the root causes, evidence, and the remediation plan.

---

## Evaluation Results: Baseline vs Fine-tuned GPT-OSS-20B v2

### Overall (any bias detected — binary)

| Metric     | Baseline | Fine-tuned v2 | Delta  |
|------------|----------|---------------|--------|
| F1         | **0.918**| 0.894         | -0.024 |
| Precision  | **0.895**| 0.863         | -0.032 |
| Accuracy   | **0.854**| 0.809         | -0.045 |

### Overall (severity grading — ordinal)

| Metric         | Baseline  | Fine-tuned v2 | Delta   |
|----------------|-----------|---------------|---------|
| Exact match    | **0.325** | 0.223         | -0.102  |
| Within-one     | **0.803** | 0.745         | -0.057  |
| Weighted kappa | **0.158** | 0.082         | -0.076  |
| MAE            | **0.949** | 1.083         | +0.134  |

### Per-Dimension Weighted Kappa

| Dimension              | Baseline  | Fine-tuned v2 | Delta  |
|------------------------|-----------|---------------|--------|
| Statistical reporting  | **0.197** | 0.166         | -0.031 |
| Spin                   | **0.186** | 0.088         | -0.098 |
| Outcome reporting      | 0.127     | **0.175**     | +0.048 |
| Conflict of interest   | **0.244** | 0.043         | -0.201 |
| Methodology            | **0.254** | 0.126         | -0.128 |

### Key Pattern

The fine-tuned model **over-predicts "high" severity** — the overall ordinal
confusion matrix shows 49 moderate→high misclassifications. The COI dimension is
a cautionary tale: binary F1 improved (+0.028) because recall jumped 0.73→0.89,
but precision collapsed (0.77→0.69) and kappa cratered (0.24→0.04). It detects
more COI but grades severity almost randomly.

Calibration error is 0.87 for **both** models — neither is well-calibrated, and
fine-tuning did not improve it.

---

## Root Cause 1: Prompt Divergence (Annotation vs Training)

The **annotation prompt** (`annotators/llm_prelabel.py`, ANNOTATION_SYSTEM_PROMPT)
has detailed operational definitions for boolean flags (relative_only, spin_level,
enrichment_design, etc.) but contains **zero severity boundary definitions**. It
asks for `"severity": "none|low|moderate|high|critical"` per domain but never
defines what each level means.

The **training prompt** (`export.py`, SYSTEM_PROMPT) has detailed per-domain
severity boundaries:

```
Statistical Reporting:
  - LOW: Minor omission, reader can still assess clinical significance
  - MODERATE: Relative measures only, reader cannot assess significance
  - HIGH: Multiple concerns suggesting intentional obfuscation

Conflict of Interest:
  - LOW: Industry involvement present but fully disclosed
  - MODERATE: Industry funding/affiliations but COI not fully disclosed
  - HIGH: Industry funding + undisclosed COI + author-sponsor affiliations
```

**Impact**: The LLM annotator (Claude) assigns severity using its own internal
calibration — which differs from the explicit boundaries the fine-tuned model is
supposed to learn. The training data teaches severity labels that don't match the
boundary definitions in the training system prompt.

**Fix**: Create a single `prompts.py` module as the canonical source for both
annotation and training prompts, with identical severity boundary definitions.
Re-annotate the full dataset with the unified prompt.

---

## Root Cause 2: Retraction Paper Contamination

The dataset includes 368 annotated retracted papers (from Retraction Watch/Crossref).
Their severity distribution:

| Severity | Count | %     |
|----------|-------|-------|
| NONE     | 136   | 37.0% |
| LOW      | 65    | 17.7% |
| MODERATE | 138   | 37.5% |
| HIGH     | 19    | 5.2%  |
| CRITICAL | 10    | 2.7%  |

**37% of retracted papers are labelled NONE severity.** This happens because:

1. The annotation prompt explicitly says "assess the abstract content normally"
   for retracted papers — retraction status doesn't auto-escalate severity.
2. Many papers are retracted for non-bias reasons: authorship disputes, plagiarism,
   duplicate publication, consent issues, publisher errors.
3. The abstract of a paper retracted for data fabrication may read perfectly well —
   the fraud isn't visible in the text.

These NONE/LOW labels for retracted papers contaminate the ground truth. A paper
retracted for data fabrication should be at minimum HIGH/CRITICAL, regardless of
how clean the abstract appears.

**Fix**: Create a retraction reason classifier (`enrichers/retraction_classifier.py`)
that maps Crossref retraction reasons to severity floors:

- Data fabrication/falsification → CRITICAL floor
- Unreliable results / statistical errors → HIGH floor
- Authorship dispute / plagiarism / consent → no floor (assess normally)

Pass this classification to the annotator and enforce severity floors at export time.

---

## Root Cause 3: Oversampling Creates Memorization

The annotation database has this severity distribution:

| Severity | Count | %     |
|----------|-------|-------|
| LOW      | 720   | 44.1% |
| MODERATE | 596   | 36.5% |
| NONE     | 282   | 17.3% |
| HIGH     | 23    | 1.4%  |
| CRITICAL | 10    | 0.6%  |

Only **23 unique HIGH** and **10 unique CRITICAL** examples exist. The
`oversample_rare_severities()` function in `export.py` duplicates these to 5%
of training data (~67 copies each).

**Impact**:
- The model memorizes those specific 33 papers and over-predicts HIGH on anything
  resembling them (explaining the moderate→high confusion pattern).
- Training distribution (5% HIGH/CRITICAL) differs from test distribution (natural
  ~2%), creating a distribution mismatch that hurts evaluation.
- Duplicated examples don't teach generalization — they teach pattern matching on
  a tiny set.

**Fix**: Remove oversampling entirely. The natural distribution is realistic —
most published RCTs genuinely are low/moderate risk. Replace random split with
stratified split to ensure proportional representation in train/val/test.

---

## Root Cause 4: Formulaic Thinking Chains

The `build_thinking_chain()` function in `export.py` generates `<think>` blocks
with templated severity explanations like:

> "Severity is LOW because the concern is minor — the reader can still assess
> clinical significance..."

These are the same template regardless of the specific paper. The model learns to
produce boilerplate rather than reason from evidence. When the template doesn't
match the severity label (because the annotation used different criteria — see
Root Cause 1), the model receives contradictory training signal.

**Fix**: Rewrite thinking chains to:
1. Reference `evidence_quotes` from the annotation
2. Count specific concerns and explicitly map to boundary definitions
3. Add cross-domain calibration reasoning at the end

---

## Remediation Plan

### Phase 1: Unify Severity Definitions (Highest Priority)

Create `prompts.py` as single source of truth for severity boundary definitions.
Both annotation and training prompts import from this module. Re-annotate the
full dataset.

### Phase 2: Retraction Reason Classification

Create `enrichers/retraction_classifier.py` to classify retraction reasons and
set severity floors. Update `build_user_message()` to pass classification context
to the annotator. Enforce floors at export time.

### Phase 3: Remove Oversampling

Remove `oversample_rare_severities()` from the export pipeline. Add stratified
splitting. Accept the natural severity distribution as the training prior.

### Phase 4: Evidence-Grounded Thinking Chains

Rewrite `_build_*_reasoning()` functions to reference specific evidence, count
concerns explicitly, and include cross-domain calibration reasoning.

### Phase 5: Cochrane RoB Calibration Anchors (Longer Term)

Expand Cochrane collection (currently only 8 papers). Fix abstract fetching.
Use expert RoB judgments as severity anchors in annotation context.

### After All Phases

1. Re-annotate the full dataset with unified prompt
2. Re-export with natural distribution (no oversampling)
3. Re-train
4. Evaluate — target: calibration error < 0.5, weighted kappa > 0.3

---

## Lessons Learned

1. **The annotation prompt IS the ground truth.** If it doesn't define severity
   boundaries, the labels are whatever the LLM felt like on that day. Defining
   boundaries only in the training prompt creates a mismatch that fine-tuning
   amplifies rather than corrects.

2. **Oversampling rare classes via duplication is counterproductive at small N.**
   With only 23 HIGH examples, duplication teaches memorization, not generalization.
   Either collect more diverse examples or accept the natural distribution.

3. **Retracted ≠ biased.** Papers are retracted for many reasons. Using retracted
   papers as positive training examples requires classifying the retraction reason
   and assigning severity accordingly.

4. **Benchmark optimization can corrupt ground truth.** Manipulating training data
   distribution to improve evaluation metrics (via oversampling, targeted collection)
   can inadvertently shift the ground truth away from reality, making the model
   worse at the actual task.
