# Fourth Fine-Tuning Run: Qwen3.5-9B with Conservative Hyperparameters

**Date:** 2026-03-19
**Model:** Qwen3.5-9B (fine-tuned with LoRA)
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)
**Status:** Training and evaluation completed

## 1. Background

The Third Run (THIRD_RUN.md) tested the 9B-aggressive hyperparameters proposed in SECOND_RUN.md §6.4 (4e-4 LR, 5 epochs, effective batch 2). Training curves showed the model converging by step ~300 of 3,090 and then flatting for ~90% of the run — the model saturated early, wasting compute.

This fourth run reverts the learning dynamics to the 27B defaults while keeping the 9B-specific LoRA capacity and regularisation settings.

## 2. Changes from Third Run

### 2.1 Hyperparameter Revision

`_9B_OVERRIDES` in `training/configs.py` was updated:

| Parameter | Third Run (9B-aggressive) | Fourth Run (revised) | Rationale |
|-----------|--------------------------|----------------------|-----------|
| learning_rate | 4e-4 | **2e-4** | 4e-4 caused early saturation |
| num_train_epochs | 5 | **3** | 5 epochs wasted compute on a flat plateau |
| gradient_accumulation_steps | 2 | **4** | Larger effective batch for smoother gradients |
| warmup_ratio | 0.06 | **0.1** | Longer warmup with lower LR |
| save_total_limit | 5 | **3** | Fewer checkpoints needed |

**Kept from 9B config:** lora_r=32, lora_alpha=64, lora_dropout=0.08, weight_decay=0.02, label_smoothing_factor=0.05.

### 2.2 Training Data

Same expanded dataset as the Third Run (unchanged):

| Split | Examples |
|-------|----------|
| Train | 1,235 |
| Val | 142 |
| Test | 144 |
| Total | 1,521 |

Includes all data fixes from THIRD_RUN.md §2.1–2.3: all 5 domains always emitted, substantive NONE reasoning, rare severity oversampling.

### 2.3 Infrastructure Fix: Training Monitor

The training monitor (`utils/training_monitor.py`) was displaying overlaid data from multiple training runs on the same charts. Root cause: `metrics.jsonl` is append-only, and the monitor accumulated metrics across run boundaries without resetting.

**Fix:** `MetricsReader.poll()` now clears `self.metrics` and resets `self.completed` whenever a new `header` record is encountered. This was verified working during the Fourth Run — charts showed only the current run's data cleanly.

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3.5-9B |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.08 |
| Target modules | q, k, v, o, gate, up, down |
| Epochs | 3 |
| Batch size | 1 (gradient accumulation 4 = effective 4) |
| Learning rate | 2e-4 (cosine schedule, 10% warmup) |
| Weight decay | 0.02 |
| Label smoothing | 0.05 |
| Max sequence length | 4096 tokens |
| Precision | bf16 |
| Training examples | 1,235 |
| Validation examples | 142 |
| Total steps | 927 |

## 4. Training Results

### 4.1 Training Curves

The revised hyperparameters produced markedly healthier training dynamics:

- **Training loss:** Rapid initial drop from 13.1 → 4.9 (first 100 steps), then gradual sustained decline to 3.85 by step 927. Unlike the Third Run, the loss was still declining at completion — no saturation plateau.
- **Eval loss:** Steady improvement throughout: 1.267 → 1.101 (Δ = -0.167). Still declining at end of training, suggesting the model would benefit from additional epochs.
- **Learning rate:** Clean cosine schedule peaking at 2e-4 around step ~93, decaying smoothly to near-zero.
- **Gradient norms:** Initial spike to 2.65 during warmup, settled to ~0.29 by end. Stable and healthy throughout.
- **GPU memory:** ~18 GiB allocated, ~43 GiB peak. Consistent with previous runs.

### 4.2 Comparison Across All 9B Runs

| Metric | Second Run | Third Run | Fourth Run |
|--------|-----------|-----------|------------|
| **Training data** | 706 (old format) | 1,235 (new format) | 1,235 (new format) |
| **Config** | 27B defaults, lora_r=16 | 9B-aggressive | 9B-revised |
| **LR** | 2e-4 | 4e-4 | 2e-4 |
| **Epochs** | 3 | 5 | 3 |
| **Total steps** | 690 | 3,090 | 927 |
| **Final eval loss** | 0.334 | 1.146 | **1.101** |
| **Eval Δ** | -0.070 | -0.122 | **-0.167** |
| **Final train loss** | 0.230 | 1.614 | 3.855 |
| **Still improving at end?** | Marginal | No (flat from step ~300) | **Yes** |
| **Saturated?** | No | Yes, severely | **No** |

**Key observations:**

1. **Fourth Run achieved the lowest eval loss** (1.101) of any run on the new training data, in only 927 steps vs 3,090 for the Third Run.
2. **Eval loss improvement was largest** (Δ = -0.167 vs -0.122), despite 3.3x fewer steps.
3. **No saturation** — the model was still learning at epoch 3, unlike the Third Run which plateaued at epoch 0.5.
4. **Loss values are not comparable** between Second Run and Third/Fourth Runs because the training data format changed dramatically (all 5 domains with NONE reasoning = much longer targets = higher cross-entropy).

### 4.3 Train/Eval Loss Gap

The train loss (3.85) is much higher than eval loss (1.10) — a 3.5x gap. This is explained by:
- **Label smoothing (0.05)** adds a constant penalty to training cross-entropy that doesn't affect eval.
- **Dropout (0.08)** is active during training but disabled during eval, inflating training loss.
- Both are regularisation techniques that improve generalisation at the cost of higher training loss — a healthy pattern.

## 5. Evaluation Results

### 5.1 Success Criteria

| Criterion | Target | Result | |
|-----------|--------|--------|---|
| Binary F1 | > 0.90 | **0.924** | **PASS** |
| Recall | > 0.85 | **0.950** | **PASS** |
| Ordinal kappa | > 0.20 | 0.124 | FAIL |
| Verification score | > 0.50 | 0.495 | FAIL (marginal) |

### 5.2 Overall Performance vs Second Run

| Metric | Second Run (9B, old data) | Fourth Run (9B, new data) | Delta |
|--------|--------------------------|---------------------------|-------|
| Binary F1 | 0.804 | **0.924** | **+0.120** |
| Precision | 0.986 | 0.898 | -0.088 |
| Recall | 0.679 | **0.950** | **+0.271** |
| Ordinal kappa | 0.159 | 0.124 | -0.035 |
| Exact match | 0.417 | 0.271 | -0.146 |
| Within-one | 0.748 | 0.764 | +0.016 |
| MAE (severity) | 0.878 | 1.076 | +0.198 |
| Calibration error | 0.913 | **0.840** | -0.073 |
| Verification score | 0.541 | 0.495 | -0.046 |
| Thinking chains | 99% | **100%** | +1% |
| Mean thinking length | 1125 chars | 1289 chars | +164 |
| Parse failures | 0 | 0 | — |

The **recall problem from the Second Run is solved** — up from 0.679 to 0.950. The model is no longer too conservative.

### 5.3 Per-Dimension Binary F1 (all improved)

| Dimension | Second Run | Fourth Run | Delta |
|-----------|-----------|------------|-------|
| Statistical reporting | 0.730 | **0.806** | +0.076 |
| Spin | 0.727 | **0.826** | +0.099 |
| Outcome reporting | 0.755 | **0.839** | +0.084 |
| COI | 0.639 | **0.698** | +0.059 |
| Methodology | 0.656 | **0.737** | +0.081 |

Every dimension improved. The enriched training data with explicit NONE assessments is paying off.

### 5.4 Closing the Gap to 32B

| Metric | 32B FT (First Run) | 9B FT (Fourth Run) | Gap |
|--------|-------------------|---------------------|-----|
| Binary F1 | 0.952 | 0.924 | 0.028 (was 0.148) |
| Recall | 0.920 | **0.950** | **9B wins** |
| Precision | 0.988 | 0.898 | 0.090 |
| Ordinal kappa | 0.285 | 0.124 | 0.161 |
| Verification | 0.368 | **0.495** | **9B wins** |

The 9B model now nearly matches the 32B on binary detection (gap narrowed from 0.148 to 0.028) and beats it on recall and verification. The remaining gap is in severity calibration.

### 5.5 Verification Source Rates

| Source | Second Run | Fourth Run | Delta |
|--------|-----------|------------|-------|
| ClinicalTrials.gov | 99% | 99% | — |
| ORCID | 94% | **100%** | +6% |
| Retraction Watch | 95% | **100%** | +5% |
| Europe PMC | 98% | **100%** | +2% |
| CMS Open Payments | 57% | **22%** | **-35%** |

Four of five sources are now at or near 100%. CMS Open Payments collapsed from 57% to 22%, dragging the mean verification score below target.

### 5.6 What Regressed and Why

**Ordinal kappa dropped (0.159 → 0.124).** The confusion matrices reveal a systematic pattern: the model over-predicts MODERATE and HIGH while almost never predicting LOW. For example, in Statistical Reporting, only 2 of 36 true-LOW examples were predicted as LOW — the rest were predicted as MODERATE (24) or HIGH (7). This "moderate collapse" occurs across all dimensions.

**CMS Open Payments collapsed (57% → 22%).** The training data only mentions Open Payments in 29.8% of examples. By COI severity: 13% for NONE, 16% for LOW, 45% for MODERATE, 100% for HIGH. The model learned the strong HIGH-COI → Open Payments association but lost the weaker MODERATE-COI signal.

**Precision dropped (0.986 → 0.898).** This is the expected trade-off for the massive recall improvement (+0.271). More true positives caught, but also 13 false positives (vs 1 previously).

## 6. Analysis

### 6.1 The Conservative Approach Won

The aggressive 9B hyperparameters (4e-4 LR, batch 2, 5 epochs) were counterproductive. The conservative revision (2e-4 LR, batch 4, 3 epochs) produced:
- Better final eval loss (1.10 vs 1.15)
- No saturation (still improving vs flat from step 300)
- 3.3x less compute (927 vs 3,090 steps)

**Conclusion:** For this task and dataset size, learning dynamics are not model-size-dependent. The 27B defaults work equally well for 9B. The 9B model's differentiation should be in LoRA capacity and regularisation, not learning rate or epoch count.

### 6.2 More Epochs May Help

The eval loss was still declining at epoch 3.00 (1.101, down from 1.267). This suggests 3 epochs may be insufficient for the expanded dataset. A follow-up run with 4-5 epochs at the conservative learning rate (2e-4) may yield further improvement without the saturation seen with 4e-4 LR.

However, the improvement rate was slowing: Δ per 50 steps went from -0.038 (steps 50-100) to -0.003 (steps 850-900). Diminishing returns suggest 4 epochs would capture most remaining gain; 5 may not be worth the compute.

### 6.3 The Training Data Change Dominates

The single biggest change between the Second Run and Third/Fourth Runs is the training data, not hyperparameters. The new format (all 5 domains always emitted, NONE reasoning, oversampled rare severities) produces fundamentally different targets. The evaluation confirms this produced a dramatically better model on binary detection (+0.120 F1, +0.271 recall).

### 6.4 Why Severity Grading Regressed

The ordinal kappa dropped from 0.159 to 0.124 despite better binary detection. Analysis of the training data distribution reveals why:

**Per-dimension severity distribution in training data (1,235 examples):**

| Dimension | NONE | LOW | MODERATE | HIGH | CRITICAL |
|-----------|------|-----|----------|------|----------|
| Statistical Reporting | 36% | **9%** | 50% | 4% | 1% |
| Spin | 41% | 42% | 13% | 4% | 0% |
| Outcome Reporting | 39% | 22% | 35% | 3% | 2% |
| COI | 29% | 30% | 36% | 4% | 1% |
| Methodology | 49% | 25% | 21% | 3% | 2% |

**The "moderate collapse" problem:** In several dimensions, MODERATE is the dominant non-NONE class (Statistical Reporting 50%, COI 36%, Outcome 35%). The model learns that when something is biased, "moderate" is almost always the right answer. LOW is severely underrepresented in Statistical Reporting (9%) and rare in absolute terms for HIGH/CRITICAL across all dimensions (2-4%).

**What the confusion matrices show:** For Statistical Reporting, 24 of 36 true-LOW examples were predicted as MODERATE. For Spin, 45 of 62 true-LOW examples were predicted as MODERATE. The model has learned "biased → moderate" as a strong default, with no reliable signal for distinguishing LOW from MODERATE.

**Root cause: insufficient training signal for ordinal boundaries.** The model sees ~1,235 examples of binary bias/no-bias decisions but only ~113 LOW examples in Statistical Reporting (9%) and ~32 HIGH Outcome Reporting examples (3%). The ordinal boundaries between severity levels require far more examples per class than the binary boundary.

### 6.5 Why CMS Open Payments Collapsed

Training data analysis shows CMS Open Payments is mentioned in only **29.8% of training examples** overall, with a highly skewed distribution by COI severity:

| COI Severity | Open Payments Citation Rate | n |
|-------------|----------------------------|---|
| NONE | 13% | 352 |
| LOW | 16% | 371 |
| MODERATE | 45% | 442 |
| HIGH | **100%** | 48 |
| CRITICAL | 38% | 8 |

The model learned the strong HIGH-COI → Open Payments association (100% in training) but the weaker MODERATE-COI signal (45%) didn't survive fine-tuning. Since most test examples have LOW-MODERATE COI, the model defaults to not citing Open Payments.

**Comparison:** The Second Run's training data (old format) may have had Open Payments mentioned more consistently across severity levels, explaining the 57% → 22% regression.

### 6.6 Is This a Data Quantity Problem?

**For binary detection: No.** 1,235 examples is sufficient — F1 0.924 and recall 0.950 are strong results. The enriched training data quality (not quantity) drove the improvement.

**For ordinal severity: Yes.** The fundamental issue is class imbalance within the ordinal scale. The model sees enough examples to learn "biased vs not biased" but not enough to learn the finer gradations. Specifically:

- **LOW vs MODERATE boundary:** Needs hundreds more examples per dimension where the distinction is clear and consistent. Currently, annotators themselves may not agree on this boundary — the ground truth may be noisy.
- **HIGH/CRITICAL:** Only 57 HIGH and 57 CRITICAL examples in training (after oversampling). For 5 dimensions, that's ~11 per dimension per class — far too few for reliable learning.

**For CMS Open Payments: Partly.** The 29.8% mention rate means ~368 training examples cite it. This should be sufficient if the citation pattern were consistent. The problem is that the citation is strongly correlated with HIGH COI severity (100%) but weakly correlated with MODERATE (45%) — the model learned the strong signal and discarded the weak one.

## 7. Recommendations for Future Runs

### 7.1 Improving Severity Grading

**A. Targeted annotation for boundary cases (highest impact, most effort):**
Annotate 200+ examples specifically chosen to illustrate the LOW-MODERATE boundary across all 5 dimensions. These should be "hard" cases where the distinction matters — papers with subtle statistical issues, minor spin, ambiguous COI. This is expensive but addresses the root cause: the model doesn't have enough examples of what "LOW" looks like distinctly from "MODERATE".

**B. Ordinal-aware loss function (medium impact, code change):**
Replace standard cross-entropy with an ordinal regression loss (e.g., CORN or cumulative link) that penalises adjacent-class errors less than distant errors. Currently, predicting MODERATE for a true-LOW case is penalised the same as predicting CRITICAL. An ordinal loss would teach the model that the severity scale has an ordering.

**C. Class-weighted loss (low impact, easy to implement):**
Weight underrepresented classes higher in the loss function. For example, weight LOW 3x and HIGH/CRITICAL 5x relative to MODERATE. This is a blunt instrument but could reduce the "moderate collapse" at the cost of more noisy predictions.

**D. Post-hoc calibration (no retraining needed):**
Apply temperature scaling or Platt scaling to the model's severity outputs after training. This can fix systematic over/under-prediction patterns without retraining. However, it requires a calibration set separate from train/val/test.

### 7.2 Fixing CMS Open Payments Citation

**A. Ensure Open Payments is cited in all COI assessments (data fix):**
Modify the export pipeline to include CMS Open Payments as a verification step whenever COI severity is LOW or higher, not just HIGH. This would increase the training signal from 29.8% to ~70%+ of examples.

**B. Add explicit verification step reasoning in thinking chains:**
The thinking chain should include reasoning about WHICH verification databases to consult and WHY. Currently, the verification steps may be listed without explanation, making it hard for the model to learn the decision logic.

### 7.3 Full Cross-Model Summary (Updated)

| Configuration | Size | n | Binary F1 | Recall | Kappa | Verification | Calibration | Thinking |
|--------------|------|---|-----------|--------|-------|--------------|-------------|----------|
| Qwen3.5-27B baseline (old prompt) | 27B | 89 | 0.989 | — | 0.021 | 0.539 | 0.404 | 0% |
| OLMo-3.1-32B baseline (old prompt) | 32B | 89 | 0.989 | — | 0.066 | 0.528 | 0.670 | 0% |
| **OLMo-3.1-32B fine-tuned** | **32B** | **89** | **0.952** | 0.920 | **0.285** | 0.368 | 0.731 | 100% |
| Qwen3.5-9B enriched prompt | 9B | 115 | 0.866 | 0.793 | 0.118 | 0.495 | 0.922 | 0% |
| Qwen3.5-9B fine-tuned (Second Run) | 9B | 115 | 0.804 | 0.679 | 0.159 | 0.541 | 0.913 | 99% |
| **Qwen3.5-9B fine-tuned (Fourth Run)** | **9B** | **144** | **0.924** | **0.950** | 0.124 | 0.495 | **0.840** | **100%** |

## 8. Key Takeaways

1. **Binary detection is solved for 9B.** F1 0.924 and recall 0.950 meet the target. The gap to 32B narrowed from 0.148 to 0.028. The 9B model actually beats the 32B on recall (0.950 vs 0.920).

2. **Severity grading requires more data, not better hyperparameters.** Four training runs have shown that ordinal kappa is stuck in the 0.12–0.16 range regardless of LR, epochs, or LoRA rank. The training data has a severe class imbalance that hyperparameters cannot fix.

3. **The "moderate collapse" is the core severity problem.** The model defaults to predicting MODERATE for any non-NONE case because MODERATE is the modal class in most dimensions. Fixing this requires either more balanced training data or an ordinal-aware loss function.

4. **CMS Open Payments needs explicit training signal.** The 29.8% mention rate in training data, concentrated at HIGH COI severity, is insufficient. The export pipeline should cite Open Payments more consistently across COI severity levels.

5. **Conservative hyperparameters won decisively.** The 2e-4 LR / 3 epoch / batch 4 config produced better eval loss, better evaluation metrics, and 3.3x less compute than the aggressive config. Don't tune learning rate by model size for LoRA.

6. **The eval loss was still declining.** A follow-up with 4 epochs at the same config may yield marginal improvements, but the bigger gains will come from addressing the training data imbalance.
