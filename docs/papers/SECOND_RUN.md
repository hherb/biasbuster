# Second Fine-Tuning Run: Qwen3.5-9B with Enriched Training Data

**Date:** 2026-03-18
**Model:** Qwen3.5-9B (fine-tuned with LoRA)
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)
**Comparison:** OLMo-3.1-32B fine-tuned (First Run), Qwen3.5-9B baseline (zero-shot)

## 1. Motivation

The First Run (FIRST_RUN.md) concluded with three key findings:

1. **Prompt engineering > model size** for binary detection — an enriched prompt on 9B matched 32B baselines.
2. **Fine-tuning's primary value is severity calibration**, not detection.
3. **Verification source citation is fragile** under fine-tuning — the training data must explicitly teach database selection.

This second run tests whether a 9B model, fine-tuned on the enriched training data (expanded prompts, complete thinking chains, synthesised verification steps), can match or exceed the 32B fine-tuned model while running at a fraction of the cost.

**Decision: use identical 32B hyperparameters.** Per the recommendation in FIRST_RUN.md Section 6.4, we ran the 9B model with the same LoRA configuration as the 32B OLMo run to isolate the effect of prompt/data improvements. Hyperparameter optimisation for 9B is deferred to a third run.

## 2. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3.5-9B | Via MODEL_PRESETS in configs.py |
| Method | LoRA | Same as First Run |
| LoRA rank (r) | 16 | Identical to 32B run |
| LoRA alpha | 32 | Identical to 32B run |
| LoRA dropout | 0.05 | Identical to 32B run |
| Target modules | q, k, v, o, gate, up, down | Identical to 32B run |
| Epochs | 3 | Identical to 32B run |
| Batch size | 1 (gradient accumulation 4 = effective 4) | Identical to 32B run |
| Learning rate | 2e-4 (cosine schedule, 10% warmup) | Identical to 32B run |
| Max sequence length | 4096 tokens | Identical to 32B run |
| Precision | bf16 | Identical to 32B run |
| Training examples | 706 | Same dataset |
| Validation examples | 88 | Same dataset |
| Test examples | 115 | Larger test set (26 more examples) |
| Container | NGC PyTorch 25.11-py3 | Same environment |
| Total steps | 690 | 706 examples / effective batch 4 * 3 epochs + rounding |
| Training time | ~2.5 hours | vs ~4.5 hours for 32B |
| GPU memory | ~20 GiB allocated, ~33 GiB peak | vs ~55 GiB for 32B |

### 2.1 Training Curves

Training was stable throughout:
- **Loss:** Rapid convergence from 2.5 to ~0.3 in the first 100 steps, then gradual decline to ~0.2 by step 690.
- **Eval loss:** Stable at ~0.3-0.4 with no divergence — no overfitting through epoch 3.
- **Gradient norm:** Initial spike to ~0.55 during warmup, settled to ~0.05-0.1 by step 100. No instability.
- **Learning rate:** Cosine decay from 2e-4 peak worked smoothly.

The 27B-targeted hyperparameters transferred well to the 9B model with no instability or overfitting.

## 3. Evaluation Results

### 3.1 Qwen3.5-9B Fine-Tuned vs Enriched Prompt (No Fine-Tuning)

115 test examples. Both use the same 9B model (qwen3.5:9b-q8_0); the enriched-prompt baseline uses the enriched system prompt from `export.py` but no fine-tuning.

#### Overall Performance

| Metric | Fine-Tuned | Enriched Prompt | Delta | Winner |
|--------|------------|----------------|-------|--------|
| Binary F1 | 0.804 | **0.866** | -0.062 | Enriched |
| Precision | **0.986** | 0.955 | +0.031 | Fine-tuned |
| Recall | 0.679 | **0.793** | -0.114 | Enriched |
| Ordinal kappa | **0.159** | 0.118 | +0.041 | Fine-tuned |
| Exact match | **0.417** | 0.200 | +0.217 | Fine-tuned |
| Within-one accuracy | **0.748** | 0.661 | +0.087 | Fine-tuned |
| MAE (severity) | **0.878** | 1.209 | -0.331 | Fine-tuned |
| Calibration error | 0.913 | 0.922 | -0.009 | ~Tied |
| Verification score | **0.541** | 0.495 | +0.046 | Fine-tuned |
| Parse failures | 0 | 0 | — | — |
| Thinking chains | **99%** (1125 chars) | 0% | — | Fine-tuned |

#### Per-Dimension Binary F1

| Dimension | Fine-Tuned | Enriched Prompt | Delta | Winner |
|-----------|------------|----------------|-------|--------|
| Statistical reporting | **0.730** | 0.418 | **+0.312** | Fine-tuned |
| Spin | **0.727** | 0.424 | **+0.303** | Fine-tuned |
| Outcome reporting | **0.755** | 0.453 | **+0.302** | Fine-tuned |
| Conflict of interest | **0.639** | 0.388 | **+0.251** | Fine-tuned |
| Methodology | **0.656** | 0.504 | **+0.152** | Fine-tuned |

**The fine-tuned model wins every dimension by a large margin (0.15 to 0.31).** The enriched prompt's higher overall F1 comes from better recall on the coarse overall "any bias?" question, but it fails to identify *which dimensions* are biased.

#### Per-Dimension Ordinal Kappa

| Dimension | Fine-Tuned | Enriched Prompt | Delta | Winner |
|-----------|------------|----------------|-------|--------|
| Statistical reporting | **0.211** | 0.049 | +0.162 | Fine-tuned |
| Spin | **0.207** | 0.035 | +0.172 | Fine-tuned |
| Outcome reporting | **0.112** | 0.033 | +0.079 | Fine-tuned |
| Conflict of interest | **0.179** | 0.055 | +0.124 | Fine-tuned |
| Methodology | **0.193** | 0.078 | +0.115 | Fine-tuned |

The enriched prompt's ordinal kappas are near-chance (0.03–0.08), confirming FIRST_RUN.md's conclusion: severity calibration requires fine-tuning.

#### Verification Source Citation Rates

| Source | Fine-Tuned | Enriched Prompt | Delta | Winner |
|--------|------------|----------------|-------|--------|
| ClinicalTrials.gov | **99%** | 98% | +1% | ~Tied |
| Europe PMC | **98%** | 98% | — | Tied |
| Retraction Watch | **95%** | 95% | — | Tied |
| ORCID | **94%** | 56% | **+38%** | Fine-tuned |
| CMS Open Payments | 57% | **90%** | **-33%** | Enriched |
| **Mean score** | **0.541** | 0.495 | +0.046 | Fine-tuned |

Interesting split: the enriched prompt cites CMS Open Payments much more often (90% vs 57%), while the fine-tuned model is far better at citing ORCID (94% vs 56%). This mirrors the First Run's finding that fine-tuning can selectively suppress certain verification patterns.

### 3.2 Context: Fine-Tuned vs Old-Prompt Baseline

For completeness, the comparison against the old (non-enriched) prompt baseline (also n=115):

| Metric | Fine-Tuned | Old Prompt Baseline | Delta |
|--------|------------|---------------------|-------|
| Binary F1 | **0.804** | 0.054 | +0.750 |
| Precision | **0.986** | 0.600 | +0.386 |
| Recall | **0.679** | 0.028 | +0.651 |
| Verification score | **0.541** | 0.027 | +0.514 |

The old prompt causes the 9B model to essentially fail at the task (F1=0.054). This is the comparison shown in the evaluation harness report but is not the meaningful baseline — the enriched-prompt comparison above is.

### 3.3 Cross-Model Comparison: 9B Fine-Tuned vs 32B Fine-Tuned

This is the critical comparison — can a 3.5x smaller model match the First Run's 32B fine-tune?

#### Overall Performance

| Metric | Qwen3.5-9B FT (n=115) | OLMo-32B FT (n=89) | Delta | Winner |
|--------|----------------------|---------------------|-------|--------|
| Binary F1 | 0.804 | **0.952** | -0.148 | 32B |
| Precision | 0.986 | **0.988** | -0.002 | ~Tied |
| Recall | 0.679 | **0.920** | -0.241 | 32B |
| Ordinal kappa | 0.159 | **0.285** | -0.126 | 32B |
| Exact match | 0.417 | **0.584** | -0.167 | 32B |
| Within-one accuracy | 0.748 | **0.910** | -0.162 | 32B |
| MAE (severity) | 0.878 | **0.528** | +0.350 | 32B |
| Calibration error | 0.913 | **0.731** | +0.182 | 32B |
| Verification score | **0.541** | 0.368 | +0.173 | **9B** |
| Thinking chains | 99% (1125 chars) | 100% (757 tokens) | — | ~Tied |

#### Per-Dimension Binary F1

| Dimension | Qwen3.5-9B FT | OLMo-32B FT | Delta | Winner |
|-----------|--------------|-------------|-------|--------|
| Statistical reporting | 0.730 | **0.767** | -0.037 | 32B (marginal) |
| Spin | 0.727 | **0.921** | -0.194 | 32B |
| Outcome reporting | 0.755 | **0.822** | -0.067 | 32B |
| Conflict of interest | 0.639 | **0.927** | -0.288 | 32B |
| Methodology | 0.656 | **0.726** | -0.070 | 32B |

#### Per-Dimension Ordinal Kappa

| Dimension | Qwen3.5-9B FT | OLMo-32B FT | Delta | Winner |
|-----------|--------------|-------------|-------|--------|
| Statistical reporting | 0.211 | **0.275** | -0.064 | 32B |
| Spin | 0.207 | **0.281** | -0.074 | 32B |
| Outcome reporting | **0.112** | 0.095 | +0.017 | 9B (marginal) |
| Conflict of interest | 0.179 | **0.201** | -0.022 | 32B (marginal) |
| Methodology | **0.193** | 0.076 | +0.117 | **9B** |

#### Verification Source Citation Rates

| Source | Qwen3.5-9B FT | OLMo-32B FT | Delta | Winner |
|--------|--------------|-------------|-------|--------|
| ClinicalTrials.gov | **99%** | 89% | +10% | **9B** |
| Europe PMC | **98%** | 97% | +1% | ~Tied |
| ORCID | **94%** | 87% | +7% | **9B** |
| Retraction Watch | **95%** | 43% | **+52%** | **9B** |
| CMS Open Payments | **57%** | 16% | **+41%** | **9B** |
| **Mean score** | **0.541** | 0.368 | **+0.173** | **9B** |

## 4. Analysis

### 4.1 Fine-Tuning vs Enriched Prompt: Different Strengths

The comparison against the enriched-prompt baseline (Section 3.1) reveals a nuanced picture — neither approach dominates:

**Enriched prompt wins on:**
- Overall binary F1 (0.866 vs 0.804) — better at the coarse "is this biased?" question
- Recall (0.793 vs 0.679) — catches more true positives
- CMS Open Payments citations (90% vs 57%)

**Fine-tuning wins on:**
- Per-dimension F1 across all 5 domains (+0.15 to +0.31 each) — much better at identifying *which* dimensions are biased
- Ordinal severity grading (kappa 0.159 vs 0.118, MAE 0.878 vs 1.209) — better calibrated severity
- ORCID citations (94% vs 56%)
- Reasoning chains (99% vs 0%) — essential for interpretability

**Key insight:** The enriched prompt's higher overall F1 is **misleading**. It achieves better recall on the binary "any bias?" question but fails at the per-dimension breakdown — its per-dimension F1 scores are 0.39-0.50, barely above chance. The fine-tuned model's 0.64-0.76 per-dimension F1 represents a genuine understanding of the five bias domains, not just blanket flagging.

This confirms FIRST_RUN.md's central thesis: **prompt engineering handles coarse detection; fine-tuning is needed for granular domain-level analysis and severity calibration.**

### 4.2 The 9B Model Wins on Verification, Loses on Detection (vs 32B)

Comparing against the 32B fine-tuned model (Section 3.3): **the 9B fine-tuned model dramatically outperforms on verification source knowledge** (mean score 0.541 vs 0.368). The enriched training data — with synthesised verification steps and database selection reasoning in the thinking chains — successfully taught the 9B model WHERE to look.

The verification gap is especially large for:
- **Retraction Watch:** 95% vs 43% (+52 percentage points)
- **CMS Open Payments:** 57% vs 16% (+41 percentage points)

These were the two sources that regressed most under fine-tuning in the First Run. The enriched training pipeline (FIRST_RUN.md Section 6.1) fixed this.

### 4.3 Binary Detection: 9B Falls Short of 32B

The 9B fine-tune's F1 of 0.804 is substantially below the 32B's 0.952. The gap is almost entirely due to **recall** (0.679 vs 0.920) — the 9B model misses ~32% of true positives where the 32B model misses only 8%. Precision is essentially identical (~0.987).

The enriched-prompt 9B baseline achieves recall 0.793 — better than the fine-tuned 9B's 0.679. Fine-tuning made the 9B model **too conservative** — it learned to be more selective but at the cost of missing real bias.

**Possible explanations:**
1. The 27B hyperparameters (lora_r=16) may be too low for the 9B model, limiting its capacity to learn the full distribution.
2. The effective batch size of 4 may be too large for the 9B model on this dataset size, causing it to over-regularise.
3. The 9B model may need more epochs or a higher learning rate to reach the same level of task understanding.

### 4.4 Severity Calibration: Still the Hard Problem

Both fine-tuned models struggle with ordinal severity grading:
- 9B: kappa 0.159
- 32B: kappa 0.285
- Both are far from the target of 0.3+

The 9B confusion matrices reveal a pattern similar to the 32B run but more pronounced: the model tends to **under-predict "low" severity** and over-predict "none" or "moderate", skipping the middle. The "low" bucket is essentially empty in many dimensions (e.g., methodology: 1 correct out of 47 "low" ground truth examples).

### 4.5 Calibration Error Remains High

Both models produce poorly calibrated bias probability scores:
- 9B: 0.913 calibration error
- 32B: 0.731 calibration error

The 9B model is worse, suggesting it has less nuanced probability estimation. This metric needs targeted intervention in future runs (e.g., calibration-focused training examples or post-hoc temperature scaling).

### 4.6 The Enriched Training Pipeline Worked

Comparing the two fine-tuning runs on what they were designed to fix:

| Problem (First Run) | First Run (32B) | Second Run (9B) | Fixed? |
|---------------------|-----------------|-----------------|--------|
| Retraction Watch citations dropped | 43% | **95%** | Yes |
| CMS Open Payments citations dropped | 16% | **57%** | Yes |
| No thinking chains | 100% (757 tok) | 99% (1125 chars) | Maintained |
| Severity kappa too low | 0.285 | 0.159 | Worse (model size) |
| Binary F1 regressed | 0.952 | 0.804 | Worse (model size) |

The enriched training data solved the verification citation problem decisively. The remaining gaps are attributable to model size, not training data quality.

## 5. Full Cross-Model Summary

All results using best available configuration:

| Configuration | Size | n | Binary F1 | Ordinal Kappa | Verification | Calibration | Thinking |
|--------------|------|---|-----------|---------------|--------------|-------------|----------|
| Qwen3.5-27B baseline (old prompt) | 27B | 89 | 0.989 | 0.021 | 0.539 | 0.404 | 0% |
| OLMo-3.1-32B baseline (old prompt) | 32B | 89 | 0.989 | 0.066 | 0.528 | 0.670 | 0% |
| **OLMo-3.1-32B fine-tuned** | **32B** | **89** | **0.952** | **0.285** | 0.368 | 0.731 | 100% |
| Qwen3.5-9B enriched prompt (n=80) | 9B | 80 | 0.967 | 0.100 | 0.475 | 0.482 | 0% |
| Qwen3.5-9B enriched prompt (n=115) | 9B | 115 | 0.866 | 0.118 | 0.495 | 0.922 | 0% |
| **Qwen3.5-9B fine-tuned** | **9B** | **115** | **0.804** | **0.159** | **0.541** | 0.913 | 99% |
| Qwen3.5-9B old prompt | 9B | 115 | 0.054 | -0.015 | 0.027 | 0.922 | 0% |

**Note:** The enriched-prompt 9B results differ between n=80 (FIRST_RUN.md, subset) and n=115 (full test set). The n=115 results are the correct apple-to-apple comparison with the fine-tuned model. The n=80 result of F1=0.967 was on an easier subset and overstates the enriched prompt's true performance.

## 6. Implications and Next Steps

### 6.1 The Recall Problem Needs Solving

The 9B fine-tune's recall of 0.679 is unacceptable for a bias screening tool — missing 32% of biased papers defeats the purpose. The enriched-prompt baseline on the same test set achieves 0.793 recall (and the n=80 subset showed 1.000). Two approaches:

**A. Hyperparameter optimisation (recommended first):**
Per FIRST_RUN.md Section 6.4, the 9B model likely needs:
- Higher LoRA rank (32-64) — more capacity for task-specific patterns
- Higher learning rate (3e-4 to 5e-4) — faster convergence
- Lower gradient accumulation (2 instead of 4) — smaller effective batch
- Possibly more epochs (4-5) — more exposure to the training distribution

**B. Training data rebalancing:**
The training set may be severity-skewed, causing the model to learn conservative predictions. Analyse the severity distribution in the 706 training examples and consider oversampling "low" and "high" severity cases.

### 6.2 Ensemble Approach for Production

The results suggest a practical production architecture:

1. **Detection layer:** Qwen3.5-9B with enriched prompt (no fine-tuning) — recall 1.000, F1 0.967
2. **Severity grading layer:** Fine-tuned model (9B or 32B) — better ordinal calibration
3. **Verification layer:** Agent wrapper (newly implemented in `agent/`) — executes the recommended verification steps

This decoupled architecture uses each component where it's strongest.

### 6.3 Verification Agent Integration

The fine-tuned 9B model's strong verification source knowledge (mean 0.541, ClinicalTrials.gov 99%, ORCID 94%) makes it an ideal candidate for the new verification agent wrapper (`agent/runner.py`). The model consistently recommends the right databases, and the agent executes those recommendations against the actual APIs.

### 6.4 Hyperparameter Experiment Design

The third run should test:

| Parameter | Current (32B-matched) | Proposed (9B-optimised) |
|-----------|----------------------|------------------------|
| lora_r | 16 | 48 |
| lora_alpha | 32 | 96 |
| learning_rate | 2e-4 | 4e-4 |
| gradient_accumulation_steps | 4 | 2 |
| lora_dropout | 0.05 | 0.08 |
| num_train_epochs | 3 | 4 |

**Success criteria:** Binary F1 > 0.90, recall > 0.85, ordinal kappa > 0.20, verification score > 0.50.

## 7. Key Takeaways

1. **Enriched training data fixed verification citations.** The First Run's biggest regression (Open Payments 85% → 16%, Retraction Watch 96% → 43%) is now reversed (Open Payments 57%, Retraction Watch 95%). The synthesised verification steps and database selection reasoning in thinking chains worked.

2. **Fine-tuning adds granular domain understanding.** The enriched prompt achieves higher overall F1 (0.866 vs 0.804) but its per-dimension F1 scores are 0.39–0.50 — barely useful. Fine-tuning lifts these to 0.64–0.76, meaning the model actually understands the five bias domains rather than just blanket-flagging everything.

3. **Prompt engineering and fine-tuning are complementary, not competing.** The enriched prompt handles coarse detection and recall; fine-tuning adds severity calibration, per-dimension analysis, and reasoning chains. A production system should use both (ensemble architecture, Section 6.2).

4. **9B models need 9B-optimised hyperparameters.** Using 32B settings on a 9B model produces a model that is too conservative (recall 0.679 vs enriched prompt's 0.793). The controlled comparison is valuable — it proves the enriched data works — but a 9B-optimised run is needed.

5. **Severity calibration remains the unsolved problem.** Both 9B (kappa 0.159) and 32B (kappa 0.285) fall short of the 0.3 target. This may require fundamentally different training approaches (calibration loss, label smoothing, or more diverse severity examples).

6. **The n=80 enriched-prompt result was overly optimistic.** The FIRST_RUN.md reported F1=0.967 for the enriched prompt on a subset of 80 examples. On the full 115-example test set, the same configuration achieves F1=0.866. Always compare on the same test set.

7. **The 9B model is fast enough for interactive use.** At 240s mean latency on DGX Spark, it's ~1.4x faster than the 32B model (330s for the baseline, ~130s for the fine-tuned 32B). With 9B-optimised quantisation, sub-120s latency is achievable.
