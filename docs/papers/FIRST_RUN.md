# First Fine-Tuning Run: Results, Analysis, and Path Forward

**Date:** 2026-03-17 / 2026-03-18
**Models tested:** OLMo-3.1-32B, Qwen3.5-27B, Qwen3.5-9B
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)

## 1. Training Configuration (OLMo-3.1-32B LoRA)

| Parameter | Value |
|-----------|-------|
| Base model | allenai/OLMo-3.1-32B-Instruct |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down |
| Epochs | 3 |
| Batch size | 1 (gradient accumulation 4 = effective 4) |
| Learning rate | 2e-4 (cosine schedule, 10% warmup) |
| Max sequence length | 4096 tokens |
| Precision | bf16 |
| Training examples | 706 |
| Validation examples | 88 |
| Test examples | 89 |
| Container | NGC PyTorch 25.11-py3 |

## 2. Evaluation Results

### 2.1 OLMo-3.1-32B: Baseline vs Fine-Tuned

Both evaluated on 89 test examples with the zero-shot prompt (baseline) and fine-tuned prompt respectively.

#### Overall Performance

| Metric | Baseline | Fine-Tuned | Delta | Winner |
|--------|----------|------------|-------|--------|
| Binary F1 | 0.989 | 0.952 | -0.037 | Baseline |
| Precision | 0.978 | 0.988 | +0.010 | Fine-tuned |
| Recall | 1.000 | 0.920 | -0.080 | Baseline |
| Ordinal kappa | 0.066 | 0.285 | +0.219 | **Fine-tuned** |
| Exact match | 0.517 | 0.584 | +0.067 | Fine-tuned |
| Within-one accuracy | 0.899 | 0.910 | +0.011 | Fine-tuned |
| MAE (severity) | 0.584 | 0.528 | -0.056 | Fine-tuned |
| Calibration error | 0.670 | 0.731 | +0.061 | Baseline |
| Verification score | 0.528 | 0.368 | -0.160 | Baseline |
| Thinking chains | 0% | 100% | — | Fine-tuned |

#### Per-Dimension Binary F1

| Dimension | Baseline | Fine-Tuned | Delta |
|-----------|----------|------------|-------|
| Statistical reporting | 0.846 | 0.767 | -0.079 |
| Spin | 0.896 | 0.921 | +0.025 |
| Outcome reporting | 0.940 | 0.822 | -0.118 |
| Conflict of interest | 0.667 | **0.927** | **+0.260** |
| Methodology | 0.852 | 0.726 | -0.126 |

#### Per-Dimension Ordinal Kappa (Severity Grading)

| Dimension | Baseline | Fine-Tuned | Delta |
|-----------|----------|------------|-------|
| Statistical reporting | 0.110 | **0.275** | +0.165 |
| Spin | 0.172 | **0.281** | +0.109 |
| Outcome reporting | 0.033 | 0.095 | +0.062 |
| Conflict of interest | 0.088 | 0.201 | +0.113 |
| Methodology | 0.010 | 0.076 | +0.066 |

#### Verification Source Citation Rates

| Source | Baseline | Fine-Tuned | Delta |
|--------|----------|------------|-------|
| ClinicalTrials.gov | 0.989 | 0.888 | -0.101 |
| Europe PMC | 1.000 | 0.966 | -0.034 |
| ORCID | 0.933 | 0.865 | -0.068 |
| Retraction Watch | 0.955 | 0.427 | **-0.528** |
| CMS Open Payments | 0.854 | 0.157 | **-0.697** |

### 2.2 Qwen3.5-27B Baseline (Zero-Shot, Old Prompt)

89 test examples with the original zero-shot prompt. Included for cross-model comparison.

| Metric | Value |
|--------|-------|
| Binary F1 | 0.989 |
| Precision | 0.978 |
| Recall | 1.000 |
| Ordinal kappa | 0.021 |
| Calibration error | 0.404 |
| Verification score | 0.539 |

Per-dimension F1: statistical 0.853, spin 0.921, outcome 0.950, COI 0.928, methodology 0.863.

### 2.3 Qwen3.5-9B: The Prompt Experiment

**This was the most significant finding of the entire evaluation cycle.**

We ran Qwen3.5-9B (q8_0 quantisation) with two different system prompts — the original short zero-shot prompt and the new enriched prompt (with operational definitions, verification database criteria, and calibration guidance). No fine-tuning was applied in either case.

#### Overall Performance

| Metric | Old Prompt (n=88) | New Enriched Prompt (n=80) | Delta |
|--------|-------------------|---------------------------|-------|
| Binary F1 | 0.455 | **0.967** | **+0.512** |
| Precision | 0.301 | 0.936 | +0.635 |
| Recall | 0.926 | 1.000 | +0.074 |
| Ordinal kappa | 0.032 | 0.100 | +0.068 |
| Calibration error | 0.160 | 0.482 | +0.322 (worse) |
| Verification score | 0.443 | 0.475 | +0.032 |

#### Per-Dimension Binary F1

| Dimension | Old Prompt | New Enriched Prompt | Delta |
|-----------|-----------|---------------------|-------|
| Statistical reporting | 0.405 | **0.880** | **+0.476** |
| Spin | 0.226 | **0.814** | **+0.588** |
| Outcome reporting | 0.328 | **0.831** | **+0.502** |
| Conflict of interest | 0.415 | **0.922** | **+0.507** |
| Methodology | 0.391 | **0.826** | **+0.435** |

#### Per-Dimension Ordinal Kappa

| Dimension | Old Prompt | New Enriched Prompt | Delta |
|-----------|-----------|---------------------|-------|
| Statistical reporting | 0.048 | 0.127 | +0.078 |
| Spin | -0.042 | 0.204 | +0.246 |
| Outcome reporting | 0.024 | **0.290** | +0.265 |
| Conflict of interest | 0.031 | **0.247** | +0.216 |
| Methodology | 0.028 | 0.221 | +0.192 |

#### Verification Source Citation Rates

| Source | Old Prompt | New Enriched Prompt | Delta |
|--------|-----------|---------------------|-------|
| ClinicalTrials.gov | 1.000 | 1.000 | — |
| Europe PMC | 0.898 | 0.900 | +0.002 |
| Retraction Watch | 0.943 | 0.963 | +0.019 |
| CMS Open Payments | 0.807 | 0.863 | +0.056 |
| ORCID | 0.330 | **0.538** | **+0.208** |

### 2.4 Cross-Model Summary Table

All results using the best available prompt for each configuration:

| Configuration | Size | Binary F1 | Ordinal Kappa | Verification | Calibration |
|--------------|------|-----------|---------------|--------------|-------------|
| Qwen3.5-27B baseline (old prompt) | 27B | 0.989 | 0.021 | 0.539 | 0.404 |
| OLMo-3.1-32B baseline (old prompt) | 32B | 0.989 | 0.066 | 0.528 | 0.670 |
| OLMo-3.1-32B fine-tuned | 32B | 0.952 | **0.285** | 0.368 | 0.731 |
| **Qwen3.5-9B (new prompt, no fine-tuning)** | **9B** | **0.967** | 0.100 | 0.475 | 0.482 |
| Qwen3.5-9B (old prompt) | 9B | 0.455 | 0.032 | 0.443 | 0.160 |

## 3. Analysis

### 3.1 What Fine-Tuning Improved (OLMo-32B)

1. **Severity grading across all 5 dimensions.** Ordinal kappa improved from near-chance (0.01-0.17) to modest agreement (0.08-0.28). The baseline dumped everything into "moderate"; the fine-tuned model learned to distinguish severity levels.

2. **Conflict of interest detection (F1 0.667 → 0.927).** The baseline missed half of all COI cases (recall 0.50 with perfect precision 1.0). The fine-tuned model catches 90% while maintaining 95% precision. Largest single-dimension improvement.

3. **Spin detection (F1 0.896 → 0.921).** Modest but consistent improvement.

4. **Reasoning chains.** 100% of fine-tuned outputs include `<think>` blocks (mean 757 tokens), enabling interpretability.

### 3.2 What Fine-Tuning Hurt (OLMo-32B)

1. **Verification source citations dropped sharply.** CMS Open Payments: 0.854 → 0.157 (82% drop). Retraction Watch: 0.955 → 0.427 (55% drop). The training data didn't consistently teach database selection.

2. **Binary F1 regressed on 3/5 dimensions.** Statistical reporting (-0.079), outcome reporting (-0.118), and methodology (-0.126). The model became too conservative.

3. **Calibration error worsened** (0.670 → 0.731).

### 3.3 The Baseline's F1 is Partially Illusory

The 32B baselines' near-perfect F1 (0.989) is misleading. They achieved 100% recall by over-flagging everything — predicting zero true negatives on several dimensions. Severity kappas near zero (0.01-0.17) show they couldn't meaningfully grade severity. They essentially said "moderate bias" for everything.

### 3.4 The Prompt is Doing Most of the Heavy Lifting

**This is the critical finding.** The enriched prompt alone — with no fine-tuning — transformed Qwen3.5-9B from unusable (F1=0.455) to competitive with 32B baselines (F1=0.967). The improvement is massive across all 5 dimensions:

- The old prompt caused the 9B model to flag everything (precision ~0.25-0.30, massive false positive rate)
- The enriched prompt with operational definitions fixed precision to 0.72-0.96 range
- Verification source citations remained strong (no regression as seen with fine-tuning)
- ORCID citation rate actually improved significantly (0.33 → 0.54)

**What this means:** The 32B baselines' good F1 was not because 32B models "understand bias better" — it was because larger models are more robust to vague prompts. Once the prompt provides explicit operational definitions, a 9B model matches or exceeds 32B baseline performance on binary detection.

### 3.5 Remaining Weakness: Severity Calibration

The one area where the prompt-only 9B model falls short is **ordinal severity grading** (kappa 0.100 vs fine-tuned OLMo's 0.285). The confusion matrices reveal the pattern:

- The 9B model with new prompt **overshoots severity** — rating "low" as "high", "moderate" as "high/critical"
- Calibration error is 0.482 (vs 0.160 with old prompt, but the old prompt's low calibration error was an artifact of low detection)
- The fine-tuned OLMo achieved better severity grading through exposure to training examples with graded labels

This confirms that **fine-tuning's primary value is teaching severity calibration, not detection**.

## 4. Root Cause Analysis

### 4.1 Training System Prompt Was Too Short

The training prompt in `export.py` was 23 lines (~320 tokens). The annotation prompt that generated the training labels was 148 lines (~1125 tokens) with 9 operational definitions. The fine-tuned model learned from an impoverished prompt and lost:

- **Verification database selection criteria** — when to cite each source
- **Operational definitions** — relative_only, surrogate endpoints, Boutron spin levels
- **Calibration guidance** — "not every industry-funded study is biased"

### 4.2 Synthetic Thinking Chains Were Incomplete

`build_thinking_chain()` only covered 3 of 5 dimensions:
- Statistical reporting (relative_only only)
- Spin (moderate/high only)
- COI (industry funding only)

Outcome reporting and methodology reasoning were entirely missing. No chain included database selection reasoning.

### 4.3 Verification Steps Passed Through Without Synthesis

`build_structured_response()` dumped `annotation.get("recommended_verification_steps", [])` verbatim. Many annotations had sparse or missing verification steps — particularly for CMS Open Payments (mentioned in only ~23% of industry-funded examples).

## 5. Implications for Model Size Strategy

The prompt experiment fundamentally changes our approach to model selection:

### 5.1 Small Models Are Viable for Detection

A 9B model with a well-crafted prompt achieves binary F1 = 0.967 — competitive with 27-32B baselines. This means:

- **7-8B models** (Qwen2.5-7B, Llama 3.1-8B, Phi-4) are very likely viable for bias detection
- **4B models** may work for binary detection but will likely struggle with:
  - Severity calibration (already hard for 9B)
  - Structured JSON output reliability
  - Verification step generation (requires factual knowledge)
- Fine-tuning should focus on what prompts can't solve: **severity calibration**

### 5.2 Inference Cost Reduction

Running a 9B model vs 32B:
- ~3.5x less VRAM
- ~3-4x faster inference
- Can run unquantised on commodity hardware
- Multiple instances possible on DGX Spark for parallel evaluation

### 5.3 Decoupled Architecture Option

If 4B models prove viable for detection but not verification:
- Use a small fine-tuned model for binary bias detection + severity grading
- Use a larger model (or rule-based system) for verification step generation
- This trades architectural complexity for inference cost savings

## 6. Approach for Next Steps

### 6.1 Immediate: Enrich Training Pipeline (Done)

These changes have already been implemented:

1. **Enriched training system prompt** in `export.py` — ported operational definitions, verification database criteria with URLs and usage conditions, Boutron spin classification, calibration guidance (~70 lines, ~800 tokens)
2. **Expanded thinking chains** to all 5 dimensions with database selection reasoning
3. **Synthesised missing verification steps** from annotation flags
4. **Synced evaluation prompt** — `evaluation/harness.py` imports from `export.py` instead of maintaining a duplicate

### 6.2 Next: Evaluate Smaller Models as Baselines

Run baseline evaluations with the enriched prompt on:
- Qwen3.5-9B (done — results in Section 2.3)
- Additional 7-9B models as candidates for fine-tuning

### 6.3 Then: Fine-Tune Best Small Model Candidate

Fine-tune the best-performing small model (likely 9B class) with:
- Enriched training data (from 6.1)
- Same LoRA configuration as the OLMo run (to isolate size effects)
- Focus evaluation on **severity calibration improvement** — this is the metric that fine-tuning should improve most

### 6.4 Unchanged Parameters

LoRA configuration, learning rate, epoch count, and other hyperparameters will remain identical across runs to isolate the effect of prompt/data improvements and model size.

## 7. Key Takeaways

1. **Prompt engineering > model size** for binary bias detection. An enriched prompt on 9B matched 32B baselines.
2. **Fine-tuning's primary value is severity calibration**, not detection. All models struggle with ordinal grading; fine-tuning is the most effective way to improve it.
3. **Verification source citation is fragile** under fine-tuning. The training data must explicitly and consistently teach database selection patterns, or the model forgets them.
4. **Headline F1 is misleading** without examining severity kappa. A model that flags everything as "moderate bias" gets high F1 but is clinically useless.
5. **9B models are strong candidates** for the next fine-tuning run, offering 3-4x inference speedup with minimal detection quality loss.
