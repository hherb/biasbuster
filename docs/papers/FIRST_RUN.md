# First Fine-Tuning Run: OLMo-3.1-32B Results and Analysis

**Date:** 2026-03-17
**Model:** OLMo-3.1-32B-Instruct (allenai/OLMo-3.1-32B-Instruct)
**Fine-tuned variant:** olmo-3.1-32b-biasbuster

## Training Configuration

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
| Hardware | NVIDIA DGX Spark (128 GB unified memory) |
| Container | NGC PyTorch 25.11-py3 |

## Results: Baseline (Zero-Shot) vs Fine-Tuned

### Overall Performance

| Metric | Baseline | Fine-Tuned | Delta | Winner |
|--------|----------|------------|-------|--------|
| Binary F1 | 0.989 | 0.952 | -0.037 | Baseline |
| Precision | 0.978 | 0.988 | +0.010 | Fine-tuned |
| Recall | 1.000 | 0.920 | -0.080 | Baseline |
| Ordinal kappa | 0.066 | 0.285 | +0.219 | Fine-tuned |
| Exact match | 0.517 | 0.584 | +0.067 | Fine-tuned |
| Within-one accuracy | 0.899 | 0.910 | +0.011 | Fine-tuned |
| MAE (severity) | 0.584 | 0.528 | -0.056 | Fine-tuned |
| Calibration error | 0.670 | 0.731 | +0.061 | Baseline |
| Verification score | 0.528 | 0.368 | -0.160 | Baseline |
| Parse failures | 0 | 0 | — | Tie |
| Thinking chains present | 0% | 100% | +100% | Fine-tuned |

### Per-Dimension Binary F1

| Dimension | Baseline | Fine-Tuned | Delta | Winner |
|-----------|----------|------------|-------|--------|
| Statistical reporting | 0.846 | 0.767 | -0.079 | Baseline |
| Spin | 0.896 | 0.921 | +0.025 | Fine-tuned |
| Outcome reporting | 0.940 | 0.822 | -0.118 | Baseline |
| Conflict of interest | 0.667 | 0.927 | +0.260 | Fine-tuned |
| Methodology | 0.852 | 0.726 | -0.126 | Baseline |

### Per-Dimension Ordinal Kappa (Severity Grading)

| Dimension | Baseline | Fine-Tuned | Delta | Winner |
|-----------|----------|------------|-------|--------|
| Statistical reporting | 0.110 | 0.275 | +0.165 | Fine-tuned |
| Spin | 0.172 | 0.281 | +0.109 | Fine-tuned |
| Outcome reporting | 0.033 | 0.095 | +0.062 | Fine-tuned |
| Conflict of interest | 0.088 | 0.201 | +0.113 | Fine-tuned |
| Methodology | 0.010 | 0.076 | +0.066 | Fine-tuned |

### Verification Source Citation Rates

| Source | Baseline | Fine-Tuned | Delta |
|--------|----------|------------|-------|
| ClinicalTrials.gov | 0.989 | 0.888 | -0.101 |
| Europe PMC | 1.000 | 0.966 | -0.034 |
| ORCID | 0.933 | 0.865 | -0.068 |
| Retraction Watch | 0.955 | 0.427 | -0.528 |
| CMS Open Payments | 0.854 | 0.157 | -0.697 |

## Analysis

### What Fine-Tuning Improved

1. **Severity grading across all 5 dimensions.** Ordinal kappa improved from near-chance (0.01-0.17) to modest agreement (0.08-0.28). The baseline model tended to dump everything into a single severity level (typically "moderate"); the fine-tuned model learned to distinguish none/low/moderate/high.

2. **Conflict of interest detection (F1 0.667 to 0.927).** The baseline missed half of all COI cases (recall 0.50 with perfect precision 1.0). The fine-tuned model catches 90% of COI while maintaining 95% precision. This is the single largest dimension-level improvement.

3. **Spin detection (F1 0.896 to 0.921).** Modest but consistent improvement, with better precision (0.885 to 0.921) while maintaining recall.

4. **Reasoning chains.** The fine-tuned model produces `<think>` blocks in 100% of outputs (mean 757 tokens), enabling interpretability. The baseline produced none.

5. **Selectivity.** The fine-tuned model is more discriminating — higher precision (0.988 vs 0.978) at the cost of recall. It no longer flags everything as biased.

### What Fine-Tuning Hurt

1. **Verification source citations dropped sharply.** Mean verification score fell from 0.528 to 0.368. The most dramatic losses:
   - CMS Open Payments: 0.854 to 0.157 (82% relative drop)
   - Retraction Watch: 0.955 to 0.427 (55% relative drop)

2. **Binary F1 regressed on 3/5 dimensions.** Statistical reporting (-0.079), outcome reporting (-0.118), and methodology (-0.126) all dropped. The model became too conservative on these dimensions.

3. **Calibration error worsened slightly** (0.670 to 0.731). The model's predicted bias probabilities are less well-calibrated than the baseline.

### The Baseline's F1 is Partially Illusory

The baseline's near-perfect F1 (0.989) is somewhat misleading. It achieved 100% recall by over-flagging: on several dimensions it predicted zero true negatives. Its severity kappas near zero (0.01-0.17) show it could not meaningfully grade severity — it essentially said "moderate bias" for everything. The fine-tuned model is more clinically useful despite the lower headline F1, because it actually differentiates severity levels and correctly identifies clean studies.

## Root Cause Analysis

### 1. Training System Prompt is Too Short

The training system prompt in `export.py` is 23 lines (~320 tokens). The annotation prompt that generated the training labels (`annotators/llm_prelabel.py`) is 148 lines (~1125 tokens) with 9 operational definitions. The fine-tuned model learned from the impoverished prompt and lost:

- **Verification database selection criteria** — the training prompt mentions databases by name but doesn't teach WHEN to cite each one (e.g., "industry-funded → check CMS Open Payments")
- **Operational definitions** — relative_only, surrogate endpoints, Boutron spin levels, COI disclosure rules are defined in the annotation prompt but absent from training
- **Calibration guidance** — "not every industry-funded study is biased" appears in the annotation prompt but not in training

### 2. Synthetic Thinking Chains are Incomplete

The `build_thinking_chain()` function in `export.py` generates reasoning from annotation flags, but only covers 3 of 5 dimensions:
- Statistical reporting (relative_only only)
- Spin (moderate/high only)
- COI (industry funding only)

Outcome reporting and methodology reasoning are entirely missing. No thinking chain includes database selection reasoning ("I should check X because Y").

### 3. Verification Steps are Passed Through Without Synthesis

The `build_structured_response()` function simply dumps `annotation.get("recommended_verification_steps", [])` verbatim. Many annotations have sparse or missing verification steps — particularly for CMS Open Payments (mentioned in only ~23% of industry-funded examples). The training data doesn't consistently teach which databases to cite.

## Approach for Second Run

### Changes to the Training Pipeline

Based on this analysis, we will make the following changes before the second fine-tuning run:

#### 1. Enrich the Training System Prompt

Port operational definitions from the annotation prompt into the training prompt in `export.py`:
- Verification database selection criteria with URLs and usage conditions
- Boutron spin classification (HIGH/MODERATE/LOW/NONE criteria)
- Operational definitions for relative_only, surrogate vs patient-centred outcomes, methodology thresholds, COI disclosure rules
- Calibration reminder

Target: ~70 lines (~800 tokens). Token budget analysis shows this fits within the 4096 max_seq_length with headroom.

#### 2. Expand Thinking Chains to All 5 Dimensions

Extend `build_thinking_chain()` to generate reasoning for:
- Outcome reporting (surrogate endpoints, outcome switching)
- Methodology (per-protocol, short follow-up, enrichment design)
- Database selection reasoning for every example ("This is industry-funded, so checking CMS Open Payments is recommended")
- A verification summary at the end of each chain listing which databases to check

#### 3. Synthesize Missing Verification Steps

Add a `_synthesize_verification_steps()` function that generates verification steps from annotation flags when the original annotation lacks them:
- Industry-funded → always include CMS Open Payments
- Always include ClinicalTrials.gov
- Undisclosed COI → suggest ORCID + Europe PMC
- Always include Retraction Watch

This ensures every training example has comprehensive verification steps, teaching the model the consistent pattern of citing all relevant databases.

#### 4. Sync Evaluation Prompt

Import the training prompt into `evaluation/harness.py` instead of maintaining a duplicate, ensuring the evaluation uses exactly the same prompt the model was trained on.

### Expected Impact

- **Verification source rates**: Should recover to baseline levels or higher, since every training example will now contain explicit database citations with selection reasoning
- **Binary F1 on regressed dimensions**: Should improve as operational definitions teach the model finer-grained detection criteria
- **Ordinal kappa**: Should maintain or improve the gains from the first run
- **Calibration**: Should improve with explicit calibration guidance in the system prompt

### Unchanged Parameters

The LoRA configuration, learning rate, epoch count, and other hyperparameters will remain identical to isolate the effect of prompt and training data improvements.
