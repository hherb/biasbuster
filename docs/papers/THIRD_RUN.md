# Third Fine-Tuning Run: 9B-Optimised Hyperparameters + Training Data Fixes

**Date:** 2026-03-19
**Model:** Qwen3.5-9B (fine-tuned with LoRA)
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)
**Status:** Configuration complete, awaiting re-export and training

## 1. Motivation

The Second Run identified three root causes for the 9B model's underperformance:

1. **Missing domain training signal.** 73.6% of training examples omitted domains with "none" severity — the model never learned what an explicit "no concern" assessment looks like. This likely caused the low recall (0.679): the model learned to stay silent rather than assess "none."

2. **Severe class imbalance.** 83% of training examples were LOW or MODERATE severity, with only 17 HIGH+CRITICAL examples. The model defaulted to conservative middle-ground predictions, hurting ordinal kappa (0.159).

3. **32B hyperparameters on a 9B model.** LoRA rank 16 provides insufficient capacity for the smaller model. Learning rate 2e-4 is too conservative. Effective batch size 4 over-regularises on 920 examples.

## 2. Changes Made

### 2.1 Training Data: Always Emit All 5 Domains (export.py)

**Problem:** `build_structured_response()` skipped domains where `severity == "none"`. The model saw 0 explicit NONE assessments for Statistical Reporting, Outcome Reporting, COI, and Methodology.

**Fix:** Every domain is now always emitted with an explicit severity label:

```
## Statistical Reporting: NONE
- No significant statistical reporting concerns identified

## Spin: NONE
- Conclusions appear to accurately reflect the reported results
```

Previously, these sections were simply absent from the training data.

### 2.2 Training Data: NONE Reasoning in Thinking Chains (export.py)

**Problem:** The `_build_*_reasoning()` functions returned early for "none" severity, producing empty reasoning. The model never learned WHY a domain has no concerns.

**Fix:** Each domain builder now produces substantive NONE reasoning that references the positive indicators:

- **Statistical:** "Statistical reporting appears adequate: absolute measures are reported; baseline risk is provided."
- **Spin:** "Spin is NONE (Boutron taxonomy) — conclusions accurately reflect the reported results with appropriate acknowledgment of limitations."
- **Outcome:** "Primary outcomes appear patient-centred. No surrogate endpoint or composite disaggregation concerns."
- **COI:** "Publicly funded study with no apparent industry conflicts." (or generic if funding type unclear)
- **Methodology:** "No significant methodological red flags identified. Study design appears appropriate."

### 2.3 Training Data: Oversample Rare Severity Classes (export.py)

**Problem:** Only 11 HIGH and 6 CRITICAL examples in training (1.8% combined). The model rarely sees these patterns.

**Fix:** New `oversample_rare_severities()` function duplicates rare classes until each has at least 5% of the training set. Applied to train split only (val/test unchanged).

| Severity | Before | After | Change |
|----------|--------|-------|--------|
| NONE | 141 | 141 | — |
| LOW | 411 | 411 | — |
| MODERATE | 351 | 351 | — |
| HIGH | 11 | ~46 | ~4x |
| CRITICAL | 6 | ~46 | ~8x |
| **Total** | **920** | **~995** | **+8%** |

### 2.4 Hyperparameters: 9B-Optimised Overrides (training/configs.py)

Model-size-aware overrides are now applied automatically when `get_config("qwen3.5-9b")` is called.

| Parameter | Second Run (32B-matched) | Third Run (9B-optimised) | Rationale |
|-----------|-------------------------|--------------------------|-----------|
| `lora_r` | 16 | **32** | More LoRA capacity for smaller model |
| `lora_alpha` | 32 | **64** | Maintain alpha/r = 2 |
| `learning_rate` | 2e-4 | **4e-4** | 9B models tolerate higher LR |
| `gradient_accumulation_steps` | 4 | **2** | Smaller effective batch (2 vs 4); more update steps |
| `num_train_epochs` | 3 | **5** | More exposure; ~2300 steps vs ~690 |
| `lora_dropout` | 0.05 | **0.08** | Combat overfitting from oversampling |
| `warmup_ratio` | 0.1 | **0.06** | Shorter warmup (~138 steps vs ~69 at old ratio) |
| `save_total_limit` | 3 | **5** | More checkpoint options for longer run |
| `weight_decay` | 0.0 | **0.02** | Regularise adapter weights (new field) |
| `label_smoothing_factor` | 0.0 | **0.05** | Soften targets for better calibration (new field) |

32B model config is unchanged — existing OLMo runs are not affected.

### 2.5 Training Pipeline: New Config Fields (train_lora.py)

`weight_decay` and `label_smoothing_factor` are now passed through to `SFTConfig` and logged in `MetricsLoggerCallback` output. These fields existed in HuggingFace TrainingArguments but were not previously exposed in our config.

## 3. Training Configuration Summary

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3.5-9B |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.08 |
| Target modules | q, k, v, o, gate, up, down |
| Epochs | 5 |
| Batch size | 1 (gradient accumulation 2 = effective 2) |
| Learning rate | 4e-4 (cosine schedule, 6% warmup) |
| Weight decay | 0.02 |
| Label smoothing | 0.05 |
| Max sequence length | 4096 tokens |
| Precision | bf16 |
| Training examples | ~995 (after oversampling) |
| Estimated steps | ~2490 (995 / 2 effective batch * 5 epochs) |
| Estimated time | ~4 hours (more steps but 9B is fast) |

## 4. Expected Impact

| Metric | Second Run (current) | Target | Driver |
|--------|---------------------|--------|--------|
| Overall recall | 0.679 | >0.85 | Explicit NONE domains teach none/concern boundary |
| Overall F1 | 0.804 | >0.88 | Better recall + maintained precision |
| Per-dimension F1 (avg) | 0.70 | >0.75 | All 5 domains always assessed |
| Ordinal kappa | 0.159 | >0.25 | Rare class oversampling + more training |
| COI F1 | 0.639 | >0.75 | NONE COI examples + higher LoRA capacity |
| Calibration error | 0.913 | <0.70 | Label smoothing + oversampling |
| Verification score | 0.541 | >0.54 | No regression expected |

## 5. Execution Steps

```bash
# 1. Re-export training data with all fixes
uv run python pipeline.py --stage export

# 2. Verify: all 5 domains present in every example
grep -c "## Statistical Reporting:" dataset/export/alpaca/train.jsonl
grep -c "## Methodology:" dataset/export/alpaca/train.jsonl
# Both should equal line count of train.jsonl

# 3. Train with 9B-optimised config (automatic via model key)
./run_training.sh qwen3.5-9b

# 4. Merge adapter
./run_merge.sh qwen3.5-9b

# 5. Export to Ollama
bash training/export_to_ollama.sh training_output/qwen3.5-9b-merged qwen3.5-9b-biasbuster-v2

# 6. Evaluate
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-9b-biasbuster-v2 --endpoint-a http://localhost:11434 \
    --model-b qwen3.5-9bb-biasbuster --endpoint-b http://localhost:11434 \
    --mode fine-tuned --sequential \
    --num-ctx 4096 \
    --output eval_results/comparison/
```

## 6. Risk Mitigation

- **Overfitting from oversampling:** Mitigated by increased dropout (0.08), weight decay (0.02), label smoothing (0.05), and best-checkpoint selection via `load_best_model_at_end`.
- **5 epochs too many:** Early stopping picks the best checkpoint by eval_loss. If the model starts overfitting at epoch 3, the step-350ish checkpoint is selected automatically.
- **Longer output from NONE sections:** Adds ~50-100 tokens per example, well within the 4096 context window. Minor inference latency increase.
