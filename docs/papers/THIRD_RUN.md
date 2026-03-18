# Third Fine-Tuning Run: Qwen3.5-9B with Expanded Dataset and Revised Hyperparameters

**Date:** 2026-03-19
**Model:** Qwen3.5-9B (fine-tuned with LoRA)
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)
**Status:** Training completed; pending evaluation

## 1. Motivation

The Second Run (SECOND_RUN.md) ended with two actionable recommendations:

1. **9B-optimised hyperparameters (Section 6.4):** Proposed aggressive settings — 4e-4 LR, 5 epochs, effective batch size 2, LoRA rank 32. These were implemented and tested.
2. **Training data fixes (Sections 2.1–2.3):** Always emit all 5 domains (including NONE), substantive NONE reasoning in thinking chains, and oversampling of rare severity classes.

Both changes were applied simultaneously. This run documents the result.

## 2. Training Data Changes

The training dataset was expanded via the ongoing annotation pipeline and the export fixes from the pre-run plan:

| | Second Run | Third Run | Delta |
|---|-----------|-----------|-------|
| Training examples | 706 | **1,235** | +529 (+75%) |
| Validation examples | 88 | **142** | +54 (+61%) |
| Test examples | 115 | **144** | +29 (+25%) |
| Total | 909 | **1,521** | +612 (+67%) |

Key data improvements (unchanged from plan):
- **All 5 domains always emitted**, including explicit NONE assessments
- **Substantive NONE reasoning** in thinking chains (previously empty)
- **Rare severity oversampling** (HIGH/CRITICAL boosted to ~5% of training set)

## 3. The 9B-Aggressive Hyperparameters Failed

### 3.1 What Was Tried

The 9B-optimised overrides from `_9B_OVERRIDES` in `training/configs.py`:

| Parameter | 27-32B Default | 9B-Aggressive (attempted) |
|-----------|---------------|---------------------------|
| learning_rate | 2e-4 | **4e-4** |
| num_train_epochs | 3 | **5** |
| gradient_accumulation_steps | 4 | **2** (effective batch 2) |
| warmup_ratio | 0.1 | **0.06** |
| lora_r | 16 | **32** |
| lora_alpha | 32 | **64** |
| lora_dropout | 0.05 | **0.08** |
| weight_decay | 0.0 | **0.02** |
| label_smoothing_factor | 0.0 | **0.05** |

### 3.2 Training Curves: Early Saturation

Training completed 3,090 steps (5 epochs). The curves revealed a clear problem:

- **Training loss** dropped rapidly from ~7.0 to ~2.5 in the first ~100 steps, then **plateaued at ~1.5–2.0 for the remaining ~2,900 steps** with no meaningful improvement.
- **Eval loss** was flat throughout (~0.8–1.0), showing the model stopped learning useful generalisations very early.
- **Gradient norms** settled to a low, steady ~0.5 — consistent with the model not making meaningful parameter updates.
- **Learning rate** ramped up to 4e-4 peak and decayed via cosine, but the high peak LR appears to have driven the model to a loss basin it couldn't escape.

In short: the model learned everything it could in the first ~10% of training, then sat idle for the remaining ~90%. Five epochs and 3,090 steps were mostly wasted compute.

### 3.3 Diagnosis

The 4e-4 learning rate combined with a small effective batch size of 2 caused the model to converge too aggressively to a local minimum. The 5 epochs compounded this — more passes over the data couldn't escape the plateau.

Comparing with the Second Run (which used the 27B defaults on 706 examples and showed gradual, sustained learning), the aggressive hyperparameters were counterproductive despite the larger dataset.

## 4. Hyperparameter Revision

Based on the training curve evidence, `_9B_OVERRIDES` was revised to keep only the genuinely 9B-specific settings while matching the 27B defaults for learning dynamics:

| Parameter | 9B-Aggressive (failed) | Revised | Rationale |
|-----------|------------------------|---------|-----------|
| learning_rate | 4e-4 | **2e-4** | Match 27B; 4e-4 caused early saturation |
| num_train_epochs | 5 | **3** | Match 27B; 5 wasted compute on a flat plateau |
| gradient_accumulation_steps | 2 | **4** | Match 27B; smoother gradients, less aggressive convergence |
| warmup_ratio | 0.06 | **0.1** | Match 27B; longer warmup with lower LR |
| save_total_limit | 5 | **3** | Match 27B; fewer checkpoints needed |
| lora_r | 32 | **32** | Kept; more capacity for smaller model |
| lora_alpha | 64 | **64** | Kept; alpha/r=2 |
| lora_dropout | 0.08 | **0.08** | Kept; regularisation |
| weight_decay | 0.02 | **0.02** | Kept; regularisation |
| label_smoothing_factor | 0.05 | **0.05** | Kept; calibration |

**Philosophy:** The 9B model should differ from 27B in LoRA capacity and regularisation, not in learning rate or epoch count. The Second Run's stable curves with 27B defaults suggest the learning dynamics are model-size-agnostic for this task.

## 5. Next Steps

1. **Re-run training** with revised hyperparameters:
   ```bash
   ./run_training.sh qwen3.5-9b
   ```

2. **Merge and export:**
   ```bash
   ./run_merge.sh qwen3.5-9b
   bash training/export_to_ollama.sh training_output/qwen3.5-9b-merged qwen3.5-9b-biasbuster
   ```

3. **Evaluate** against Second Run results. Success criteria (from SECOND_RUN.md §6.4):
   - Binary F1 > 0.90
   - Recall > 0.85
   - Ordinal kappa > 0.20
   - Verification score > 0.50

4. **If the loss plateau persists** even with conservative hyperparameters, investigate:
   - Curriculum learning (easy examples first)
   - Per-dimension loss weighting
   - Larger LoRA rank (64+)
   - Whether the 9B model's capacity is genuinely saturated for this task

## 6. Infrastructure Fix

The training monitor was displaying overlaid data from the current and previous training runs on the same charts.

**Root cause:** `metrics.jsonl` is append-only. When a new training run starts for the same model, the new header and metrics are appended after the previous run's data. The monitor (`utils/training_monitor.py`) accumulated all metrics without resetting across run boundaries.

**Fix:** `MetricsReader.poll()` now clears `self.metrics` and resets `self.completed` whenever a new `header` record is encountered, ensuring only the latest run's data is displayed.

## 7. Key Takeaways

1. **Aggressive hyperparameters hurt the 9B model.** The SECOND_RUN.md §6.4 proposal (4e-4 LR, 5 epochs, small batch) was well-motivated but wrong in practice. The training curves provide clear evidence of early saturation.

2. **Learning dynamics appear model-size-agnostic.** The 27B defaults (2e-4 LR, 3 epochs, effective batch 4) produced stable gradual learning for both 32B and 9B models on this task. The performance gap between model sizes comes from capacity, not hyperparameters.

3. **The 9B-specific value is in LoRA capacity and regularisation.** Higher rank (32 vs 16), dropout (0.08 vs 0.05), weight decay (0.02), and label smoothing (0.05) are justified for a smaller model. The learning rate and epoch count are not.

4. **Monitor your training curves.** The saturation was obvious from the charts but would have been invisible without the real-time monitor. The `metrics.jsonl` + NiceGUI dashboard caught this early enough to correct course without wasting a full evaluation cycle.
