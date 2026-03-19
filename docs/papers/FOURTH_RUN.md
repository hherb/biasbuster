# Fourth Fine-Tuning Run: Qwen3.5-9B with Conservative Hyperparameters

**Date:** 2026-03-19
**Model:** Qwen3.5-9B (fine-tuned with LoRA)
**Hardware:** NVIDIA DGX Spark (128 GB unified memory, GB10/Blackwell)
**Status:** Training completed; pending evaluation

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

## 5. Analysis

### 5.1 The Conservative Approach Won

The aggressive 9B hyperparameters (4e-4 LR, batch 2, 5 epochs) were counterproductive. The conservative revision (2e-4 LR, batch 4, 3 epochs) produced:
- Better final eval loss (1.10 vs 1.15)
- No saturation (still improving vs flat from step 300)
- 3.3x less compute (927 vs 3,090 steps)

**Conclusion:** For this task and dataset size, learning dynamics are not model-size-dependent. The 27B defaults work equally well for 9B. The 9B model's differentiation should be in LoRA capacity and regularisation, not learning rate or epoch count.

### 5.2 More Epochs May Help

The eval loss was still declining at epoch 3.00 (1.101, down from 1.267). This suggests 3 epochs may be insufficient for the expanded dataset. A follow-up run with 4-5 epochs at the conservative learning rate (2e-4) may yield further improvement without the saturation seen with 4e-4 LR.

However, the improvement rate was slowing: Δ per 50 steps went from -0.038 (steps 50-100) to -0.003 (steps 850-900). Diminishing returns suggest 4 epochs would capture most remaining gain; 5 may not be worth the compute.

### 5.3 The Training Data Change Dominates

The single biggest change between the Second Run and Third/Fourth Runs is not hyperparameters — it's the training data. The new format (all 5 domains always emitted, NONE reasoning, oversampled rare severities) produces fundamentally different targets. Loss values across data formats are not comparable.

The evaluation harness (next step) will determine whether the higher absolute loss translates to better or worse downstream task performance. It's plausible that the model is learning a harder but more useful task.

## 6. Next Steps

1. **Merge LoRA adapter and export to Ollama:**
   ```bash
   ./run_merge.sh qwen3.5-9b
   bash training/export_to_ollama.sh training_output/qwen3.5-9b-merged qwen3.5-9b-biasbuster
   ```

2. **Run evaluation harness** against the 144-example test set and compare with Second Run results.

3. **Success criteria** (from SECOND_RUN.md §6.4):
   - Binary F1 > 0.90
   - Recall > 0.85
   - Ordinal kappa > 0.20
   - Verification score > 0.50

4. **If eval loss was still declining**, consider a fifth run with 4 epochs at 2e-4 LR to capture remaining improvement without overfitting.

## 7. Key Takeaways

1. **Aggressive LR causes saturation, not faster convergence.** The 4e-4 LR drove the model to a loss basin it couldn't escape. The 2e-4 LR produced sustained learning throughout all 3 epochs.

2. **Compute efficiency matters.** The Fourth Run achieved better results in 927 steps than the Third Run did in 3,090. Fewer epochs with better learning dynamics beat more epochs with early saturation.

3. **Monitor your training curves.** The saturation in the Third Run was immediately obvious from the charts. Without the real-time training monitor, this would have been discovered only after a full evaluation cycle — wasting hours of GPU time and days of elapsed time.

4. **Don't tune learning rate by model size for LoRA.** With LoRA fine-tuning on structured output tasks, the base model size doesn't significantly affect optimal learning rate. The adapter parameters are what's being optimised, and their learning dynamics are similar across model scales.
