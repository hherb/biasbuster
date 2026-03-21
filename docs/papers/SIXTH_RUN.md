# Sixth Run: GPT-OSS 20B MoE Fine-Tuning (LoRA, Attention-Only)

**Date:** 2026-03-21
**Model:** openai/gpt-oss-20b (21B total params, 3.6B active, 32 experts, top-4 routing)
**Hardware:** DGX Spark (ARM/Blackwell/GB10, 128 GB unified memory) via NGC Docker
**Training framework:** TRL SFTTrainer + PEFT LoRA
**Status:** Training completed, results under analysis

## 1. Background

The Fifth Run (FIFTH_RUN.md) established gpt-oss:20b as the strongest baseline ever evaluated -- achieving F1 0.918 and recall 0.941 *without any fine-tuning*, outperforming even the fine-tuned BiasBuster v5 (Qwen3.5-9B) on the 157-example test set. This made it the top fine-tuning priority.

This sixth run is the first attempt at LoRA fine-tuning the GPT-OSS 20B MoE model.

## 2. Infrastructure Work

Several issues had to be resolved before training could begin:

### 2.1 MXFP4 Weight Dequantization

GPT-OSS stores its MoE expert weights in MXFP4 (4-bit floating point) format. The MXFP4 backward pass is not implemented in PyTorch, so gradients cannot flow through frozen expert layers during LoRA training. The fix:

```python
from transformers import Mxfp4Config
quantization_config = Mxfp4Config(dequantize=True)
```

This dequantizes MXFP4 expert weights to BF16 on load, enabling gradient flow. Reference: [OpenAI fine-tuning cookbook](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers).

### 2.2 Attention Implementation

GPT-OSS uses alternating dense and locally banded sparse attention patterns. The OpenAI cookbook specifies `attn_implementation="eager"` rather than PyTorch SDPA. Our default was SDPA; this was changed to `"eager"` for GPT-OSS via `_MOE_OVERRIDES` in `configs.py`.

### 2.3 Device Placement (accelerate Offloading)

Initial attempts with `device_map="auto"` caused accelerate to offload parameters to the meta device, producing:

```
RuntimeError: Function MmBackward0 returned an invalid gradient at index 1
  - expected device meta but got cuda:0
```

The fix: default to `device_map={"": 0}` (single GPU), and for MXFP4 models that require `device_map="auto"`, constrain to GPU 0 via `max_memory` to prevent offloading. The dequantized model (~42 GB in BF16) fits in the DGX Spark's 128 GB unified memory.

### 2.4 Ollama Export Template

GPT-OSS uses the Harmony response format (`<|start|>`/`<|end|>` tokens), not ChatML (`<|im_start|>`/`<|im_end|>`). The export script was hardcoding ChatML for all models. Fixed by detecting `gpt-oss` in the model name and skipping the ChatML template override, letting Ollama use the tokenizer's built-in Harmony template.

## 3. Training Configuration (Initial)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | openai/gpt-oss-20b | Best baseline performer |
| LoRA targets | q_proj, k_proj, v_proj, o_proj | Attention-only (skip expert FFNs and router to avoid expert collapse) |
| LoRA rank (r) | 16 | Conservative |
| LoRA alpha | 32 | alpha/r = 2 |
| LoRA dropout | 0.05 | Light regularization |
| Learning rate | 1e-5 | Conservative for MoE (50x lower than dense default) |
| LR scheduler | Cosine | Standard |
| Warmup ratio | 0.1 | Standard |
| Epochs | 3 | Standard |
| Batch size | 1 | Memory-constrained |
| Gradient accumulation | 4 | Effective batch = 4 |
| Max sequence length | 4096 | Standard |
| Weight decay | 0.01 | Light regularization |
| Gradient checkpointing | True | Memory savings |
| MXFP4 dequantize | True | Required for backward pass |
| Attention implementation | eager | Required for GPT-OSS attention pattern |
| Trainable parameters | 7,962,624 / 20,922,719,808 (0.04%) | Extremely sparse LoRA |
| Training examples | 1,347 | Same dataset as prior runs |
| Validation examples | 155 | Same dataset as prior runs |

## 4. Training Results

### 4.1 Training Curves

Training completed ~1,200 steps (3 epochs x ~337 steps/epoch with grad_accum=4).

**Training loss:**
- Starts at ~4.8
- Drops rapidly to near 0 by step ~200
- Remains flat at ~0 for the remaining ~1,000 steps

**Eval loss:**
- Tracks training loss closely, also reaching near 0 by step ~200
- No divergence between train and eval loss

**Gradient norms:**
- Start at ~5, decline to ~0.5 by step ~200
- Collapse to near 0 after step ~300

**GPU memory:**
- Allocated: ~40 GiB (stable)
- Max allocated: ~53 GiB (peak)
- Well within the 128 GB unified memory budget

### 4.2 Analysis

The training dynamics show clear signs of **rapid saturation**:

1. **Convergence in ~200 steps.** The model memorizes the 1,347-example training set within roughly half of the first epoch. Steps 200-1,200 are wasted compute with near-zero gradients.

2. **Near-zero loss is suspicious.** For a complex structured-output task (multi-domain bias JSON with `<think>` reasoning chains), a training loss of essentially 0 indicates the model is reproducing training examples verbatim rather than learning generalizable patterns.

3. **0.04% trainable parameters is extremely sparse.** Only 7.9M parameters (all in attention layers) are being updated out of 20.9B total. This is unusually low -- the model has enormous frozen capacity that learns nothing, while the tiny LoRA adapter memorizes the training data.

4. **Eval loss tracking train loss** could mean either (a) genuine generalization, or (b) the eval set is too similar to training data. Given the near-zero values, overconfidence/poor calibration is the greater concern.

### 4.3 Comparison to Prior Runs

| Run | Model | Trainable % | Convergence | Final Loss | Concern |
|-----|-------|:-----------:|:-----------:|:----------:|---------|
| Run 1 | OLMo-3.1-32B | ~0.12% | ~500 steps | ~0.3 | Moderate overfitting |
| Run 2 | Qwen3.5-9B | ~0.22% | ~800 steps | ~0.5 | Some overfitting |
| Run 4 | Qwen3.5-9B | ~0.22% | ~600 steps | ~0.2 | Good convergence |
| **Run 6** | **gpt-oss-20b** | **0.04%** | **~200 steps** | **~0.0** | **Rapid saturation** |

GPT-OSS converges 3-4x faster than prior models despite having the smallest fraction of trainable parameters. This is likely because:
- The MoE model's strong pretrained representations require minimal adaptation
- The attention-only LoRA is too conservative -- the model adapts its attention patterns almost instantly to the small dataset
- The 1e-5 LR, while "conservative" for MoE, may still be too high given the model's strong baseline

## 5. Revised Configuration

Based on the rapid saturation observed, the `_MOE_OVERRIDES` in `configs.py` are updated:

| Parameter | Initial | Revised | Rationale |
|-----------|:-------:|:-------:|-----------|
| Learning rate | 1e-5 | **5e-6** | Slower convergence for better calibration |
| Epochs | 3 | **1** | Model converges in ~200 steps; 3 epochs wastes 2/3 of compute |
| LoRA dropout | 0.05 | **0.1** | Stronger regularization to combat memorization |
| Weight decay | 0.01 | **0.02** | Additional regularization |

Parameters kept unchanged:
- LoRA rank 16 / alpha 32 — increasing rank would risk faster memorization
- Target modules (attention-only) — adding expert FFNs risks expert collapse
- Gradient accumulation 4 — effective batch size of 4 is reasonable
- Warmup ratio 0.1 — still needed for LR schedule stability

### 5.1 Expected Impact

With 1 epoch and ~337 effective steps:
- Convergence should happen around step 100-150 (vs the previous 200), but with the lower LR and higher dropout, the loss should plateau at a healthier ~0.2-0.4 range instead of collapsing to 0
- The model should remain better calibrated, with less overconfident predictions
- Total training time should be ~1/3 of the initial run

## 6. Key Takeaways

1. **GPT-OSS 20B MoE can be fine-tuned on DGX Spark** once MXFP4 dequantization and device placement are handled correctly. GPU memory usage (~40-53 GiB) is comfortable.

2. **The model's strong baseline means it needs less training, not more.** Three epochs is excessive for a model that already achieves F1 0.918 zero-shot. One carefully calibrated epoch should suffice.

3. **Attention-only LoRA at 0.04% trainable params is viable** but requires aggressive regularization (high dropout, low LR, short training) to prevent memorization of the small dataset.

4. **Evaluation is the next critical step.** The revised hyperparameters need validation via the evaluation harness on the 157-example test set. The key question: does the fine-tuned gpt-oss:20b add `<think>` reasoning chains while maintaining or improving upon the 0.918 baseline F1?

## 7. Next Steps

1. **Re-run training** with the revised hyperparameters (1 epoch, LR 5e-6, dropout 0.1, weight_decay 0.02)
2. **Merge adapter** and export to Ollama
3. **Evaluate** on the 157-example test set (same as Fifth Run) for direct comparison with the unfine-tuned baseline
4. **Compare thinking chains** -- verify that the fine-tuned model produces `<think>` reasoning while the baseline does not
5. If evaluation looks good, consider whether expanding the training dataset would yield further gains beyond the attention-only LoRA ceiling
