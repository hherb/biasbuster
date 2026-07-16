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

## 7. Export Strategy: Adapter Overlay (Preserving MXFP4)

### 7.1 Problem with Full Merge

The initial export approach — merge adapter into base, save as BF16, re-quantize to GGUF — has two problems for GPT-OSS:

1. **`save_pretrained()` fails** because the MXFP4→BF16 reverse transform is not implemented (`NotImplementedError` in `revert_weight_conversion`). Workaround: clear `quantization_config` before saving — but this produces a 42 GB BF16 model.

2. **Ollama supports MXFP4 natively** for GPT-OSS (~14 GB), so dequantizing to BF16 and re-quantizing to Q4/Q8 is wasteful and lossy.

### 7.2 Solution: ADAPTER Directive

Ollama's `ADAPTER` Modelfile directive applies a LoRA adapter on top of the base model at load time — no merge needed:

```dockerfile
FROM gpt-oss:20b
ADAPTER /path/to/final_adapter/
```

This preserves the base model's native MXFP4 format while applying our attention-only LoRA adapter. The adapter directory (`final_adapter/`) contains `adapter_config.json` + `adapter_model.safetensors` — produced directly by the training script.

A new export script (`training/export_adapter_to_ollama.sh`) implements this:

```bash
bash training/export_adapter_to_ollama.sh gpt-oss:20b \
    training_output/gpt-oss-20b-lora/final_adapter \
    gpt-oss-20b-biasbuster
```

### 7.3 When to Use Which Export Path

| Path | Script | Use Case |
|------|--------|----------|
| Adapter overlay | `export_adapter_to_ollama.sh` | GPT-OSS (MXFP4 preserved, ~14 GB + tiny adapter) |
| Full merge + safetensors | `export_to_ollama.sh` | Dense models (Qwen, OLMo) |
| Full merge + GGUF | `export_to_ollama.sh --gguf Q4_K_M` | Dense models when disk/memory constrained |

## 8. Evaluation Results (Post-Revised Training)

Training was re-run with the revised hyperparameters (1 epoch, LR 5e-6, dropout 0.1, weight_decay 0.02), and the adapter was exported to Ollama via `export_adapter_to_ollama.sh` (preserving MXFP4 base + LoRA overlay, ~14 GB total). The fine-tuned model (`hherb/gpt-oss-20b-biasbuster:latest`) was evaluated on the same 157-example test set as the Fifth Run, enabling direct comparison with the unfine-tuned baseline.

### 8.1 Overall Performance

| Metric | gpt-oss:20b (baseline) | gpt-oss-20b-biasbuster (fine-tuned) | Δ |
|--------|:---:|:---:|:---:|
| Binary F1 | 0.918 | **0.938** | +0.020 |
| Precision | **0.895** | 0.883 | -0.012 |
| Recall | 0.941 | **1.000** | +0.059 |
| Accuracy | 0.854 | **0.885** | +0.031 |
| Ordinal κ | **0.158** | 0.042 | -0.116 |
| Calibration error | 0.866 | 0.866 | 0.000 |
| Verification score | 0.591 | **0.624** | +0.033 |
| Parse failures | 0 | 0 | — |
| Mean latency (s) | **76.7** | 123.6 | +46.9 |
| Tokens/sec | **31.8** | 27.6 | -4.2 |
| Thinking chains | 0% | 0% | — |

**Key finding:** Fine-tuning improved binary detection (F1 +0.020, recall to perfect 1.000) and mean verification score (+0.033), but **severely degraded severity grading** (κ 0.158 → 0.042). The fine-tuned model did not produce `<think>` reasoning chains, despite the training data containing them.

### 8.2 Per-Dimension Binary F1

| Dimension | Baseline | Fine-Tuned | Δ |
|-----------|:---:|:---:|:---:|
| Statistical reporting | 0.805 | 0.800 | -0.005 |
| Spin | 0.748 | **0.795** | +0.047 |
| Outcome reporting | 0.752 | **0.788** | +0.036 |
| COI | 0.751 | **0.828** | +0.077 |
| Methodology | 0.793 | 0.797 | +0.004 |

The fine-tuned model wins or ties on every dimension. The largest gain is in **COI detection (+0.077 F1)**, driven by massively improved recall (0.734 → 0.973). Spin and outcome reporting also improved meaningfully.

### 8.3 Statistical Significance (Pairwise Tests)

| Test | Dimension | p-value | Significant? | Winner |
|------|-----------|---------|:---:|:---:|
| McNemar | Statistical reporting (binary) | 0.739 | No | Tie |
| McNemar | Spin (binary) | 0.669 | No | Tie |
| McNemar | Outcome reporting (binary) | 0.695 | No | Tie |
| McNemar | COI (binary) | 0.252 | No | Tie |
| McNemar | Methodology (binary) | 0.846 | No | Tie |
| Wilcoxon | Statistical reporting (ordinal) | **0.049** | **Yes** | **Baseline** |
| Wilcoxon | Spin (ordinal) | 0.118 | No | Tie |
| Wilcoxon | Outcome reporting (ordinal) | 0.973 | No | Tie |
| Wilcoxon | COI (ordinal) | **0.017** | **Yes** | **Baseline** |
| Wilcoxon | Methodology (ordinal) | **<0.001** | **Yes** | **Baseline** |

**No binary detection differences reach significance** — the fine-tuned model's improvements in spin, outcome, and COI are trends, not proven gains at n=157.

**Three ordinal tests are significant, all favouring the baseline.** The fine-tuned model's severity grading is significantly worse on statistical reporting (MAE 0.777 → 0.949, p=0.049), COI (MAE 0.771 → 0.994, p=0.017), and methodology (MAE 0.713 → 1.121, p<0.001).

### 8.4 Ordinal Severity Comparison

| Dimension | Metric | Baseline | Fine-Tuned | Winner |
|-----------|--------|:---:|:---:|:---:|
| Statistical reporting | κ | **0.197** | 0.153 | Baseline |
| Statistical reporting | MAE | **0.777** | 0.949 | Baseline |
| Spin | κ | **0.186** | 0.119 | Baseline |
| Spin | MAE | **1.185** | 1.344 | Baseline |
| Outcome reporting | κ | **0.127** | 0.068 | Baseline |
| Outcome reporting | MAE | 1.089 | 1.102 | ~Tie |
| COI | κ | **0.244** | 0.048 | Baseline |
| COI | MAE | **0.771** | 0.994 | Baseline |
| Methodology | κ | **0.254** | 0.062 | Baseline |
| Methodology | MAE | **0.713** | 1.121 | Baseline |

The baseline wins severity grading on every dimension. The fine-tuned model systematically over-predicts severity — when it detects bias, it defaults to "moderate" or "high" regardless of ground truth.

### 8.5 Verification Source Citation Rates

| Source | Baseline | Fine-Tuned | Δ |
|--------|:---:|:---:|:---:|
| CMS Open Payments | **95.5%** | 89.8% | -5.7% |
| ClinicalTrials.gov | 93.6% | **96.2%** | +2.6% |
| ORCID | **97.5%** | 84.7% | -12.8% |
| Retraction Watch | **95.5%** | 77.7% | -17.8% |
| Europe PMC | **97.5%** | 90.5% | -7.0% |
| **Mean score** | 0.591 | **0.624** | +0.033 |

The baseline cites individual verification sources more consistently, but the fine-tuned model has a higher mean verification *score* (0.624 vs 0.591), suggesting it uses citations more appropriately when it does include them.

### 8.6 Efficiency

| Metric | Baseline | Fine-Tuned |
|--------|:---:|:---:|
| Mean latency | **76.7s** | 123.6s |
| Tokens/sec | **31.8** | 27.6 |
| Error rate | 0% | 0% |

The fine-tuned model is ~60% slower, likely generating longer outputs.

## 9. Analysis

### 9.1 The Attention-Only LoRA Ceiling

The evaluation reveals a clear pattern: fine-tuning with attention-only LoRA at 0.04% trainable parameters **improved the model's detection sensitivity** (perfect recall, better COI/spin F1) but **destroyed its severity calibration** (κ dropped from 0.158 to 0.042). This is the same trade-off seen in Run 1 (OLMo-3.1-32B), but more extreme.

The likely mechanism: attention-only LoRA can redirect the model's attention to bias-relevant features in the abstract (improving detection), but the tiny adapter lacks capacity to encode the nuanced ordinal severity scale. The frozen expert FFN weights — which contain the model's domain knowledge about severity gradations — cannot be updated, so the model defaults to its training data's modal severity class (MODERATE).

### 9.2 Missing Thinking Chains

The fine-tuned model produces 0% `<think>` reasoning chains despite the training data containing them. This is likely because:
1. The attention-only LoRA cannot redirect output formatting behaviour that is controlled by the frozen FFN/MoE layers
2. The Harmony chat template may interact differently with chain-of-thought generation
3. One epoch of training at 5e-6 LR may be insufficient to teach the model a new output structure

This is the most disappointing result — chain-of-thought reasoning was a primary motivation for fine-tuning beyond the strong baseline.

### 9.3 Comparison with Prior Runs

| Configuration | Type | Size | n | Binary F1 | Recall | Ordinal κ | Verification | Thinking |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-oss:20b baseline | Baseline | 20B MoE | 157 | 0.918 | 0.941 | **0.158** | 0.591 | 0% |
| **gpt-oss:20b fine-tuned (Run 6)** | **Fine-tuned** | **20B MoE** | **157** | **0.938** | **1.000** | 0.042 | **0.624** | **0%** |
| Qwen3.5-9B fine-tuned (Run 4) | Fine-tuned | 9B | 144 | 0.924 | 0.950 | 0.124 | 0.495 | 100% |
| OLMo-3.1-32B fine-tuned (Run 1) | Fine-tuned | 32B | 89 | 0.952 | 0.920 | 0.285 | 0.368 | 100% |

Run 6 achieves the **highest binary F1 (0.938) and recall (1.000) of any model evaluated**, but the **lowest severity κ (0.042)** of any fine-tuned model, and **no thinking chains**. It is the best screening model (misses nothing) but the worst calibrated assessor.

## 10. Key Takeaways

1. **Attention-only LoRA on MoE improves detection but degrades calibration.** The 0.04% trainable parameter fraction is sufficient to redirect attention to bias-relevant features, but insufficient to teach severity grading or output formatting (thinking chains).

2. **The fine-tuned gpt-oss:20b is the strongest binary detector.** F1 0.938 with perfect recall makes it the best screening model. For use cases where "flag everything suspicious" is acceptable, it outperforms all other configurations.

3. **The unfine-tuned gpt-oss:20b remains the best calibrated model.** κ 0.158 and balanced per-dimension severity grading make it the best choice when severity ratings matter.

4. **Chain-of-thought training failed for MoE attention-only LoRA.** A different approach is needed — either including expert FFN layers in LoRA targets (risking expert collapse), using a higher-rank adapter, training for more steps, or using a higher learning rate.

5. **Production recommendation: ensemble.** Use the fine-tuned model for binary screening (perfect recall), then use the unfine-tuned baseline for severity assessment and verification recommendations on flagged papers. This combines the strengths of both.

## 11. Next Steps

1. **Investigate thinking chain failure.** Test whether increasing LoRA rank (32 → 64), training for 2-3 epochs, or adding MLP layers to LoRA targets produces `<think>` chains without expert collapse.

2. **Test ensemble approach.** Build a two-stage pipeline: fine-tuned gpt-oss:20b for screening → unfine-tuned gpt-oss:20b for severity assessment.

3. **Expand LoRA targets cautiously.** Try including gate_proj/up_proj/down_proj for non-expert MLP layers only (skip expert FFNs and router). This would increase trainable parameters while preserving MoE routing stability.

4. **Consider longer training with current config.** The 1-epoch, 5e-6 LR setting may be too conservative for teaching new output formatting. Try 2-3 epochs to see if thinking chains emerge with more exposure.
