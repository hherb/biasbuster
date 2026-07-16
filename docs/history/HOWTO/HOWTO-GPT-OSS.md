# HOWTO: Fine-Tune GPT-OSS 20B for BiasBuster

End-to-end guide for LoRA fine-tuning OpenAI's GPT-OSS 20B (MoE) on
DGX Spark and deploying via Ollama with MXFP4 preserved.

**Hardware tested:** NVIDIA DGX Spark (ARM/Blackwell/GB10, 128 GB unified memory)
**Model:** openai/gpt-oss-20b (21B total params, 3.6B active, 32 experts, top-4 routing)
**Result:** ~12.8 GB merged model with native MXFP4 expert weights

## Prerequisites

```bash
# Project setup
cd ~/src/biasbuster
uv sync

# Pull base model into Ollama (for serving after merge)
ollama pull gpt-oss:20b

# Ensure HF cache is user-owned (Docker runs may leave root-owned files)
sudo chown -R $USER:$USER ~/.cache/huggingface/

# Training data must be exported first
uv run python pipeline.py --stage export
```

## 1. Training

### 1.1 Architecture Considerations

GPT-OSS is a Mixture-of-Experts model. Key differences from dense models:

| Aspect | Dense (Qwen, OLMo) | GPT-OSS MoE |
|--------|:---:|:---:|
| LoRA targets | All 7 (q,k,v,o,gate,up,down) | Attention only (q,k,v,o) |
| Active params | 100% | 17% (3.6B of 21B) |
| Expert weights | BF16 | MXFP4 (4-bit microscaling) |
| Learning rate | 2e-4 | 5e-6 |
| Trainable % | ~0.2% | 0.04% |
| Risk | Overfitting | Expert collapse, rapid saturation |

LoRA targets only attention layers. Expert FFN weights and the router are
frozen to avoid expert collapse (where all tokens route to the same experts).

### 1.2 MXFP4 Dequantization

GPT-OSS stores expert weights in MXFP4 format. The MXFP4 backward pass is
not implemented in PyTorch, so we dequantize to BF16 on load for training:

```python
from transformers import Mxfp4Config
quantization_config = Mxfp4Config(dequantize=True)
```

This is handled automatically by `_MOE_OVERRIDES` in `training/configs.py`.
Reference: [OpenAI fine-tuning cookbook](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers).

### 1.3 Run Training

```bash
# Full training run (inside NGC Docker container)
./run_training.sh gpt-oss-20b

# Smoke test (5 steps)
./run_training.sh gpt-oss-20b --max-steps 5

# Override hyperparameters from CLI
./run_training.sh gpt-oss-20b --lr 5e-6 --epochs 1 --lora-rank 16
```

The script launches the NGC PyTorch container, installs TRL/PEFT, and runs
`training/train_lora.py` with MoE-specific configuration automatically applied.

### 1.4 Output Format: Single-Channel with `<think>` Tags

GPT-OSS uses the Harmony multi-channel response format at the template level,
but fine-tuning with attention-only LoRA **cannot teach the model to use
Harmony channels** (analysis vs final). The model falls back to producing
literal `<think>` tags in the `content` field instead of using channel tokens.

Therefore, GPT-OSS training data keeps everything in a single channel:
`<think>` reasoning blocks and JSON output are both in `content`. The
`make_formatting_func()` in `data_utils.py` detects Harmony tokenizers
and skips the channel split automatically.

See `docs/ROUND_3.md` §4.5 for the Harmony channel experiment and
`docs/MISTAKES_TO_ROUND_3.md` §5 for why it failed.

### 1.5 Current Hyperparameters

Set in `training/configs.py` `_MOE_OVERRIDES`, tuned through Runs 6-9:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 5e-6 | Halved from 1e-5 after Run 6 saturated in ~200 steps |
| num_train_epochs | 2 | Run 7 showed loss still descending at 1 epoch; gradient collapse in Run 8 was from 8e-6 LR, not epoch count |
| lora_r / alpha | 32 / 64 | Doubled from 16/32 — more capacity for dual-channel output |
| lora_dropout | 0.1 | Doubled from 0.05 to combat rapid memorisation |
| weight_decay | 0.02 | Doubled from 0.01 for additional regularisation |
| attn_implementation | eager | Required for GPT-OSS attention pattern (not SDPA) |
| mxfp4_dequantize | True | Required for backward pass through frozen expert layers |

### 1.5 GPU Memory

Training uses ~40 GiB allocated / ~53 GiB peak. Fits comfortably in the
DGX Spark's 128 GB unified memory.

### 1.6 Monitor Training

In a separate terminal (on the host, not in Docker):

```bash
uv run python -m utils.training_monitor --metrics-dir training_output/gpt-oss-20b-lora
```

Opens a web dashboard at http://127.0.0.1:8001 with live loss curves,
learning rate schedule, GPU memory, and gradient norms.

## 2. Merging (Surgical, MXFP4 Preserved)

### 2.1 Why Not Standard Merge?

The standard merge path (`merge_adapter.py`) loads the full model through
transformers, which:
1. Dequantizes MXFP4 expert weights to BF16 (~42 GB)
2. Calls `save_pretrained()` which tries to revert the dequantization
3. Fails with `NotImplementedError` in `revert_weight_conversion`

Even bypassing `save_pretrained()`, the output would be ~42 GB BF16 instead
of ~14 GB MXFP4.

### 2.2 Surgical Merge

`training/merge_adapter_surgical.py` operates directly on the safetensors
files at the byte level:

1. **Copies** each shard file byte-for-byte (MXFP4 experts → untouched)
2. **Parses** the safetensors JSON header to locate attention tensor offsets
3. **Reads** attention tensors as raw BF16 bytes → float32
4. **Applies** LoRA delta: `merged = base + (lora_B @ lora_A) * (alpha / r)`
5. **Writes** float32 → BF16 bytes back in place at the same offset

No torch dependency. Uses only numpy, huggingface_hub, and safetensors
(header parsing). Runs on the host without Docker.

```bash
# Clean any previous failed merge attempts
sudo rm -rf training_output/gpt-oss-20b-merged/

# Run (auto-detects gpt-oss and uses surgical merge)
./run_merge.sh gpt-oss-20b
```

Output: `training_output/gpt-oss-20b-merged/` (~12.8 GB, MXFP4 preserved)

### 2.3 What the Surgical Merge Produces

```
training_output/gpt-oss-20b-merged/
├── model-00000-of-00002.safetensors   # shard 0 (attention merged, experts untouched)
├── model-00001-of-00002.safetensors   # shard 1
├── model-00002-of-00002.safetensors   # shard 2
├── model.safetensors.index.json       # tensor→shard mapping (copied from base)
├── config.json                        # model config (quantization_config intact)
├── tokenizer_config.json              # tokenizer config
├── tokenizer.json                     # tokenizer vocabulary
├── chat_template.jinja                # Harmony response format template
├── generation_config.json             # generation defaults
└── special_tokens_map.json            # special token definitions
```

## 3. Export to Ollama

### 3.1 Import Merged Model

```bash
bash training/export_to_ollama.sh \
    training_output/gpt-oss-20b-merged \
    gpt-oss-20b-biasbuster
```

The export script detects `gpt-oss` in the model name and:
- **Extracts the full Harmony template** from the base `gpt-oss:20b` Ollama
  model (378 lines including tool calling, reasoning levels, and response
  format) and injects it into the Modelfile for the merged model
- Sets stop tokens and all parameters from the base model
- Requires `gpt-oss:20b` to be already pulled in Ollama

**Important:** Ollama does NOT pick up the chat template from the model's
`tokenizer_config.json` or `chat_template.jinja` when importing safetensors.
Without the correct Harmony template, the model produces incoherent output
(raw token soup). The export script handles this automatically by extracting
the template from the base Ollama model.

Ollama serves MXFP4 natively — the ~12.8 GB model runs without additional
quantization.

### 3.2 Verify

```bash
ollama list | grep gpt-oss-20b-biasbuster
ollama run gpt-oss-20b-biasbuster

# Verify the Harmony template was applied (should NOT show "{{ .Prompt }}")
ollama show gpt-oss-20b-biasbuster --modelfile | head -5
```

### 3.3 Alternative: GGUF Quantization (if MXFP4 not supported)

If your Ollama version doesn't support MXFP4, fall back to GGUF:

```bash
# Requires llama.cpp built in ./llama.cpp/
./run_merge.sh gpt-oss-20b --quantize q8_0    # ~21 GB, near-lossless
./run_merge.sh gpt-oss-20b --quantize Q4_K_M  # ~12 GB, good quality
```

Note: this path uses the standard merge (`merge_adapter.py`), not the
surgical merge, so MXFP4 is lost. The GGUF quantization compensates.

## 4. Evaluation

**Important:** The evaluation harness must use the Ollama **native API**
(`/api/chat`) for GPT-OSS models. The OpenAI-compatible endpoint
(`/v1/chat/completions`) does not return the Harmony `thinking` field,
causing all analysis-channel reasoning to be silently lost.

The harness auto-detects Ollama endpoints on port 11434 and uses the native
API automatically. If using a non-standard port, pass `--num-ctx` to force
the native path.

```bash
# Single model evaluation (uses Ollama native API automatically)
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gpt-oss-20b-biasbuster \
    --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --max-tokens 8000 \
    --output eval_results/

# Compare against baseline (sequential: one model at a time)
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gpt-oss-20b-biasbuster \
    --endpoint-a http://localhost:11434 \
    --model-b gpt-oss:20b \
    --endpoint-b http://localhost:11434 \
    --mode fine-tuned \
    --max-tokens 8000 \
    --sequential \
    --output eval_results/
```

**Pre-flight sanity check** (do this before every eval run):

```bash
# Verify the model returns thinking + content fields
curl -s http://localhost:11434/api/chat -d '{
  "model": "gpt-oss-20b-biasbuster",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": false
}' | python3 -m json.tool | grep -A2 '"thinking"'
```

If `thinking` is missing, the Harmony template was not applied correctly.
See Troubleshooting below.

Key metrics to compare against the unfine-tuned baseline:
- Binary F1 (baseline: 0.925)
- Recall (baseline: 87.5%)
- Verification score (baseline: 0.460)
- Thinking chain presence (baseline: 0%, fine-tuned target: >80%)

### 4.1 Checkpoint/Resume

Evaluation results are saved incrementally to the SQLite database. If the
process is interrupted (e.g., server shutdown), restart without
`--force-reevaluation` and it will skip already-evaluated examples:

```bash
# Resume (same command, no --force-reevaluation)
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gpt-oss-20b-biasbuster \
    --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --max-tokens 8000 \
    --output eval_results/
```

## 5. Troubleshooting

### Training fails with "expected device meta but got cuda:0"

`device_map="auto"` caused accelerate to offload parameters. Fixed in
`train_lora.py`: defaults to `device_map={"": 0}`, switches to `"auto"`
with `max_memory` constraint only for MXFP4 models.

### Merge fails with NotImplementedError in revert_weight_conversion

`save_pretrained()` tries to reverse MXFP4→BF16 dequantization, which
isn't implemented. Use the surgical merge (`run_merge.sh` auto-detects
gpt-oss and uses it).

### Permission errors on training_output/ or .cache/huggingface/

Docker runs create root-owned files. Fix with:

```bash
sudo chown -R $USER:$USER training_output/ ~/.cache/huggingface/
```

### Ollama: "unsupported architecture" with ADAPTER directive

Ollama doesn't support runtime LoRA adapters for GPT-OSS yet (neither
safetensors nor GGUF format). Use the merge + import path instead.

### Ollama: "loras are not yet implemented"

Same as above — Ollama can serve GPT-OSS base models in MXFP4 but
adapter overlay isn't implemented for this architecture. Merge first.

### Ollama: model produces incoherent output / token soup

The Harmony chat template was not applied. Ollama does NOT auto-detect
the template from `tokenizer_config.json` or `chat_template.jinja` in
imported safetensors. Check with:

```bash
ollama show gpt-oss-20b-biasbuster --modelfile | grep "^TEMPLATE"
```

If it shows `TEMPLATE {{ .Prompt }}`, the template is a bare passthrough.
Re-export with the fixed script which extracts the 378-line Harmony template
from the base `gpt-oss:20b` model:

```bash
bash training/export_to_ollama.sh \
    training_output/gpt-oss-20b-merged \
    gpt-oss-20b-biasbuster
```

Requires `gpt-oss:20b` to be pulled in Ollama first (`ollama pull gpt-oss:20b`).

### `uv: command not found` when running with sudo

The surgical merge doesn't need sudo. Run `./run_merge.sh gpt-oss-20b`
as your regular user. If output dirs have root ownership, fix permissions
first (see above).

### Training saturates too quickly (loss → 0 in ~200 steps)

The model's strong baseline (F1 0.925 zero-shot) means it needs very
little adaptation. Current mitigations:
- 2 epochs (down from 3)
- LR 5e-6 (down from 1e-5)
- Dropout 0.1 (up from 0.05)
- Weight decay 0.02 (up from 0.01)

If still saturating, try LR 2e-6 or increase the training dataset.

### Evaluation shows F1 near 0.65 or recall < 50%

Almost certainly the evaluation harness is not capturing the Harmony
`thinking` field. Check:

1. Is the endpoint using the Ollama native API? (port 11434 auto-detects)
2. Does `curl` to the model return a `thinking` field? (see §4 above)
3. Is `--max-tokens` >= 8000? (analysis channel can exhaust smaller budgets)
4. Check `thinking_present_rate` in the evaluation output — should be >0%

See `docs/MISTAKES_TO_ROUND_3.md` for the full analysis.

### Evaluation shows 0% thinking_present_rate

The model is not producing analysis-channel output. Possible causes:

1. **Template mismatch**: training data put everything in `content` instead
   of splitting `<think>` into `thinking` field. Check `data_utils.py`
   `_split_think_block()` and run `validate_harmony_channels()`.
2. **Ollama template not applied**: `ollama show model --modelfile` should
   NOT show `TEMPLATE {{ .Prompt }}`. Re-export with `export_to_ollama.sh`.
3. **OpenAI-compat endpoint used**: the `/v1/chat/completions` path doesn't
   return `thinking`. Use the native `/api/chat` path (auto-detected on
   port 11434).

## 6. File Reference

| File | Purpose |
|------|---------|
| `training/configs.py` | `_MOE_OVERRIDES` — MoE-specific hyperparameters |
| `training/train_lora.py` | TRL/PEFT training with MXFP4 dequantization |
| `training/merge_adapter_surgical.py` | Byte-level merge preserving MXFP4 |
| `training/merge_adapter.py` | Standard merge (dense models, not for GPT-OSS) |
| `training/export_to_ollama.sh` | Import merged model into Ollama |
| `training/export_adapter_to_ollama.sh` | Adapter overlay (future, when Ollama supports it) |
| `run_training.sh` | NGC Docker training wrapper |
| `run_merge.sh` | Auto-detects model type and calls surgical or standard merge |
| `docs/papers/FIFTH_RUN.md` | Baseline evaluation of unfine-tuned gpt-oss:20b |
| `docs/papers/SIXTH_RUN.md` | First fine-tuning run analysis and hyperparameter tuning |
| `docs/ROUND_3.md` | Runs 7-9: hyperparameter tuning and Harmony template fix |
| `docs/MISTAKES_TO_ROUND_3.md` | Evaluation harness issues that produced misleading results |
| `docs/PREPARING_ROUND_4.md` | Decision tree and checklist for the next training round |

## 7. Architecture Notes

### MXFP4 Format

MXFP4 (Microscaling FP4) is a 4-bit floating-point format standardised by
the Open Compute Project. Each element uses E2M1 encoding (1 sign bit,
2 exponent bits, 1 mantissa bit) with a shared E8M0 block scaling exponent
per 32 elements. In the safetensors files, MXFP4 data is stored as:

- `*_blocks`: uint8 tensors containing packed FP4 values
- `*_scales`: uint8 tensors containing E8M0 shared exponents

Only the MoE expert layers (gate_up_proj, down_proj) use MXFP4. All other
layers (attention, router, norms, embeddings, lm_head) are BF16.

### Harmony Response Format

GPT-OSS uses the Harmony response format (not ChatML). Key tokens:
- `<|start|>`, `<|end|>` — message boundaries
- `<|channel|>` — output channel (analysis, commentary, etc.)
- `<|message|>` — message content start

The chat template is stored in `chat_template.jinja` and applied
automatically by `tokenizer.apply_chat_template()` during training.
Ollama picks it up from the tokenizer config in the merged model.

### LoRA Merge Math

For each attention layer targeted by LoRA:

```
merged_weight = base_weight + (lora_B @ lora_A) * (alpha / r)
```

Where:
- `lora_A`: shape (r, in_features), F32
- `lora_B`: shape (out_features, r), F32
- `alpha / r`: scaling factor (32 / 16 = 2.0)
- `base_weight`: BF16, converted to F32 for addition, back to BF16

The surgical merge performs this arithmetic on raw bytes without loading
the model through transformers, avoiding all MXFP4 dequantization issues.
