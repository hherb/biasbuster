# 8b. Fine-Tuning with LoRA on Apple Silicon (MLX)

**What you'll do:** Fine-tune a base model using QLoRA on an Apple Silicon Mac via the MLX framework. This is an alternative to [Chapter 8](08_training.md) (DGX Spark / NGC Docker) -- choose whichever matches your hardware.

## When to Use This Path

| | DGX Spark (Chapter 8) | Apple Silicon (this chapter) |
|---|---|---|
| **Hardware** | NVIDIA GPU (DGX Spark, A100, etc.) | Apple M1/M2/M3/M4 Mac |
| **Container** | NGC Docker required | No Docker -- runs natively |
| **Models** | Full bf16 (27-32B params) | Quantized 4-bit/8-bit (9-27B) |
| **Speed** | Faster | Slower but accessible |
| **Training quality** | Full precision LoRA | QLoRA (quantized base + full LoRA) |

Both paths produce models exportable to Ollama via the same pipeline, and both write identical `metrics.jsonl` for the training monitor.

## Prerequisites

- Exported training data at `dataset/export/alpaca/{train,val,test}.jsonl` (from [Chapter 6](06_export.md))
- macOS on Apple Silicon (M1 or later)
- 64 GB or 128 GB unified RAM (see model sizing below)
- Python 3.12+ with `uv`

## Step 1: Install MLX Dependencies

```bash
uv sync --group mlx
```

This installs `mlx-lm` and its dependencies (`mlx`, `transformers`, etc.). The `mlx` package only works on Apple Silicon -- it will fail on Intel Macs.

## Step 2: Choose a Model

The MLX path uses pre-quantized models from the [mlx-community](https://huggingface.co/mlx-community) on HuggingFace. Models are downloaded automatically on first use.

### Qwen Models

| Preset Key | Base Model | Quant | RAM (est.) | Recommended Mac |
|------------|-----------|-------|-----------|-----------------|
| `qwen3.5-9b-4bit` | Qwen3.5-9B | 4-bit | ~10 GB | 64 GB (comfortable) |
| `qwen3.5-9b-8bit` | Qwen3.5-9B | 8-bit | ~15 GB | 64 GB (comfortable) |
| `qwen3.5-27b-4bit` | Qwen3.5-27B | 4-bit | ~25 GB | 64 GB (tight) / 128 GB |
| `qwen3.5-27b-8bit` | Qwen3.5-27B | 8-bit | ~38 GB | 128 GB only |

### GPT-OSS MoE Models

| Preset Key | Base Model | Quant | RAM (est.) | Recommended Mac |
|------------|-----------|-------|-----------|-----------------|
| `gpt-oss-20b-4bit` | gpt-oss-20b | 4-bit | ~10 GB | 64 GB+ |
| `gpt-oss-20b-8bit` | gpt-oss-20b | 8-bit | ~20 GB | 128 GB |

GPT-OSS is a Mixture-of-Experts model (32 experts, top-4 routing, 21B total / 3.6B active parameters). Despite its large total parameter count, the sparse activation makes it memory-efficient during inference.

**Recommendations:**
- **64 GB Mac:** Start with `qwen3.5-9b-4bit` for fast iteration, or `gpt-oss-20b-4bit` for MoE. Move to 8-bit for better quality once workflow is validated.
- **128 GB Mac:** Use `qwen3.5-27b-4bit` for the best quality/speed tradeoff. Use `qwen3.5-27b-8bit` or `gpt-oss-20b-8bit` for maximum precision.

## Step 3: Smoke Test

Run a quick 5-iteration test to verify everything works:

```bash
./run_training_mlx.sh qwen3.5-9b-4bit --max-iters 5
```

This will:
1. Download model weights from HuggingFace (~5 GB for 9B-4bit)
2. Convert alpaca JSONL to MLX-lm's chat format (automatic, one-time)
3. Apply LoRA adapters and run 5 training iterations
4. Save adapter checkpoint and `metrics.jsonl`

You should see output like:

```
==> MLX LoRA Training (Apple Silicon)
==> Model: qwen3.5-9b-4bit

Loading model: mlx-community/Qwen3.5-9B-4bit
Trainable: 3,407,872 / 9,241,321,472 (0.04%) parameters
Starting MLX LoRA training...
Iter 1: Train loss 2.431, LR 0.000040, It/s 0.82
...
Training complete!
  Adapter saved to: training_output/qwen3.5-9b-4bit-mlx-lora/adapters.safetensors
```

If the smoke test succeeds, proceed to a full training run.

## Step 4: Full Training Run

```bash
# 64 GB Mac
./run_training_mlx.sh qwen3.5-9b-4bit
./run_training_mlx.sh gpt-oss-20b-4bit

# 128 GB Mac
./run_training_mlx.sh qwen3.5-27b-4bit
./run_training_mlx.sh gpt-oss-20b-8bit
```

Training runs for 3 epochs by default. With 920 training examples, that's roughly 2,760 iterations. Expected wall-clock time:

| Model | Mac | Time (approx.) |
|-------|-----|----------------|
| Qwen3.5-9B-4bit | M3 Max 64 GB | 1-2 hours |
| GPT-OSS-20B-4bit | M3 Max 64 GB | 1-3 hours |
| Qwen3.5-27B-4bit | M3 Max 128 GB | 4-8 hours |

### Data Conversion

The training script automatically converts your alpaca-format data to MLX-lm's chat format on first run:

```
dataset/export/alpaca/train.jsonl  →  dataset/export/chat/train.jsonl
dataset/export/alpaca/val.jsonl    →  dataset/export/chat/valid.jsonl  (note: renamed)
dataset/export/alpaca/test.jsonl   →  dataset/export/chat/test.jsonl
```

Conversion is skipped on subsequent runs if the output files are up-to-date.

## Hyperparameters

All hyperparameters are in `training/configs_mlx.py`. The LoRA settings match the DGX Spark pipeline for comparable results:

### LoRA Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lora_rank` | 16 | Same as PyTorch path |
| `lora_scale` | 32.0 | MLX equivalent of `lora_alpha` |
| `lora_dropout` | 0.05 | Same as PyTorch path |
| `lora_num_layers` | 16 | Number of transformer layers with LoRA |

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_train_epochs` | 3 | Computed as iterations internally |
| `batch_size` | 1-2 | Model-dependent (set by preset) |
| `learning_rate` | 2e-4 | With cosine decay + 10% warmup |
| `max_seq_length` | 4096 | Max tokens per example |
| `mask_prompt` | True | Loss only on assistant output |

GPT-OSS MoE models automatically receive a conservative learning rate (1e-5) to avoid expert collapse.

### Reporting / Checkpointing

| Parameter | Value |
|-----------|-------|
| `steps_per_report` | 10 |
| `steps_per_eval` | 50 |
| `steps_per_save` | 50 |

## Step 5: Monitor Training

The MLX callback writes the same `metrics.jsonl` format, so the existing training monitor works unchanged:

```bash
# In a separate terminal
uv run python -m utils.training_monitor \
    --metrics-dir training_output/qwen3.5-9b-4bit-mlx-lora
```

Opens a web dashboard at http://localhost:8080 with loss curves, learning rate, memory usage, and hyperparameters.

See [TRAINING_INTERPRETATION.md](../../TRAINING_INTERPRETATION.md) for how to read the charts. The same healthy/warning indicators from Chapter 8 apply here.

## Step 6: Resume After Interruption

```bash
./run_training_mlx.sh qwen3.5-9b-4bit --resume
```

The script loads the latest adapter weights and continues training.

## Output Directory

```
training_output/qwen3.5-9b-4bit-mlx-lora/
├── adapter_config.json        # LoRA configuration
├── adapters.safetensors       # Final adapter weights
├── metrics.jsonl              # Training metrics (for monitor)
└── checkpoint-*/              # Periodic checkpoints
    └── adapters.safetensors
```

## Next Step: Merge & Deploy

After training completes, proceed to merge and deploy. The MLX path has its own merge script, but the Ollama export is shared:

```bash
# Merge (de-quantises to full precision)
./run_merge_mlx.sh qwen3.5-9b-4bit

# Export to Ollama with GGUF quantisation
./run_merge_mlx.sh qwen3.5-9b-4bit --quantize Q4_K_M
```

Or merge and export separately:

```bash
# 1. Merge
./run_merge_mlx.sh qwen3.5-9b-4bit

# 2. Export to Ollama (same script as DGX Spark path)
bash training/export_to_ollama.sh \
    training_output/qwen3.5-9b-4bit-merged \
    qwen3.5-9b-biasbuster \
    --gguf Q4_K_M
```

Then continue with [Chapter 9 (Merging, Quantising & Deploying)](09_merge_and_deploy.md) for GGUF quantisation details and Ollama deployment verification, and [Chapter 10 (Evaluating Fine-Tuned Models)](10_evaluation.md) for evaluation.

## Troubleshooting

### Out of Memory

If your Mac becomes unresponsive or you see memory pressure:

1. Use a smaller model preset (27B → 9B, or 8-bit → 4-bit)
2. Close other applications -- MLX uses unified memory shared with the OS and all apps
3. Check Activity Monitor → Memory Pressure graph (yellow/red = trouble)

### Model Download Fails

```bash
# Login to HuggingFace if required
uv run huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=hf_your_token_here
```

### mlx-lm Not Found

```bash
# Ensure you installed the MLX dependency group
uv sync --group mlx

# Verify it's available
uv run python -c "import mlx_lm; print(mlx_lm.__version__)"
```

### Training Is Very Slow

- Verify you're on Apple Silicon: `uname -m` should show `arm64`
- Close GPU-heavy apps (video editors, games, browsers with many tabs)
- If examples are shorter than 4096 tokens, consider reducing `max_seq_length` in the config

### Data Conversion Errors

```bash
# Re-export alpaca data if corrupted
uv run python pipeline.py --stage export

# Force reconversion by removing cached chat data
rm -rf dataset/export/chat/
```
