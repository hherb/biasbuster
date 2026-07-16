# Fine-Tuning on Apple Silicon with MLX

Step-by-step guide to LoRA/QLoRA fine-tuning of BiasBuster models on Apple
Silicon Macs using the MLX framework. No Docker or NVIDIA GPU required.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4 series)
- Python 3.12+
- `uv` package manager
- Sufficient unified RAM (see [Model Presets](#model-presets) for sizing)

## Step 1: Install Dependencies

```bash
# Install base project dependencies + MLX group
uv sync --group mlx
```

This installs `mlx-lm` and its dependencies (`mlx`, `transformers`, etc.).

## Step 2: Prepare Training Data

If you haven't already built the training dataset, run the full pipeline first:

```bash
uv run python pipeline.py --stage all
```

Then export to fine-tuning format:

```bash
uv run python pipeline.py --stage export
```

This creates alpaca-format JSONL files in `dataset/export/alpaca/`:
- `train.jsonl` (920 examples)
- `val.jsonl` (115 examples)
- `test.jsonl` (115 examples)

The MLX training script automatically converts these to MLX-lm's chat format
on first run (stored in `dataset/export/chat/`).

## Step 3: Choose a Model Preset

| Preset | Model | Quantization | RAM Usage (est.) | Recommended Mac |
|--------|-------|--------------|-----------------|-----------------|
| `qwen3.5-9b-4bit` | Qwen3.5-9B | 4-bit | ~10 GB | 64 GB (comfortable) |
| `qwen3.5-9b-8bit` | Qwen3.5-9B | 8-bit | ~15 GB | 64 GB (comfortable) |
| `qwen3.5-27b-4bit` | Qwen3.5-27B | 4-bit | ~25 GB | 64 GB (tight) / 128 GB |
| `qwen3.5-27b-8bit` | Qwen3.5-27B | 8-bit | ~38 GB | 128 GB only |

All models are pre-quantized MLX weights from the
[mlx-community](https://huggingface.co/mlx-community) on HuggingFace —
downloaded automatically on first use.

**Recommendation:**
- **64 GB Mac**: Start with `qwen3.5-9b-4bit` for fast iteration, or
  `qwen3.5-9b-8bit` for better quality with comfortable headroom.
- **128 GB Mac**: Use `qwen3.5-27b-4bit` for the best quality/speed tradeoff,
  or `qwen3.5-27b-8bit` if you want maximum precision.

## Step 4: Smoke Test (Optional but Recommended)

Run 5 training iterations to verify everything works before committing to a
full run:

```bash
./run_training_mlx.sh qwen3.5-9b-4bit --max-iters 5
```

This will:
1. Download the model weights (~5 GB for 9B-4bit)
2. Convert the alpaca data to chat format
3. Apply LoRA adapters (rank=16, scale=32)
4. Run 5 training iterations
5. Save adapter checkpoint and metrics

Expected output:
```
==> MLX LoRA Training (Apple Silicon)
==> Model: qwen3.5-9b-4bit

Loading model: mlx-community/Qwen3.5-9B-4bit
Trainable: 3,407,872 / 9,241,321,472 (0.04%) parameters
Starting MLX LoRA training...
Iter 1: Train loss 2.431, LR 0.000040, It/s 0.82
Iter 2: Train loss 2.387, LR 0.000080, It/s 0.85
...
Training complete!
```

## Step 5: Full Training Run

```bash
# 64 GB Mac
./run_training_mlx.sh qwen3.5-9b-4bit

# 128 GB Mac
./run_training_mlx.sh qwen3.5-27b-4bit
```

Training runs for 3 epochs by default. With 920 training examples and
batch_size=1, that's ~2,760 iterations. Expect:
- **9B-4bit**: ~1-2 hours on M3 Max
- **27B-4bit**: ~4-8 hours on M3 Max/Ultra

### LoRA Configuration

The MLX training uses matching LoRA settings to the DGX Spark pipeline for
comparable results:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA scale (alpha) | 32.0 |
| LoRA dropout | 0.05 |
| LoRA layers | 16 (9B) / 16-32 (27B) |
| Learning rate | 2e-4 (cosine decay + warmup) |
| Warmup ratio | 10% |
| Batch size | 1-2 (model dependent) |
| Max sequence length | 4096 |
| Gradient checkpointing | Enabled |

## Step 6: Monitor Training

The MLX training writes `metrics.jsonl` in the same format as the DGX Spark
pipeline, so the existing training monitor works unchanged:

```bash
# In a separate terminal
uv run python -m utils.training_monitor \
    --metrics-dir training_output/qwen3.5-9b-4bit-mlx-lora
```

Opens a NiceGUI dashboard at http://localhost:8080 showing loss curves,
learning rate schedule, memory usage, and hyperparameters.

See [TRAINING_INTERPRETATION.md](TRAINING_INTERPRETATION.md) for how to read
the dashboard charts.

## Step 7: Resume After Interruption

If training is interrupted (Ctrl+C, crash, or system sleep), resume from the
latest adapter checkpoint:

```bash
./run_training_mlx.sh qwen3.5-9b-4bit --resume
```

Checkpoints are saved every 50 iterations by default.

## Step 8: Merge Adapter

After training, fuse the LoRA adapter into the base model:

```bash
./run_merge_mlx.sh qwen3.5-9b-4bit
```

This de-quantizes the fused model to full precision (required for
GGUF conversion and Ollama import). Output goes to
`training_output/qwen3.5-9b-4bit-merged/`.

## Step 9: Export to Ollama

Import the merged model into Ollama for inference:

```bash
# Full-precision safetensors import
bash training/export_to_ollama.sh \
    training_output/qwen3.5-9b-4bit-merged \
    qwen3.5-9b-biasbuster

# Or with GGUF quantization (smaller, faster)
./run_merge_mlx.sh qwen3.5-9b-4bit --quantize Q4_K_M
```

## Step 10: Evaluate

Run the evaluation harness against the fine-tuned model served via Ollama:

```bash
# Start Ollama (if not already running)
ollama serve

uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-9b-biasbuster \
    --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --output eval_results/mlx_qwen3.5-9b/
```

## Troubleshooting

### Out of Memory

If you see memory pressure warnings or the Mac becomes unresponsive:

1. **Use a smaller model**: Switch from 27B to 9B, or from 8-bit to 4-bit
2. **Close other applications**: MLX uses unified memory shared with the OS
3. **Check Activity Monitor**: Look at "Memory Pressure" graph — yellow/red
   means trouble

### Model Download Fails

Models are downloaded from HuggingFace on first use. If downloads fail:

```bash
# Login to HuggingFace (some models require it)
uv run huggingface-cli login

# Or set the token as an environment variable
export HF_TOKEN=hf_your_token_here
```

### Training Is Very Slow

- **Check that you're on Apple Silicon** — MLX does not work on Intel Macs
- **Ensure no other GPU-heavy apps** are running (video editors, games)
- **Reduce max_seq_length** if your examples are shorter than 4096 tokens

### Data Conversion Issues

If the automatic alpaca→chat conversion fails:

```bash
# Check that alpaca exports exist
ls -la dataset/export/alpaca/

# Re-export if needed
uv run python pipeline.py --stage export

# Delete cached chat data to force reconversion
rm -rf dataset/export/chat/
```

## Output Directory Structure

After a complete training run:

```
training_output/qwen3.5-9b-4bit-mlx-lora/
├── adapter_config.json      # LoRA configuration
├── adapters.safetensors     # Final LoRA adapter weights
├── metrics.jsonl            # Training metrics (for monitor)
└── checkpoint-*/            # Periodic checkpoints
    └── adapters.safetensors

training_output/qwen3.5-9b-4bit-merged/
├── model-*.safetensors      # Full merged model weights
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer
└── tokenizer_config.json
```

## Comparison: MLX vs DGX Spark Training

| | DGX Spark (PyTorch) | Apple Silicon (MLX) |
|---|---|---|
| **Framework** | TRL/PEFT + PyTorch | mlx-lm + MLX |
| **Models** | Qwen3.5-27B, OLMo-3.1-32B (full bf16) | Qwen3.5 9B/27B (4-bit/8-bit QLoRA) |
| **GPU Memory** | 128 GB unified (Blackwell) | 64-128 GB unified (Apple) |
| **Container** | NGC Docker required | Native macOS (no Docker) |
| **Precision** | bf16 (no quantization) | 4-bit/8-bit quantized base + full-precision LoRA |
| **Speed** | Faster (dedicated GPU) | Slower but accessible |
| **Training monitor** | Same `metrics.jsonl` | Same `metrics.jsonl` |
| **Ollama export** | Same pipeline | Same pipeline |
