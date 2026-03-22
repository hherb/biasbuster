# 8. Fine-Tuning with LoRA

**What you'll do:** Fine-tune a base model using LoRA (Low-Rank Adaptation) on your exported training data. Training runs inside an NGC Docker container on an NVIDIA GPU.

> **Apple Silicon?** If you're training on an M-series Mac instead of an NVIDIA GPU, see [Chapter 8b: Fine-Tuning on Apple Silicon (MLX)](08b_training_mlx.md) instead.

## Prerequisites

- Exported training data at `dataset/export/alpaca/{train,val,test}.jsonl`
- Docker with GPU support (NVIDIA Container Toolkit)
- Sufficient GPU memory (tested on DGX Spark with 128 GB)
- The NGC PyTorch container image: `nvcr.io/nvidia/pytorch:25.11-py3`

## Supported Models

| Preset Key | HuggingFace ID | Download Size | Architecture |
|------------|----------------|---------------|-------------|
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | ~54 GB | Dense |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | ~18 GB | Dense |
| `olmo-3.1-32b` | `allenai/OLMo-3.1-32B-Instruct` | ~64 GB | Dense |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | ~14 GB (MXFP4) | MoE (32 experts, top-4) |

## Download the Base Model from HuggingFace

Training requires the full-precision base model weights from HuggingFace (not the quantized Ollama versions used for evaluation). These are large downloads.

### Option A: Pre-download (Recommended)

Pre-downloading avoids surprises during training startup and lets you verify the download completed successfully:

```bash
# Install the HuggingFace CLI (if not already available)
uv tool install huggingface_hub

# Download the model you plan to fine-tune
hf download Qwen/Qwen3.5-27B
# or
hf download allenai/OLMo-3.1-32B-Instruct
# or
hf download openai/gpt-oss-20b
```

Models are cached in `~/.cache/huggingface/hub/`. The training script mounts this directory into the Docker container, so the model is available without re-downloading.

### Option B: Automatic Download

If you skip the manual download, HuggingFace `transformers` will download the model automatically when training starts. This works but has drawbacks:

- The download happens inside the Docker container, with no progress indication beyond log output
- If the download is interrupted, you must restart training from scratch (the partial download is not reliably resumed inside Docker)
- For very large models, the download can take over an hour depending on your connection

### Gated Models

Some HuggingFace models require accepting a license agreement before download. If you encounter an access error:

1. Visit the model page on [huggingface.co](https://huggingface.co) (e.g., `https://huggingface.co/Qwen/Qwen3.5-27B`)
2. Accept the license/terms if prompted
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Log in locally:

```bash
hf login
```

The token is stored in `~/.cache/huggingface/token` and is automatically available inside the training container.

### Verifying the Download

```bash
# List cached models
hf scan-cache

# Check a specific model is complete
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/
```

A complete download will have a single snapshot directory containing `model.safetensors.*` shards, `config.json`, and tokenizer files.

## Start Training

```bash
# Dense models
./run_training.sh qwen3.5-27b
./run_training.sh qwen3.5-9b
./run_training.sh olmo-3.1-32b

# MoE model
./run_training.sh gpt-oss-20b
```

The script:
1. Launches an NGC Docker container with GPU access
2. Installs TRL, PEFT, and transformers inside the container
3. Runs `training/train_lora.py` with the model preset and default hyperparameters
4. Saves checkpoints and metrics to `training_output/{model}-lora/`

### Additional Options

```bash
# Resume from the latest checkpoint
./run_training.sh qwen3.5-27b --resume

# Smoke test (5 steps only)
./run_training.sh qwen3.5-27b --max-steps 5

# Override hyperparameters from the command line
./run_training.sh qwen3.5-27b --lr 1e-4 --epochs 5 --lora-rank 32
```

## Hyperparameters

All hyperparameters are centralised in `training/configs.py`. The base defaults target 27-32B dense models; 9B and MoE models receive automatic overrides.

### LoRA Configuration (Base Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank (adaptation dimension) |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | Regularisation dropout |
| `target_modules` | q,k,v,o,gate,up,down | All attention + MLP layers |

### Training Configuration (Base Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 3 | Full passes through training data |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `lr_scheduler_type` | cosine | Cosine annealing with warmup |
| `warmup_ratio` | 0.1 | 10% of steps for LR warmup |
| `max_seq_length` | 4096 | Max tokens per training example |
| `bf16` | True | bfloat16 mixed precision |
| `gradient_checkpointing` | True | Trades compute for memory |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |

### 9B Model Overrides

Applied automatically when the model key contains "9b":

| Parameter | Override | Rationale |
|-----------|---------|-----------|
| `lora_r` | 32 | More LoRA capacity for smaller model |
| `lora_alpha` | 64 | Maintain alpha/r = 2 |
| `lora_dropout` | 0.08 | Higher to combat overfitting |
| `weight_decay` | 0.02 | Additional regularisation |
| `label_smoothing_factor` | 0.05 | Soften targets for better calibration |

### GPT-OSS MoE Overrides

Applied automatically for `gpt-oss-20b`. The MoE architecture requires special handling:

| Parameter | Override | Rationale |
|-----------|---------|-----------|
| `target_modules` | q,k,v,o only | Skip expert FFNs and router for stable training |
| `learning_rate` | 5e-6 | Conservative LR to avoid expert collapse |
| `lora_dropout` | 0.1 | Combat rapid memorisation |
| `num_train_epochs` | 1 | Model converges within half an epoch |
| `weight_decay` | 0.02 | Additional regularisation |
| `mxfp4_dequantize` | True | Backward pass not implemented for MXFP4 |
| `attn_implementation` | eager | Required per OpenAI cookbook |

### Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_steps` | 50 | Checkpoint every 50 steps |
| `eval_steps` | 50 | Evaluate on validation set every 50 steps |
| `save_total_limit` | 3 | Keep only 3 most recent checkpoints |
| `logging_steps` | 10 | Log metrics every 10 steps |

## Monitor Training

While training runs in Docker, monitor progress from the host in a separate terminal:

```bash
uv run python -m utils.training_monitor \
    --metrics-dir training_output/qwen3.5-27b-lora
```

This opens a NiceGUI web dashboard showing:

- **Training loss curve** -- should drop steeply then plateau
- **Validation loss** -- should track training loss (slight lag is normal)
- **Learning rate schedule** -- smooth cosine decay
- **GPU memory** -- should be flat after initial steps
- **Gradient norms** -- monitors training stability
- **ETA** -- estimated time to completion

### Healthy Training Indicators

- Training loss drops from ~2.3 to ~0.5 over 3 epochs
- Validation loss follows training loss without diverging significantly
- Final token accuracy around 85%
- No sudden spikes in gradient norms
- GPU memory usage is stable

### Warning Signs

- Validation loss increasing while training loss decreases = overfitting
- Loss stuck at a plateau = learning rate too low or data issues
- NaN/Inf in gradient norms = training instability

## Output Directory

```
training_output/qwen3.5-27b-lora/
├── checkpoint-50/              # Periodic checkpoints
├── checkpoint-100/
├── checkpoint-150/             # Only 3 most recent kept
├── final_adapter/              # Best checkpoint (lowest eval loss)
│   ├── adapter_config.json     # LoRA configuration
│   ├── adapter_model.bin       # LoRA weights (~50-100 MB)
│   └── tokenizer files
├── metrics.jsonl               # Live metrics (read by training monitor)
└── trainer_state.json          # HuggingFace trainer state
```

The `final_adapter/` directory contains the best checkpoint, automatically selected by lowest evaluation loss. This is what you'll merge into the base model in the next step.

## Resume Training

If training is interrupted (crash, timeout, etc.), resume from the latest checkpoint:

```bash
./run_training.sh qwen3.5-27b --resume
```

The trainer automatically detects the most recent checkpoint in the output directory and continues from there.

## Next Step

[Merging, Quantizing & Deploying](09_merge_and_deploy.md) -- merge the LoRA adapter into the base model and deploy to Ollama.
