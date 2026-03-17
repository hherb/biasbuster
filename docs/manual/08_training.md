# 8. Fine-Tuning with LoRA

**What you'll do:** Fine-tune a base model using LoRA (Low-Rank Adaptation) on your exported training data. Training runs inside an NGC Docker container.

## Prerequisites

- Exported training data at `dataset/export/alpaca/{train,val,test}.jsonl`
- Docker with GPU support (NVIDIA Container Toolkit)
- Sufficient GPU memory (tested on DGX Spark with 128 GB)
- The NGC PyTorch container image: `nvcr.io/nvidia/pytorch:25.11-py3`

## Start Training

```bash
# Train from scratch
./run_training.sh qwen3.5-27b

# Or with OLMo
./run_training.sh olmo-3.1-32b
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
```

## Hyperparameters

All hyperparameters are centralized in `training/configs.py`. The defaults are tuned for 27-32B parameter models on a single GPU:

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank (adaptation dimension) |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | Regularization dropout |
| `target_modules` | q,k,v,o,gate,up,down | All attention + MLP layers |

### Training Configuration

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
