# 11. Fine-Tuning Workbench (GUI)

**What you'll do:** Use the browser-based GUI to configure, train, evaluate, and export fine-tuned models -- all from a single interface.

The Fine-Tuning Workbench wraps the command-line training, evaluation, and export workflows (Chapters 8-10) in a NiceGUI web application with live monitoring.

## Launch

```bash
uv run python -m gui
uv run python -m gui --port 9090   # custom port
```

Open `http://localhost:8080` (or your custom port) in a browser. The header badge shows your detected platform and available backends (e.g., "Linux (trl)" or "Darwin (mlx)").

## Tab 1: Settings

Configure your training run before starting:

### Model Selection

- **Model preset** -- dropdown of all available presets. On Linux, shows TRL presets (qwen3.5-27b, qwen3.5-9b, olmo-3.1-32b, gpt-oss-20b). On macOS, shows MLX presets (qwen3.5-9b-4bit, gpt-oss-20b-4bit, etc.)
- Selecting a preset auto-populates hyperparameters with the model-specific defaults from `training/configs.py` or `training/configs_mlx.py`

### Hyperparameter Overrides

All overridable from the GUI:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 2e-4 | Adjusted per model (5e-6 for GPT-OSS MoE) |
| Epochs | 3 | Number of training epochs |
| LoRA Rank | 16 | Adaptation dimension |
| Batch Size | 1 | Per-device batch size |
| Gradient Accumulation | 4 | Effective batch multiplier |
| Max Sequence Length | 8192 | Maximum tokens per training example |

### Data Paths

- **Train / Val / Test files** -- file pickers pointing to your exported JSONL files (default: `dataset/export/alpaca/`)

Settings are persisted to `~/.biasbuster/gui_settings.json` and restored on next launch.

## Tab 2: Evaluation

Run model evaluation without leaving the GUI:

- **Endpoint A / Model A** -- configure the primary model endpoint (default: Ollama at localhost:11434)
- **Endpoint B / Model B** -- optional second model for comparison
- **Mode** -- zero-shot or fine-tuned
- **Run Evaluation** -- launches `evaluation.run` as a subprocess with streaming output
- **Results** -- displays F1, kappa, calibration, and per-dimension metrics when complete

## Tab 3: Fine-Tuning

Train models with live monitoring:

- **Start Training** -- launches the training subprocess (Docker on Linux, native on macOS)
- **Live Charts** -- loss curves, learning rate schedule, GPU memory, and gradient norms update in real time by reading `metrics.jsonl` (uses the same `MetricsReader` as `utils/training_monitor.py`)
- **Output Log** -- streaming subprocess output
- **Resume** -- resume from the latest checkpoint if training was interrupted

## Tab 4: Export

Merge and deploy the trained model:

- **Merge Adapter** -- triggers the merge process (Docker or surgical, depending on model)
- **Quantization** -- dropdown to select GGUF quantization type (Q4_K_M, q8_0, f16, etc.)
- **Ollama Model Name** -- customise the name for the deployed model
- **Export to Ollama** -- runs the full merge → quantize → import pipeline

## Platform Detection

The Workbench automatically detects your platform and adjusts available options:

| Platform | Backends | Docker Required | Notes |
|----------|----------|----------------|-------|
| Linux (DGX Spark) | TRL (PyTorch) | Yes | Full-precision LoRA in NGC container |
| macOS (Apple Silicon) | MLX | No | QLoRA with pre-quantized models |

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
uv run python -m gui --port 9090
```

### Charts Not Updating

The training monitor reads `metrics.jsonl` from the training output directory. If charts are empty:
- Verify training is actually running and producing output
- Check that the training output directory matches what's configured in Settings

### Settings Not Persisting

Settings are stored in `~/.biasbuster/gui_settings.json`. If the file is not writable, settings will be lost between sessions. Check permissions on the `~/.biasbuster/` directory.
