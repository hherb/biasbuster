#!/bin/bash
# Run MLX LoRA/QLoRA fine-tuning on Apple Silicon (no Docker needed).
#
# Usage:
#   ./run_training_mlx.sh qwen3.5-27b-4bit              # full training
#   ./run_training_mlx.sh qwen3.5-9b-4bit --resume      # resume from checkpoint
#   ./run_training_mlx.sh qwen3.5-9b-4bit --max-iters 5 # smoke test
#
# Available model presets:
#   qwen3.5-9b-4bit   — 64GB Mac (comfortable, ~10GB)
#   qwen3.5-9b-8bit   — 64GB Mac (comfortable, ~15GB)
#   qwen3.5-27b-4bit  — 64GB Mac (tight) / 128GB Mac (~25GB)
#   qwen3.5-27b-8bit  — 128GB Mac only (~38GB)
#
# Prerequisites:
#   uv sync --group mlx
#
# After training, merge the adapter and export to Ollama:
#   ./run_merge_mlx.sh qwen3.5-27b-4bit
#   bash training/export_to_ollama.sh training_output/qwen3.5-27b-4bit-merged qwen3.5-27b-biasbuster

set -euo pipefail

VALID_MODELS="qwen3.5-9b-4bit qwen3.5-9b-8bit qwen3.5-27b-4bit qwen3.5-27b-8bit"

MODEL="${1:?Usage: $0 <model-preset> [extra args...]
Available presets: $VALID_MODELS}"
shift

# Validate model key
FOUND=0
for m in $VALID_MODELS; do
    if [[ "$m" == "$MODEL" ]]; then
        FOUND=1
        break
    fi
done
if [[ "$FOUND" -eq 0 ]]; then
    echo "Error: Unknown model preset '$MODEL'" >&2
    echo "Available: $VALID_MODELS" >&2
    exit 1
fi

echo "==> MLX LoRA Training (Apple Silicon)"
echo "==> Model: $MODEL"
echo "==> Extra args: $*"
echo ""

# Check that mlx-lm is installed
if ! uv run python -c "import mlx_lm" 2>/dev/null; then
    echo "Error: mlx-lm not installed. Run: uv sync --group mlx" >&2
    exit 1
fi

# Run training — "$@" preserves argument quoting correctly
uv run python -m training.train_lora_mlx \
    --model "$MODEL" \
    "$@"
