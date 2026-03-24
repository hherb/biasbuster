#!/bin/bash
# Run LoRA fine-tuning inside the NGC PyTorch container.
#
# Usage:
#   ./run_training.sh qwen3.5-27b              # train from scratch
#   ./run_training.sh olmo-3.1-32b --resume    # resume from checkpoint
#   ./run_training.sh qwen3.5-27b --max-steps 5  # smoke test (5 steps)
#
# Requires: user in the 'docker' group (sudo usermod -aG docker $USER)
#
# After training, merge the adapter and export to Ollama:
#   ./run_merge.sh qwen3.5-27b
#   bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster

set -euo pipefail

MODEL="${1:?Usage: $0 <qwen3.5-27b|qwen3.5-9b|olmo-3.1-32b|gpt-oss-20b> [extra args...]}"
shift
EXTRA_ARGS="$*"

IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
PROJECT_DIR="/home/hherb/src/biasbuster"
HF_CACHE="/home/hherb/.cache/huggingface"

echo "==> Model: $MODEL"
echo "==> Extra args: $EXTRA_ARGS"
echo "==> Container: $IMAGE"
echo ""

docker run --gpus all --rm -it \
    --shm-size=16g \
    -v "$PROJECT_DIR":/workspace/biasbuster \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace/biasbuster \
    "$IMAGE" \
    bash -c "
        pip install --quiet trl peft datasets 'transformers>=4.57' &&
        python -m training.train_lora \
            --model $MODEL \
            --train-file dataset/export/alpaca/train.jsonl \
            --val-file dataset/export/alpaca/val.jsonl \
            --output-dir training_output/${MODEL}-lora \
            $EXTRA_ARGS
    "
