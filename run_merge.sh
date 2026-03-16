#!/bin/bash
# Merge a LoRA adapter into the base model inside the NGC container.
#
# Usage:
#   ./run_merge.sh qwen3.5-27b
#   ./run_merge.sh olmo-3.1-32b

set -euo pipefail

MODEL="${1:?Usage: $0 <qwen3.5-27b|olmo-3.1-32b>}"

IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
PROJECT_DIR="/home/hherb/src/biasbuster"
HF_CACHE="/home/hherb/.cache/huggingface"

echo "==> Merging adapter for: $MODEL"
echo ""

sudo docker run --gpus all --rm -it \
    --shm-size=16g \
    -v "$PROJECT_DIR":/workspace/biasbuster \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace/biasbuster \
    "$IMAGE" \
    bash -c "
        pip install --quiet peft 'transformers>=4.57' &&
        python -m training.merge_adapter --model $MODEL
    "
