#!/bin/bash
# Merge a LoRA adapter into the base model inside the NGC container,
# optionally quantize to GGUF and import into Ollama.
#
# Usage:
#   ./run_merge.sh qwen3.5-27b
#   ./run_merge.sh olmo-3.1-32b --quantize q8_0
#   ./run_merge.sh olmo-3.1-32b --quantize Q4_K_M

set -euo pipefail

MODEL="${1:?Usage: $0 <qwen3.5-27b|qwen3.5-9b|olmo-3.1-32b|gpt-oss-20b> [--quantize <quant>]}"
shift

QUANTIZE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantize)
            QUANTIZE="${2:?--quantize requires a type (e.g. q8_0, Q4_K_M)}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
PROJECT_DIR="/home/hherb/src/biasbuster"
HF_CACHE="/home/hherb/.cache/huggingface"

echo "==> Merging adapter for: $MODEL"
echo ""

if echo "$MODEL" | grep -qi "gpt-oss"; then
    # GPT-OSS: surgical merge preserving MXFP4 expert weights.
    # Runs on host (no Docker needed) — operates directly on safetensors
    # files, only modifying attention layers and copying everything else
    # byte-for-byte.  Output stays ~14 GB with native MXFP4.
    echo "==> Using surgical merge (MXFP4 preservation)"
    uv run python -m training.merge_adapter_surgical --model "$MODEL"
else
    # Dense models (Qwen, OLMo): standard merge inside NGC container
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
fi

if [[ -n "$QUANTIZE" ]]; then
    MERGED_DIR="training_output/${MODEL}-merged"
    OLLAMA_NAME="${MODEL}-biasbuster"
    echo ""
    echo "==> Quantizing merged model to $QUANTIZE and importing into Ollama..."
    bash training/export_to_ollama.sh "$MERGED_DIR" "$OLLAMA_NAME" --gguf "$QUANTIZE"
fi
