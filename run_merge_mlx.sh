#!/bin/bash
# Fuse an MLX LoRA adapter into the base model, optionally export to Ollama.
#
# Usage:
#   ./run_merge_mlx.sh qwen3.5-27b-4bit
#   ./run_merge_mlx.sh qwen3.5-27b-4bit --quantize Q4_K_M

set -euo pipefail

MODEL="${1:?Usage: $0 <model-preset> [--quantize <quant>]}"
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

echo "==> Fusing MLX adapter for: $MODEL"
echo ""

# Fuse with de-quantize (needed for Ollama/GGUF export)
uv run python -m training.merge_adapter_mlx --model "$MODEL" --de-quantize

if [[ -n "$QUANTIZE" ]]; then
    MERGED_DIR="training_output/${MODEL}-merged"
    OLLAMA_NAME="${MODEL}-biasbuster"
    echo ""
    echo "==> Quantizing merged model to $QUANTIZE and importing into Ollama..."
    bash training/export_to_ollama.sh "$MERGED_DIR" "$OLLAMA_NAME" --gguf "$QUANTIZE"
fi
