#!/bin/bash
# Export a merged fine-tuned model to Ollama.
#
# Usage:
#   bash training/export_to_ollama.sh <merged-dir> <ollama-model-name> [--gguf <quant>]
#
# Examples:
#   # Path A: Import safetensors directly (full precision)
#   bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster
#
#   # Path B: Convert to GGUF first (quantized, smaller)
#   bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster --gguf Q4_K_M
#
# Requires:
#   - Ollama running on the host
#   - For --gguf: llama.cpp built in ./llama.cpp/ (convert_hf_to_gguf.py + llama-quantize)

set -euo pipefail

MERGED_DIR="${1:?Usage: $0 <merged-dir> <ollama-model-name> [--gguf <quant>]}"
MODEL_NAME="${2:?Usage: $0 <merged-dir> <ollama-model-name> [--gguf <quant>]}"
GGUF_QUANT=""

# Parse optional --gguf flag
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gguf)
            GGUF_QUANT="${2:?--gguf requires a quantization type (e.g. Q4_K_M)}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

MERGED_DIR="$(realpath "$MERGED_DIR")"

if [[ -n "$GGUF_QUANT" ]]; then
    # =========================================================================
    # Path B: Convert to GGUF, optionally quantize, then import
    # =========================================================================
    LLAMA_CPP="$(dirname "$0")/../llama.cpp"
    if [[ ! -d "$LLAMA_CPP" ]]; then
        echo "ERROR: llama.cpp not found at $LLAMA_CPP" >&2
        echo "Clone it: git clone https://github.com/ggml-org/llama.cpp.git" >&2
        exit 1
    fi

    GGUF_DIR="$(dirname "$MERGED_DIR")/gguf"
    mkdir -p "$GGUF_DIR"

    QUANT_GGUF="$GGUF_DIR/${MODEL_NAME}-${GGUF_QUANT}.gguf"

    # For quantization types supported by convert_hf_to_gguf.py (f32, f16,
    # bf16, q8_0, auto), convert directly in one pass — avoids creating a
    # ~64 GB fp16 intermediate that doubles peak memory and disk usage.
    DIRECT_TYPES="q8_0 f32 f16 bf16 auto Q8_0 F32 F16 BF16"
    QUANT_LOWER="$(echo "$GGUF_QUANT" | tr '[:upper:]' '[:lower:]')"

    if echo "$DIRECT_TYPES" | tr ' ' '\n' | grep -iqx "$GGUF_QUANT"; then
        echo "==> Converting directly to GGUF ($GGUF_QUANT) — single-pass, no fp16 intermediate..."
        python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" \
            --outfile "$QUANT_GGUF" --outtype "$QUANT_LOWER"
    else
        # Two-pass: fp16 conversion then llama-quantize (for Q4_K_M, Q5_K_M, etc.)
        FP16_GGUF="$GGUF_DIR/${MODEL_NAME}-f16.gguf"

        echo "==> Converting to GGUF (fp16)..."
        python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" \
            --outfile "$FP16_GGUF" --outtype f16

        echo "==> Quantizing to $GGUF_QUANT..."
        "$LLAMA_CPP/build/bin/llama-quantize" "$FP16_GGUF" "$QUANT_GGUF" "$GGUF_QUANT"

        echo "    Intermediate fp16 GGUF kept at: $FP16_GGUF"
    fi

    echo "==> Creating Ollama model: $MODEL_NAME"
    MODELFILE="$(mktemp /tmp/Modelfile.XXXXXX)"
    echo "FROM $QUANT_GGUF" > "$MODELFILE"
    ollama create "$MODEL_NAME" -f "$MODELFILE"
    rm -f "$MODELFILE"

else
    # =========================================================================
    # Path A: Import safetensors directly via hard-link workaround
    # =========================================================================
    # Ollama rejects symlinks as "insecure path". Since the merged model
    # directory contains real files (not HF cache symlinks), we can use it
    # directly — but create a hard-link copy just in case.
    RESOLVED="$(mktemp -d /tmp/${MODEL_NAME}-resolved.XXXXXX)"

    echo "==> Creating hard-link directory..."
    for f in "$MERGED_DIR"/*; do
        if [[ -f "$f" ]]; then
            ln "$(realpath "$f")" "$RESOLVED/$(basename "$f")"
        fi
    done

    echo "==> Creating Ollama model: $MODEL_NAME"
    MODELFILE="$(mktemp /tmp/Modelfile.XXXXXX)"
    echo "FROM $RESOLVED/" > "$MODELFILE"
    ollama create "$MODEL_NAME" -f "$MODELFILE"
    rm -f "$MODELFILE"

    echo "==> Cleaning up hard-link directory..."
    rm -rf "$RESOLVED"
fi

echo ""
echo "Done! Model available as: $MODEL_NAME"
echo "Verify: ollama list | grep $MODEL_NAME"
echo "Test:   curl -s http://localhost:11434/v1/models | python3 -m json.tool"
