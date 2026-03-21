#!/bin/bash
# Export a LoRA adapter to Ollama WITHOUT merging — preserves the base model's
# native format (e.g. MXFP4 for GPT-OSS).
#
# Instead of merging adapter weights into a full-size model and re-quantizing,
# this script uses Ollama's ADAPTER directive to apply the LoRA adapter on top
# of the base model at load time.  This is the preferred export path for
# GPT-OSS because:
#   - Base model stays in MXFP4 (~14 GB, Ollama serves it natively)
#   - No dequantize → merge → re-quantize pipeline
#   - Adapter is tiny (a few MB of attention-layer deltas)
#
# Usage:
#   bash training/export_adapter_to_ollama.sh <base-ollama-model> <adapter-dir> <ollama-model-name>
#
# Examples:
#   # GPT-OSS 20B with LoRA adapter (MXFP4 preserved)
#   bash training/export_adapter_to_ollama.sh gpt-oss:20b \
#       training_output/gpt-oss-20b-lora/final_adapter \
#       gpt-oss-20b-biasbuster
#
#   # Qwen 9B with LoRA adapter
#   bash training/export_adapter_to_ollama.sh qwen3.5:9b \
#       training_output/qwen3.5-9b-lora/final_adapter \
#       qwen3.5-9b-biasbuster
#
# Prerequisites:
#   - Ollama running on the host
#   - Base model already pulled: ollama pull gpt-oss:20b
#   - Adapter directory contains adapter_config.json + adapter_model.safetensors

set -euo pipefail

BASE_MODEL="${1:?Usage: $0 <base-ollama-model> <adapter-dir> <ollama-model-name>}"
ADAPTER_DIR="${2:?Usage: $0 <base-ollama-model> <adapter-dir> <ollama-model-name>}"
MODEL_NAME="${3:?Usage: $0 <base-ollama-model> <adapter-dir> <ollama-model-name>}"

ADAPTER_DIR="$(realpath "$ADAPTER_DIR")"

# Verify adapter files exist
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "ERROR: adapter_config.json not found in $ADAPTER_DIR" >&2
    echo "Expected files: adapter_config.json, adapter_model.safetensors" >&2
    exit 1
fi

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
    echo "ERROR: adapter_model.safetensors not found in $ADAPTER_DIR" >&2
    exit 1
fi

# Ollama expects the adapter file to be named model.safetensors (not
# adapter_model.safetensors as PEFT produces).  Create a staging directory
# with the correct naming to avoid modifying the original adapter.
# Ref: https://github.com/ollama/ollama/issues/13314
STAGING="$(mktemp -d /tmp/${MODEL_NAME}-adapter.XXXXXX)"
ln "$ADAPTER_DIR/adapter_model.safetensors" "$STAGING/model.safetensors" 2>/dev/null \
    || cp "$ADAPTER_DIR/adapter_model.safetensors" "$STAGING/model.safetensors"
cp "$ADAPTER_DIR/adapter_config.json" "$STAGING/"

# Verify base model is available in Ollama
if ! ollama show "$BASE_MODEL" >/dev/null 2>&1; then
    echo "ERROR: Base model '$BASE_MODEL' not found in Ollama" >&2
    echo "Pull it first: ollama pull $BASE_MODEL" >&2
    rm -rf "$STAGING"
    exit 1
fi

echo "==> Base model:   $BASE_MODEL"
echo "==> Adapter:      $ADAPTER_DIR"
echo "==> Output model: $MODEL_NAME"
echo ""

# Build Modelfile with ADAPTER directive
MODELFILE="$(mktemp /tmp/Modelfile.XXXXXX)"
cat > "$MODELFILE" <<MODELFILE_EOF
FROM ${BASE_MODEL}
ADAPTER ${STAGING}

PARAMETER stop "<|end|>"
PARAMETER stop "<|endoftext|>"
MODELFILE_EOF

echo "==> Creating Ollama model with adapter overlay..."
ollama create "$MODEL_NAME" -f "$MODELFILE"
rm -f "$MODELFILE"
rm -rf "$STAGING"

echo ""
echo "Done! Model available as: $MODEL_NAME"
echo "  Base model ($BASE_MODEL) format preserved (e.g. MXFP4)"
echo "  Adapter applied at load time — no merge needed"
echo ""
echo "Verify: ollama list | grep $MODEL_NAME"
echo "Test:   ollama run $MODEL_NAME"
