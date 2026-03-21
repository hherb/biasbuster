#!/bin/bash
# End-to-end: merge LoRA adapter → (optional GGUF quantize) → create Ollama model.
#
# Detects the platform automatically:
#   - Linux (DGX Spark): uses run_merge.sh (Docker-based dense merge or surgical MoE merge)
#   - macOS (Apple Silicon): uses run_merge_mlx.sh (MLX fuse with de-quantize)
#
# Usage:
#   ./lora2ollama.sh qwen3.5-27b
#   ./lora2ollama.sh qwen3.5-27b --quantize Q4_K_M
#   ./lora2ollama.sh qwen3.5-27b --ollama-model-name my-bias-detector
#   ./lora2ollama.sh qwen3.5-9b-4bit --quantize Q4_K_M --ollama-model-name qwen9b-bias
#
# Options:
#   --quantize <type>          GGUF quantization type (e.g. q8_0, Q4_K_M). Optional.
#   --ollama-model-name <name> Name for the Ollama model. Default: <model>-biasbuster

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────────────

MODEL="${1:?Usage: $0 <model-preset> [--quantize <type>] [--ollama-model-name <name>]}"
shift

QUANTIZE=""
OLLAMA_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantize)
            QUANTIZE="${2:?--quantize requires a type (e.g. q8_0, Q4_K_M)}"
            shift 2
            ;;
        --ollama-model-name)
            OLLAMA_NAME="${2:?--ollama-model-name requires a name}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 <model-preset> [--quantize <type>] [--ollama-model-name <name>]" >&2
            exit 1
            ;;
    esac
done

# Default Ollama model name
OLLAMA_NAME="${OLLAMA_NAME:-${MODEL}-biasbuster}"

# ── Detect platform ─────────────────────────────────────────────────────────

OS="$(uname -s)"

# MLX presets have suffixes like -4bit/-8bit; TRL presets don't
is_mlx_preset() {
    [[ "$1" =~ -(4|8)bit$ ]]
}

# ── Step 1: Merge adapter ───────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LoRA → Ollama: $MODEL"
echo "║  Ollama name:   $OLLAMA_NAME"
[[ -n "$QUANTIZE" ]] && echo "║  Quantization:  $QUANTIZE"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [[ "$OS" == "Darwin" ]] || is_mlx_preset "$MODEL"; then
    echo "==> Platform: macOS / MLX backend"
    echo "==> Step 1/2: Fusing MLX adapter..."
    bash "$SCRIPT_DIR/run_merge_mlx.sh" "$MODEL"
    MERGED_DIR="training_output/${MODEL}-merged"
elif [[ "$OS" == "Linux" ]]; then
    echo "==> Platform: Linux / TRL backend (DGX Spark)"
    echo "==> Step 1/2: Merging LoRA adapter..."
    bash "$SCRIPT_DIR/run_merge.sh" "$MODEL"
    MERGED_DIR="training_output/${MODEL}-merged"
else
    echo "ERROR: Unsupported platform: $OS" >&2
    exit 1
fi

# ── Step 2: Export to Ollama ─────────────────────────────────────────────────

echo ""
echo "==> Step 2/2: Exporting to Ollama as '$OLLAMA_NAME'..."

EXPORT_ARGS=("$MERGED_DIR" "$OLLAMA_NAME")
if [[ -n "$QUANTIZE" ]]; then
    EXPORT_ARGS+=(--gguf "$QUANTIZE")
fi

bash "$SCRIPT_DIR/training/export_to_ollama.sh" "${EXPORT_ARGS[@]}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All done! Model available as: $OLLAMA_NAME"
echo ""
echo "  Test it:"
echo "    ollama run $OLLAMA_NAME"
echo "════════════════════════════════════════════════════════════"
