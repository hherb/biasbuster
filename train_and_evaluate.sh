#!/bin/bash
# End-to-end fine-tuning pipeline: train → merge → Ollama export → evaluate.
#
# Automatically versions each run by querying the SQLite database for existing
# versioned model names and incrementing the version number.
#
# Usage:
#   ./train_and_evaluate.sh gpt-oss-20b                          # full run
#   ./train_and_evaluate.sh gpt-oss-20b --datadir path/to/data   # custom data
#   ./train_and_evaluate.sh gpt-oss-20b -- --max-steps 5         # smoke test
#   ./train_and_evaluate.sh qwen3.5-27b --baseline qwen3.5:27b   # explicit baseline
#
# Arguments:
#   $1                  Base model preset key (gpt-oss-20b, qwen3.5-27b, etc.)
#   --datadir <path>    Training data directory (default: dataset/export/alpaca/)
#   --baseline <model>  Ollama baseline model name (default: auto-derived)
#   -- <args...>        Extra args passed to train_lora.py (--max-steps, --lr, etc.)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

BASE_MODEL="${1:?Usage: $0 <model> [--datadir <path>] [--baseline <model>] [-- extra training args...]}"
shift

DATADIR=""
BASELINE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datadir)
            DATADIR="${2:?--datadir requires a path}"
            shift 2
            ;;
        --baseline)
            BASELINE="${2:?--baseline requires a model name}"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        *)
            EXTRA_ARGS="$*"
            break
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Derive paths and names
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
DB="$PROJECT_DIR/dataset/biasbuster.db"
DATADIR="${DATADIR:-$PROJECT_DIR/dataset/export/alpaca}"

IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
OLLAMA_ENDPOINT="http://localhost:11434"

# Canonicalize DATADIR to absolute path (needed for Docker path remapping)
DATADIR="$(cd "$DATADIR" && pwd)"
TRAIN_FILE="$DATADIR/train.jsonl"
VAL_FILE="$DATADIR/val.jsonl"
TEST_FILE="$DATADIR/test.jsonl"

# Accept both preset keys (gpt-oss-20b) and Ollama names (gpt-oss:20b).
# Normalise to preset key for training/merge, and Ollama name for evaluation.
if [[ "$BASE_MODEL" == *":"* ]]; then
    # User passed Ollama name (gpt-oss:20b) — derive preset key
    BASELINE="${BASELINE:-$BASE_MODEL}"
    BASE_MODEL="${BASE_MODEL//:/-}"
else
    # User passed preset key (gpt-oss-20b) — derive Ollama name
    # gpt-oss-20b → gpt-oss:20b  (split on LAST dash only)
    BASELINE="${BASELINE:-${BASE_MODEL%-*}:${BASE_MODEL##*-}}"
fi

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

fail() { echo "ERROR: $*" >&2; exit 1; }

# Validate model name contains only safe characters (used in SQL and shell commands)
if [[ ! "$BASE_MODEL" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    fail "Invalid model name '$BASE_MODEL' — only alphanumeric, dots, dashes, and underscores allowed"
fi

command -v sqlite3 >/dev/null 2>&1 || fail "sqlite3 not found"
command -v uv >/dev/null 2>&1      || fail "uv not found"
[[ -f "$DB" ]]                     || fail "Database not found: $DB"
[[ -f "$TRAIN_FILE" ]]             || fail "Training file not found: $TRAIN_FILE"
[[ -f "$VAL_FILE" ]]               || fail "Validation file not found: $VAL_FILE"
[[ -f "$TEST_FILE" ]]              || fail "Test file not found: $TEST_FILE"

# Docker is needed for training (all models) and merge (dense models)
command -v docker >/dev/null 2>&1 || fail "docker not found (required for training)"

# Check Ollama is running and baseline model is available
command -v ollama >/dev/null 2>&1 || fail "ollama not found"
OLLAMA_MODELS="$(ollama list 2>/dev/null || true)"
if ! echo "$OLLAMA_MODELS" | grep -qF "$BASELINE"; then
    fail "Baseline model '$BASELINE' not found in Ollama. Pull it first: ollama pull $BASELINE"
fi

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

MAX_VERSION=$(sqlite3 "$DB" "
    SELECT COALESCE(MAX(ver), 0) FROM (
        SELECT CAST(SUBSTR(model_id, INSTR(model_id, 'biasbusterV') + 11) AS INTEGER) AS ver
        FROM eval_outputs
        WHERE model_id LIKE '%${BASE_MODEL}-biasbusterV%'
        UNION ALL
        SELECT CAST(SUBSTR(model_name, INSTR(model_name, 'biasbusterV') + 11) AS INTEGER) AS ver
        FROM annotations
        WHERE model_name LIKE '%${BASE_MODEL}-biasbusterV%'
    );
")

NEXT_VERSION=$((MAX_VERSION + 1))
VERSIONED_NAME="${BASE_MODEL}-biasbusterV${NEXT_VERSION}"
LORA_DIR="training_output/${BASE_MODEL}-lora-V${NEXT_VERSION}"
MERGED_DIR="training_output/${BASE_MODEL}-merged-V${NEXT_VERSION}"
EVAL_OUTPUT="eval_results/${VERSIONED_NAME}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            BiasBuster Train & Evaluate Pipeline             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Base model:       $BASE_MODEL"
echo "║  Version:          V${NEXT_VERSION} (prev max: V${MAX_VERSION})"
echo "║  Versioned name:   $VERSIONED_NAME"
echo "║  LoRA output:      $LORA_DIR/"
echo "║  Merged output:    $MERGED_DIR/"
echo "║  Baseline model:   $BASELINE"
echo "║  Training data:    $DATADIR/"
echo "║  Eval output:      $EVAL_OUTPUT/"
if [[ -n "$EXTRA_ARGS" ]]; then
echo "║  Extra args:       $EXTRA_ARGS"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Train LoRA adapter
# ---------------------------------------------------------------------------

# Paths inside Docker: the project is mounted at /workspace/biasbuster,
# so remap host-absolute DATADIR paths to their container equivalents.
# LORA_DIR and MERGED_DIR are already relative (resolve from container workdir).
CONTAINER_WORKDIR="/workspace/biasbuster"
CONTAINER_TRAIN_FILE="${TRAIN_FILE/#$PROJECT_DIR/$CONTAINER_WORKDIR}"
CONTAINER_VAL_FILE="${VAL_FILE/#$PROJECT_DIR/$CONTAINER_WORKDIR}"

echo "=== Stage 1/4: Training LoRA adapter ==="
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
            --model $BASE_MODEL \
            --train-file $CONTAINER_TRAIN_FILE \
            --val-file $CONTAINER_VAL_FILE \
            --output-dir $LORA_DIR \
            $EXTRA_ARGS
    "

# Validate training output
if [[ ! -d "$LORA_DIR/final_adapter" ]]; then
    fail "Training did not produce final_adapter/ in $LORA_DIR"
fi

echo ""
echo "=== Stage 1/4: Training complete ==="
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Merge adapter
# ---------------------------------------------------------------------------

echo "=== Stage 2/4: Merging adapter ==="
echo ""

if [[ "${BASE_MODEL,,}" == *gpt-oss* ]]; then
    # GPT-OSS: surgical merge on host (preserves MXFP4 expert weights)
    uv run python -m training.merge_adapter_surgical \
        --model "$BASE_MODEL" \
        --adapter-path "$LORA_DIR/final_adapter" \
        --output-dir "$MERGED_DIR"
else
    # Dense models: standard merge in Docker
    docker run --gpus all --rm -it \
        --shm-size=16g \
        -v "$PROJECT_DIR":/workspace/biasbuster \
        -v "$HF_CACHE":/root/.cache/huggingface \
        -w /workspace/biasbuster \
        "$IMAGE" \
        bash -c "
            pip install --quiet peft 'transformers>=4.57' &&
            python -m training.merge_adapter \
                --model $BASE_MODEL \
                --adapter-path $LORA_DIR/final_adapter \
                --output-dir $MERGED_DIR
        "
fi

# Validate merge output
if [[ ! -d "$MERGED_DIR" ]] || [[ -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]]; then
    fail "Merge did not produce output in $MERGED_DIR"
fi

echo ""
echo "=== Stage 2/4: Merge complete ==="
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Export to Ollama
# ---------------------------------------------------------------------------

echo "=== Stage 3/4: Exporting to Ollama as $VERSIONED_NAME ==="
echo ""

bash "$PROJECT_DIR/training/export_to_ollama.sh" "$MERGED_DIR" "$VERSIONED_NAME"

# Validate Ollama model exists
OLLAMA_MODELS_POST="$(ollama list 2>/dev/null || true)"
if ! echo "$OLLAMA_MODELS_POST" | grep -qF "$VERSIONED_NAME"; then
    fail "Ollama export failed — model '$VERSIONED_NAME' not found in ollama list"
fi

echo ""
echo "=== Stage 3/4: Ollama export complete ==="
echo ""

# ---------------------------------------------------------------------------
# Stage 4: Evaluate (fine-tuned vs baseline, sequential)
# ---------------------------------------------------------------------------

echo "=== Stage 4/4: Evaluating $VERSIONED_NAME vs $BASELINE ==="
echo ""

uv run python -m evaluation.run \
    --test-set "$TEST_FILE" \
    --model-a "$VERSIONED_NAME" \
    --endpoint-a "$OLLAMA_ENDPOINT" \
    --model-b "$BASELINE" \
    --endpoint-b "$OLLAMA_ENDPOINT" \
    --mode fine-tuned \
    --sequential \
    --num-ctx 4096 \
    --output "$EVAL_OUTPUT"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                        Complete!                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:    $VERSIONED_NAME"
echo "║  Results:  $EVAL_OUTPUT/"
echo "╚══════════════════════════════════════════════════════════════╝"
