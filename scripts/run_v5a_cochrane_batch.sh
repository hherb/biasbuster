#!/usr/bin/env bash
# Run V5A decomposed annotation across all Cochrane-tagged papers that
# already have JATS XML cached locally, for a single model. Idempotent:
# papers already annotated for this model in the DB are skipped.
#
# Designed to be run in two terminals in parallel:
#     ./scripts/run_v5a_cochrane_batch.sh anthropic
#     ./scripts/run_v5a_cochrane_batch.sh ollama:gemma4:26b-a4b-it-q8_0
#
# The two backends don't compete for the same resource (Sonnet is API-
# bound, gemma4 is local), so wall-clock = max(sonnet_time, gemma4_time).
#
# Logs go to v5a_batch_<safe_model>.log in the project root, with one
# line per PMID prefixed by an ISO timestamp.

set -uo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: $0 <model>" >&2
    echo "  examples:" >&2
    echo "    $0 anthropic" >&2
    echo "    $0 ollama:gemma4:26b-a4b-it-q8_0" >&2
    exit 2
fi

DB_PATH="${BIASBUSTER_DB:-dataset/biasbuster.db}"
CACHE_DIR="${HOME}/.biasbuster/downloads/pmid"

# DB stores annotations under "<model>_fulltext_decomposed".
# Special-case: "anthropic" is stored as "anthropic_fulltext_decomposed"
# (without a model suffix), matching the existing 15-paper run.
if [[ "$MODEL" == "anthropic" ]]; then
    DB_MODEL_TAG="anthropic_fulltext_decomposed"
    SAFE_MODEL="anthropic"
else
    DB_MODEL_TAG="${MODEL}_fulltext_decomposed"
    SAFE_MODEL=$(echo "$MODEL" | tr ':/' '__')
fi

LOG="v5a_batch_${SAFE_MODEL}.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf "%s %s\n" "$(ts)" "$*" | tee -a "$LOG"; }

if [[ ! -f "$DB_PATH" ]]; then
    log "ERROR: DB not found at $DB_PATH"
    exit 1
fi

# Build the candidate PMID list: Cochrane-tagged AND has cached JATS AND
# not yet annotated under DB_MODEL_TAG.
mapfile -t COCHRANE_PMIDS < <(
    sqlite3 "$DB_PATH" \
        "SELECT pmid FROM papers
         WHERE source LIKE 'cochrane%'
           AND COALESCE(excluded, 0) = 0
         ORDER BY pmid"
)

mapfile -t ANNOTATED_PMIDS < <(
    sqlite3 "$DB_PATH" \
        "SELECT pmid FROM annotations WHERE model_name = '$DB_MODEL_TAG'"
)

# Build a set of already-annotated PMIDs for O(1) lookup.
declare -A IS_ANNOTATED
for p in "${ANNOTATED_PMIDS[@]}"; do
    IS_ANNOTATED["$p"]=1
done

# Filter: must have JATS cached AND not already annotated.
TODO=()
SKIPPED_NO_JATS=0
SKIPPED_DONE=0
for pmid in "${COCHRANE_PMIDS[@]}"; do
    if [[ ! -s "${CACHE_DIR}/${pmid}.jats.xml" ]]; then
        SKIPPED_NO_JATS=$((SKIPPED_NO_JATS + 1))
        continue
    fi
    if [[ -n "${IS_ANNOTATED[$pmid]:-}" ]]; then
        SKIPPED_DONE=$((SKIPPED_DONE + 1))
        continue
    fi
    TODO+=("$pmid")
done

log "model=$MODEL db_tag=$DB_MODEL_TAG"
log "candidates=${#COCHRANE_PMIDS[@]} cached_jats_papers=${#TODO[@]}+${SKIPPED_DONE}_done"
log "skipped: no_jats=$SKIPPED_NO_JATS already_annotated=$SKIPPED_DONE"
log "to annotate now: ${#TODO[@]}"

if [[ ${#TODO[@]} -eq 0 ]]; then
    log "nothing to do, exiting."
    exit 0
fi

# Per-paper run. Continue past failures (they get logged and we move on)
# so one bad paper doesn't kill an 8-hour run.
OK=0
FAIL=0
for i in "${!TODO[@]}"; do
    pmid="${TODO[$i]}"
    n=$((i + 1))
    log "[$n/${#TODO[@]}] $pmid: starting"
    start=$SECONDS
    if uv run python annotate_single_paper.py \
            --pmid "$pmid" \
            --decomposed \
            --model "$MODEL" \
            --source cochrane_rob \
            >>"$LOG" 2>&1; then
        elapsed=$((SECONDS - start))
        log "[$n/${#TODO[@]}] $pmid: ok (${elapsed}s)"
        OK=$((OK + 1))
    else
        elapsed=$((SECONDS - start))
        log "[$n/${#TODO[@]}] $pmid: FAILED (${elapsed}s)"
        FAIL=$((FAIL + 1))
    fi
done

log "done. ok=$OK fail=$FAIL total=${#TODO[@]}"
