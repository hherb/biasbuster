#!/usr/bin/env bash
#
# Calibration paper test — sequentially run the full 4-mode comparison
# across 4 calibration papers for Claude (ground truth) and three local
# models (gpt-oss 120b, gpt-oss 20b, gemma4 26B).
#
# Designed to be started once and left alone — suitable for
# ``nohup bash scripts/run_calibration_test.sh > calibration.log 2>&1 &``
# or running in a detached tmux/screen session.
#
# Test matrix
# -----------
# Papers (see docs/two_step_approach/INITIAL_FINDINGS_V3.md §6.2):
#     32382720 — LOW-RoB anchor (EClinicalMedicine rTMS, all 5 Cochrane
#                domains = low). Must rate low or none.
#     39777610 — Industry-funded LOW (Dermavant tapinarof phase 3).
#                Primary test of whether Round 10 trigger (d)
#                over-fires on well-conducted industry RCTs.
#     39905419 — SOME_CONCERNS (BMC CAM balneotherapy post-COVID).
#                Middle-of-scale anchor.
#     39691748 — HIGH (J Pain Research lidocaine patch). HIGH
#                generalisation test — different archetype from the
#                Seed Health probiotic paper that motivated v3.
#
# Stages:
#   Stage 1 — Claude full text on each paper (4 runs, ~3 min each)
#   Stage 2 — compare_singlecall_twocall.py on each local model
#             with all 4 PMIDs + --full-text → 4 modes × 4 papers
#             per model = 16 annotations per model × 3 models = 48.
#
# All outputs land in the same SQLite DB under these tags:
#   anthropic_fulltext                            # Stage 1 GT
#   ollama_gpt-oss_120b_abstract_singlecall       # a1
#   ollama_gpt-oss_120b_abstract_twocall          # a2
#   ollama_gpt-oss_120b_fulltext_singlecall       # f1
#   ollama_gpt-oss_120b_fulltext_twocall          # f2
#   ollama_gpt-oss_20b_*   (same 4 tags)
#   ollama_gemma4_26b-a4b-it-q8_0_*   (same 4 tags)
#
# NOTE: compare_singlecall_twocall.py DELETES and re-inserts on every
# run, so re-running this script from scratch overwrites whatever was
# there. annotate_single_paper.py (Stage 1) skips if the annotation
# already exists — safe to re-run without --force.

set -u  # abort on undefined vars, but not on command failure (one run
        # crashing should not take down the whole matrix)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CALIBRATION_PMIDS=(32382720 39777610 39905419 39691748)
CALIBRATION_PMIDS_CSV=$(IFS=, ; echo "${CALIBRATION_PMIDS[*]}")

# Run local models smallest-first so partial failures still leave us
# with the faster/easier results to look at.
LOCAL_MODELS=(
    "ollama:gpt-oss:20b"
    "ollama:gemma4:26b-a4b-it-q8_0"
    "ollama:gpt-oss:120b"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="calibration_test_logs_${TIMESTAMP}"
OUTPUT_DIR="calibration_test_results_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

MAIN_LOG="${LOG_DIR}/main.log"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$MAIN_LOG"
}

section() {
    log ""
    log "============================================================"
    log "$*"
    log "============================================================"
}

safe_model_tag() {
    # ollama:gpt-oss:120b → ollama_gpt-oss_120b  (matches
    # compare_singlecall_twocall.py's _safe_model_tag)
    echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g'
}

# ---------------------------------------------------------------------
# Stage 1 — Claude full-text ground truth
# ---------------------------------------------------------------------

section "STAGE 1 — Claude full-text ground truth (4 papers)"
log "Output dir: $OUTPUT_DIR"
log "Log dir:    $LOG_DIR"

STAGE1_OK=0
STAGE1_FAIL=0
for pmid in "${CALIBRATION_PMIDS[@]}"; do
    log ""
    log "[Claude GT] PMID $pmid — starting"
    claude_log="${LOG_DIR}/claude_ft_${pmid}.log"
    claude_out="${OUTPUT_DIR}/claude_ft_${pmid}.json"

    uv run python annotate_single_paper.py \
        --pmid "$pmid" \
        --model anthropic \
        --full-text \
        --source calibration_test \
        --output "$claude_out" \
        > "$claude_log" 2>&1
    annotate_rc=$?

    # annotate_single_paper.py exits 0 both on success AND when it
    # skips an already-annotated paper. Verify the annotation row
    # actually exists in the DB either way.
    db_check=$(uv run python -c "
import sqlite3
from config import Config
conn = sqlite3.connect(Config().db_path)
cur = conn.cursor()
cur.execute('SELECT overall_severity, overall_bias_probability FROM annotations WHERE pmid=? AND model_name=?',
            ('$pmid', 'anthropic_fulltext'))
row = cur.fetchone()
conn.close()
if row is None:
    print('MISSING')
else:
    print(f'sev={row[0]} prob={row[1]:.2f}')
" 2>/dev/null)

    if [[ "$db_check" == MISSING ]]; then
        log "[Claude GT] PMID $pmid — annotation missing (annotate rc=$annotate_rc); see $claude_log"
        STAGE1_FAIL=$((STAGE1_FAIL + 1))
    else
        log "[Claude GT] PMID $pmid — OK ($db_check)"
        STAGE1_OK=$((STAGE1_OK + 1))
    fi
done

log ""
log "Stage 1 complete: $STAGE1_OK OK, $STAGE1_FAIL failed"
if [[ $STAGE1_OK -eq 0 ]]; then
    log "No Claude ground truth was produced — aborting before local-model stage"
    exit 1
fi

# ---------------------------------------------------------------------
# Stage 2 — local models, all 4 modes × 4 papers each
# ---------------------------------------------------------------------

section "STAGE 2 — local models × 4 modes × 4 papers"

STAGE2_OK=0
STAGE2_FAIL=0
for model in "${LOCAL_MODELS[@]}"; do
    model_safe=$(safe_model_tag "$model")
    model_log="${LOG_DIR}/${model_safe}.log"
    model_out="${OUTPUT_DIR}/${model_safe}.json"

    log ""
    log "[$model] starting — 4 modes × 4 papers = 16 annotations"
    log "[$model] log: $model_log"
    log "[$model] output: $model_out"
    model_t0=$(date +%s)

    if uv run python scripts/compare_singlecall_twocall.py \
            --pmids "$CALIBRATION_PMIDS_CSV" \
            --model "$model" \
            --full-text \
            --source calibration_test \
            --output "$model_out" \
            > "$model_log" 2>&1 ; then
        elapsed=$(( $(date +%s) - model_t0 ))
        log "[$model] OK in ${elapsed}s"
        STAGE2_OK=$((STAGE2_OK + 1))
    else
        elapsed=$(( $(date +%s) - model_t0 ))
        log "[$model] FAILED after ${elapsed}s; see $model_log"
        STAGE2_FAIL=$((STAGE2_FAIL + 1))
    fi
done

log ""
log "Stage 2 complete: $STAGE2_OK/${#LOCAL_MODELS[@]} models OK"

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

section "DONE — summary"

log ""
log "DB tags now populated for PMIDs: ${CALIBRATION_PMIDS_CSV}"
log ""
log "Per-paper headline severity across all annotation tags:"

uv run python << PY 2>&1 | tee -a "$MAIN_LOG"
import sqlite3
from config import Config

pmids = [$(printf "'%s'," "${CALIBRATION_PMIDS[@]}" | sed 's/,$//')]
tags = [
    ('Claude GT',         'anthropic_fulltext'),
    ('120b a1',           'ollama_gpt-oss_120b_abstract_singlecall'),
    ('120b a2',           'ollama_gpt-oss_120b_abstract_twocall'),
    ('120b f1',           'ollama_gpt-oss_120b_fulltext_singlecall'),
    ('120b f2',           'ollama_gpt-oss_120b_fulltext_twocall'),
    ('20b a1',            'ollama_gpt-oss_20b_abstract_singlecall'),
    ('20b a2',            'ollama_gpt-oss_20b_abstract_twocall'),
    ('20b f1',            'ollama_gpt-oss_20b_fulltext_singlecall'),
    ('20b f2',            'ollama_gpt-oss_20b_fulltext_twocall'),
    ('gemma4 a1',         'ollama_gemma4_26b-a4b-it-q8_0_abstract_singlecall'),
    ('gemma4 a2',         'ollama_gemma4_26b-a4b-it-q8_0_abstract_twocall'),
    ('gemma4 f1',         'ollama_gemma4_26b-a4b-it-q8_0_fulltext_singlecall'),
    ('gemma4 f2',         'ollama_gemma4_26b-a4b-it-q8_0_fulltext_twocall'),
]

conn = sqlite3.connect(Config().db_path)
cur = conn.cursor()

header = f"{'label':<14}"
for p in pmids:
    header += f"  {p:<16}"
print(header)
print("-" * len(header))

for label, tag in tags:
    row_str = f"{label:<14}"
    for pmid in pmids:
        cur.execute(
            "SELECT overall_severity, overall_bias_probability FROM annotations "
            "WHERE pmid=? AND model_name=?", (pmid, tag))
        r = cur.fetchone()
        if r is None:
            cell = "—"
        else:
            sev, prob = r
            cell = f"{sev}/{prob:.2f}" if prob is not None else f"{sev}"
        row_str += f"  {cell:<16}"
    print(row_str)
conn.close()
PY

log ""
log "Logs:    $LOG_DIR/"
log "Outputs: $OUTPUT_DIR/"
log "Main log: $MAIN_LOG"
log ""
log "Next step: score each local-model cell against the Claude GT in the"
log "same column. Papers where every local model matches Claude are good"
log "calibration anchors; cells where a local model diverges upward on a"
log "LOW paper are the Round 10 regression signals to investigate."
