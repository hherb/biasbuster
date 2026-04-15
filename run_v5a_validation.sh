#!/bin/bash
# V5A validation run — 16 papers (15 Cochrane RoB + 1 manual import).
# All have full text cached locally so no network risk overnight.
#
# The pilot set (5 papers) is a subset, so we get:
#   - 5 control papers (re-annotate; --force ensures fresh run)
#   - 11 new papers for the expanded validation
#
# Usage:  nohup ./run_v5a_validation.sh > v5a_validation.log 2>&1 &

set -u

# All 16 papers with cached JATS full text, Cochrane RoB + 1 manual
PMIDS="32318744 32382720 32841300 33033445 34435694 35409805 36110787 \
36518417 37184828 38546803 39555048 39676466 39691748 39777610 \
39905419 41750436"

MODELS="anthropic ollama:gemma4:26b-a4b-it-q8_0 ollama:gpt-oss:20b"

START=$(date +%s)

for model in $MODELS; do
  for pmid in $PMIDS; do
    echo ""
    echo "=============================================================="
    echo "$(date) | model=$model | pmid=$pmid"
    echo "=============================================================="
    uv run python annotate_single_paper.py \
      --pmid "$pmid" --model "$model" --decomposed --force
  done
done

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=============================================================="
echo "$(date) | annotation complete in ${ELAPSED}s — generating report"
echo "=============================================================="
uv run python -m biasbuster.pipeline --stage compare --models \
  anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed,ollama:gpt-oss:20b_fulltext_decomposed

echo ""
echo "=============================================================="
echo "$(date) | computing Cochrane-agreement report"
echo "=============================================================="
uv run python compare_vs_cochrane.py --models \
  anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed,ollama:gpt-oss:20b_fulltext_decomposed

echo ""
echo "=============================================================="
echo "$(date) | done"
echo "=============================================================="
