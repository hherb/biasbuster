#!/bin/bash
# V5A decomposed-pipeline evaluation — run all 5 test papers on all 3
# models, then generate the comparison report. Sequential because the
# local Ollama models share the same GPU.
#
# Usage:  nohup ./run_v5a_eval.sh > v5a_eval.log 2>&1 &

set -u

PMIDS="32382720 39691748 39777610 39905419 41750436"
MODELS="anthropic ollama:gemma4:26b-a4b-it-q8_0 ollama:gpt-oss:20b"

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

echo ""
echo "=============================================================="
echo "$(date) | generating comparison report"
echo "=============================================================="
uv run python -m biasbuster.pipeline --stage compare --models \
  anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed,ollama:gpt-oss:20b_fulltext_decomposed

echo ""
echo "=============================================================="
echo "$(date) | done"
echo "=============================================================="
