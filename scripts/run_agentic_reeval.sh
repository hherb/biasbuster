#!/bin/bash
# Re-annotate with new review scaffold prompts — run both local models
# on all 5 papers with --force to overwrite old annotations.
# Sequential because both models share the same GPU.

PMIDS="32382720 39691748 39777610 39905419 41750436"
MODELS="ollama:gemma4:26b-a4b-it-q8_0 ollama:gpt-oss:20b"

for model in $MODELS; do
  for pmid in $PMIDS; do
    echo "=== $(date) | $model | PMID $pmid ==="
    uv run python annotate_single_paper.py --pmid "$pmid" --model "$model" --agentic --force
    echo ""
  done
done

echo "=== $(date) | Running comparison ==="
uv run python -m biasbuster.pipeline --stage compare \
  --models anthropic_fulltext_agentic,ollama:gemma4:26b-a4b-it-q8_0_fulltext_agentic,ollama:gpt-oss:20b_fulltext_agentic

echo "=== $(date) | Done ==="

