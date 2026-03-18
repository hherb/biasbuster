# 10. Evaluating Fine-Tuned Models

**What you'll do:** Evaluate your fine-tuned model against the zero-shot baseline to measure improvement, using the same test set and metrics.

## Available Fine-Tuned Models

| Model | Size | Base | Key Strengths | Key Weaknesses | Docs |
|-------|------|------|---------------|----------------|------|
| olmo-3.1-32b-biasbuster | 32B | OLMo-3.1-32B-Instruct | Best binary F1 (0.952), best severity kappa (0.285) | Low verification citations (Open Payments 16%) | FIRST_RUN.md |
| qwen3.5-9bb-biasbuster | 9B | Qwen3.5-9B | Best verification knowledge (mean 0.541), 1.4x faster | Lower recall (0.679), worse calibration (0.913) | SECOND_RUN.md |

## Run Fine-Tuned Evaluation

With the fine-tuned model served via Ollama:

```bash
# 32B model
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --num-ctx 4096 \
    --output eval_results/fine_tuned/

# 9B model
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-9bb-biasbuster --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --num-ctx 4096 \
    --output eval_results/comparison/
```

Key difference from baseline: `--mode fine-tuned` uses the training system prompt (matching what the model was trained on) instead of the generic zero-shot prompt.

## Compare Against Baseline

### Method 1: Re-Analysis with Both Outputs

Copy the baseline outputs alongside the fine-tuned outputs and re-analyse:

```bash
# Copy baseline outputs into the fine-tuned results directory
cp eval_results/zero_shot/olmo-3.1:32b-instruct-q8_0_outputs.jsonl \
   eval_results/fine_tuned/

# Re-analyse all outputs together (no inference needed)
uv run python -m evaluation.run \
    --reanalyse eval_results/fine_tuned/ \
    --test-set dataset/export/alpaca/test.jsonl \
    --output eval_results/fine_tuned/
```

This generates a comparison report between the fine-tuned and baseline models.

### Method 2: Run Both Models

If both models are available via Ollama:

```bash
# Compare fine-tuned vs baseline (same model family)
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-9bb-biasbuster --endpoint-a http://localhost:11434 \
    --model-b qwen3.5:9b-q8_0 --endpoint-b http://localhost:11434 \
    --mode fine-tuned --sequential \
    --num-ctx 4096 \
    --output eval_results/comparison/

# Compare two fine-tuned models head-to-head
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --model-b qwen3.5-9bb-biasbuster --endpoint-b http://localhost:11434 \
    --mode fine-tuned --sequential \
    --num-ctx 4096 \
    --output eval_results/comparison/
```

## Understanding the Comparison Report

The generated Markdown report (`comparison_fine-tuned_*.md`) shows side-by-side metrics:

### What to Look For

**Binary detection (F1, precision, recall):**
- Fine-tuning should maintain or improve recall (catching all biased abstracts)
- Precision may improve as the model learns to distinguish severity levels
- Watch for the **precision/recall tradeoff**: fine-tuning tends to increase precision but reduce recall (the 9B model went from recall 1.000 zero-shot to 0.679 fine-tuned). If recall drops below 0.85, the model is too conservative for screening use.

**Ordinal severity (weighted kappa, MAE):**
- This is where fine-tuning should show the biggest gains
- Zero-shot models typically have near-chance kappa (0.02-0.07)
- A well-trained model should achieve kappa > 0.3
- Current best: OLMo-32B fine-tuned at kappa 0.285. This remains the hardest metric to improve.

**Per-dimension F1:**
- Check each domain independently
- COI detection was a common weakness in zero-shot models but improved dramatically with fine-tuning (32B: 0.667 → 0.927)
- Methodology and statistical reporting are the hardest dimensions (both models score < 0.77)

**Calibration error (ECE):**
- Should decrease after fine-tuning, but in practice it has worsened for both models (32B: 0.670 → 0.731, 9B: 0.913)
- May require post-hoc calibration (temperature scaling) or calibration-focused training

**Verification source knowledge:**
- Critical metric for the verification agent wrapper (`agent/runner.py`)
- The enriched training data dramatically improved this: 9B fine-tuned achieves ClinicalTrials.gov 99%, ORCID 94%, Retraction Watch 95%
- CMS Open Payments remains the weakest source (57% for 9B, 16% for 32B)

**Reasoning quality:**
- Fine-tuned models should produce `<think>` reasoning blocks (absent in zero-shot)
- Check `thinking_present_rate` and `mean_thinking_length`
- Both fine-tuned models achieve ~100% thinking chain presence

### Statistical Significance

The report includes McNemar's test (binary) and Wilcoxon signed-rank test (ordinal) for each dimension. Differences marked as "significant" have p < 0.05. With 89-115 test examples, small improvements may not reach significance -- this doesn't mean they're not real.

### Current Performance Benchmarks

For reference, the best achieved metrics across all runs:

| Metric | Best Value | Model | Run |
|--------|-----------|-------|-----|
| Binary F1 | 0.952 | OLMo-32B fine-tuned | First Run |
| Recall | 0.920 | OLMo-32B fine-tuned | First Run |
| Ordinal kappa | 0.285 | OLMo-32B fine-tuned | First Run |
| Verification score | 0.541 | Qwen3.5-9B fine-tuned | Second Run |
| Thinking chains | 100% | OLMo-32B fine-tuned | First Run |
| Inference speed | 240s mean | Qwen3.5-9B fine-tuned | Second Run |

## Output Files

```
eval_results/fine_tuned/                            # First Run (OLMo-32B)
├── olmo-3.1-32b-biasbuster_outputs.jsonl           # Raw model responses
├── olmo-3.1-32b-biasbuster_evaluation.json         # Per-model metrics

eval_results/comparison/                            # Second Run (Qwen3.5-9B)
├── qwen3.5-9bb-biasbuster:latest_outputs.jsonl     # Fine-tuned raw responses
├── qwen3.5-9bb-biasbuster:latest_evaluation.json   # Fine-tuned metrics
├── qwen3.5:9b-q8_0_outputs.jsonl                   # Baseline raw responses
├── qwen3.5:9b-q8_0_evaluation.json                 # Baseline metrics
├── comparison_fine-tuned_2026-03-18.json            # Full comparison data
├── comparison_fine-tuned_2026-03-18.md              # Human-readable report
└── comparison_fine-tuned_2026-03-18.csv             # Spreadsheet export
```

## Re-Analyse Without Re-Running Inference

If you improve the scoring or parsing logic:

```bash
uv run python -m evaluation.run \
    --reanalyse eval_results/fine_tuned/ \
    --test-set dataset/export/alpaca/test.jsonl \
    --output eval_results/fine_tuned/
```

This re-parses saved raw outputs and recomputes all metrics without calling the model again.

## Iterating

If results are unsatisfactory:

1. **Review failure cases** -- inspect raw outputs in the JSONL file for patterns (broken JSON, missing domains, wrong severity)
2. **Augment training data** -- add more examples for weak dimensions (e.g., COI detection)
3. **Adjust hyperparameters** -- try more epochs, different learning rate, or larger LoRA rank. See SECOND_RUN.md Section 6.4 for 9B-optimised hyperparameter recommendations.
4. **Re-train and re-evaluate** -- the pipeline supports quick iteration:

```bash
# Train with adjusted settings (9B or 32B)
./run_training.sh qwen3.5-9b
# or: ./run_training.sh olmo-3.1-32b

# Merge and export to Ollama
./run_merge.sh qwen3.5-9b
bash training/export_to_ollama.sh training_output/qwen3.5-9b-merged qwen3.5-9b-biasbuster

# Re-evaluate
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-9bb-biasbuster --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --force-reevaluation \
    --num-ctx 4096 \
    --output eval_results/comparison/
```

Use `--force-reevaluation` to re-run all examples even if previous results exist in the database.

## Testing with the Verification Agent

After evaluation, test the fine-tuned model with the verification agent demo to see it execute recommended verification steps against real APIs:

```bash
uv run python -m utils.agent_demo --model qwen3.5-9bb-biasbuster --port 8082
```

The agent wrapper parses verification steps from the model's output and dispatches them to ClinicalTrials.gov, ORCID, Europe PMC, CMS Open Payments, and Retraction Watch. See `agent/` package for details.

## Pipeline Stage: Compare Annotations

For comparing multiple annotator models (not fine-tuned vs baseline, but e.g., Anthropic vs DeepSeek annotations):

```bash
uv run python pipeline.py --stage compare
```

This computes inter-model agreement using the same metrics framework, generating a report in `dataset/annotation_comparison/`.
