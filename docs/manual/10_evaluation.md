# 10. Evaluating Fine-Tuned Models

**What you'll do:** Evaluate your fine-tuned model against the zero-shot baseline to measure improvement, using the same test set and metrics.

## Run Fine-Tuned Evaluation

With the fine-tuned model served via Ollama:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --num-ctx 4096 \
    --output eval_results/fine_tuned/
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
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --model-b olmo-3.1:32b-instruct-q8_0 --endpoint-b http://localhost:11434 \
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

**Ordinal severity (weighted kappa, MAE):**
- This is where fine-tuning should show the biggest gains
- Zero-shot models typically have near-chance kappa (0.02-0.07)
- A well-trained model should achieve kappa > 0.3

**Per-dimension F1:**
- Check each domain independently
- COI detection is a common weakness in zero-shot models
- Fine-tuning on examples with CMS Open Payments citations should improve it

**Calibration error (ECE):**
- Should decrease after fine-tuning
- A well-calibrated model's predicted probabilities match actual bias rates

**Verification source knowledge:**
- Fine-tuned models should cite specific databases more consistently
- Look for improvements in CMS Open Payments and ClinicalTrials.gov citation rates

**Reasoning quality:**
- Fine-tuned models should produce `<think>` reasoning blocks (absent in zero-shot)
- Check `thinking_present_rate` and `mean_thinking_length`

### Statistical Significance

The report includes McNemar's test (binary) and Wilcoxon signed-rank test (ordinal) for each dimension. Differences marked as "significant" have p < 0.05. With only 89 test examples, small improvements may not reach significance -- this doesn't mean they're not real.

## Output Files

```
eval_results/fine_tuned/
├── olmo-3.1-32b-biasbuster_outputs.jsonl         # Raw model responses
├── olmo-3.1-32b-biasbuster_evaluation.json        # Per-model metrics
├── comparison_fine-tuned_2026-03-17.json           # Full comparison data
├── comparison_fine-tuned_2026-03-17.md             # Human-readable report
└── comparison_fine-tuned_2026-03-17.csv            # Spreadsheet export
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
3. **Adjust hyperparameters** -- try more epochs, different learning rate, or larger LoRA rank
4. **Re-train and re-evaluate** -- the pipeline supports quick iteration:

```bash
# Train with adjusted settings
./run_training.sh olmo-3.1-32b

# Merge and quantize
./run_merge.sh olmo-3.1-32b --quantize q8_0

# Re-evaluate
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --force-reevaluation \
    --num-ctx 4096 \
    --output eval_results/fine_tuned/
```

Use `--force-reevaluation` to re-run all examples even if previous results exist in the database.

## Pipeline Stage: Compare Annotations

For comparing multiple annotator models (not fine-tuned vs baseline, but e.g., Anthropic vs DeepSeek annotations):

```bash
uv run python pipeline.py --stage compare
```

This computes inter-model agreement using the same metrics framework, generating a report in `dataset/annotation_comparison/`.
