# 10. Evaluating Fine-Tuned Models

**What you'll do:** Evaluate your fine-tuned model against the zero-shot baseline to measure improvement, using the same test set and metrics.

## Run Fine-Tuned Evaluation

With the fine-tuned model served via Ollama:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a <your-model>-biasbuster --endpoint-a http://localhost:11434 \
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
cp eval_results/zero_shot/<base-model>_outputs.jsonl \
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
    --model-a qwen3.5-27b-biasbuster --endpoint-a http://localhost:11434 \
    --model-b qwen3.5:27b-q8_0 --endpoint-b http://localhost:11434 \
    --mode fine-tuned --sequential \
    --num-ctx 4096 \
    --output eval_results/comparison/

# Compare two fine-tuned models head-to-head
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a olmo-3.1-32b-biasbuster --endpoint-a http://localhost:11434 \
    --model-b gpt-oss-20b-biasbuster --endpoint-b http://localhost:11434 \
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
- Watch for the **precision/recall tradeoff**: fine-tuning can increase precision but reduce recall. If recall drops below 0.85, the model may be too conservative for screening use.

**Ordinal severity (weighted kappa, MAE):**
- This is where fine-tuning should show the biggest gains
- Zero-shot models typically have near-chance kappa (0.02-0.07)
- A well-trained model should achieve kappa > 0.3
- This remains the hardest metric to improve

**Per-dimension F1:**
- Check each domain independently
- COI detection is often a weakness in zero-shot models but improves dramatically with fine-tuning
- Methodology and statistical reporting tend to be the hardest dimensions

**Calibration error (ECE):**
- Should ideally decrease after fine-tuning
- May require post-hoc calibration (temperature scaling) if worsened

**Verification source knowledge:**
- Critical metric for the verification agent (see below)
- The enriched training data should dramatically improve citation rates for ClinicalTrials.gov, ORCID, and Retraction Watch
- CMS Open Payments tends to be the weakest source

**Reasoning quality:**
- Fine-tuned models should produce `<think>` reasoning blocks (absent in zero-shot)
- Check `thinking_present_rate` and `mean_thinking_length`

### Statistical Significance

The report includes McNemar's test (binary) and Wilcoxon signed-rank test (ordinal) for each dimension. Differences marked as "significant" have p < 0.05. With small test sets, real improvements may not reach significance.

## Output Files

```
eval_results/fine_tuned/
├── <model>_outputs.jsonl           # Raw model responses
├── <model>_evaluation.json         # Per-model metrics
├── comparison_fine-tuned_<date>.json   # Full comparison data
├── comparison_fine-tuned_<date>.md     # Human-readable report
└── comparison_fine-tuned_<date>.csv    # Spreadsheet export
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
4. **Re-train and re-evaluate** -- use the auto-versioned orchestrator for quick iteration:

```bash
# One command: train → merge → Ollama export → evaluate (auto-versioned)
./train_and_evaluate.sh gpt-oss-20b

# Smoke test (5 training steps)
./train_and_evaluate.sh gpt-oss-20b -- --max-steps 5

# Custom training data
./train_and_evaluate.sh gpt-oss-20b --datadir dataset_v2/export/alpaca

# Pass hyperparameter overrides to train_lora.py
./train_and_evaluate.sh gpt-oss-20b -- --lr 1e-5 --epochs 3
```

The script queries the database for existing `{model}-biasbusterV{n}` entries,
increments the version number, and evaluates the new model against the unmodified
baseline via Ollama with `--sequential` mode (single GPU). Accepts both preset
keys (`gpt-oss-20b`) and Ollama names (`gpt-oss:20b`).

For manual control over individual steps, see sections [8](08_training.md),
[9](09_merge_and_deploy.md), and the evaluation commands above.

## Testing with the Verification Agent

After evaluation, test the fine-tuned model with the verification agent demo to see it execute recommended verification steps against real APIs:

```bash
uv run python -m utils.agent_demo --model qwen3.5-27b-biasbuster --port 8082
```

The agent wrapper (`agent/` package) parses verification steps from the model's output and dispatches them to real APIs:
- **ClinicalTrials.gov** -- registered outcomes for switching detection
- **ORCID** -- author affiliation and employment history
- **Europe PMC** -- funder metadata and full-text
- **CMS Open Payments** -- physician payments from pharmaceutical companies
- **Crossref / Retraction Watch** -- retraction and correction status

The agent then feeds verification results back to the model for a refined assessment.

## Pipeline Stage: Compare Annotations

For comparing multiple annotator models (not fine-tuned vs baseline, but e.g., Anthropic vs DeepSeek annotations):

```bash
uv run python pipeline.py --stage compare
```

This computes inter-model agreement using the same metrics framework, generating a report in `dataset/annotation_comparison/`.
