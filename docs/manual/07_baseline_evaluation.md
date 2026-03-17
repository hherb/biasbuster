# 7. Establishing a Baseline

**What you'll do:** Evaluate base models in zero-shot mode (no fine-tuning) to establish performance baselines. This lets you measure how much fine-tuning improves each model.

## Prerequisites

- Exported test set at `dataset/export/alpaca/test.jsonl`
- Models served via Ollama (or any OpenAI-compatible endpoint)

### Serve Models with Ollama

```bash
# Pull the base models (if not already available)
ollama pull qwen3.5:27b-q8_0
ollama pull olmo-3.1:32b-instruct-q8_0

# Ollama serves on http://localhost:11434 by default
ollama serve
```

## Run Baseline Evaluation

### Single Model

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5:27b-q8_0 --endpoint-a http://localhost:11434 \
    --mode zero-shot \
    --num-ctx 4096 \
    --output eval_results/zero_shot/
```

### Two Models (Sequential)

On a single GPU, run models one at a time:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5:27b-q8_0 --endpoint-a http://localhost:11434 \
    --model-b olmo-3.1:32b-instruct-q8_0 --endpoint-b http://localhost:11434 \
    --mode zero-shot --sequential \
    --num-ctx 4096 \
    --output eval_results/zero_shot/
```

In sequential mode, the tool evaluates model A first, then prompts you to swap the served model before evaluating model B. It polls the endpoint until the new model is ready.

### Two Models (Simultaneous)

If you have enough GPU memory for both:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5:27b-q8_0 --endpoint-a http://localhost:11434 \
    --model-b olmo-3.1:32b-instruct-q8_0 --endpoint-b http://localhost:11435 \
    --mode zero-shot \
    --num-ctx 4096 \
    --output eval_results/zero_shot/
```

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | zero-shot | Use `zero-shot` for base models, `fine-tuned` for trained models |
| `--num-ctx` | model default | Ollama context window. Set to 4096 for a large speedup when abstracts are short |
| `--temperature` | 0.1 | Low value for reproducible outputs |
| `--max-tokens` | 4000 | Maximum response length |
| `--sequential` | off | Run models one at a time (single GPU) |
| `--think` / `--no-think` | auto | Enable/disable extended reasoning (Qwen3+ only) |
| `--force-reevaluation` | off | Re-run all examples, ignoring saved checkpoints |

## Checkpoint/Resume

The evaluation harness saves results incrementally to both JSONL files and the SQLite database. If interrupted, re-running the same command will skip already-evaluated examples. Use `--force-reevaluation` to re-run everything.

## Output Files

After evaluation completes, the output directory contains:

```
eval_results/zero_shot/
├── qwen3.5:27b-q8_0_outputs.jsonl              # Raw model responses
├── olmo-3.1:32b-instruct-q8_0_outputs.jsonl
├── qwen3.5:27b-q8_0_evaluation.json             # Per-model metrics
├── olmo-3.1:32b-instruct-q8_0_evaluation.json
├── comparison_zero-shot_2026-03-17.json          # Full comparison data
├── comparison_zero-shot_2026-03-17.md            # Human-readable report
└── comparison_zero-shot_2026-03-17.csv           # Spreadsheet export
```

## Understanding the Metrics

### Binary Detection (Any Bias vs None)

| Metric | What It Means |
|--------|---------------|
| **F1** | Harmonic mean of precision and recall. 1.0 is perfect. |
| **Precision** | Of abstracts flagged as biased, how many actually are? |
| **Recall** | Of actually biased abstracts, how many were flagged? |

### Ordinal Severity Agreement

| Metric | What It Means |
|--------|---------------|
| **Weighted kappa** | Agreement on severity level, accounting for chance. 0.0 = chance, 1.0 = perfect. |
| **MAE** | Mean absolute error on the 0--4 severity scale. Lower is better. |
| **Exact match** | Fraction where predicted severity exactly matches ground truth. |
| **Within-one** | Fraction within one severity level of ground truth. |

### Per-Dimension Breakdown

Each of the five bias domains gets its own binary F1 and ordinal kappa. The comparison report flags statistically significant differences between models using McNemar's test (binary) and Wilcoxon signed-rank test (ordinal).

### Calibration Error

Expected Calibration Error (ECE) measures whether the model's predicted bias probability matches the actual rate. Lower is better. A model that says "70% chance of bias" should be correct about 70% of the time.

### Verification Source Knowledge

Tracks how often each model cites specific verification databases:

| Source | What It Covers |
|--------|----------------|
| CMS Open Payments | US physician payments from pharmaceutical companies |
| ClinicalTrials.gov | Registered trial protocols for outcome switching detection |
| ORCID | Author employment history and affiliations |
| Retraction Watch | Known retracted/corrected papers |
| Europe PMC | Funder metadata and full-text availability |

## Reading the Comparison Report

Open the generated Markdown report (`comparison_zero-shot_*.md`) for a quick summary. Key sections:

1. **Overall performance table** -- F1, precision, recall, kappa, calibration
2. **Per-dimension F1** -- which model wins each domain (with significance)
3. **Verification source coverage** -- which model is better at citing databases
4. **Efficiency** -- mean latency, tokens/second, error rate
5. **Bottom line** -- overall winner based on dimension wins

## Next Step

[Fine-Tuning with LoRA](08_training.md) -- train the base model on your exported dataset to improve bias detection.
