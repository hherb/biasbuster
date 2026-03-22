# 6. Exporting Training Data

**What you'll do:** Convert validated annotations from the database into fine-tuning formats with chain-of-thought reasoning and verification steps.

## Run Export

```bash
uv run python pipeline.py --stage export
```

## Output Structure

Export creates training files in two formats with 80/10/10 train/val/test splits:

```
dataset/export/
├── alpaca/                    # Primary format (used for training)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── metadata.json
└── sharegpt/                  # Alternative multi-turn format
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── metadata.json
```

A third format, `openai_chat`, is available programmatically via `export.export_dataset(fmt="openai_chat")` but is not produced by the default pipeline export.

**Config:**
- `train_split` (default: 0.8) -- fraction for training
- `val_split` (default: 0.1) -- fraction for validation (test = remainder)
- `export_seed` (default: 42) -- random seed for reproducible splits

## Format Details

### Alpaca Format (Primary)

Each line in the JSONL file contains:

```json
{
  "system": "You are a biomedical research integrity analyst...",
  "instruction": "Assess the following clinical trial abstract for potential bias:\n\nTitle: ...\nPMID: ...\n\nAbstract: ...",
  "input": "",
  "output": "<think>...</think>\n\n## Statistical Reporting: HIGH\n..."
}
```

### ShareGPT Format

Multi-turn conversation format:

```json
{
  "conversations": [
    {"from": "system", "value": "You are a biomedical research integrity analyst..."},
    {"from": "human", "value": "Assess the following clinical trial abstract..."},
    {"from": "gpt", "value": "<think>...</think>\n\n## Statistical Reporting..."}
  ]
}
```

## The Training Output

Each training example's output section contains two key components:

### Reasoning Chain (`<think>` tags)

The model learns to reason step by step before providing its assessment:

```
<think>
The abstract reports effect sizes using only relative measures
without providing absolute risk reduction or NNT. This makes it
impossible to assess the clinical significance of the findings
without knowing the baseline risk.

Key text: "Treatment reduced cardiovascular events by 23% (HR 0.77,
95% CI 0.65-0.92, p=0.004)"
</think>
```

### Structured Assessment

After reasoning, the output includes per-domain severity ratings, evidence quotes, and verification steps:

```
## Statistical Reporting: HIGH
- Only relative measures reported (HR 0.77)
- No absolute risk reduction or NNT provided
- No baseline event rate given

## Recommended Verification Steps
1. Check ClinicalTrials.gov for registered primary outcomes
2. Search CMS Open Payments for author payment records
3. Verify author affiliations via ORCID
```

## Retraction Severity Floors

During export, retracted papers automatically receive minimum severity floors based on their retraction reason category (e.g., data fabrication receives CRITICAL, plagiarism receives HIGH). This is applied by the retraction classifier before format conversion, ensuring the training data accurately reflects the severity of known-biased papers.

## Verify Exported Data

```bash
# Check file sizes and line counts
wc -l dataset/export/alpaca/*.jsonl

# Preview a training example
head -1 dataset/export/alpaca/train.jsonl | python3 -m json.tool | head -20
```

## Next Step

[Establishing a Baseline](07_baseline_evaluation.md) -- evaluate base models in zero-shot mode before fine-tuning to measure improvement.
