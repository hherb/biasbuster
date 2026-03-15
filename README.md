# BMLibrarian Bias Detection Dataset Builder

A toolkit for building curated training datasets to fine-tune LLMs for detecting
bias in biomedical abstracts, and for evaluating fine-tuned models head-to-head.
Designed for use with BMLibrarian.

## Architecture

```
bias_dataset_builder/
├── collectors/
│   ├── retraction_watch.py    # Retracted papers via Crossref API
│   ├── cochrane_rob.py        # Cochrane Risk of Bias assessments
│   ├── spin_detector.py       # Heuristic pre-screening for spin indicators
│   └── clinicaltrials_gov.py  # Outcome switching detection via registry
├── enrichers/
│   ├── author_coi.py          # Author conflict-of-interest verification
│   ├── funding_checker.py     # Funding source classification
│   └── effect_size_auditor.py # Relative vs absolute reporting analysis
├── annotators/
│   └── llm_prelabel.py        # API-based pre-labelling with Claude
├── schemas/
│   └── bias_taxonomy.py       # Structured bias taxonomy and labels
├── evaluation/
│   ├── harness.py             # Inference runner (OpenAI-compatible APIs)
│   ├── scorer.py              # Output parsing and ground truth attachment
│   ├── metrics.py             # Per-dimension and aggregate metrics
│   ├── comparison.py          # Statistical comparison + report generation (JSON/MD/CSV)
│   ├── run.py                 # CLI entry point (--model-a / --model-b)
│   └── selftest.py            # Self-test with synthetic data
├── utils/
│   ├── completeness_checker.py # Check labelling coverage per model
│   ├── agreement_analyzer.py   # Inter-model agreement metrics
│   └── review_gui.py           # NiceGUI web-based CSV review tool
├── config.py                  # Configuration and API keys
├── pipeline.py                # Orchestration pipeline
└── export.py                  # Export to fine-tuning formats (Alpaca, ShareGPT)
```

## Bias Taxonomy

The model is trained on a multi-dimensional bias assessment:

1. **Statistical Reporting Bias**
   - Sole/emphasis on relative risk reduction without absolute measures
   - Missing NNT/NNH
   - Baseline risk omission
   - Selective p-value reporting

2. **Spin in Conclusions**
   - Claims not supported by primary outcome
   - Inappropriate causal language from observational data
   - Focus on secondary/subgroup analyses when primary failed
   - Boutron classification: none/low/moderate/high

3. **Outcome Reporting**
   - Surrogate vs patient-centred outcomes
   - Outcome switching (vs registry)
   - Composite endpoint disaggregation missing

4. **Conflict of Interest Signals**
   - Industry funding without disclosure
   - Author-pharma payment patterns
   - Ghost authorship indicators

5. **Methodological Red Flags**
   - Inappropriate comparator (placebo when active exists)
   - Enrichment design without acknowledgment
   - Per-protocol only (no ITT)
   - Premature stopping

## Verification Sources (for model training)

The model should learn WHERE to look for corroboration:

- **CMS Open Payments** (openpaymentsdata.cms.gov) - US physician payments
- **ClinicalTrials.gov** - Registered outcomes vs published outcomes
- **WHO ICTRP** - International trial registry search
- **Crossref/Retraction Watch** - Retraction status
- **ORCID** - Author affiliation history
- **EuroPMC funding data** - Funder metadata
- **Cochrane RoB database** - Expert risk assessments

## Usage

### Dataset Building Pipeline

```bash
# Set up environment and install dependencies
uv sync

# Configure
cp config.example.py config.py
# Edit config.py with your API keys

# Run full pipeline (collect → enrich → annotate → export)
uv run python pipeline.py --stage all

# Or run individual stages
uv run python pipeline.py --stage collect
uv run python pipeline.py --stage enrich
uv run python pipeline.py --stage annotate
uv run python pipeline.py --stage export
```

### Pipeline Flow

```mermaid
flowchart TD
    subgraph Collect
        C1[Crossref / Retraction Watch]
        C2[PubMed RCTs by MeSH]
        C3[Cochrane RoB via Europe PMC]
    end

    subgraph Enrich
        E1[Effect Size Auditor]
        E2[Funding Checker]
        E3[Author COI]
        E4[Outcome Switching]
    end

    subgraph Annotate
        A1[Claude API]
        A2[DeepSeek / OpenAI-compat]
    end

    HR[Human Review<br/>edit CSV]

    subgraph Export
        X1[Alpaca + think chains]
        X2[ShareGPT]
        X3[OpenAI chat]
    end

    subgraph Compare
        M1[Per-dimension F1 & kappa]
        M2[McNemar / Wilcoxon tests]
        M3[Markdown report]
    end

    Collect -- raw/*.jsonl --> Enrich
    Enrich -- high/low suspicion.jsonl --> Annotate
    Annotate -- annotated.jsonl + review.csv --> HR
    HR -- validated labels --> Export
    Export -- test.jsonl --> Compare
    Annotate -. model outputs .-> Compare
```

Human review (editing the CSV) is a manual step between annotate and export.

## Dataset Utilities

Three tools support the human-review workflow between annotation and export.

### Completeness Checker

Reports labelling coverage — how many enriched abstracts each model has annotated,
and which PMIDs are missing.

```bash
# Show completion against configured annotation limits
uv run python -m utils.completeness_checker

# Show progress against full enriched set (ignore config caps)
uv run python -m utils.completeness_checker --no-limits

# List individual missing PMIDs
uv run python -m utils.completeness_checker --show-missing

# Check specific models only
uv run python -m utils.completeness_checker --models anthropic,deepseek
```

Example output (default: shows progress vs configured caps):

```
Source                    |            anthropic |             deepseek
-----------------------------------------------------------------------
cochrane_rob              |     6/6     (100.0%) |     6/6     (100.0%)
high_suspicion            |   393/394   ( 99.7%) |   394/394   (100.0%)
low_suspicion             |   300/300   (100.0%) |   300/300   (100.0%)  [of 2733 enriched]
retracted_papers          |   199/200   ( 99.5%) |   200/200   (100.0%)  [of 514 enriched]
-----------------------------------------------------------------------
TOTAL                     |   898/900   ( 99.8%) |   900/900   (100.0%)
```

### Inter-Model Agreement Analyzer

Compares two models' annotations on shared PMIDs — per-dimension severity
agreement, Cohen's weighted kappa, flag-level agreement, and the most divergent
cases. Lighter-weight than the full evaluation pipeline (no ground truth needed).

```bash
# Default: compare anthropic vs deepseek
uv run python -m utils.agreement_analyzer

# Specify models and save report
uv run python -m utils.agreement_analyzer --model-a anthropic --model-b deepseek --output report.md

# Show more divergent cases
uv run python -m utils.agreement_analyzer --top-divergent 20
```

Reports overall severity kappa, per-dimension exact/within-one agreement,
flag-level agreement rates (e.g. `relative_only`, `spin_level`, `funding_type`),
probability MAE and Pearson r, and the top most-divergent cases ranked by
bias probability difference.

### CSV Review GUI

Web-based tool for reviewing and validating annotation CSVs. Opens in your
browser with an editable AG Grid table. Built with NiceGUI.

```bash
# Open a specific review CSV
uv run python -m utils.review_gui dataset/labelled/anthropic/high_suspicion_review.csv

# Browse available CSVs interactively
uv run python -m utils.review_gui

# Use a different port
uv run python -m utils.review_gui --port 9090
```

Features:
- **In-grid editing** of `HUMAN_VALIDATED`, `HUMAN_OVERRIDE_SEVERITY`, and
  `HUMAN_NOTES` columns (double-click to edit)
- **Color-coded rows**: green for validated, yellow for overridden severity
- **Quick filter** and column sorting
- **"Next Unvalidated"** button to jump to the next unreviewed row
- **Detail panel** showing full reasoning text on row selection
- **Atomic save** (writes to temp file, then renames) to prevent data loss
- **Stats bar** showing validation progress

## Evaluation Harness

The `evaluation/` module provides a standardised framework for comparing
fine-tuned bias detection models. It supports zero-shot baselines and
post-fine-tuning evaluation, producing per-dimension metrics and head-to-head
statistical comparisons.

### Motivation: Dual-Model Comparison

The evaluation harness was built to rigorously compare candidate base models for
fine-tuning — specifically **OLMo 3.1 32B** and **Qwen3.5 27B**. These models
represent two distinct approaches:

| | OLMo 3.1 32B | Qwen3.5 27B |
|---|---|---|
| **Training data** | Fully open (Dolma 3: 238M academic PDFs, Semantic Scholar corpus) | Closed, large-scale multilingual |
| **Biomedical strength** | Heavy academic/scientific bias in pretraining data | Broader general coverage |
| **Transparency** | Full: data, code, checkpoints, intermediate weights all public | Model weights only |
| **Context window** | ~32K–128K | 262K native |
| **Modality** | Text only | Multimodal (text, images, video) |
| **Fine-tuning** | SFT and DPO checkpoints released separately | Standard instruct checkpoint |
| **Parameters** | 32B (~64GB bf16) | 27B (~54GB bf16) |

For a research integrity tool, OLMo's full transparency and academic-heavy
training data are compelling — you can audit what the model was trained on and
cite it in publications. Qwen3.5 likely has stronger raw benchmark performance
and offers multimodality if table/figure analysis is needed later.

The evaluation harness lets you answer empirically which model performs better on
*this specific task* rather than relying on general benchmarks.

### Evaluation Pipeline

```
Test Examples (JSONL with ground truth)
    │
    ▼
EvalHarness — query models via OpenAI-compatible API (SGLang, vLLM, Ollama)
    │
    ▼
scorer.py — parse outputs (JSON or free-text), extract <think> reasoning
    │
    ▼
metrics.py — per-dimension binary F1, ordinal kappa, flag accuracy, calibration
    │
    ▼
comparison.py — McNemar (binary) and Wilcoxon (ordinal) pairwise tests
    │
    ▼
Reports — JSON + Markdown with confusion matrices, radar data, winner summary
```

### What the Reports Measure

- **Per-dimension F1 and weighted kappa** — separately for statistical reporting,
  spin, COI, outcome reporting, and methodology. Shows which model handles
  specific bias types best.
- **Severity confusion matrices** — reveals whether a model systematically
  under-rates or over-rates bias severity.
- **Flag-level accuracy** — binary accuracy on specific flags like
  `relative_only`, `nnt_reported`, `baseline_risk_reported`.
- **Verification source coverage** — what percentage of the time each model
  recommends checking Open Payments, ClinicalTrials.gov, ORCID, etc.
- **Calibration** — whether the model's bias probability estimates match observed
  rates (Expected Calibration Error).
- **Reasoning quality** — `<think>` block presence and length for reasoning
  models.
- **Statistical significance** — McNemar's test (binary detection) and Wilcoxon
  signed-rank (ordinal severity) determine whether differences are real.

### Running Evaluations

**Self-test** (no model endpoints needed — validates the scoring pipeline):

```bash
uv run python -m evaluation.selftest
```

**Serve models** (example with SGLang on a DGX Spark):

```bash
# Terminal 1: Qwen3.5-27B
python -m sglang.launch_server --model-path Qwen/Qwen3.5-27B \
    --port 8000 --reasoning-parser qwen3

# Terminal 2: OLMo-3.1-32B (run sequentially if memory-constrained)
python -m sglang.launch_server --model-path allenai/Olmo-3.1-32B-Instruct \
    --port 8001
```

**Zero-shot baseline** (before any fine-tuning):

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-27b --endpoint-a http://localhost:8000 \
    --model-b olmo-3.1-32b --endpoint-b http://localhost:8001 \
    --mode zero-shot \
    --output eval_results/zero_shot/
```

**Post-fine-tuning evaluation** (same test set, LoRA-adapted models):

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-27b-lora --endpoint-a http://localhost:8000 \
    --model-b olmo-3.1-32b-lora --endpoint-b http://localhost:8001 \
    --mode fine-tuned \
    --output eval_results/finetuned/
```

**Re-score saved outputs** (no inference needed):

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --reanalyse eval_results/zero_shot/ \
    --output eval_results/zero_shot_rescored/
```

### Controlled Comparison Protocol

For a rigorous head-to-head, keep these variables identical across models:

- LoRA rank, alpha, target modules, learning rate, batch size, epochs
- Temperature, max tokens, top-p at inference time
- Test set (10% held out, never seen during training)

The only variable that should differ is the base model. Evaluate each bias
dimension separately — one model might excel at relative-vs-absolute detection
while the other is better at spin classification.

The **zero-shot baseline** (before fine-tuning) is particularly informative: it
measures how much each model already knows about bias detection from pretraining
alone. If OLMo outperforms Qwen on bias detection zero-shot despite weaker
general benchmarks, that directly demonstrates the value of Dolma's
academic-heavy training corpus.
