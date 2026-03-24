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
│   ├── effect_size_auditor.py # Relative vs absolute reporting analysis
│   └── retraction_classifier.py # Retraction reason → severity floor mapping
├── annotators/
│   ├── __init__.py            # Shared utilities, retraction notice filter
│   ├── llm_prelabel.py        # Anthropic Claude annotator
│   └── openai_compat.py       # OpenAI-compatible annotator (DeepSeek, etc.)
├── schemas/
│   └── bias_taxonomy.py       # Structured bias taxonomy and labels
├── evaluation/
│   ├── harness.py             # Inference runner (OpenAI-compatible APIs)
│   ├── scorer.py              # Output parsing and ground truth attachment
│   ├── metrics.py             # Per-dimension and aggregate metrics
│   ├── comparison.py          # Statistical comparison + report generation (JSON/MD/CSV)
│   ├── run.py                 # CLI entry point (--model-a / --model-b)
│   └── selftest.py            # Self-test with synthetic data
├── training/
│   ├── train_lora.py          # LoRA fine-tuning (SFTTrainer + PEFT, DGX Spark)
│   ├── train_lora_mlx.py      # LoRA/QLoRA fine-tuning (MLX, Apple Silicon)
│   ├── configs.py             # Hyperparameters and model presets (PyTorch)
│   ├── configs_mlx.py         # Hyperparameters and model presets (MLX)
│   ├── callbacks.py           # MetricsLoggerCallback → metrics.jsonl (PyTorch)
│   ├── callbacks_mlx.py       # MLXMetricsLoggerCallback → metrics.jsonl (MLX)
│   ├── data_utils.py          # Alpaca JSONL loading, chat formatting, MLX conversion
│   ├── merge_adapter.py       # Merge LoRA adapter (PyTorch/PEFT)
│   ├── merge_adapter_mlx.py   # Fuse LoRA adapter (MLX)
│   └── export_to_ollama.sh    # Convert merged model to Ollama (safetensors/GGUF)
├── gui/
│   ├── __main__.py             # Entry point: uv run python -m gui
│   ├── app.py                  # Main page with 4-tab layout
│   ├── state.py                # Shared state, platform detection, settings persistence
│   ├── process_runner.py       # Async subprocess wrapper with output streaming
│   ├── settings_tab.py         # Model selector, hyperparameters, data paths
│   ├── training_tab.py         # Training with live loss/LR/GPU charts
│   ├── evaluation_tab.py       # Evaluation runner and results display
│   └── export_tab.py           # Merge adapter, GGUF quantise, Ollama import
├── utils/
│   ├── completeness_checker.py # Check labelling coverage per model
│   ├── agreement_analyzer.py   # Inter-model agreement metrics
│   ├── review_gui.py           # NiceGUI web-based review tool (DB-backed)
│   └── training_monitor.py     # Real-time training dashboard (NiceGUI)
├── database.py                # SQLite backend (single source of truth)
├── prompts.py                 # Single source of truth for annotation/training prompts
├── seed_database.py           # Post-collection cleanup (RW reasons, abstracts, notices)
├── seed_export.py             # Seed data export/import (disaster recovery)
├── backfill_cochrane_domains.py # Backfill per-domain RoB ratings (checkpoint/resume)
├── reprocess_rob.py           # Re-resolve failed Cochrane PMID resolutions
├── annotate_single_paper.py   # Import & annotate one paper by PMID or DOI
├── migrate_jsonl_to_sqlite.py # Idempotent JSONL → SQLite migration script
├── config.py                  # Configuration and API keys
├── pipeline.py                # Orchestration pipeline
├── export.py                  # Export to fine-tuning formats (Alpaca, ShareGPT)
├── run_training.sh            # Launch LoRA training in NGC Docker
├── run_training_mlx.sh        # Launch LoRA training on Apple Silicon (MLX)
├── run_merge.sh               # Launch adapter merge in NGC Docker
├── run_merge_mlx.sh           # Launch adapter merge on Apple Silicon (MLX)
├── MLX_TRAINING.md            # Step-by-step MLX fine-tuning guide
└── TRAINING_INTERPRETATION.md # Guide to interpreting training monitor charts
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

## Retracted Papers Strategy

Retracted papers are handled in two distinct ways:

- **Retraction notices** (bare "This article has been retracted" text with no
  original research content) are **filtered out** before annotation. The
  `is_retraction_notice()` function in `annotators/__init__.py` detects these
  by title/abstract patterns and length heuristics. They have no assessable
  content for bias detection.

- **Original papers that were later retracted** are high-value training examples.
  Their flaws were serious enough to warrant retraction, making them excellent
  ground truth for bias assessment. The retraction watch collector follows the
  Crossref `update-to` relationship from retraction notices back to the original
  paper's DOI and fetches the original abstract from PubMed.

The annotation system prompt (principle 5 in `ANNOTATION_SYSTEM_PROMPT`) instructs
models to assess retracted papers on their actual content rather than
automatically assigning "critical" severity based on retraction status alone.

## Annotation Prompt — Operational Definitions

The annotation system prompt in `annotators/llm_prelabel.py` includes detailed
operational definitions (principles 5–9) that resolve ambiguities causing
inter-model disagreement:

| Principle | Topic | Key Clarification |
|-----------|-------|-------------------|
| 5 | Retraction notices | Bare notices → skip; original content with retraction metadata → assess normally |
| 6 | Absolute vs relative | Raw event rates in both arms (e.g. "84% vs 36%") count as absolute measures |
| 7 | Surrogate vs patient-centred | Process measures (dose modifications, lab values) are surrogates; mortality/QoL/functional status are patient-centred |
| 8 | Methodology thresholds | Domain-specific follow-up adequacy cutoffs (e.g. <12 months for chronic disease = short) |
| 9 | COI disclosure | Funding source alone is insufficient for `coi_disclosed = true`; requires explicit author-level COI statements |

These definitions were added after observing ~55% disagreement between Claude and
DeepSeek on 898 shared annotations, driven primarily by ambiguous handling of
retraction notices, relative/absolute measure classification, and methodology
severity thresholds.

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

# Run full pipeline (collect → seed → enrich → annotate → export)
uv run python pipeline.py --stage all

# Or run individual stages
uv run python pipeline.py --stage collect       # retraction watch + PubMed RCTs + Cochrane RoB
uv run python pipeline.py --stage collect-rob   # Cochrane RoB only (+ fetch abstracts)
uv run python pipeline.py --stage seed          # enrich RW reasons + fetch abstracts + clean notices
uv run python pipeline.py --stage enrich
uv run python pipeline.py --stage annotate
uv run python pipeline.py --stage annotate --models anthropic,deepseek  # multi-model
uv run python pipeline.py --stage export
uv run python pipeline.py --stage compare       # compare models vs human labels
```

### Single-Paper Import & Annotation

For ad-hoc additions to the dataset (e.g. a paper you stumbled upon, or retrying
a failed annotation), use the standalone single-paper tool:

```bash
# Import (if needed) and annotate by PMID — uses DeepSeek by default
uv run python annotate_single_paper.py --pmid 41271640

# Use Anthropic Claude instead
uv run python annotate_single_paper.py --pmid 41271640 --model anthropic

# Import by DOI — resolves to PMID via NCBI, then fetches from PubMed
uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001

# Tag the source for newly imported papers (default: manual_import)
uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001 --source cochrane_rob

# Re-annotate a paper that already has an annotation
uv run python annotate_single_paper.py --pmid 41271640 --force
```

The script performs the full pipeline for a single paper:

1. **Resolve** — If given `--doi`, converts to PMID via the NCBI ID Converter API
2. **Fetch** — If the paper is not already in the database, fetches it from PubMed and stores it
3. **Validate** — Rejects papers with no abstract or bare retraction notices
4. **Enrich** — Runs the effect-size audit heuristic and stores the suspicion level
5. **Annotate** — Sends to the chosen LLM backend and stores the result

If the paper already exists in the database, steps 1-2 are skipped. If an
annotation already exists for that model, the script skips unless `--force` is
given (which deletes the old annotation first).

### Data Storage

All pipeline data is stored in a single SQLite database (`dataset/biasbuster.db`
by default). The schema has four tables:

| Table | Purpose | Key |
|-------|---------|-----|
| `papers` | All collected papers (retracted, RCT, Cochrane) with RoB domain ratings, review metadata | `pmid` |
| `enrichments` | Heuristic analysis results (effect size audit, outcome switching) | `pmid` |
| `annotations` | LLM bias assessments (one row per paper per model) | `(pmid, model_name)` |
| `human_reviews` | Human validation decisions | `(pmid, model_name)` |
| `eval_outputs` | Evaluation harness results (per model+mode) | `(pmid, model_id, mode)` |

Cochrane RoB papers use `upsert_cochrane_paper()` which preserves PubMed-fetched
titles/abstracts while updating domain ratings and review metadata on re-runs.

Legacy JSONL data can be imported with the migration script:

```bash
uv run python migrate_jsonl_to_sqlite.py
uv run python migrate_jsonl_to_sqlite.py --data-dir dataset --db-path dataset/biasbuster.db
```

### Pipeline Flow

```mermaid
flowchart TD
    subgraph Collect
        C1[Crossref / Retraction Watch]
        C2[PubMed RCTs by MeSH]
        C3[Cochrane RoB via Europe PMC]
    end

    DB[(SQLite DB<br/>biasbuster.db)]

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

    HR[Human Review<br/>NiceGUI tool]

    subgraph Export
        X1[Alpaca + think chains]
        X2[ShareGPT]
        X3[OpenAI chat]
    end

    subgraph Train
        T1[LoRA Fine-tuning<br/>SFTTrainer + PEFT]
        T2[Merge Adapter]
        T3[Export to Ollama]
    end

    TM[Training Monitor<br/>NiceGUI dashboard]

    subgraph Compare
        M1[Per-dimension F1 & kappa]
        M2[McNemar / Wilcoxon tests]
        M3[Markdown report]
    end

    SD[Seed<br/>RW reasons + abstracts<br/>+ notice cleanup]

    Collect --> DB
    DB --> SD --> DB
    DB --> Enrich --> DB
    DB --> Annotate --> DB
    DB --> HR --> DB
    DB --> Export
    Export --> T1 --> T2 --> T3
    T1 -.->|metrics.jsonl| TM
    DB --> Compare
```

Human review (using the NiceGUI web tool) is a manual step between annotate and export.

## Dataset Utilities

Three tools support the human-review workflow between annotation and export.
All read from and write to the SQLite database.

### Completeness Checker

Reports annotation coverage — how many available abstracts each model has
annotated, with progress against configured limits.

```bash
# Show completion against configured annotation limits
uv run python -m utils.completeness_checker

# Show progress against full available set (ignore config caps)
uv run python -m utils.completeness_checker --no-limits

# Check specific models only
uv run python -m utils.completeness_checker --models anthropic,deepseek

# Use a specific database
uv run python -m utils.completeness_checker --db-path dataset/biasbuster.db
```

Example output (default: shows progress vs configured caps):

```
Source                    |            anthropic |             deepseek
-----------------------------------------------------------------------
cochrane_rob              |     6/6     (100.0%) |     6/6     (100.0%)
high_suspicion            |   393/394   ( 99.7%) |   394/394   (100.0%)
low_suspicion             |   300/300   (100.0%) |   300/300   (100.0%)  [of 2733 available]
retracted_papers          |   199/200   ( 99.5%) |   200/200   (100.0%)  [of 514 available]
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

# Use a specific database
uv run python -m utils.agreement_analyzer --db-path dataset/biasbuster.db
```

Reports overall severity kappa, per-dimension exact/within-one agreement,
flag-level agreement rates (e.g. `relative_only`, `spin_level`, `funding_type`),
probability MAE and Pearson r, and the top most-divergent cases ranked by
bias probability difference.

### Review GUI

Web-based tool for reviewing and validating annotations. Reads from and saves
directly to the SQLite database. Opens in your browser with an editable AG Grid
table. Built with NiceGUI.

```bash
# Review a specific model's annotations
uv run python -m utils.review_gui --model anthropic

# Browse available models interactively
uv run python -m utils.review_gui

# Use a different port or database
uv run python -m utils.review_gui --model deepseek --port 9090 --db-path dataset/biasbuster.db
```

Features:
- **In-grid editing** of `HUMAN_VALIDATED`, `HUMAN_OVERRIDE_SEVERITY`, and
  `HUMAN_NOTES` columns (double-click to edit)
- **Color-coded rows**: green for validated, yellow for overridden severity
- **Quick filter** and column sorting
- **"Next Unvalidated"** button to jump to the next unreviewed row
- **Detail panel** showing full reasoning text on row selection
- **Export CSV** button for offline review in spreadsheets
- **Direct DB save** — changes are written to the `human_reviews` table
- **Stats bar** showing validation progress

## Fine-Tuning Workbench (GUI)

A NiceGUI-based web application that provides a graphical interface for the
full fine-tuning workflow — from model selection through training, evaluation,
and export. Designed for users who prefer not to use the CLI directly.

```bash
# Launch the workbench (opens at http://localhost:8080)
uv run python -m gui

# Use a different port
uv run python -m gui --port 9090
```

The workbench has four tabs:

### Settings Tab

- **Model selector** — dropdown populated from available presets, auto-detected
  by platform (TRL presets on Linux, MLX presets on macOS, all on Windows)
- **Hyperparameters** — learning rate, epochs, LoRA rank, batch size, gradient
  accumulation, max sequence length. Selecting a model auto-populates with
  preset defaults; changes are passed through to the training scripts
- **Data paths** — training/validation/test JSONL file paths
- **Evaluation endpoints** — model names and OpenAI-compatible endpoint URLs
- Settings persist to `~/.biasbuster/gui_settings.json` across sessions

### Evaluation Tab

- **Run Baseline** (zero-shot) or **Run Fine-Tuned** evaluations
- **Re-analyse** previously saved results without re-running inference
- Streams subprocess log output in real time
- Displays per-dimension metrics tables (F1, Precision, Recall, Kappa)
- Shows comparison reports if two models are evaluated

### Fine-Tuning Tab

- **Start/Stop** training with the selected model and hyperparameters
- Live charts (same as the standalone training monitor):
  - Training & eval loss curves
  - Learning rate schedule
  - GPU memory usage (shown automatically when data is available)
  - Gradient norm
- Progress bar with step count, epoch, and ETA
- Hyperparameters table populated from the training run
- Raw process log in a collapsible panel

### Export Tab

- **Merge Adapter** — merges the LoRA adapter into the base model
- **GGUF Export** — converts the merged model to GGUF with selectable
  quantisation (Q4_K_M, Q5_K_M, q8_0, f16, bf16)
- **Ollama Import** — imports the merged model directly into Ollama
- Each operation runs independently with its own progress log

### Cross-Platform Support

| Platform | Training | Evaluation | Export |
|----------|----------|------------|--------|
| Linux (DGX Spark) | TRL/PEFT via `uv run` | Yes | Yes |
| macOS (Apple Silicon) | MLX via `uv run` | Yes | Yes |
| Windows | Not supported | Yes | Limited |

## Fine-Tuning (LoRA)

The `training/` module fine-tunes candidate base models using LoRA adapters.
Training runs inside the NGC PyTorch container on DGX Spark (the only source of
aarch64 + CUDA PyTorch). Both models use identical LoRA hyperparameters for a
controlled comparison — only the base model differs.

### Quick Start

```bash
# 1. Train (each takes ~2-4 hours on DGX Spark)
./run_training.sh qwen3.5-27b
./run_training.sh olmo-3.1-32b

# 2. Merge LoRA adapter into base model
./run_merge.sh qwen3.5-27b
./run_merge.sh olmo-3.1-32b

# 3. Export to Ollama (safetensors — full precision)
bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster
bash training/export_to_ollama.sh training_output/olmo-3.1-32b-merged olmo-3.1-32b-biasbuster

# 3b. Or export as quantized GGUF (smaller, faster inference)
bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster-q4 --gguf Q4_K_M

# 4. Evaluate fine-tuned models
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-27b-biasbuster --endpoint-a http://localhost:11434 \
    --model-b olmo-3.1-32b-biasbuster --endpoint-b http://localhost:11434 \
    --mode fine-tuned --sequential --num-ctx 4096 \
    --output eval_results/fine_tuned/
```

### Smoke Test

Run 5 training steps to verify the setup without waiting hours:

```bash
./run_training.sh qwen3.5-27b --max-steps 5
```

### Checkpoint/Resume

Training saves checkpoints every 50 steps. To resume after interruption:

```bash
./run_training.sh qwen3.5-27b --resume
```

### Training Monitor

A real-time NiceGUI dashboard that visualises training progress by reading the
`metrics.jsonl` file written by `MetricsLoggerCallback`. Run it on the host
while training runs in the NGC Docker container (the training output directory
is volume-mounted).

```bash
# Auto-detect latest metrics.jsonl under training_output/
uv run python -m utils.training_monitor

# Point to a specific run
uv run python -m utils.training_monitor --metrics-dir training_output/olmo-3.1-32b-lora

# Custom port and refresh interval
uv run python -m utils.training_monitor --port 8081 --refresh 5
```

The dashboard shows:
- **Training & eval loss** curves (train loss as line, eval loss as scatter points)
- **Learning rate schedule** (warmup + cosine decay)
- **GPU memory** (allocated vs peak)
- **Gradient norm** over time
- **Hyperparameters table** (all LoRA and training config)
- **Progress bar** with step count, epoch, and ETA

See [TRAINING_INTERPRETATION.md](TRAINING_INTERPRETATION.md) for a detailed
guide to interpreting every metric and parameter on the dashboard.

### LoRA Configuration (Controlled Comparison)

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q,k,v,o_proj + gate,up,down_proj |
| Learning rate | 2e-4 (cosine schedule) |
| Effective batch size | 4 (1 × 4 grad accum) |
| Epochs | 3 |
| Precision | bf16 (no quantization) |
| Max sequence length | 4096 |

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
