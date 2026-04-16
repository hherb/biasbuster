# BiasBuster

**A risk-of-bias analysis pipeline for biomedical randomised trials.**
Reads a paper's full text (or abstract if full text isn't available),
extracts structured facts with one LLM call, assesses bias across
5 domains with a second LLM call, and produces a JSON or Markdown
report with specific verification steps pointing at external databases
(ClinicalTrials.gov, CMS Open Payments, ORCID, etc.). Ships with a CLI
tool, the analysis library, a dataset-building pipeline, optional
LoRA fine-tuning infrastructure, and a GUI workbench. Designed for use
with BMLibrarian.

**BiasBuster assesses *risk of bias*, not *proof of bias*** — and is
intentionally more aggressive than Cochrane RoB 2 on the Conflict of
Interest domain, which Cochrane deliberately excludes. See
[COI Design Rationale](docs/two_step_approach/DESIGN_RATIONALE_COI.md)
for the full justification.

## Project Status (April 2026)

**The V5A decomposed pipeline is the current recommended architecture
for local models.** gemma4-26B running through V5A achieves perfect
weighted-kappa agreement (κ = 1.000) with Cochrane RoB 2 expert labels
on the two directly-comparable domains (methodology, outcome reporting)
across a 15-paper validation cohort.

### Architecture evolution

| Version | Architecture | Status |
|---------|-------------|--------|
| **v3** — two-call | LLM extraction → LLM assessment (single prompt) | Production-grade for abstract-only; ceiling on multi-paper calibration due to LLM arithmetic errors |
| **v4** — agentic | LLM extraction → LLM agent with `run_mechanical_assessment` tool | Works for Claude; **fails for small local models** (gemma4/gpt-oss rubber-stamp the tool output without contextual review) |
| **v5A** — decomposed | LLM extraction → Python mechanical rules → **per-domain focused LLM override calls** → Python synthesis | **Recommended for local models.** Closes the small-model gap by giving each LLM call a single narrow task (3-field JSON output) instead of the v4 agent's "review 5 domains and emit 30-field JSON" |

### V5A validation results (16 papers, 15 Cochrane)

| Metric | gemma4-26B | gpt-oss-20B |
|--------|------------|-------------|
| Overall severity κ vs Sonnet | +0.429 (pass ≥ 0.30) | +0.158 (fail) |
| Methodology κ vs Cochrane experts | **1.000** | 0.000 |
| Outcome-reporting κ vs Cochrane experts | **1.000** | **1.000** |
| COI κ vs Sonnet | 0.868 | 0.598 |
| Overall κ vs Cochrane (raw) | +0.027 | +0.118 |

The raw overall κ vs Cochrane is low for all models by design —
biasbuster's COI policy extension (see
[DESIGN_RATIONALE_COI.md](docs/two_step_approach/DESIGN_RATIONALE_COI.md))
rates industry-funded trials HIGH where Cochrane rates them LOW because
Cochrane RoB 2 does not assess COI. The per-domain methodology and
outcome-reporting kappas are the clean signal: both Sonnet and gemma4
track the experts perfectly on the domains Cochrane actually assesses.

Disagreement analysis on 80 domain-level comparisons showed **86% of
gemma4–Sonnet disagreements trace to Stage 1 extraction quality**
(gemma4 missing facts that Sonnet catches), not Stage 3 override
judgment. Zero calibration drift was observed — when both models
extract the same facts, they produce identical override decisions.

**Recommendation: gemma4-26B + V5A** for fully-local deployment.
gpt-oss-20B is not recommended (fails methodology agreement with
Cochrane experts). Fine-tuning is not needed.

Full empirical details:
[docs/three_step_approach/V5A_RESULTS.md](docs/three_step_approach/V5A_RESULTS.md),
[docs/three_step_approach/OVERVIEW.md](docs/three_step_approach/OVERVIEW.md).
Earlier architecture history:
[docs/two_step_approach/INITIAL_FINDINGS_V3.md](docs/two_step_approach/INITIAL_FINDINGS_V3.md),
[docs/two_step_approach/V4_AGENT_DESIGN.md](docs/two_step_approach/V4_AGENT_DESIGN.md).

## Quick Start — Analyse a Publication

```bash
# Install
uv sync

# Pull the recommended local model (once, ~18 GB on disk)
ollama pull gemma4:26b-a4b-it-q8_0

# Analyse by PMID with the recommended local model
biasbuster 12345678 --model ollama:gemma4:26b-a4b-it-q8_0

# Analyse by DOI with markdown report
biasbuster 10.1016/j.example.2024.01.001 \
    --model ollama:gemma4:26b-a4b-it-q8_0 \
    --format markdown

# Cloud fallback when you want the strongest available model
biasbuster 12345678 --model anthropic:claude-sonnet-4-6

# Analyse a local PDF or JATS XML file
biasbuster ./paper.pdf --format markdown --model ollama:gemma4:26b-a4b-it-q8_0
biasbuster ./paper.xml --model deepseek:deepseek-reasoner

# Save markdown report to a file
biasbuster 12345678 --model ollama:gemma4:26b-a4b-it-q8_0 --format markdown -o report.md

# Full analysis with external verification (ClinicalTrials.gov, CMS, ORCID, etc.)
biasbuster 12345678 --model ollama:gemma4:26b-a4b-it-q8_0 --verify --format markdown
```

See [docs/BIASBUSTER_CLI.md](docs/BIASBUSTER_CLI.md) for full CLI
documentation.

> **On the default model**: for historical reasons the CLI's built-in
> `DEFAULT_MODEL` still points at a legacy fine-tuned alias
> (`ollama:qwen3.5-9b-biasbuster`) produced by this project's earlier
> training pipeline. If you haven't fine-tuned that model yourself,
> pass `--model ollama:gemma4:26b-a4b-it-q8_0` (or any other model
> of your choice) explicitly, or set `model.default` in
> `~/.biasbuster/config.toml`.

## Package Structure

All source code lives under the `biasbuster/` Python package, installable via
`pip install .` or `uv sync`. The package is PyPI-ready with hatchling as the
build backend.

```
biasbuster/                        # Top-level Python package
├── cli/                           # CLI tool for single-publication analysis
│   ├── main.py                    # Entry point (biasbuster command)
│   ├── settings.py                # TOML config loading with env/CLI overrides
│   ├── content.py                 # Identifier resolution, content acquisition via bmlib
│   ├── chunking.py                # Section-based (JATS) + token-window chunking
│   ├── analysis.py                # Single-pass and map-reduce LLM assessment
│   ├── verification.py            # Wrapper around agent verification pipeline
│   ├── formatting.py              # JSON and Markdown output formatters
│   └── pdf_extract.py             # PDF text extraction via pdfplumber
├── agent/                         # Verification agent (tool-augmented assessment)
│   ├── runner.py                  # Agent loop: assess → verify → refine
│   ├── model_client.py            # Ollama API client with retry logic
│   ├── agent_config.py            # Agent configuration dataclass
│   ├── verification_planner.py    # Deterministic verification step synthesis
│   ├── tool_router.py             # Regex-based step → tool call routing
│   └── tools.py                   # Tool wrappers (ClinicalTrials.gov, CMS, ORCID, etc.)
├── collectors/                    # Data source collectors (async)
│   ├── retraction_watch.py        # Retracted papers via Crossref API + PubMed
│   ├── cochrane_rob.py            # Cochrane Risk of Bias assessments
│   ├── pubmed_xml.py              # PubMed XML parsing utilities
│   ├── spin_detector.py           # Heuristic pre-screening for spin indicators
│   └── clinicaltrials_gov.py      # Outcome switching detection via registry
├── enrichers/                     # Heuristic analysis modules
│   ├── effect_size_auditor.py     # Relative vs absolute reporting analysis
│   ├── funding_checker.py         # Funding source classification
│   ├── author_coi.py              # Author conflict-of-interest verification
│   └── retraction_classifier.py   # Retraction reason → severity floor mapping
├── annotators/                    # LLM annotation backends
│   ├── __init__.py                # Shared utilities, JSON repair, retraction filter
│   ├── llm_prelabel.py            # Anthropic Claude annotator
│   └── openai_compat.py           # OpenAI-compatible annotator (DeepSeek, etc.)
├── schemas/
│   └── bias_taxonomy.py           # Structured bias taxonomy and labels
├── evaluation/                    # Model evaluation harness
│   ├── harness.py                 # Inference runner (OpenAI-compatible APIs)
│   ├── scorer.py                  # Output parsing and ground truth attachment
│   ├── metrics.py                 # Per-dimension and aggregate metrics
│   ├── comparison.py              # Statistical comparison + report generation
│   ├── run.py                     # CLI entry point (--model-a / --model-b)
│   └── selftest.py                # Self-test with synthetic data
├── training/                      # LoRA fine-tuning pipeline
│   ├── train_lora.py              # SFTTrainer + PEFT (DGX Spark)
│   ├── train_lora_mlx.py          # MLX LoRA/QLoRA (Apple Silicon)
│   ├── configs.py / configs_mlx.py
│   ├── callbacks.py / callbacks_mlx.py
│   ├── data_utils.py              # Alpaca JSONL loading, chat formatting
│   ├── merge_adapter.py / merge_adapter_mlx.py
│   └── export_to_ollama.sh
├── gui/                           # Fine-Tuning Workbench (NiceGUI)
│   ├── __main__.py                # Entry point: uv run python -m biasbuster.gui
│   ├── app.py                     # 4-tab layout (settings, training, eval, export)
│   ├── state.py                   # Platform detection, settings persistence
│   └── *.py                       # Tab modules
├── utils/                         # Utilities and review tools
│   ├── review_gui.py              # NiceGUI web-based review tool (DB-backed)
│   ├── training_monitor.py        # Real-time training dashboard
│   ├── completeness_checker.py    # Annotation coverage checker
│   └── agreement_analyzer.py      # Inter-model agreement metrics
├── crowd/                         # Crowdsourced human annotation platform
├── database.py                    # SQLite backend (single source of truth)
├── prompts.py                     # Canonical annotation/training prompts
├── pipeline.py                    # Orchestration pipeline (collect→export)
└── export.py                      # Export to fine-tuning formats
```

### Key Dependencies

| Dependency | Purpose |
|------------|---------|
| [bmlib](https://github.com/hherb/bmlib) | Publication retrieval, JATS parsing, LLM abstraction (multi-provider) |
| httpx | Async HTTP client for external APIs |
| anthropic | Anthropic Claude SDK (annotation) |
| pdfplumber | PDF text extraction (CLI tool) |
| nicegui | Web UI for review tool and training workbench |
| tokenizers | Token counting |

### Installation

```bash
# Clone and set up
git clone https://github.com/hherb/biasbuster.git
cd biasbuster
uv sync

# Configure (edit config.py with API keys)
cp config.example.py config.py

# The biasbuster CLI command is now available
biasbuster --help
```

## Bias Taxonomy

The pipeline assesses papers along five independent domains, each
rated on a `none / low / moderate / high / critical` severity scale
with its own per-flag evidence quotes. The overall severity is the
maximum of the domain severities; the overall bias probability is a
separately-calibrated numeric value that can sit below the
categorical rating to express "structural risk present but
methodology otherwise acceptable" (see
[DESIGN_RATIONALE_COI.md](docs/two_step_approach/DESIGN_RATIONALE_COI.md)
for how the category+probability combination is used).

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
   - Structural COI: sponsor-employed/shareholder authors on
     industry-funded trials trigger a hard-HIGH rating regardless
     of methodology quality — see
     [docs/two_step_approach/DESIGN_RATIONALE_COI.md](docs/two_step_approach/DESIGN_RATIONALE_COI.md)
     for the justification. **BiasBuster assesses *risk of bias*,
     not *proof of bias*** — and is intentionally more aggressive
     than Cochrane RoB 2 on this domain, which Cochrane
     deliberately excludes from its methodology-focused assessment.

5. **Methodological Red Flags**
   - Inappropriate comparator (placebo when active exists)
   - Enrichment design without acknowledgment
   - Per-protocol only (no ITT)
   - Premature stopping

## BiasBuster CLI

The `biasbuster` command analyses individual publications for risk of bias.
It accepts PMIDs, DOIs, or local files and produces structured JSON or Markdown
reports.

### How It Works

The **V5A decomposed pipeline** is the recommended architecture for
local models. Cloud models (Claude) can also use V5A, or the v4
agentic loop which works well for large models. Each paper goes
through these stages:

1. **Content acquisition** — fetches full text via bmlib's 3-tier
   fallback chain (Europe PMC JATS XML → Unpaywall PDF →
   abstract-only fallback). JATS is preferred because it gives
   semantic section boundaries.
2. **JATS back-matter extraction** — extracts funding, COI
   disclosures, and acknowledgments from `<back>` elements that
   standard body parsers miss. This is the channel by which the
   pipeline sees sponsor-employed authors.
3. **Stage 1 — Extraction**: for each section (full-text mode) or
   for the abstract (abstract mode), the model extracts structured
   facts — sample size, attrition, analysis population, outcomes
   with types and p-values, funding source, author affiliations,
   etc. No bias judgement at this stage, only facts. Full-text
   mode merges per-section extractions into a single structured
   object.
4. **Stage 2 — Mechanical assessment** (V5A): the merged extraction
   is fed to a pure-Python rule engine (`biasbuster/assessment/`)
   that applies deterministic severity thresholds and produces a
   draft per-domain assessment with full provenance.
5. **Stage 3 — Per-domain override review** (V5A): for each domain
   the mechanical rules flagged as moderate or higher *and*
   overridable, one focused LLM call asks "does this rule genuinely
   apply to THIS paper?" with a 3-field JSON output. Calls run in
   parallel. Non-overridable domains (structural COI triggers)
   skip the LLM entirely.
6. **Stage 4 — Synthesis**: Python assembles the final assessment
   from the mechanical draft + per-domain decisions, runs hard-rule
   enforcement, and optionally generates a reasoning summary via
   one small LLM call.
5. **Verification** (optional, `--verify`) — cross-checks against
   ClinicalTrials.gov, CMS Open Payments, ORCID, Europe PMC,
   Retraction Watch.
6. **Output** — JSON (default) or Markdown report.

The split matters because extraction accuracy is measurable against
ground truth (numbers either match the paper or don't), assessment
logic can be improved without re-running extraction, and smaller
models can reliably do one narrow task per call rather than the
entire combined problem.

Legacy v1 single-call paths (`--single-call`, for both abstract and
full text) are still present as fallbacks but are **strongly
discouraged** — the full-text single-call path collapses to ~50%
agreement with Claude across all tested model families and
produces generic "no major red flags" moderate verdicts
regardless of the input. See
[INITIAL_FINDINGS_V3.md §4.1](docs/two_step_approach/INITIAL_FINDINGS_V3.md)
for empirical details.

### Model Selection

Models use bmlib's `provider:model_name` format:

```bash
# Recommended: gemma4 26B (local, best validated match to Claude on this task)
biasbuster 12345678 --model ollama:gemma4:26b-a4b-it-q8_0

# Alternatives — also validated against Claude on the motivating failure case
biasbuster 12345678 --model ollama:gpt-oss:20b
biasbuster 12345678 --model ollama:gpt-oss:120b

# Cloud fallback for papers where local-model quality matters most
biasbuster 12345678 --model anthropic:claude-sonnet-4-6
biasbuster 12345678 --model deepseek:deepseek-reasoner

# Bare model names default to Ollama (auto-detected)
biasbuster 12345678 --model gpt-oss:20b
```

Known provider prefixes: anthropic, ollama, openai, deepseek, mistral,
gemini. Bare model names without a prefix default to Ollama.

**Which model should I use?** Based on the V5A 16-paper validation
against Cochrane RoB 2 expert labels (see
[V5A_RESULTS.md](docs/three_step_approach/V5A_RESULTS.md)),
**gemma4 26B** is the recommended local model. It achieves perfect
weighted-kappa agreement (κ = 1.000) with Cochrane experts on the
methodology and outcome-reporting domains. gpt-oss 20B is not
recommended (fails on methodology accuracy vs Cochrane).

Fine-tuning is not needed. The V5A decomposed pipeline with gemma4
26B achieves expert-level accuracy on shared Cochrane domains
without any fine-tuning. The remaining gap vs Sonnet traces to
Stage 1 extraction quality, not assessment judgment.

### Download Caching

Downloaded abstracts and full-text files are cached in `~/.biasbuster/downloads/`
so repeated analyses of the same paper skip network calls. Use `--force-download`
to re-fetch after a correction or retraction.

### Configuration

Settings are read from `~/.biasbuster/config.toml` with environment variable and
CLI flag overrides. See [docs/BIASBUSTER_CLI.md](docs/BIASBUSTER_CLI.md) for the
full config reference.

## Dataset Building Pipeline (optional)

You only need this section if you want to build training data for
fine-tuning, generate a curated bias-assessment corpus for research,
or compare multiple annotators head-to-head against a ground-truth
set. For ad-hoc analysis of individual papers, use the CLI above.

```bash
# Run full pipeline (collect → seed → enrich → annotate → export)
uv run python -m biasbuster.pipeline --stage all

# Or run individual stages
uv run python -m biasbuster.pipeline --stage collect
uv run python -m biasbuster.pipeline --stage seed
uv run python -m biasbuster.pipeline --stage enrich
uv run python -m biasbuster.pipeline --stage annotate
uv run python -m biasbuster.pipeline --stage annotate --models anthropic,deepseek
uv run python -m biasbuster.pipeline --stage export
uv run python -m biasbuster.pipeline --stage compare
```

### Single-Paper Import & Annotation

```bash
uv run python -m biasbuster.annotate_single_paper --pmid 41271640
uv run python -m biasbuster.annotate_single_paper --pmid 41271640 --model anthropic
uv run python -m biasbuster.annotate_single_paper --doi 10.1016/j.example.2024.01.001
uv run python -m biasbuster.annotate_single_paper --pmid 41271640 --force
```

### Data Storage

All pipeline data is stored in a single SQLite database (`dataset/biasbuster.db`).

| Table | Purpose | Key |
|-------|---------|-----|
| `papers` | Collected papers with RoB domain ratings, review metadata | `pmid` |
| `enrichments` | Heuristic analysis results (effect size audit, outcome switching) | `pmid` |
| `annotations` | LLM bias assessments (one row per paper per model) | `(pmid, model_name)` |
| `human_reviews` | Human validation decisions | `(pmid, model_name)` |
| `eval_outputs` | Evaluation harness results | `(pmid, model_id, mode)` |

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

    CLI[biasbuster CLI<br/>Single-paper analysis]

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
    T3 --> CLI
```

## Retracted Papers Strategy

- **Retraction notices** (bare "This article has been retracted" text) are
  **filtered out** by `is_retraction_notice()`. They have no assessable content.
- **Original papers that were later retracted** are high-value training examples.
  The collector follows the Crossref `update-to` relationship back to the
  original DOI and fetches the original abstract from PubMed.

## Verification Sources

The model learns WHERE to look for corroboration:

- **CMS Open Payments** (openpaymentsdata.cms.gov) — US physician payments
- **ClinicalTrials.gov** — Registered outcomes vs published outcomes
- **ORCID** — Author affiliation history
- **Europe PMC** — Funder metadata, full-text COI sections
- **Crossref / Retraction Watch** — Retraction status
- **Cochrane RoB database** — Expert risk assessments

## Dataset Utilities

### Completeness Checker

```bash
uv run python -m biasbuster.utils.completeness_checker
uv run python -m biasbuster.utils.completeness_checker --no-limits
```

### Inter-Model Agreement Analyzer

```bash
uv run python -m biasbuster.utils.agreement_analyzer
uv run python -m biasbuster.utils.agreement_analyzer --model-a anthropic --model-b deepseek
```

### Review GUI

```bash
uv run python -m biasbuster.utils.review_gui --model anthropic
```

## Fine-Tuning (optional)

> **Do you need this?** Almost certainly not. The V5A decomposed
> pipeline with gemma4-26B achieves expert-level accuracy on shared
> Cochrane RoB 2 domains **without any fine-tuning** (see
> [V5A Results](docs/three_step_approach/V5A_RESULTS.md)). The V5A
> pipeline moves bias-assessment logic to deterministic Python rules,
> leaving the LLM only the extraction and narrow override tasks —
> which gemma4-26B handles reliably out of the box.
>
> Reasons you might still want to fine-tune:
> - **Smaller deployment**: a fine-tuned 9B model uses less VRAM
>   (~8 GB vs 24+ GB for 26B), which matters on constrained hardware.
> - **Specialised domains**: your papers come from a subfield where
>   you want domain-specific extraction exemplars.
>
> If those don't apply, just use
> `biasbuster ... --model ollama:gemma4:26b-a4b-it-q8_0` and skip
> this section.

### Fine-Tuning Workbench (GUI)

```bash
uv run python -m biasbuster.gui
uv run python -m biasbuster.gui --port 9090
```

4-tab NiceGUI application: Settings → Training (live charts) → Evaluation → Export.

### Fine-Tuning (LoRA)

```bash
# Train (DGX Spark)
./run_training.sh qwen3.5-27b

# Train (Apple Silicon)
./run_training_mlx.sh qwen3.5-27b-4bit

# One-command train → merge → Ollama → evaluate
./train_and_evaluate.sh gpt-oss-20b
```

### Training Monitor

```bash
uv run python -m biasbuster.utils.training_monitor
```

## Evaluation Harness

Use this for head-to-head comparison of annotators against a
ground-truth test set — relevant mostly when fine-tuning or
comparing multiple base models. Not needed for ad-hoc analysis.

```bash
# Self-test
uv run python -m biasbuster.evaluation.selftest

# Compare two models head-to-head against a labelled test set
uv run python -m biasbuster.evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gemma4:26b-a4b-it-q8_0 --endpoint-a http://localhost:11434 \
    --model-b gpt-oss:20b --endpoint-b http://localhost:11434 \
    --sequential --num-ctx 4096 \
    --output eval_results/v3_comparison/
```

For reliability-testing a single model × paper combination (N runs
of the same full-text two-call pipeline on the same paper), see
`scripts/reliability_test_fulltext.py`. For the calibration matrix
(multiple papers × multiple modes × multiple models), see
`scripts/run_calibration_test.sh`.

## Documentation

- [CLI Reference](docs/BIASBUSTER_CLI.md) — full `biasbuster` command documentation
- [User Manual](docs/manual/index.md) — step-by-step guide through the entire pipeline
- [Training Guide](docs/TRAINING.md) — LoRA fine-tuning details
- [MLX Training](docs/MLX_TRAINING.md) — Apple Silicon training guide
- [Model Card](docs/MODEL_CARD.md) — fine-tuned model documentation
- [COI Design Rationale](docs/two_step_approach/DESIGN_RATIONALE_COI.md) — why BiasBuster's COI domain is intentionally more aggressive than Cochrane RoB 2 (*risk of bias*, not *proof of bias*)
- [V5A Decomposed Pipeline — Results](docs/three_step_approach/V5A_RESULTS.md) — 16-paper validation of V5A: gemma4 κ=1.000 vs Cochrane experts on methodology/outcome-reporting, disagreement diagnostic showing 86% of remaining gaps are extraction quality not judgment
- [V5A Decomposed Pipeline — Design](docs/three_step_approach/V5A_DECOMPOSED.md) — architecture, per-domain override prompts, reused code, verification plan
- [V5A/V5B Overview](docs/three_step_approach/OVERVIEW.md) — problem statement, decision tree, success criteria
- [v4 Tool-Calling Agent Design](docs/two_step_approach/V4_AGENT_DESIGN.md) — v4 agentic architecture (works for Claude, not for small local models — superseded by V5A for local deployment)
- [v3 Two-Call Findings](docs/two_step_approach/INITIAL_FINDINGS_V3.md) — empirical history of the v3 architecture: 10 rounds of prompt iteration, 3-family verification, calibration test
