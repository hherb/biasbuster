# BiasBuster User Manual

BiasBuster is a toolkit for building curated training datasets to fine-tune large language models for detecting bias in biomedical research abstracts, and for analysing individual publications. It collects abstracts from multiple sources, enriches them with heuristic analysis, annotates them with LLMs, and exports structured training data. The fine-tuned model learns to assess bias across five domains and suggest specific verification steps citing real databases.

## Package Structure

All source code lives under the `biasbuster/` Python package. The project uses hatchling as the build backend and is installable via `pip install .` or `uv sync`. The `biasbuster` CLI command is registered as a console script entry point.

```bash
# Install
uv sync

# The biasbuster command is now available
biasbuster --help

# All modules are under the biasbuster namespace
uv run python -m biasbuster.pipeline --stage all
uv run python -m biasbuster.gui
uv run python -m biasbuster.evaluation.run --help
```

## Prerequisites

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/) package manager
- **bmlib** -- shared library for biomedical literature tools (publication retrieval, JATS parsing, LLM abstraction). Installed automatically as an editable local dependency.
- **Training hardware** (one of):
  - NVIDIA GPU with Docker (DGX Spark, A100, etc.) -- for full-precision LoRA
  - Apple Silicon Mac with 64-128 GB RAM (M1/M2/M3/M4) -- for QLoRA via MLX
- **Ollama** (for serving fine-tuned models and for the CLI tool with local models)
- **API keys** -- Anthropic and/or DeepSeek (for LLM annotation and CLI analysis); NCBI API key (optional, increases PubMed rate limits)
- **llama.cpp** (for GGUF quantization; clone into project root)

## Workflow Overview

The complete pipeline takes you from raw PubMed abstracts to a fine-tuned, deployed bias detection model, plus a CLI tool for analysing individual publications:

```
Collect ──> Seed ──> Enrich ──> Annotate ──> Human Review ──> Export
                                                                 │
                                    Baseline Eval <── Serve Models │
                                          │                       │
                                    Fine-Tune (LoRA) <────────────┘
                                          │
                                    Merge & Quantize
                                          │
                                    Deploy to Ollama
                                          │
                                    Evaluate Fine-Tuned Model
                                          │
                                    Verification Agent
                                          │
                                    biasbuster CLI
                                    (single-publication analysis)
```

A **Fine-Tuning Workbench** GUI (`uv run python -m biasbuster.gui`) wraps the training, evaluation, and export steps in a browser-based interface with live monitoring.

## Table of Contents

1. [Installation & Configuration](01_setup.md)
2. [Harvesting Training Data](02_data_collection.md)
3. [Heuristic Enrichment](03_enrichment.md)
4. [LLM Annotation](04_annotation.md)
5. [Creating Ground Truth (Human Review)](05_human_review.md)
6. [Exporting Training Data](06_export.md)
7. [Establishing a Baseline](07_baseline_evaluation.md)
8. [Fine-Tuning with LoRA](08_training.md) (NVIDIA GPU / DGX Spark)
8b. [Fine-Tuning with LoRA on Apple Silicon](08b_training_mlx.md) (MLX -- alternative to 8)
9. [Merging, Quantizing & Deploying](09_merge_and_deploy.md)
10. [Evaluating Fine-Tuned Models](10_evaluation.md)
11. [Fine-Tuning Workbench (GUI)](11_workbench_gui.md)
12. [Analysing Publications (CLI)](12_cli.md)

## Quick Start — Analyse a Publication

The fastest way to use BiasBuster is the CLI tool, which analyses a single publication for risk of bias:

```bash
# By PMID
biasbuster 12345678

# By DOI with markdown output
biasbuster 10.1016/j.example.2024.01.001 --format markdown

# With a cloud model
biasbuster 12345678 --model anthropic:claude-sonnet-4-6

# Local JATS XML file
biasbuster ./paper.xml --format markdown

# Full analysis with external verification
biasbuster 12345678 --verify --save --format markdown
```

See [Chapter 12: Analysing Publications (CLI)](12_cli.md) for full details.

## Quick Start — Build a Training Dataset

If you want to run the entire data pipeline in one command:

```bash
uv run python -m biasbuster.pipeline --stage all
```

This runs stages 1-5 (collect, seed, enrich, annotate, export) sequentially. However, you will typically want to run each stage individually and review the results between steps. The manual pages above walk you through each stage in detail.
