# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BMLibrarian Bias Detection Dataset Builder — a toolkit for building curated training datasets to fine-tune LLMs for detecting bias in biomedical abstracts. The fine-tuned model learns to assess bias across 5 domains and suggest specific verification steps citing real databases.

## Commands

```bash
# Set up environment and install dependencies
uv sync

# Configure (edit config.py with API keys)
cp config.example.py config.py

# Run full pipeline
uv run python pipeline.py --stage all

# Run individual stages
uv run python pipeline.py --stage collect
uv run python pipeline.py --stage enrich
uv run python pipeline.py --stage annotate
uv run python pipeline.py --stage export

# Individual module demos (each has __main__ block)
uv run python -m collectors.spin_detector
uv run python -m enrichers.effect_size_auditor
uv run python -m enrichers.funding_checker

# Add a new dependency
uv add <package>
```

There is no formal test suite — modules have `if __name__ == "__main__":` demo blocks that serve as smoke tests.

## Architecture

### Pipeline Stages (pipeline.py)

4-stage async workflow orchestrated by `pipeline.py`:

1. **Collect** — Fetch abstracts from external APIs (Crossref/Retraction Watch, PubMed RCTs by MeSH domain, Cochrane RoB assessments via Europe PMC)
2. **Enrich** — Run heuristic analysis (effect size auditing, outcome switching via ClinicalTrials.gov) to bucket abstracts into high/low suspicion
3. **Annotate** — Send abstracts to Claude API (via `anthropic` SDK) for structured 5-domain bias assessment; output JSONL + CSV for human review
4. **Export** — Convert human-validated annotations to fine-tuning formats (alpaca with `<think>` chains, sharegpt, openai_chat) with 80/10/10 train/val/test splits

Human review (editing the CSV) is a manual step between Annotate and Export.

### Module Pattern

- **Collectors** (`collectors/`): Async classes using `httpx.AsyncClient` with rate limiting. Each fetches from a specific source (Crossref, PubMed, ClinicalTrials.gov, Europe PMC). Return typed dataclasses. PubMed XML parsing functions (`parse_pubmed_xml`, `parse_pubmed_xml_batch`) are standalone module-level functions in `retraction_watch.py`.
- **Enrichers** (`enrichers/`): Mostly synchronous regex/heuristic processors. `effect_size_auditor` scores reporting bias 0-1. `funding_checker` classifies funding sources. `author_coi` is async (queries ORCID, Europe PMC, CMS Open Payments).
- **Annotators** (`annotators/`): `LLMAnnotator` sends abstracts to Claude (`claude-sonnet-4-20250514`) via the `anthropic` async SDK. Parses JSON responses into `BiasAssessment` objects. Semaphore-based concurrency (default 3).
- **Schemas** (`schemas/`): `bias_taxonomy.py` defines the full bias taxonomy as dataclasses and enums. `schemas/__init__.py` exports `extract_abstract_sections()` used by both `spin_detector` and `effect_size_auditor`.

### Data Flow

All stages use JSONL (one JSON object per line). Flow:
```
raw/*.jsonl → enriched/{high,low}_suspicion.jsonl → labelled/*_annotated.jsonl + review.csv → export/{alpaca,sharegpt}/{train,val,test}.jsonl
```

### Key Design Decisions

- **Verification-focused training**: The export format includes `<think>` reasoning chains and verification steps citing specific databases (CMS Open Payments, ClinicalTrials.gov, ORCID, etc.). This teaches the model WHERE to look, not just what to flag.
- **Multi-source ground truth**: Retracted papers (known positives), Cochrane RoB (expert assessments), and heuristic-mined PubMed RCTs provide diverse training signal.
- **Boutron spin classification**: Spin detection uses the established Boutron taxonomy (none/low/moderate/high).
- **Config is a dataclass** (`config.py`): Contains all API endpoints, collection limits, MeSH focus domains, and output directory paths. `config.py` is gitignored — copy `config.example.py` to get started.
- **Single source of truth for SYSTEM_PROMPT**: The canonical training system prompt lives in `export.py`. `schemas/bias_taxonomy.py` imports it lazily to avoid duplication.

### External APIs Used

Crossref (retracted papers), PubMed E-utilities (abstract search/fetch), ClinicalTrials.gov v2 (outcome switching, sponsor data), Europe PMC (Cochrane reviews, funder data), ORCID (author affiliations), CMS Open Payments (physician payments), Anthropic Claude API (annotation).
