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

# Run full pipeline (stages 1-4)
uv run python pipeline.py --stage all

# Run individual stages
uv run python pipeline.py --stage collect
uv run python pipeline.py --stage enrich
uv run python pipeline.py --stage annotate
uv run python pipeline.py --stage annotate --models anthropic,deepseek  # multi-model
uv run python pipeline.py --stage export
uv run python pipeline.py --stage compare   # compare models vs human labels

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

5-stage async workflow orchestrated by `pipeline.py`:

1. **Collect** — Fetch abstracts from external APIs (Crossref/Retraction Watch, PubMed RCTs by MeSH domain, Cochrane RoB assessments via Europe PMC)
2. **Enrich** — Run heuristic analysis (effect size auditing, outcome switching via ClinicalTrials.gov) to bucket abstracts into high/low suspicion
3. **Annotate** — Send abstracts to one or more LLMs for structured 5-domain bias assessment; store annotations in SQLite. Supports multiple backends via `--models` flag (default: anthropic). Each annotation is saved incrementally to the DB via an `on_result` callback. Retraction notices are filtered out via `is_retraction_notice()` before annotation.
4. **Export** — Convert human-validated annotations to fine-tuning formats (alpaca with `<think>` chains, sharegpt, openai_chat) with 80/10/10 train/val/test splits
5. **Compare** — Score each model's annotations against human ground truth (or inter-model agreement if no human labels). Generates per-dimension F1, Cohen's kappa, McNemar's significance tests, and a Markdown comparison report.

Human review (using the NiceGUI web tool) is a manual step between Annotate and Export.

### Module Pattern

- **Collectors** (`collectors/`): Async classes using `httpx.AsyncClient` with rate limiting. Each fetches from a specific source (Crossref, PubMed, ClinicalTrials.gov, Europe PMC). Return typed dataclasses. PubMed XML parsing functions (`parse_pubmed_xml`, `parse_pubmed_xml_batch`) are standalone module-level functions in `retraction_watch.py`.
- **Enrichers** (`enrichers/`): Mostly synchronous regex/heuristic processors. `effect_size_auditor` scores reporting bias 0-1. `funding_checker` classifies funding sources. `author_coi` is async (queries ORCID, Europe PMC, CMS Open Payments).
- **Annotators** (`annotators/`): Two backends sharing prompt, user-message construction, and output utilities via `annotators/__init__.py`:
  - `LLMAnnotator` (`llm_prelabel.py`) — Anthropic Claude via the `anthropic` async SDK
  - `OpenAICompatAnnotator` (`openai_compat.py`) — any OpenAI-compatible API (DeepSeek, vLLM, SGLang, etc.) via `httpx`
  - Both use the same `ANNOTATION_SYSTEM_PROMPT` and `build_user_message()` for comparable outputs
  - Both support incremental save via `on_result` callback and checkpoint/resume in `annotate_batch()`
  - `annotators/__init__.py` also contains `is_retraction_notice()` for filtering bare retraction notices
- **Schemas** (`schemas/`): `bias_taxonomy.py` defines the full bias taxonomy as dataclasses and enums. `schemas/__init__.py` exports `extract_abstract_sections()` used by both `spin_detector` and `effect_size_auditor`.
- **Evaluation** (`evaluation/`): Harness for running models, scoring outputs, computing metrics (binary F1, ordinal kappa, calibration, verification quality), and generating head-to-head comparison reports with statistical tests.

### Data Flow

All pipeline data lives in a single SQLite database (`dataset/biasbuster.db`). Schema:
```
papers (pmid PK) → enrichments (pmid PK) → annotations (pmid, model_name PK) → human_reviews (pmid, model_name PK)
```

Export stage reads from the DB and writes training files:
```
DB → export/{alpaca,sharegpt}/{train,val,test}.jsonl
```

Legacy JSONL data can be imported with `migrate_jsonl_to_sqlite.py` (idempotent, uses INSERT OR IGNORE).

### Key Design Decisions

- **Verification-focused training**: The export format includes `<think>` reasoning chains and verification steps citing specific databases (CMS Open Payments, ClinicalTrials.gov, ORCID, etc.). This teaches the model WHERE to look, not just what to flag.
- **Multi-source ground truth**: Retracted papers (known positives), Cochrane RoB (expert assessments), and heuristic-mined PubMed RCTs provide diverse training signal.
- **Boutron spin classification**: Spin detection uses the established Boutron taxonomy (none/low/moderate/high).
- **Config is a dataclass** (`config.py`): Contains all API endpoints, collection limits, MeSH focus domains, and `db_path`. `config.py` is gitignored — copy `config.example.py` to get started.
- **SQLite as single source of truth**: `database.py` provides the `Database` class with schema-enforced PMID uniqueness, atomic upserts, WAL mode for concurrent reads, and foreign key constraints. All pipeline stages read/write via `Database` methods instead of file I/O.
- **Single source of truth for SYSTEM_PROMPT**: The canonical annotation prompt lives in `annotators/llm_prelabel.py` (`ANNOTATION_SYSTEM_PROMPT`). The training system prompt lives in `export.py`. `schemas/bias_taxonomy.py` imports it lazily to avoid duplication.
- **Shared annotator utilities**: `annotators/__init__.py` contains `build_user_message()`, `_ensure_parsed()`, `is_retraction_notice()`, `parse_llm_json()`, and `strip_markdown_fences()` — shared across all backends to eliminate duplication and ensure consistent behaviour.
- **Incremental annotation persistence**: `annotate_batch()` accepts an `on_result` callback; the pipeline passes a function that calls `db.insert_annotation()` per result, so annotations survive mid-batch crashes.

### External APIs Used

Crossref (retracted papers), PubMed E-utilities (abstract search/fetch), ClinicalTrials.gov v2 (outcome switching, sponsor data), Europe PMC (Cochrane reviews, funder data), ORCID (author affiliations), CMS Open Payments (physician payments), Anthropic Claude API (annotation), DeepSeek API (annotation, OpenAI-compatible).
