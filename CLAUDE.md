# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## STOP AND CHECK — Before Writing Any Code

These are non-negotiable. Violations have caused real data loss and wasted resources.

- **Will this process run >2 minutes?** → Do NOT run it in Claude's shell. Print the command for the user to run in their own terminal. The user decides when to stop.
- **Does this process produce results over time?** → It MUST save incrementally with checkpoint/resume support. Use `on_result` callbacks, periodic DB commits, or flush-every-N patterns. Never batch results in memory to save at the end.
- **Am I truncating/trimming data that needs to be analyzed?** → NEVER. If data exceeds a context limit, use chunk & map-reduce, or skip the item with logging. Truncation silently destroys the information we need.
- **Am I adding features beyond what was asked?** → Don't. Fix first, enhance later. No speculative error handling, no premature abstractions, no "while I'm here" refactors.
- **Does this touch the annotation or training prompt?** → Both MUST use `prompts.py` as the single source of truth. Never inline prompt text.

## Project Overview

BMLibrarian Bias Detection Dataset Builder — a toolkit for building curated training datasets to fine-tune LLMs for detecting bias in biomedical abstracts. The fine-tuned model learns to assess bias across 5 domains and suggest specific verification steps citing real databases.

## Hardware Environment
This project runs either on a DGX Spark (ARM/Blackwell/GB10 architecture) or on an Apple M series processor. Do not suggest x86-specific solutions. Always consider ARM compatibility, GPU memory constraints, and quantization requirements when recommending ML/inference approaches.

## Python Coding Standards
1. We prefer pure functions over complex code with side effects. Ideally packaged as reusable library modules.
2. Doc strings and type hints are mandatory in programming languages supporting them
3. No magic numbers. Use configuration / constants modules
4. Unit tests are generally required
5. All network / API functions need to attempt to gracefully recover with retires with exponential backoff (up to  MAX_RETRIES)
6. All caught errors must be shown to the user in the UI and logged
7. All errors discovered need to be fixed as priority, not postponed
8. We aim at keeping individual code files below 500 lines if feasible. Refactor early into separate modules if growth anticipated
9. All display sizing should be relative to allow a large variety of screen sizes/formats/DPI constellations. Do not needlessly waste screen real estate by hard coding display item sizes.
10. Anything displayed on the screen as output needs to be user selectable for copy & paste
11. we use uv for python libraries / environments. Do not use pip / venv directly 
11. If in doubt about imlementation, pause and ask the user. Ideally, propose the solutions you have been thinking about.
12. After making code changes, always check for None/null handling in new code paths. Use explicit None checks rather than relying on dict.get() defaults or optional chaining alone.

### UI Development
When editing UI code (NiceGUI, AG Grid), test each change in isolation before moving on. Do not introduce new features while fixing bugs — fix first, then enhance.

### Debugging Approach
When debugging, ask clarifying questions before jumping to a diagnosis. Confirm the user's actual problem description before proposing fixes. Do not assume the environment is headless or x86 without checking.

### Pipeline & Data Processing Conventions
Always implement retry logic and incremental saves/commits when building data pipelines or annotation workflows. Never silently skip errors or produce duplicate entries.

## Commands

```bash
# Set up environment and install dependencies
uv sync

# Configure (edit config.py with API keys)
cp config.example.py config.py

# Run full pipeline (stages 1-5)
uv run python pipeline.py --stage all

# Run individual stages
uv run python pipeline.py --stage collect
uv run python pipeline.py --stage collect-rob # Cochrane RoB only (+ fetch abstracts)
uv run python pipeline.py --stage seed       # enrich RW reasons + fetch abstracts + clean notices
uv run python seed_database.py               # standalone (same as --stage seed)
uv run python seed_database.py --step enrich-rw   # just retraction reasons from RW CSV
uv run python seed_database.py --step fetch-abs   # just missing abstracts from PubMed
uv run python seed_database.py --step clean       # just retraction notice cleanup
uv run python pipeline.py --stage enrich
uv run python pipeline.py --stage annotate
uv run python pipeline.py --stage annotate --models anthropic,deepseek  # multi-model
uv run python pipeline.py --stage export
uv run python pipeline.py --stage compare   # compare models vs human labels

# Individual module demos (each has __main__ block)
uv run python -m collectors.spin_detector
uv run python -m enrichers.effect_size_auditor
uv run python -m enrichers.funding_checker

# Training (inside NGC Docker on DGX Spark)
./run_training.sh qwen3.5-27b              # full training run
./run_training.sh olmo-3.1-32b --resume    # resume from checkpoint
./run_training.sh gpt-oss-20b              # MoE model (attention-only LoRA)
./run_training.sh qwen3.5-27b --max-steps 5  # smoke test

# Merge LoRA adapter and export to Ollama
./run_merge.sh qwen3.5-27b                     # dense models: standard merge in Docker
./run_merge.sh gpt-oss-20b                     # MoE: surgical merge preserving MXFP4 (~14 GB)
bash training/export_to_ollama.sh training_output/qwen3.5-27b-merged qwen3.5-27b-biasbuster
bash training/export_to_ollama.sh training_output/gpt-oss-20b-merged gpt-oss-20b-biasbuster

# Training monitor (run on host while training runs in Docker)
uv run python -m utils.training_monitor --metrics-dir training_output/qwen3.5-27b-lora

# MLX Training (on Apple Silicon Macs — no Docker)
uv sync --group mlx                                    # install MLX dependencies
./run_training_mlx.sh qwen3.5-9b-4bit                  # full training (64GB Mac)
./run_training_mlx.sh qwen3.5-27b-4bit --max-iters 5   # smoke test
./run_training_mlx.sh qwen3.5-27b-8bit --resume        # resume (128GB Mac)
./run_training_mlx.sh gpt-oss-20b-4bit                  # MoE model (64GB+ Mac)
./run_training_mlx.sh gpt-oss-20b-8bit                  # MoE model (128GB Mac)

# MLX merge and export to Ollama
./run_merge_mlx.sh qwen3.5-27b-4bit
./run_merge_mlx.sh qwen3.5-27b-4bit --quantize Q4_K_M  # with GGUF quantization

# Fine-Tuning Workbench GUI (all-in-one: settings → train → evaluate → export)
uv run python -m gui
uv run python -m gui --port 9090

# Add a new dependency
uv add <package>
```

There is no formal test suite — modules have `if __name__ == "__main__":` demo blocks that serve as smoke tests.

## Architecture

### Pipeline Stages

7-stage workflow with two orchestrators:

**Data pipeline** (`pipeline.py` — async):
1. **Collect** — Fetch abstracts from external APIs (Crossref/Retraction Watch, PubMed RCTs by MeSH domain, Cochrane RoB assessments via Europe PMC)
1b. **Seed** (`seed_database.py`) — Post-collection cleanup: enrich retraction reasons from Retraction Watch CSV (structured ~111-category vocabulary), fetch missing abstracts from PubMed (Cochrane papers), remove bare retraction notices. Idempotent and reproducible.
2. **Enrich** — Run heuristic analysis (effect size auditing, outcome switching via ClinicalTrials.gov) to bucket abstracts into high/low suspicion
3. **Annotate** — Send abstracts to one or more LLMs for structured 5-domain bias assessment; store annotations in SQLite. Supports multiple backends via `--models` flag (default: anthropic). Each annotation is saved incrementally to the DB via an `on_result` callback. Retraction notices are filtered out via `is_retraction_notice()` before annotation.
4. **Export** — Convert human-validated annotations to fine-tuning formats (alpaca with `<think>` chains, sharegpt, openai_chat) with 80/10/10 train/val/test splits
5. **Compare** — Score each model's annotations against human ground truth (or inter-model agreement if no human labels). Generates per-dimension F1, Cohen's kappa, McNemar's significance tests, and a Markdown comparison report.

Human review (using the NiceGUI web tool) is a manual step between Annotate and Export.

**Training pipeline** (`training/` — two backends):
6. **Train** — LoRA fine-tuning with two backends:
   - **DGX Spark** (NGC Docker): `train_lora.py` via TRL's `SFTTrainer` for Qwen3.5-27B / OLMo-3.1-32B / GPT-OSS-20B.
   - **Apple Silicon** (native macOS): `train_lora_mlx.py` via `mlx_lm.tuner` for pre-quantized Qwen models (9B/27B in 4-bit/8-bit QLoRA) and GPT-OSS-20B MoE (4-bit/8-bit).
   Both backends write identical `metrics.jsonl` for live monitoring via the NiceGUI dashboard (`utils/training_monitor.py`). After training, merge adapter and export to Ollama (`export_to_ollama.sh`).
   **GPT-OSS MoE special handling**: MXFP4 expert weights are dequantized to BF16 on load via `Mxfp4Config(dequantize=True)` (backward pass not implemented for MXFP4). LoRA targets attention layers only (skip expert FFNs/router). Uses `attn_implementation="eager"` per OpenAI cookbook. Ollama export uses the Harmony chat template (not ChatML).

### Module Pattern

- **Collectors** (`collectors/`): Async classes using `httpx.AsyncClient` with rate limiting. Each fetches from a specific source (Crossref, PubMed, ClinicalTrials.gov, Europe PMC). Return typed dataclasses. PubMed XML parsing functions (`parse_pubmed_xml`, `parse_pubmed_xml_batch`) are standalone module-level functions in `retraction_watch.py`. `cochrane_rob.py` exports `rob_assessment_to_paper_dict()` (shared pure function) and `collect_rob_dataset()` supports `skip_pmids`/`skip_pmcids` for efficient re-runs.
- **Enrichers** (`enrichers/`): Mostly synchronous regex/heuristic processors. `effect_size_auditor` scores reporting bias 0-1. `funding_checker` classifies funding sources. `author_coi` is async (queries ORCID, Europe PMC, CMS Open Payments). `retraction_classifier` classifies retraction reasons and assigns severity floors (see `docs/MISTAKES_ROUND_1_AND_FIXES.md`).
- **Annotators** (`annotators/`): Two backends sharing prompt, user-message construction, and output utilities via `annotators/__init__.py`:
  - `LLMAnnotator` (`llm_prelabel.py`) — Anthropic Claude via the `anthropic` async SDK
  - `OpenAICompatAnnotator` (`openai_compat.py`) — any OpenAI-compatible API (DeepSeek, vLLM, SGLang, etc.) via `httpx`
  - Both use the same `ANNOTATION_SYSTEM_PROMPT` and `build_user_message()` for comparable outputs
  - Both support incremental save via `on_result` callback and checkpoint/resume in `annotate_batch()`
  - `annotators/__init__.py` also contains `is_retraction_notice()` for filtering bare retraction notices
- **Schemas** (`schemas/`): `bias_taxonomy.py` defines the full bias taxonomy as dataclasses and enums. `schemas/__init__.py` exports `extract_abstract_sections()` used by both `spin_detector` and `effect_size_auditor`.
- **Evaluation** (`evaluation/`): Harness for running models, scoring outputs, computing metrics (binary F1, ordinal kappa, calibration, verification quality), and generating head-to-head comparison reports with statistical tests.
- **Training** (`training/`): LoRA fine-tuning pipeline with two backends:
  - **PyTorch/TRL** (DGX Spark): `train_lora.py` using TRL's `SFTTrainer`, `configs.py` for hyperparameters (with `_MOE_OVERRIDES` for GPT-OSS: MXFP4 dequantize, eager attn, conservative LR), `callbacks.py` for metrics, `merge_adapter.py` for adapter fusion.
  - **MLX** (Apple Silicon): `train_lora_mlx.py` using `mlx_lm.tuner`, `configs_mlx.py` for MLX-specific presets (Qwen 9B/27B in 4-bit/8-bit, GPT-OSS-20B in 4-bit/8-bit MoE), `callbacks_mlx.py` bridging to the same `metrics.jsonl` format, `merge_adapter_mlx.py` for adapter fusion via `mlx_lm.fuse`.
  - Shared: `data_utils.py` handles alpaca JSONL loading, chat template formatting, and alpaca→chat format conversion for MLX-lm.
- **Training Monitor** (`utils/training_monitor.py`): NiceGUI web dashboard that reads `metrics.jsonl` and displays live loss curves, learning rate schedule, GPU memory, gradient norms, and hyperparameters. Run with `uv run python -m utils.training_monitor`.
- **Fine-Tuning Workbench** (`gui/`): NiceGUI 4-tab GUI (`uv run python -m gui`) wrapping the entire fine-tuning workflow. `state.py` handles platform detection and settings persistence (`~/.biasbuster/gui_settings.json`). `process_runner.py` provides an async subprocess wrapper. Tab modules (`settings_tab.py`, `training_tab.py`, `evaluation_tab.py`, `export_tab.py`) each build their UI and launch operations as subprocesses. The training tab reuses `MetricsReader` from `utils/training_monitor.py` for live chart updates. Both training scripts (`train_lora.py`, `train_lora_mlx.py`) accept optional `--lr`, `--epochs`, `--lora-rank`, `--batch-size`, `--grad-accum`, `--max-seq-len` CLI args for GUI-driven hyperparameter overrides.

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

Training pipeline reads exported JSONL and writes to its own directory:
```
export/alpaca/{train,val,test}.jsonl → training/ → training_output/<model>-lora/
    ├── checkpoint-*/          # periodic checkpoints (adapter + optimiser state)
    ├── metrics.jsonl          # live training metrics (read by training_monitor.py)
    ├── final_adapter/         # best adapter (auto-selected by lowest eval_loss)
    └── trainer_state.json     # HuggingFace trainer state with eval history
```

### Key Design Decisions

- **Verification-focused training**: The export format includes `<think>` reasoning chains and verification steps citing specific databases (CMS Open Payments, ClinicalTrials.gov, ORCID, etc.). This teaches the model WHERE to look, not just what to flag.
- **Multi-source ground truth**: Retracted papers (known positives), Cochrane RoB (expert assessments), and heuristic-mined PubMed RCTs provide diverse training signal.
- **Boutron spin classification**: Spin detection uses the established Boutron taxonomy (none/low/moderate/high).
- **Config is a dataclass** (`config.py`): Contains all API endpoints, collection limits, MeSH focus domains, and `db_path`. `config.py` is gitignored — copy `config.example.py` to get started.
- **SQLite as single source of truth**: `database.py` provides the `Database` class with schema-enforced PMID uniqueness, atomic upserts, WAL mode for concurrent reads, and foreign key constraints. All pipeline stages read/write via `Database` methods instead of file I/O.
- **Single source of truth for prompts**: `prompts.py` contains all severity boundary definitions, domain criteria, and verification database recommendations. Both `ANNOTATION_SYSTEM_PROMPT` (used by annotators) and `TRAINING_SYSTEM_PROMPT` (used by export) share identical severity boundaries. `annotators/llm_prelabel.py` and `export.py` both import from `prompts.py`. See `docs/MISTAKES_ROUND_1_AND_FIXES.md` for why this unification matters.
- **Shared annotator utilities**: `annotators/__init__.py` contains `build_user_message()`, `_ensure_parsed()`, `is_retraction_notice()`, `parse_llm_json()`, and `strip_markdown_fences()` — shared across all backends to eliminate duplication and ensure consistent behaviour.
- **Shared Cochrane persistence**: `collectors/cochrane_rob.py` exports `rob_assessment_to_paper_dict()` (pure function for `RoBAssessment` → paper dict conversion) used by all Cochrane save paths. `database.py` provides `upsert_cochrane_paper()` with `INSERT ON CONFLICT DO UPDATE` that always updates Cochrane-authoritative fields (domain ratings, review metadata) while preserving PubMed-fetched data (title, abstract) via CASE/COALESCE guards. Empty strings cannot blank existing review metadata.
- **Incremental annotation persistence**: `annotate_batch()` accepts an `on_result` callback; the pipeline passes a function that calls `db.insert_annotation()` per result, so annotations survive mid-batch crashes.

### External APIs Used

Crossref (retracted papers), PubMed E-utilities (abstract search/fetch), ClinicalTrials.gov v2 (outcome switching, sponsor data), Europe PMC (Cochrane reviews, funder data), ORCID (author affiliations), CMS Open Payments (physician payments), Anthropic Claude API (annotation), DeepSeek API (annotation, OpenAI-compatible).
