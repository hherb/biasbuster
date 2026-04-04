# BiasBuster CLI

Command-line tool for analysing biomedical publications for risk of bias. Accepts a PubMed ID, DOI, or local file, fetches the best available content (full text or abstract), runs a structured 5-domain bias assessment via an LLM, and outputs a detailed report in JSON or Markdown.

## Installation

The CLI is installed as part of the biasbuster package:

```bash
uv sync
```

This registers the `biasbuster` command via the project entry point. You can also run it directly:

```bash
uv run python -m cli.main <identifier> [options]
```

### Dependencies

- **bmlib** (local, editable) -- publication retrieval, JATS parsing, LLM abstraction
- **pdfplumber** -- PDF text extraction for local files
- **httpx** -- HTTP client (transitive via bmlib)

## Quick Start

```bash
# Analyse a paper by PMID (uses default model from config)
biasbuster 12345678

# Analyse by DOI with markdown output
biasbuster 10.1016/j.example.2024.01.001 --format markdown

# Use a specific cloud model
biasbuster 12345678 --model anthropic:claude-sonnet-4-6

# Use a local fine-tuned model via Ollama
biasbuster 12345678 --model ollama:qwen3.5-9b-biasbuster

# Analyse a local JATS XML file
biasbuster ./paper.xml --format markdown

# Analyse a local PDF
biasbuster ./paper.pdf --model deepseek:deepseek-reasoner

# Full analysis with verification pipeline and DB persistence
biasbuster 12345678 --verify --save --format markdown
```

## Usage

```
biasbuster <identifier> [options]
```

### Positional Argument

`<identifier>` -- one of:

| Format | Examples | Detection |
|--------|----------|-----------|
| **PMID** | `12345678`, `PMID:12345678` | Bare integer or `PMID:` prefix |
| **DOI** | `10.1016/j.example.2024`, `doi:10.1016/...`, `https://doi.org/10.1016/...` | Starts with `10.`, `doi:` prefix, or `doi.org` URL |
| **Local file** | `./paper.pdf`, `./paper.xml`, `./paper.jats` | File extension `.pdf`, `.xml`, or `.jats` |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | from config | LLM model as `provider:model_name`. See [Model Selection](#model-selection). |
| `--format {json,markdown}` | `json` | Output format. |
| `--verify` | off | Enable the verification pipeline (cross-checks against external databases). |
| `--save` | off | Persist results to the BiasBuster SQLite database. |
| `--db PATH` | `dataset/biasbuster.db` | Database path (only with `--save`). |
| `--config PATH` | `~/.biasbuster/config.toml` | Config file path. |
| `--email EMAIL` | from config | Contact email for API polite pools (PubMed, Unpaywall). |
| `-v` / `--verbose` | off | Debug logging to stderr. |
| `-q` / `--quiet` | off | Suppress progress messages; output only the result. |

## Model Selection

Models are specified as `provider:model_name` strings, matching bmlib's LLM client routing format:

| Provider | Example | Requirements |
|----------|---------|--------------|
| **ollama** | `ollama:qwen3.5-9b-biasbuster` | Ollama running locally |
| **anthropic** | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` env var or config |
| **deepseek** | `deepseek:deepseek-reasoner` | `DEEPSEEK_API_KEY` env var or config |
| **openai** | `openai:gpt-4o` | API key in config |

A bare model name without a provider prefix (e.g. `qwen3.5-9b-biasbuster`) defaults to the `ollama` provider.

## Configuration

Settings are read from `~/.biasbuster/config.toml` (override with `--config`). Values are resolved in this priority order:

1. **CLI flags** (highest)
2. **Environment variables**
3. **Config file**
4. **Built-in defaults** (lowest)

### Example Config File

```toml
[model]
default = "ollama:qwen3.5-9b-biasbuster"

[api_keys]
anthropic = "sk-ant-..."      # or set ANTHROPIC_API_KEY env var
deepseek = "sk-..."           # or set DEEPSEEK_API_KEY env var
ncbi = ""                     # or set NCBI_API_KEY env var

[endpoints]
ollama = "http://localhost:11434"
deepseek_base = "https://api.deepseek.com"

[general]
email = "you@example.com"     # required for Unpaywall/PubMed polite pool
db_path = "dataset/biasbuster.db"

[verify]
crossref_mailto = "you@example.com"
```

### Environment Variables

| Variable | Maps to |
|----------|---------|
| `ANTHROPIC_API_KEY` | `[api_keys] anthropic` |
| `DEEPSEEK_API_KEY` | `[api_keys] deepseek` |
| `NCBI_API_KEY` | `[api_keys] ncbi` |
| `BIASBUSTER_EMAIL` | `[general] email` |

## How It Works

### Content Acquisition

The tool resolves the identifier and fetches the best available content:

1. **PMID/DOI**: Fetches metadata from PubMed, then attempts structured JATS full text from Europe PMC. Falls back through bmlib's 3-tier chain (Europe PMC XML -> Unpaywall PDF -> DOI resolve). If only the abstract is available, proceeds with abstract-only analysis.
2. **Local JATS XML** (`.xml`, `.jats`): Parsed via bmlib's `JATSParser` into structured sections (abstract, introduction, methods, results, discussion, etc.).
3. **Local PDF** (`.pdf`): Text extracted via pdfplumber.

### Analysis Modes

**Abstract-only (single-pass)**: When only the abstract is available, it is sent to the LLM in a single call using the canonical 5-domain bias assessment prompt from `prompts.py`.

**Full-text (map-reduce)**: When full text is available, the document is chunked and analysed in two phases:

- **Map phase**: Each section/chunk is analysed independently for bias signals relevant to its content (e.g. Methods sections are checked for methodology red flags, Results for statistical reporting issues, Discussion for spin).
- **Reduce phase**: All per-section findings are synthesised into a single unified 5-domain assessment.

Chunking strategy:
- **JATS articles**: Split by semantic sections (`body_sections` from the parsed `JATSArticle`). Large sections are sub-chunked with token windows.
- **Plain text (PDFs)**: Split into overlapping token windows (~3000 tokens, 200 token overlap) with boundary-aware breaking at paragraph or sentence boundaries.

### Bias Assessment Domains

The assessment covers 5 domains, each rated NONE / LOW / MODERATE / HIGH / CRITICAL:

1. **Statistical Reporting** -- relative-only measures, selective p-values, subgroup emphasis
2. **Spin** -- Boutron taxonomy: do conclusions match results?
3. **Outcome Reporting** -- patient-centred vs surrogate, outcome switching
4. **Conflict of Interest** -- funding disclosure, industry affiliations, COI transparency
5. **Methodology** -- comparator choice, enrichment design, per-protocol only, premature stopping

Plus an overall severity, calibrated bias probability (0.0--1.0), confidence level, and recommended verification steps.

### Verification Pipeline (`--verify`)

When `--verify` is enabled, the tool runs a second pass:

1. Parses the initial assessment to extract bias flags
2. Synthesises verification steps (deterministic rules, not LLM-based)
3. Executes verification tools concurrently (max 3 in parallel):
   - **ClinicalTrials.gov** -- registered outcomes, sponsor, protocol amendments
   - **CMS Open Payments** -- physician payment records
   - **ORCID** -- author affiliation histories
   - **Europe PMC** -- full-text funding/COI disclosures
   - **Retraction Watch / Crossref** -- retraction status
   - **Effect size audit** -- local heuristic reporting bias check
4. Sends tool results back to the LLM for a refined assessment

## Output Formats

### JSON (default)

```json
{
  "metadata": {
    "identifier": "12345678",
    "identifier_type": "pmid",
    "title": "Example Clinical Trial Title",
    "pmid": "12345678",
    "doi": "10.1234/example",
    "model": "anthropic:claude-sonnet-4-6",
    "verified": false,
    "content_type": "abstract",
    "timestamp": "2026-04-04T12:00:00+00:00"
  },
  "assessment": {
    "statistical_reporting": { "severity": "moderate", "relative_only": true, ... },
    "spin": { "severity": "low", "spin_level": "low", ... },
    "outcome_reporting": { "severity": "none", ... },
    "conflict_of_interest": { "severity": "high", ... },
    "methodology": { "severity": "low", ... },
    "overall_severity": "moderate",
    "overall_bias_probability": 0.45,
    "reasoning": "...",
    "confidence": "medium",
    "recommended_verification_steps": ["..."]
  },
  "verification": null
}
```

When `--verify` is used, the `"verification"` key contains tool results and the refined assessment.

### Markdown (`--format markdown`)

A human-readable report with:
- Paper metadata header
- Overall risk summary table (severity, probability, confidence)
- Per-domain sections with flags, severity, and evidence quotes
- Recommended verification steps
- Verification results and refined assessment (if `--verify`)

## Database Persistence (`--save`)

When `--save` is passed, the tool stores results in the BiasBuster SQLite database:

- Inserts the paper record (if not already present) with source `cli_import`
- Stores the annotation under model label `cli_{provider}_{model_name}`
- Integrates with the existing dataset pipeline for human review via the NiceGUI web tool

## Architecture

```
cli/
  __init__.py       Package init
  main.py           Entry point: argparse, config loading, orchestration
  config.py         TOML config loading, env var resolution, defaults
  content.py        Identifier resolution, content acquisition via bmlib
  chunking.py       Section-based (JATS) + token-window fallback chunking
  pdf_extract.py    PDF text extraction via pdfplumber
  analysis.py       Single-pass and map-reduce LLM assessment
  verification.py   Wrapper around agent/ verification pipeline
  formatting.py     JSON and Markdown output formatters
```

### Key Dependencies

| Need | Source | What |
|------|--------|------|
| Full-text retrieval | bmlib | `fulltext.FullTextService.fetch_fulltext()` |
| JATS parsing | bmlib | `fulltext.JATSParser.parse()` -> `JATSArticle` |
| Full-text caching | bmlib | `fulltext.FullTextCache` |
| LLM chat (all providers) | bmlib | `llm.LLMClient.chat()` with `provider:model` routing |
| Bias assessment prompt | biasbuster | `prompts.ANNOTATION_SYSTEM_PROMPT` |
| User message construction | biasbuster | `annotators.build_user_message()` |
| JSON parsing/repair | biasbuster | `annotators.parse_llm_json()` |
| Verification tools | biasbuster | `agent.verification_planner`, `agent.tool_router`, `agent.tools` |

## Tests

```bash
uv run python -m pytest tests/test_cli_config.py tests/test_cli_content.py tests/test_cli_chunking.py tests/test_cli_formatting.py -v
```

26 unit tests covering configuration loading, identifier classification, text chunking, and output formatting.
