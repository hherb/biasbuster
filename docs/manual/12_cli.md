# 12. Analysing Publications (CLI)

The `biasbuster` command-line tool analyses individual biomedical publications for risk of bias. It fetches the best available content (full text or abstract), runs a structured 5-domain bias assessment via an LLM, and outputs a detailed report.

This is the primary user-facing tool — it wraps the entire analysis pipeline into a single command.

## Basic Usage

```bash
biasbuster <identifier> [options]
```

The identifier can be a **PMID**, **DOI**, or **local file path**:

```bash
# PubMed ID (bare or prefixed)
biasbuster 12345678
biasbuster PMID:12345678

# DOI (bare, prefixed, or as URL)
biasbuster 10.1016/j.example.2024.01.001
biasbuster doi:10.1016/j.example.2024
biasbuster https://doi.org/10.1016/j.example.2024

# Local files
biasbuster ./paper.pdf
biasbuster ./paper.xml
biasbuster ./paper.jats
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | from config | LLM model as `provider:model_name` |
| `--format {json,markdown}` | `json` | Output format |
| `--verify` | off | Enable external verification pipeline |
| `--save` | off | Persist results to the BiasBuster SQLite database |
| `--db PATH` | `dataset/biasbuster.db` | Database path (with `--save`) |
| `--config PATH` | `~/.biasbuster/config.toml` | Config file path |
| `--email EMAIL` | from config | Contact email for API polite pools |
| `-v` / `--verbose` | off | Debug logging to stderr |
| `-q` / `--quiet` | off | Suppress progress, output result only |

## Model Selection

Models are specified as `provider:model_name`, using bmlib's unified LLM client:

| Provider | Example | Requirement |
|----------|---------|-------------|
| ollama | `ollama:qwen3.5-9b-biasbuster` | Ollama running locally |
| anthropic | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| deepseek | `deepseek:deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| openai | `openai:gpt-4o` | API key in config |
| mistral | `mistral:mistral-large-latest` | API key in config |
| gemini | `gemini:gemini-pro` | API key in config |

**Ollama model names with colons** (e.g. `gpt-oss:20b`) are automatically detected as Ollama models. Only recognised provider prefixes (anthropic, ollama, openai, deepseek, mistral, gemini) trigger provider routing — everything else defaults to Ollama.

## Configuration

Create `~/.biasbuster/config.toml`:

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
email = "you@example.com"     # required for Unpaywall and NCBI polite pools
db_path = "dataset/biasbuster.db"

[verify]
crossref_mailto = "you@example.com"
```

**Resolution order** (highest priority first): CLI flag → environment variable → config file → built-in default.

| Environment Variable | Maps to |
|---------------------|---------|
| `ANTHROPIC_API_KEY` | `[api_keys] anthropic` |
| `DEEPSEEK_API_KEY` | `[api_keys] deepseek` |
| `NCBI_API_KEY` | `[api_keys] ncbi` |
| `BIASBUSTER_EMAIL` | `[general] email` |

## How Content Is Acquired

The tool resolves the identifier and fetches the best available content through a multi-tier fallback:

1. **PMID/DOI → structured JATS full text**: Discovers the PMC ID via Europe PMC, fetches JATS XML, and parses it into structured sections (Introduction, Methods, Results, Discussion, etc.) via bmlib's `JATSParser`.
2. **JATS back-matter extraction**: JATS XML places funding and COI disclosures in `<back>` elements rather than `<body>`. The tool explicitly extracts `<funding-group>`, `<notes notes-type="COI-statement">`, `<ack>`, and author contribution notes so they are not missed during analysis.
3. **PDF fallback**: If no JATS XML is available, bmlib's `FullTextService` tries Unpaywall for an open-access PDF. Text is extracted via pdfplumber.
4. **Abstract-only fallback**: If no full text is available, the abstract from PubMed is used.

For local files, `.xml`/`.jats` files are parsed via `JATSParser` and `.pdf` files via pdfplumber.

## Analysis Modes

### Abstract-Only (Single-Pass)

When only the abstract is available, it is sent to the LLM in a single call using the canonical 5-domain bias assessment prompt. This includes any available metadata (authors, funding, MeSH terms).

### Full-Text (Map-Reduce)

When full text is available, the document is analysed in two phases:

**Map phase** — each section is analysed independently:
- Methods → methodology red flags (comparator, enrichment, per-protocol)
- Results → statistical reporting (relative-only, selective p-values)
- Discussion/Conclusions → spin (Boutron taxonomy)
- Funding, COI & Disclosures → conflict of interest signals

**Reduce phase** — all per-section findings are synthesised into a unified 5-domain assessment with overall severity, bias probability, and confidence.

**Chunking strategy:**
- **JATS articles**: split by semantic sections from `body_sections`, plus a dedicated back-matter chunk for funding/COI/disclosures. Large sections are sub-chunked with token windows.
- **Plain text (PDFs)**: split into overlapping token windows (~3000 tokens, 200 token overlap) with boundary-aware breaking at paragraph or sentence boundaries.

## The Five Assessment Domains

Each domain is rated NONE / LOW / MODERATE / HIGH / CRITICAL:

| Domain | What It Checks |
|--------|---------------|
| **Statistical Reporting** | Relative-only measures, missing NNT/ARR, selective p-values, subgroup emphasis |
| **Spin** | Boutron taxonomy — do conclusions match results? Causal language, extrapolation |
| **Outcome Reporting** | Patient-centred vs surrogate, outcome switching, composite disaggregation |
| **Conflict of Interest** | Funding disclosure, industry affiliations, COI transparency |
| **Methodology** | Comparator choice, enrichment design, per-protocol only, premature stopping |

The output also includes:
- **Overall severity** (aggregated across domains)
- **Bias probability** (0.0–1.0 calibrated estimate)
- **Confidence** (low/medium/high)
- **Recommended verification steps** (specific databases to check)

## Verification Pipeline (`--verify`)

When `--verify` is enabled, the tool runs a second pass after the initial assessment:

1. Parses the initial assessment to extract bias flags
2. Synthesises verification steps (deterministic rules, not LLM-based)
3. Executes verification tools concurrently (up to 3 in parallel):
   - **ClinicalTrials.gov** — registered outcomes, sponsor, protocol amendments
   - **CMS Open Payments** — physician payment records from pharma/device companies
   - **ORCID** — author affiliation histories for undisclosed industry ties
   - **Europe PMC** — full-text funding and COI disclosure sections
   - **Retraction Watch / Crossref** — retraction status, expressions of concern
   - **Effect size audit** — local heuristic analysis of reporting patterns
4. Sends verification results back to the LLM for a **refined assessment**

The refined assessment may upgrade or downgrade severity ratings based on what the verification tools found.

## Output Formats

### JSON (default)

```bash
biasbuster 12345678 > report.json
```

Structured output with `metadata`, `assessment` (5-domain + overall), and `verification` (null unless `--verify` is used). Suitable for programmatic consumption and piping to other tools.

### Markdown

```bash
biasbuster 12345678 --format markdown
```

Human-readable report with paper metadata header, overall risk summary table, per-domain sections with flags and evidence quotes, recommended verification steps, and (if `--verify`) verification results with a before/after comparison.

## Database Persistence (`--save`)

```bash
biasbuster 12345678 --save
biasbuster 12345678 --save --db /path/to/custom.db
```

When `--save` is passed, the tool:
- Inserts the paper record (if not already present) with source `cli_import`
- Stores the annotation under model label `cli_{provider}_{model_name}`
- Integrates with the existing dataset pipeline — the paper can then be reviewed via the NiceGUI review tool and included in training data export

## Examples

### Quick assessment of a known paper

```bash
biasbuster 12345678 --format markdown --model anthropic:claude-sonnet-4-6
```

### Batch analysis with shell scripting

```bash
# Analyse a list of PMIDs
while read pmid; do
    biasbuster "$pmid" -q > "reports/${pmid}.json"
done < pmids.txt
```

### Full analysis with verification, saved to database

```bash
biasbuster https://doi.org/10.3390/antibiotics15020138 \
    --verify \
    --save \
    --format markdown \
    --model ollama:qwen3.5-9b-biasbuster
```

### Analyse a local JATS file from a journal download

```bash
biasbuster ./downloaded_article.xml --format markdown
```

## Troubleshooting

**"Unknown provider" error** — If your Ollama model name contains a colon (e.g. `gpt-oss:20b`), just use it directly. Only known provider names (anthropic, ollama, openai, deepseek, mistral, gemini) are treated as prefixes. Use `ollama:gpt-oss:20b` if you want to be explicit.

**"No content found" error** — The paper may not have an abstract in PubMed or may not be in Europe PMC. Try providing the DOI, or download the PDF locally and pass the file path.

**COI not detected** — For JATS full-text analysis, funding and COI disclosures are extracted from the XML back-matter (`<funding-group>`, `<notes notes-type="COI-statement">`). If the paper was analysed as abstract-only (no full text available), COI information from the full text will not be visible. Use `--verbose` to see which content type was used.

**Empty email warnings** — The NCBI ID Converter API requires a contact email. Set `email` in your config file or pass `--email you@example.com`. Without it, DOI-to-PMID resolution falls back to Europe PMC search (which works but is slower).

---

Next: Return to the [User Manual index](index.md)
