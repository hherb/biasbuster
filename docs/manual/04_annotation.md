# 4. LLM Annotation

**What you'll do:** Send collected abstracts to one or more LLMs for structured bias assessment across five domains.

## Run Annotation

```bash
# Default: Anthropic Claude only
uv run python pipeline.py --stage annotate

# Multiple backends
uv run python pipeline.py --stage annotate --models anthropic,deepseek
```

## The Five Bias Domains

Each abstract is assessed across these dimensions:

| Domain | What It Detects |
|--------|----------------|
| **Statistical Reporting** | Reliance on relative measures (HR, OR) without absolute measures (ARR, NNT). Selective p-value reporting, subgroup emphasis. |
| **Spin** | Conclusions that don't match results. Claims of benefit when primary outcome was not significant. Uses the Boutron taxonomy (none/low/moderate/high). |
| **Outcome Reporting** | Surrogate endpoints without validation, composite endpoints not disaggregated, evidence of outcome switching from registry. |
| **Conflict of Interest** | Undisclosed funding, industry author affiliations, missing COI statements. |
| **Methodology** | Inappropriate comparator, enrichment design, per-protocol analysis only, premature stopping, short follow-up. |

Each domain receives a severity rating: **none**, **low**, **moderate**, **high**, or **critical**.

## How Annotation Works

### Source Selection

The annotator pulls abstracts from four pools, each with a configurable limit:

```python
annotation_max_per_source = {
    "high_suspicion": 500,    # High reporting bias score
    "retracted_papers": 200,  # Known-biased positives
    "cochrane_rob": 300,      # Expert-assessed
    "low_suspicion": 300,     # Negative examples
}
```

### Filtering

Bare retraction notices (e.g., "This article has been retracted") are automatically filtered out before annotation. They contain no assessable research content. Original papers that were later retracted -- which have full abstracts intact -- are kept and assessed normally.

### User Message

Each abstract is sent to the LLM with rich context:

- PMID, title, abstract text
- Authors (first 5 with affiliations)
- Funding/grants
- Journal and MeSH terms
- Retraction reasons (if applicable)
- Heuristic pre-screen results from enrichment (reporting pattern, score, flags)

### Incremental Persistence

Each annotation is saved to the database immediately after the LLM responds, via an `on_result` callback. If the process crashes mid-batch, already-completed annotations are preserved. Re-running the command will skip them.

## Annotation Backends

### Anthropic Claude (default)

Uses the `anthropic` async SDK. Configure in `config.py`:

```python
anthropic_api_key = ""           # Or set ANTHROPIC_API_KEY env var
annotation_model = "claude-sonnet-4-6"
annotation_max_tokens = 4000
```

### DeepSeek (OpenAI-compatible)

Uses `httpx` with the OpenAI-compatible API. Configure:

```python
deepseek_api_key = ""            # Or set DEEPSEEK_API_KEY env var
deepseek_api_base = "https://api.deepseek.com"
deepseek_model = "deepseek-reasoner"
deepseek_max_tokens = 4000
```

### Rate Limiting

- `annotation_delay` (default: 1.0s) -- delay between API calls
- `annotation_concurrency` (default: 3) -- maximum concurrent requests

Both backends include retry logic with exponential backoff (up to 3 retries).

## What Gets Stored

Annotations are stored in the `annotations` table:

| Field | Description |
|-------|-------------|
| `pmid` | PubMed ID |
| `model_name` | Which backend produced this ("anthropic" or "deepseek") |
| `annotation` | Full JSON assessment (all 5 domains, severity, flags, evidence quotes) |
| `overall_severity` | Aggregate severity across all domains |
| `overall_bias_probability` | 0.0 to 1.0 probability estimate |
| `confidence` | Model's self-assessed confidence (low/medium/high) |

## Verify Annotations

```bash
uv run python -c "
from database import Database
db = Database('dataset/biasbuster.db')
with db:
    for model in ['anthropic', 'deepseek']:
        pmids = db.get_annotated_pmids(model)
        print(f'{model}: {len(pmids)} annotations')
"
```

## Next Step

[Creating Ground Truth (Human Review)](05_human_review.md) -- validate and correct LLM annotations using the web-based review tool.
