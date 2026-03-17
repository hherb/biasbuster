# 2. Harvesting Training Data

**What you'll do:** Collect clinical trial abstracts from three complementary sources to build a diverse training dataset.

## Run Collection

```bash
uv run python pipeline.py --stage collect
```

This fetches abstracts from all three sources and stores them in the SQLite database. The process is resumable -- if interrupted, re-running the command will skip already-collected papers.

## Data Sources

### Retraction Watch (Known-Biased Positives)

Retracted papers serve as known-positive training examples. The collector:

1. Queries the Crossref API for papers with `update-type:retraction`
2. Resolves original paper DOIs (not the retraction notice itself)
3. Converts DOIs to PMIDs via the NCBI ID Converter
4. Fetches full abstracts from PubMed in batches of 200

**Config:**
- `retraction_watch_max` (default: 2000) -- maximum retracted papers to collect

### PubMed RCTs (General Population)

Randomized controlled trials from PubMed, filtered by medical domain. These provide the bulk of training examples and span a range of bias levels.

For each domain in `focus_domains`, the collector searches PubMed:

```
"{domain}"[MeSH Terms] AND "randomized controlled trial"[Publication Type]
AND "{pubmed_rct_start_date}"[Date - Publication] : "today"[Date - Publication]
```

**Config:**
- `focus_domains` -- list of MeSH terms to search (default: 7 domains including cardiovascular, oncology, diabetes, etc.)
- `spin_screening_max` (default: 5000) -- total abstracts across all domains
- `pubmed_rct_start_date` (default: "2020/01/01") -- earliest publication date

### Cochrane Risk of Bias (Expert Ground Truth)

Cochrane systematic reviews contain expert risk-of-bias assessments using the RoB 2.0 framework. The collector:

1. Searches Europe PMC for recent Cochrane reviews mentioning "risk of bias"
2. Parses full-text XML to extract per-study RoB judgments across 5 domains
3. Resolves study identifiers (e.g., "Smith 2020") to PubMed PMIDs

**Config:**
- `cochrane_max_reviews` (default: 50) -- Cochrane reviews to search
- `cochrane_rob_max` (default: 1000) -- maximum studies to extract
- `cochrane_min_year` (default: 2018) -- earliest review year

## What Gets Stored

All papers are inserted into the `papers` table in SQLite with fields:

| Field | Description |
|-------|-------------|
| `pmid` | PubMed ID (primary key) |
| `title`, `abstract` | Paper text |
| `journal`, `year` | Publication metadata |
| `authors` | JSON array with names and affiliations |
| `grants`, `mesh_terms` | Funding and MeSH term data |
| `source` | Which collector produced this record |
| `retraction_reasons` | Why the paper was retracted (if applicable) |
| `overall_rob` | Cochrane risk-of-bias judgment (if applicable) |

## Verify Collection

Check how many papers were collected:

```bash
uv run python -c "
from database import Database
db = Database('dataset/biasbuster.db')
with db:
    for source in ['retraction_watch', 'pubmed_rct', 'cochrane_rob']:
        pmids = db.get_paper_pmids(source=source)
        print(f'{source}: {len(pmids)} papers')
"
```

## Rate Limits

The collectors respect external API rate limits:

| API | Delay | Notes |
|-----|-------|-------|
| Crossref | 0.5s | Polite pool requires `crossref_mailto` |
| PubMed | 0.35s | With NCBI API key; 0.5s without |
| ClinicalTrials.gov | 0.5s | Used during enrichment stage |

## Next Step

[Heuristic Enrichment](03_enrichment.md) -- analyse collected abstracts for statistical reporting patterns and outcome switching.
