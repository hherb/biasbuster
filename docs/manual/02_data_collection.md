# 2. Harvesting Training Data

**What you'll do:** Collect clinical trial abstracts from three complementary sources to build a diverse training dataset, then clean and enrich the collected data.

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

1. Searches Europe PMC for open-access systematic reviews mentioning "risk of bias"
2. Extracts per-study RoB judgments via regex patterns; falls back to LLM-based extraction (DeepSeek reasoner, chunk & map-reduce) when regex finds nothing
3. Resolves study identifiers to PubMed PMIDs using a multi-layer strategy

#### PMID Resolution

Study IDs in systematic reviews vary widely -- `"Smith 2020"`, `"Smith et al. 2020"`, `"Smith et al. [28]"`. The collector normalises these and applies five resolution strategies in priority order:

| Layer | Strategy | Description |
|-------|----------|-------------|
| 1a | Bracket-reference lookup | Match `[28]` to `<ref>` XML element by label, read inline `<pub-id>` |
| 1b | Author+year from refs | Match normalised surname+year against the review's reference list |
| 1c | Surname-only from refs | If exactly one ref matches the surname (when no year is available) |
| 2 | DOI → PMID | PubMed `esearch` with `DOI[DOI]` for studies with a DOI but no PMID |
| 3 | Author+year PubMed search | Broad PubMed search by author name and publication year |

Resolution is done per-review (not batched at the end) so results can be saved incrementally via the `on_result` callback.

**Config:**
- `cochrane_max_reviews` (default: 200) -- Cochrane reviews to search
- `cochrane_rob_max` (default: 1000) -- maximum studies to extract
- `cochrane_min_year` (default: 2015) -- earliest review year

### Cochrane Data Persistence

Cochrane RoB papers use `upsert_cochrane_paper()` instead of the standard `INSERT OR IGNORE`. This ensures that re-running collection always updates domain-level RoB ratings and review metadata, while preserving PubMed-fetched titles and abstracts that were populated by a later seed step. The conversion from `RoBAssessment` to paper dict is handled by a shared pure function (`rob_assessment_to_paper_dict()`) used across all scripts that save Cochrane data.

To backfill per-domain RoB ratings (D1-D5) for papers that were collected before domain-level extraction was added:

```bash
uv run python backfill_cochrane_domains.py
```

This long-running script (~5-6 hours due to DeepSeek API calls) supports checkpoint/resume — interrupted runs can be restarted without re-processing already completed reviews.

## Post-Collection Cleanup (Seed)

After collection, run the seed step to clean and enrich the raw data:

```bash
uv run python pipeline.py --stage seed
```

Or run it standalone with individual steps:

```bash
uv run python seed_database.py                    # all steps
uv run python seed_database.py --step enrich-rw   # just retraction reasons
uv run python seed_database.py --step fetch-abs   # just missing abstracts
uv run python seed_database.py --step clean       # just retraction notice filter
```

The seed step performs three idempotent operations:

1. **enrich-rw** -- Downloads the Retraction Watch CSV from Crossref Labs and enriches `retraction_reasons` with structured reason codes from the controlled RW vocabulary (~111 categories)
2. **fetch-abs** -- Fetches missing abstracts from PubMed for Cochrane RoB papers (and any paper with an empty abstract but a valid PMID)
3. **clean** -- Flags bare retraction notices ("This article has been retracted...") that contain no assessable research content. Original papers that were later retracted are kept.

## What Gets Stored

All papers are inserted into the `papers` table in SQLite with fields:

| Field | Description |
|-------|-------------|
| `pmid` | PubMed ID (primary key) |
| `title`, `abstract` | Paper text |
| `journal`, `year` | Publication metadata |
| `authors` | JSON array with names and affiliations |
| `grants`, `mesh_terms` | Funding and MeSH term data |
| `subjects` | Subject categories |
| `source` | Which collector produced this record |
| `retraction_reasons` | Why the paper was retracted (if applicable) |
| `overall_rob` | Cochrane risk-of-bias judgment (if applicable) |
| `randomization_bias` | RoB 2 domain D1 rating (if applicable) |
| `deviation_bias` | RoB 2 domain D2 rating (if applicable) |
| `missing_outcome_bias` | RoB 2 domain D3 rating (if applicable) |
| `measurement_bias` | RoB 2 domain D4 rating (if applicable) |
| `reporting_bias` | RoB 2 domain D5 rating (if applicable) |
| `cochrane_review_pmid` | Source Cochrane review PMID (if applicable) |
| `cochrane_review_doi` | Source Cochrane review DOI (if applicable) |
| `cochrane_review_title` | Source Cochrane review title (if applicable) |
| `domain` | Medical domain (MeSH-based) |
| `excluded` | Soft-delete flag for filtered records |

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
