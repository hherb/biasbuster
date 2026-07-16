# Extracting Expert RoB 2 / QUADAS-2 Ratings from Published Reviews

**Purpose:** One-page recipe for ingesting per-study × per-domain expert ratings from a candidate systematic review into the biasbuster pipeline. Follow this when you find a new review and want to add it to the faithfulness benchmark.

---

## 1. Triage the candidate

Before extracting, confirm the review is usable:

- **Per-study traffic-light figure**: each included study has its own row of coloured dots for every domain. Summary bar plots alone are not enough — you need the granular ratings. Check the main text *and* supplementary figures/spreadsheets.
- **Domain alignment**: RoB 2 has 5 domains (randomization, deviations, missing data, measurement, reporting); QUADAS-2 has 4 (patient selection, index test, reference standard, flow and timing). The figure captions usually spell out which tool was used.
- **Resolvable included studies**: author + year is enough if all studies are in PubMed. Check by searching ~3 at random on PubMed — if any aren't indexed, plan for per-study PMID resolution work.
- **Open-access full text**: JATS XML on Europe PMC is preferred (the pipeline fetches it automatically). PDFs via PMC work too but require a separate download step.
- **Review-PMID + DOI present**: needed for `cochrane_review_pmid` / `cochrane_review_doi` columns so the rating_source can be attributed. Without them, rows will be skipped by the backfill script (see the pilot-set limitation in `docs/papers/medrxiv_quadas_rob2_assessors.md`).

## 2. CSV column schema

Create a new file at `dataset/manual_verification_sets/<slug>_<date>.csv`. Example slug: `rob2_bcg_vaccine_20260501`.

**Required columns** (same for both methodologies):

| Column | Meaning | Example |
|---|---|---|
| `study_id` | 1..N sequence | `1` |
| `trial_name` | Human-readable label (author + year) | `Smith 2021` |
| `pmid` | PubMed ID of the included study | `35123456` |
| `pmc_url` | Full PMC URL if the study has one; empty otherwise | `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8012345/` |
| `source_review` | Descriptive name of the review that graded the study | `Smith 2024 BCG meta-analysis` |
| `cochrane_review_pmid` | The **review's** PMID (not the included study's) | `41689765` |
| `cochrane_review_doi` | The review's DOI | `10.1080/21645515.2025.2500000` |

**Rating columns — RoB 2** (5 domains + overall):

| Column | Vocabulary | Rule-of-thumb map from figure |
|---|---|---|
| `randomization` | `low` / `some_concerns` / `high` | Green dot / yellow dot / red dot on D1 |
| `deviations` | same | D2 |
| `missing_data` | same | D3 |
| `measurement` | same | D4 |
| `reporting` | same | D5 |
| `overall` | same | Overall column on the figure |

**Rating columns — QUADAS-2**: leave the 5 RoB-2-named columns empty; the ROBINS-I / QUADAS-2 per-domain mappings don't line up. Use `overall` for the overall **bias** rating (QUADAS-2 vocab: `low` / `unclear` / `high`). Per-domain QUADAS-2 extraction is on the roadmap but not in the current CSV importer.

## 3. Worked example — RoB 2 row

For Deng 2024's Radwan 2021 entry (reference):

```csv
10,Radwan et al. 2021,34059568,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8185267/,Deng 2024 plyometric meta-analysis,38760392,10.1038/s41598-024-61905-7,some_concerns,low,low,low,low,some_concerns
```

Fields in order: `study_id=10, trial_name=Radwan et al. 2021, pmid=34059568, pmc_url=..., source_review=Deng 2024 plyometric meta-analysis, cochrane_review_pmid=38760392, cochrane_review_doi=10.1038/s41598-024-61905-7, randomization=some_concerns, deviations=low, missing_data=low, measurement=low, reporting=low, overall=some_concerns`.

See `dataset/manual_verification_sets/rob2_deng2024_20260421.csv` for the full file this row came from.

## 4. Extraction workflow (figure-by-eye)

For each study row in the traffic-light figure:

1. Record the first author + year exactly as printed. Look up the PMID on PubMed (`<AuthorLastname> <Year> <KeyTerm>`).
2. For each domain column, colour → vocabulary:
   - 🟢 green (often "+") → `low`
   - 🟡 yellow (often "-" or "?") → `some_concerns` (RoB 2) / `unclear` (QUADAS-2)
   - 🔴 red (often "X") → `high`
3. Cross-check at least one row against the paper's prose. Figures are published by image; one transcription error per row is easy to make. The prose usually says things like "of the X trials, N were rated low on D2" — those counts should match your extraction.
4. Flag any symbol you can't match (e.g. half-green/half-yellow) as needing a second look.

## 5. Resolve PMIDs

The import script handles DOI → PMID resolution internally via NCBI's ID converter. If the figure uses PMIDs or DOIs directly, you're done. If it uses first author + year, run this to check:

```bash
uv run python -c "
import httpx
r = httpx.get(
    'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
    params={'db': 'pubmed', 'term': 'Smith 2021 plyometric[Title/Abstract]',
            'retmode': 'json', 'retmax': 3},
    timeout=20,
)
print(r.json()['esearchresult']['idlist'])
"
```

## 6. Import and run

```bash
# 1. Commit the CSV to git (source-of-truth principle)
git add dataset/manual_verification_sets/<your_csv>.csv
git commit -m "data(rob2): <review> expert ratings"

# 2. Import into the DB (fetches PubMed metadata + Europe PMC full text for each PMID)
uv run python -m scripts.import_rob2_verification_set \
    --csv dataset/manual_verification_sets/<your_csv>.csv \
    --verification-set <slug> \
    --db dataset/biasbuster_recovered.db

# 3. Backfill the expert ratings into expert_methodology_ratings
uv run python scripts/backfill_rob2_expert_ratings.py \
    --db dataset/biasbuster_recovered.db \
    --added-by "<reviewer>-extraction-<date>"

# 4. Annotate with the AI assessor (resume-safe; only new papers get hit)
uv run python scripts/batch_annotate_pmids.py \
    --pmids-file <your_pmid_list.txt> \
    --model anthropic \
    --methodology cochrane_rob2 \
    --decomposed \
    --db dataset/biasbuster_recovered.db

# 5. Run the faithfulness harness
uv run python -m biasbuster.evaluation.methodology_faithfulness \
    --methodology cochrane_rob2 \
    --model anthropic_fulltext_decomposed \
    --db dataset/biasbuster_recovered.db \
    --output reports/<review>_faithfulness_$(date +%Y%m%d)
```

For a QUADAS-2 review, swap `cochrane_rob2` → `quadas_2` in steps 4 and 5 (and skip step 3, which is RoB 2-specific; the QUADAS-2 backfill happens during the ingest path).

## 7. Validation checklist

Before handing results back to the paper, confirm:

- [ ] Every row's `pmid` resolves to a real PubMed record (`Database.get_paper(pmid)` returns non-empty).
- [ ] `expert_methodology_ratings` has one row per paper × domain (check: `SELECT COUNT(*) FROM expert_methodology_ratings WHERE rating_source LIKE '%<review_pmid>%'`).
- [ ] The assessor ran to completion for as many papers as have full text available (track the `annotated=N failed=M` count).
- [ ] The harness output's paired-n matches your expected overlap (if 10 papers have expert ratings and 8 have AI annotations, paired should be 8, not 10).
- [ ] Per-domain confusion matrices are sane (no ordinal rating outside the methodology's vocabulary).

## 8. Common pitfalls

- **"Some concerns" ≠ "unclear"**: RoB 2 uses `some_concerns`; QUADAS-2 uses `unclear`. Do not mix.
- **Per-outcome vs per-study**: RoB 2 is technically per-outcome (you can assess the same trial differently for different outcomes). Most reviews collapse to per-study; note which the figure is.
- **Case-control diagnostic studies**: QUADAS-2 Q1.2 penalises case-control designs. Many diagnostic accuracy studies ARE case-control; this is the single biggest source of expert-vs-algorithm disagreement in our dataset. Flag these — they're valuable for the paper's argument, not noise.
- **"Not reported" vs "N/A"**: if the paper doesn't describe blinding of outcome assessors, the signalling answer is `no` or `no-information`, not `yes`. The corresponding domain should usually be `some_concerns`, not `low`.

## 9. Next step after extraction

Once the harness report is in `reports/<review>_faithfulness_<date>/`, paste the overall + per-domain kappa numbers into the preprint draft at `docs/papers/medrxiv_quadas_rob2_assessors.md` under §3 (Results).
