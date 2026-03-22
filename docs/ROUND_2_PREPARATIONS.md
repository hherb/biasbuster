# Round 2 Training Data Preparations

This document records every change made to the data pipeline, prompting
strategy, and export workflow between Round 1 and Round 2 of fine-tuning.

For the analysis of what went wrong in Round 1, see
[MISTAKES_ROUND_1_AND_FIXES.md](MISTAKES_ROUND_1_AND_FIXES.md).

---

## 1. Unified Severity Definitions (Critical Fix)

**Problem**: Round 1 used different severity definitions for annotation vs
training.  The annotation prompt had zero severity boundary definitions; the
training prompt had detailed per-domain boundaries.

**Change**: Created `prompts.py` as a single source of truth.

| Component | Before | After |
|-----------|--------|-------|
| `annotators/llm_prelabel.py` | 150-line inline prompt, no severity boundaries | Imports `ANNOTATION_SYSTEM_PROMPT` from `prompts.py` |
| `export.py` | 90-line inline prompt with boundaries | Imports `TRAINING_SYSTEM_PROMPT` from `prompts.py` |
| `schemas/bias_taxonomy.py` | Lazy import from `export.py` | Lazy import from `prompts.py` |

Both prompts now share identical:
- **Severity scale** (NONE/LOW/MODERATE/HIGH/CRITICAL with definitions)
- **Per-domain severity boundaries** (statistical, spin, outcome, COI, methodology)
- **Verification database recommendations**
- **Retraction severity floor principle** (new — see below)
- **Calibration note** (enhanced — explicitly states most RCTs are LOW/MODERATE)

**Impact**: Every annotation now uses the same severity criteria that the
model will be trained on.  The mismatch that caused ground truth corruption
in Round 1 is eliminated.

**Reproduction**: Annotations must be re-generated with the unified prompt.

---

## 2. Retraction Reason Classification

**Problem**: Round 1 had 136/368 retracted papers labelled as NONE severity
because the annotator assessed abstract content only, while many papers were
retracted for data fabrication or unreliable results (invisible in the abstract).

**Changes**:

### 2a. Structured retraction reasons from Retraction Watch CSV

`seed_database.py` step `enrich-rw` downloads the full Retraction Watch CSV
from Crossref Labs (~69k entries) and enriches the `retraction_reasons` field
in the papers table with the controlled RW vocabulary (~111 categories).

Before: all 799 retracted papers had only the generic `["Retraction"]` label.
After: 676/767 papers have structured reasons like `["Falsification/Fabrication
of Data", "Misconduct by Author"]`.

### 2b. Retraction classifier (`enrichers/retraction_classifier.py`)

Maps retraction reason strings to severity floors:

| Floor | Retraction reasons | Coverage |
|-------|-------------------|----------|
| CRITICAL | Paper mill, fabrication, falsification, misconduct | 24% |
| HIGH | Unreliable results, data concerns, manipulation | 36% |
| MODERATE | Statistical errors, peer review concerns, unknown | 28% |
| None (no floor) | Duplication, plagiarism, referencing, author objection | 11% |

### 2c. Retraction context in annotation prompt

`annotators/__init__.py` `build_user_message()` now passes the classified
retraction reason and severity floor to the annotator LLM:

```
RETRACTION CLASSIFICATION: This paper was retracted. Reason category:
unreliable results. Severity floor: HIGH. The overall severity MUST be
at least HIGH, regardless of how the abstract reads.
```

### 2d. Severity floor enforcement at export time

`export.py` `_apply_retraction_floors()` bumps annotation severity up to
the retraction floor before generating training data.  The original
annotation in the database is never modified.

**Reproduction**: Run `seed_database.py --step enrich-rw` after collection.

---

## 3. Cochrane RoB Expansion

**Problem**: Round 1 had only 8 Cochrane RoB papers, all with empty abstracts.
Expert RoB judgments are the gold standard for severity calibration.

**Changes**:

### 3a. Broader systematic review search

The Cochrane collector (`collectors/cochrane_rob.py`) now searches:
- Open-access PMC systematic reviews that used RoB 2 (not just Cochrane reviews)
- Wider year range (2015-2026, was 2018-2026)
- More reviews per domain (200 total, was 50)

### 3b. Multi-layer PMID resolution

Study ID resolution uses up to five strategies, applied per-review (not
batched at the end) so that `on_result` can save incrementally:

| Layer | Strategy | How it works |
|-------|----------|-------------|
| 1a | **Bracket-reference lookup** | LLM returns `ref_number` (e.g. 28); matched to `<ref>` XML element by `<label>` or `id="CR28"` to read `<pub-id pub-id-type="pmid">` directly |
| 1b | **Author+year from refs** | Normalised first-author surname + year matched against `<ref>` list (`<name><surname>` + `<year>`) |
| 1c | **Surname-only from refs** | If exactly one `<ref>` matches the surname (no year in study ID), use that ref |
| 2 | **DOI → PMID** | For studies with a DOI but no PMID from layer 1, `esearch` with `DOI[DOI]` |
| 3 | **Author+year PubMed search** | Relaxed PubMed search (no RCT publication type filter) as final fallback |

Study IDs are normalised via `_normalize_study_id()` which handles
`"Author et al., 2020 [28]"`, `"Author et al. [28]"`, `"Author 2020a"`,
etc. by stripping "et al." and extracting the bracket reference number.

Resolution rate improved from 22% to 52%.

### 3b-fix. "et al." and bracket-reference handling (2026-03-22)

**Problem**: Many systematic reviews cite studies as `"Manwaring et al. [28]"`
rather than `"Manwaring 2024"`.  All three original resolution layers used a
regex (`(\w+)\s+(year)`) that failed on "et al." — the year was never
extracted, and the bracket reference number was ignored entirely.  PMIDs
sitting in the XML `<ref>` elements went unmatched.

**Changes**:

- **LLM prompt** now requests `ref_number` (the bracket citation number)
- **`_normalize_study_id()`** helper parses all variants into
  `(surname, year, ref_number)`, tolerating "et al." and bracket refs
- **`_is_valid_study_id()`** accepts study IDs with only a ref number
  (no year needed if bracket ref is present)
- **`extract_included_study_refs()`** extracts `ref_label` from XML
  `<label>` elements or `<ref id="CR28">` attributes
- **`_resolve_pmids_from_refs()`** rewritten with three sub-strategies:
  bracket-ref lookup (highest priority), author+year, surname-only
- **`resolve_study_pmids()`** uses normalised study IDs for PubMed search

### 3c. LLM-based RoB extraction (fallback)

When regex-based extraction finds no study-level RoB judgments in a review,
the full text is sent to DeepSeek reasoner for structured extraction.

Uses **chunk & map-reduce** (never truncation): the document is split into
overlapping chunks, each sent to the LLM independently, results merged
and deduplicated.

### 3d. Abstract fetching for Cochrane papers

`seed_database.py` step `fetch-abs` fetches abstracts from PubMed for any
paper with an empty abstract.  Cochrane papers now have real abstracts
for annotation.

### 3e. Cochrane RoB context in annotation prompt

`annotators/__init__.py` `build_user_message()` now passes Cochrane RoB
domain judgments when available:

```
Cochrane RoB 2 expert assessment: overall=high, randomization=some_concerns,
measurement=high. Use these expert judgments to calibrate your severity ratings.
```

### 3f. LLM extraction cache and targeted re-resolution (2026-03-23)

**Problem**: LLM-based RoB extraction costs tokens (DeepSeek reasoner).  When
PMID resolution fails, the extracted study IDs are discarded — re-running the
pipeline re-spends tokens on the same reviews.  In the initial run, 70/132
extracted studies were lost to failed resolution.

**Changes**:

- **LLM result cache** (`dataset/llm_rob_cache.json`): `extract_rob_via_llm()`
  now caches raw LLM responses keyed by PMCID.  On re-run, cached reviews skip
  the LLM entirely and go straight to resolution.  Cache is saved incrementally
  after each LLM call.
- **`reprocess_rob.py`**: Standalone script that re-processes specific reviews
  with the improved resolution code.  Can parse a pipeline log file to
  automatically identify reviews with incomplete resolution.

```bash
# Re-process reviews with incomplete resolution from a prior run:
uv run python reprocess_rob.py --from-log log.txt

# Or target specific PMCIDs:
uv run python reprocess_rob.py PMC12987355 PMC11152306 PMC12942296
```

The script fetches the review XML (free), uses cached LLM results (no tokens),
runs multi-layer PMID resolution with the improved code, saves new papers to
the database, and fetches their abstracts from PubMed.

**New CLI**: `python pipeline.py --stage collect-rob` runs Cochrane/RoB
collection only (no retraction watch or PubMed RCTs).

**Reproduction**: Run `pipeline.py --stage collect-rob` then `seed_database.py --step fetch-abs`.
After a run, recover lost studies with `reprocess_rob.py --from-log log.txt`.

---

## 4. Retraction Notice Cleanup

**Problem**: 33 bare retraction notices (abstract = "This article has been
retracted…") were in the database, adding noise.

**Change**: `seed_database.py` step `clean` permanently removes bare retraction
notices from the papers table using the same `is_retraction_notice()` function
the annotation stage already used for filtering.

Before: 799 retracted papers.  After: 767 (32 notices removed).

**Reproduction**: Run `seed_database.py --step clean` after collection.

---

## 5. Natural Severity Distribution (No Oversampling)

**Problem**: Round 1 duplicated ~23 HIGH and ~10 CRITICAL examples to 5%
of training data via `oversample_rare_severities()`.  This caused
memorization and train/test distribution mismatch.

**Change**: Removed the oversampling call from `export_dataset()`.  Training
uses the natural severity distribution.

Also replaced random shuffle-and-split with **stratified splitting** by
severity class, ensuring proportional representation in train/val/test.

Export metadata now includes `severity_distribution` and
`train_severity_distribution` for tracking.

**Reproduction**: Automatic — export always uses natural distribution now.

---

## 6. Evidence-Grounded Thinking Chains

**Problem**: Round 1 thinking chains used formulaic templates that didn't
reference specific evidence from the annotation.

**Changes** to `export.py` `build_thinking_chain()`:

- **Concern counting**: "2 concern(s) identified: relative_only + subgroup_emphasis"
- **Evidence quotes**: References `evidence_quotes` from the annotation when available
- **Boundary citation**: "Per boundary definition: reader cannot assess clinical significance"
- **Cross-domain calibration**: "3/5 domains have concerns; highest is MODERATE"
- **Retraction reasoning**: "This paper was retracted (data fabrication). Severity floor: CRITICAL"

---

## 7. Pipeline Changes Summary

### New files

| File | Purpose |
|------|---------|
| `prompts.py` | Single source of truth for severity boundaries |
| `enrichers/retraction_classifier.py` | Retraction reason → severity floor mapping |
| `seed_database.py` | Reproducible post-collection cleanup (3 steps) |
| `reprocess_rob.py` | Targeted re-resolution of failed Cochrane reviews (parses log, uses LLM cache) |
| `docs/MISTAKES_ROUND_1_AND_FIXES.md` | Round 1 post-mortem |
| `docs/ROUND_2_PREPARATIONS.md` | This document |

### Modified files

| File | Change |
|------|--------|
| `annotators/llm_prelabel.py` | Import prompt from `prompts.py` |
| `annotators/__init__.py` | Retraction classification + Cochrane RoB context in `build_user_message()` |
| `export.py` | Import prompt from `prompts.py`; remove oversampling; stratified split; retraction floors; evidence-grounded thinking chains |
| `schemas/bias_taxonomy.py` | Updated lazy import |
| `pipeline.py` | Added `seed` and `collect-rob` stages |
| `collectors/cochrane_rob.py` | Broader search, 5-layer PMID resolution (bracket-ref + author+year + surname-only + DOI + PubMed search), LLM extraction with `ref_number` + result caching, study ID normalisation ("et al." / bracket / initials handling), junk filter |
| `config.example.py` | Updated defaults (deepseek-reasoner, cochrane_max_reviews=200, cochrane_min_year=2015) |
| `CLAUDE.md` | Updated architecture docs, commands, module descriptions |

---

## 8. Reproducing the Round 2 Dataset From Scratch

```bash
# 1. Set up environment
uv sync
cp config.example.py config.py
# Edit config.py: add API keys (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, NCBI_API_KEY)
# Set crossref_mailto to your email

# 2. Collect raw data
uv run python pipeline.py --stage collect        # retraction watch + PubMed RCTs + Cochrane

# 3. Expand Cochrane/RoB collection (LLM-assisted, uses DeepSeek reasoner)
uv run python pipeline.py --stage collect-rob

# 4. Seed: enrich retraction reasons, fetch missing abstracts, clean notices
uv run python seed_database.py                    # all 3 steps

# 5. Heuristic enrichment
uv run python pipeline.py --stage enrich

# 6. Annotate with unified prompt (severity boundaries in BOTH annotation and training prompts)
uv run python pipeline.py --stage annotate

# 7. Export for training (natural distribution, no oversampling, stratified split)
uv run python pipeline.py --stage export

# Or run everything in one shot (collect → seed → enrich → annotate → export):
uv run python pipeline.py --stage all
```

### Key differences from Round 1 reproduction

| Step | Round 1 | Round 2 |
|------|---------|---------|
| Annotation prompt | No severity boundaries | Full severity boundaries from `prompts.py` |
| Retraction reasons | Generic "Retraction" for all | Structured RW vocabulary (~111 categories) |
| Retraction handling | Assess abstract content only | Severity floors based on retraction reason |
| Cochrane data | 8 papers, no abstracts | 23+ papers with abstracts, expert RoB context |
| Export oversampling | 5% minimum per severity class | Natural distribution (no oversampling) |
| Export split | Random shuffle | Stratified by severity class |
| Thinking chains | Formulaic templates | Evidence-grounded with concern counts |
| Cochrane RoB extraction | Regex only | Regex first, LLM fallback (chunk & map-reduce) |
| Cochrane PMID resolution | Author+year regex only | 5-layer: bracket-ref, author+year, surname-only, DOI, PubMed search |

### Expected outcomes

- **Calibration error**: < 0.5 (was 0.87)
- **Weighted kappa**: > 0.3 (was 0.08-0.16)
- **Severity distribution**: realistic (most papers LOW/MODERATE, no artificial inflation)
- **Ground truth quality**: retracted papers correctly rated, Cochrane anchors, unified boundaries
