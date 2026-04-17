# Cochrane Corpus Integrity Forensics

**Date:** 2026-04-17
**Scope:** how the `dataset/biasbuster.db` Cochrane corpus came to contain
untrustworthy rating-to-PMID mappings.
**Status:** findings inform the rebuild plan; no fixes applied in this
document.

This is an incident report, not a defence. The goal is to document the
failure modes precisely enough that (a) the next harvest does not repeat
them and (b) the lessons transfer to any future pipeline that distils
expert judgements from open-access scientific literature.

---

## 1. Observed symptoms

The Phase 1 fidelity audit
(`scripts/phase1_fidelity_audit.py`, run 2026-04-17) showed that of 23
papers in the DB carrying a `cochrane_review_pmid` back-link, **all 23
back-link to documents that are not primary Cochrane systematic
reviews**:

| Review PMID | Back-linked trials | What the document actually is |
|---|---:|---|
| 40986224 | 15 | World Congress on Osteoporosis 2025 — conference proceedings book |
| 38298189 | 4 | Overview of 17 systematic reviews (PRIO-harms) — no primary RoB assessments |
| 32727739 | 2 | Protocol paper for a planned review — the review itself is not yet done |
| 41507508 | 2 | One actual systematic review using RoB 2 |

The LLM extraction cache (`dataset/llm_rob_cache.json`) contains **zero
entries** for any of the four PMCIDs those PMIDs resolve to, meaning
the cached LLM path did not produce the ratings now in the DB. Separately,
the earlier PublicationType audit
(`scripts/audit_publication_types.py`) found 28 non-trial papers in the
active cohort — case reports, reviews, letters, editorials — all
carrying a Cochrane `overall_rob` rating.

Combined: the DB carries hundreds of ratings whose provenance we cannot
reconstruct from the cached artefacts, and the back-links we do have
point at non-review documents. The ratings exist, but their connection
to a real Cochrane RoB 2 assessment cannot be established from the data
we have.

## 2. How the pipeline was supposed to work

Per `biasbuster/collectors/cochrane_rob.py` (`collect_rob_dataset` at
line 989), the intended dataflow was:

1. Query Europe PMC for Cochrane reviews that use RoB 2.
2. For each returned review, fetch the full-text JATS.
3. Run a regex extractor over the text to pull `(study_id,
   overall_rob)` tuples from RoB statements.
4. If regex returns nothing, run an LLM fallback extractor that returns
   structured JSON tuples including per-domain ratings.
5. Parse the review's reference list to build
   `(author, year) → PMID` and `[ref_num] → PMID` lookups.
6. Resolve each `study_id` to a trial PMID via bracket-ref, then
   author+year, then a PubMed author+year search as a final fallback.
7. Persist the trial with its ratings and a back-link to the source
   review.

Each step alone is defensible; the corruption arises from how the
defects in each step compose.

## 3. Defects identified

Each of the following is independently sufficient to produce wrong
ratings. In practice they compose and amplify each other.

### 3.1 The Europe PMC search query is too permissive

At `cochrane_rob.py:261-267`:

```
SRC:PMC AND OPEN_ACCESS:Y AND HAS_FT:Y AND PUB_YEAR:[2018 TO 2026]
  AND ("risk-of-bias" OR "risk of bias 2" OR "RoB 2")
  AND ("included studies" OR "randomized controlled" OR "randomized")
```

Any open-access PMC document from 2018-onward that mentions a
bias-related phrase and a randomisation-related phrase matches. This
includes — demonstrably — all four documents in the back-linked set:

- A systematic-review **protocol** mentions RoB 2 as part of its
  methods plan (PMID 32727739).
- An **overview of reviews** mentions RoB 2 to summarise how the
  underlying reviews assessed their trials (PMID 38298189).
- A **conference proceedings book** contains hundreds of abstracts,
  some of which mention bias in methods or discussion sections
  (PMID 40986224).
- An actual systematic review using RoB 2 (PMID 41507508).

The query carried no constraint on PubMed PublicationType or
source-journal identity (e.g. `ISSN:1469-493X` for the Cochrane
Database of Systematic Reviews). As a result, the search returned a
population of candidate documents of which only a minority were the
kind of document the downstream extractor assumed.

### 3.2 The regex extractor is greedy and context-free

At `cochrane_rob.py:510-522`:

```python
rob_patterns = [
    r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)\s+(?:was|were)\s+'
    r'(?:judged|rated|assessed|considered)\s+...',
    r'(\w+)\s+risk\s+of\s+bias\s*\(([A-Z][a-z]+[^)]*(?:19|20)\d{2}[a-z]?)\)',
    r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)\s*[:–\-]\s*'
    r'(low|high|unclear|some concerns)\s+risk',
    r'(high|low|unclear)\s+risk\s+of\s+bias[^.]*?'
    r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?)',
    r'([A-Z][a-z]+\s+(?:19|20)\d{2}[a-z]?).{0,200}?overall.{0,50}?'
    r'(low|high|some\s+concerns)',
]
```

These patterns match text anywhere in the document. They have no check
for structural context — table vs prose, RoB-2-summary vs literature-
review-citation. Pattern 4 is particularly dangerous: it scans up to a
sentence boundary for a `(level) risk of bias ... (Name Year)` sequence,
which is a normal way to cite a prior study's published limitation
("Smith 2020 reported that previously-used assays had high risk of bias
in older populations"). In a conference-proceedings body or an overview
paper, many such sentences legitimately exist and are *not* Stage A's
own RoB judgements.

### 3.3 The PMID resolver falls back to fuzzy bibliography matching

At `cochrane_rob.py:941-945`:

```python
# Strategy 3: surname-only (only if unambiguous)
candidates = refs_by_author.get(author_lower, [])
if len(candidates) == 1:
    if _apply_ref(a, candidates[0]):
        matched += 1
```

When a regex match produces `("Smith", "high")` without a bracket
reference number, and the document's bibliography contains exactly one
reference whose first author is any surname "Smith", the match is
applied regardless of year, title, or publication type. In a Cochrane
review's reference list this fallback is usually benign (the included
studies are the overwhelming majority of Smith-type surnames). In a
conference proceedings or overview paper the bibliography reflects
*all* the underlying reviews' bibliographies merged together — hundreds
to thousands of references. Surname-uniqueness becomes a coincidence,
not a mapping.

### 3.4 Author+year PubMed search silently picks the first hit when
multiple match

At `cochrane_rob.py:854-861`:

```python
if len(pmids) == 1:
    assessment.pmid = pmids[0]
elif pmids:
    assessment.pmid = pmids[0]  # Take first as candidate
    logger.debug(f"Multiple PMID candidates for {assessment.study_id}: {pmids}")
```

For any prolific author, `author+year` returns many PMIDs. The pipeline
silently keeps the first (which is PubMed's most-recent-first ordering
— not necessarily the one the review actually assessed). The disambig-
uation step referenced in the inline comment ("would need title matching")
was never implemented. The ambiguity is logged at DEBUG level, so it
never surfaced in production logs.

### 3.5 The oversized-document guard was added after the corpus was
harvested

Git history:

| Commit | Date | Change |
|---|---|---|
| 8d6f7dd | 2026-03-15 | DB migrated to SQLite (pre-existing JSONL harvest imported) |
| 926a816 | 2026-03-23 | "Consolidate Cochrane RoB persistence and fix silent data loss" |
| **1534197** | **2026-03-24** | **"Add oversized document guard and update docs for current DB state"** |

The `MAX_FULLTEXT_BYTES = 2_400_000` guard at `cochrane_rob.py:160-164`
was added on 2026-03-24. The Cochrane corpus was harvested between
2026-03-15 and 2026-03-23 (per the DB's `collected_at` range). The
5.9 MB World Congress on Osteoporosis proceedings (PMID 40986224) was
therefore processed without the guard: the full 5.9 MB of conference
abstracts and bibliographies was fed to the regex extractor and the
reference resolver.

### 3.6 No post-harvest publication-type validation

The trial PMIDs produced by §3.3 and §3.4 were persisted without any
final check of what PubMed actually says that PMID is. Case reports,
reviews, letters, and editorials are kept verbatim. Our PublicationType
audit (2026-04-17) found 28 non-trial papers in the active cohort and
202 ambiguous — every one of which passed Stage A without objection.

### 3.7 Back-link and rating fields were populated without an atomic
invariant

The `cochrane_review_pmid` column is attached to an assessment at
`cochrane_rob.py:1083-1086`, *before* PMID resolution runs:

```python
for a in assessments:
    a.cochrane_review_pmid = review.get("pmid", "")
    a.cochrane_review_doi = review.get("doi", "")
    a.cochrane_review_title = review.get("title", "")
```

If the review was a protocol, an overview, or a proceedings, the trial
inherits that review's PMID as its "source" even when the actual rating
tuple originated via regex patterns that had nothing to do with the
review's own judgement. The DB has no invariant enforcing that a paper
carrying a rating must also carry a verifiable source review; the
`upsert_cochrane_paper` method writes whichever fields are present.

## 4. How the defects compose

Putting §3.1 – §3.7 together reconstructs the observed corruption:

1. **Search over-match (§3.1)** admits documents that are not primary
   systematic reviews — protocols, overviews, conference proceedings.
2. **No size guard yet (§3.5)** lets the harvester process a 5.9 MB
   conference proceedings end to end.
3. **Context-free regex (§3.2)** treats every "(Name Year)" mention
   near a "risk of bias" token as an own-review judgement, producing
   many spurious `(study_id, overall_rob)` tuples from unrelated
   abstracts inside the proceedings.
4. **Fuzzy reference matching (§3.3)** then binds each spurious tuple
   to a PMID by surname-unique lookup in the proceedings' combined
   bibliography of thousands of references.
5. **Silent PubMed fallback (§3.4)** catches the rest — picking an
   arbitrary "author (year)" paper from PubMed as the trial identity.
6. **No publication-type check (§3.6)** keeps case reports, letters,
   and editorials in the output, because the pipeline never asks PubMed
   "is this actually a trial?".
7. **Atomic invariant missing (§3.7)** records the document's PMID as
   the "source review" even though the ratings never came from that
   document's own RoB 2 table.

The result is a corpus where roughly 9% of rows are definitely wrong
(non-trials), a further ~25% are likely wrong (PubMed-ambiguous
PublicationType, plus the survivors of the fuzzy-binding steps), and
the remaining rows are correct *but indistinguishable* from the wrong
ones given only the data the DB persists. The latter is the critical
point: without additional provenance we cannot, by inspection of the
DB alone, identify which rows are trustworthy.

## 5. What survived and is still usable

Not everything is compromised. Five pieces of work remain useful as-is:

1. **The V5A pipeline and prompts** (`biasbuster/assessment/`,
   `biasbuster/prompts_v5a.py`). These are independent of the
   Cochrane corpus and were validated on cached full-text JATS files
   that are real RCTs (including 16 papers that were individually
   sanity-checked during earlier rounds).
2. **The Sonnet and gemma4 V5A annotations for the 121-paper cohort.**
   The annotations themselves are reproducible: if a paper's JATS
   is a real trial, the V5A output is a valid bias assessment of
   that trial, independent of whether the DB's `overall_rob`
   column was ever correct. These annotations support inter-model
   agreement analysis (Sonnet vs gemma4, κ = +0.562 on N = 121,
   +0.420 on the 26 confirmed trials) without any dependence on
   Cochrane labels.
3. **The PublicationType audit output**
   (`dataset/cochrane_pubtype_audit.csv`). A clean, PMID-keyed
   classification we can use to filter the cohort before the next run.
4. **The full-text JATS cache.** Every JATS XML in
   `~/.biasbuster/downloads/pmid/` is canonical Europe PMC content
   and unaffected by the Stage A defects. A rebuilt Stage A would
   re-use these files without re-fetching.
5. **The manuscript sidecars we have already drafted** —
   `METHODOLOGY.md`, `PRIOR_APPROACHES.md`, this `FORENSICS.md`.
   The data has changed; the architecture and prior-work narratives
   have not.

## 6. Lessons

Five rules for the rebuild. Each is a direct consequence of a defect
in §3.

### 6.1 Restrict the source document set at query time

Do not search for "anything that mentions RoB 2". Search for documents
that are structurally Cochrane reviews: either restrict to the
`Cochrane Database of Systematic Reviews` ISSN, or filter Europe PMC
results to PubMed PublicationType `Systematic Review` / `Meta-Analysis`,
or both. Every non-review document admitted into the harvest is a
vector for §3.2–§3.4 errors.

### 6.2 Extract only from verifiably-structured RoB tables

The regex patterns in §3.2 proved to match any prose that happens to
discuss bias. A rewritten Stage A should:

- Locate RoB 2 summary tables *structurally*: a JATS `<table>` element
  whose header row mentions all five RoB 2 domains, or a section
  heading like "Risk of bias in included studies".
- Require each extracted row to carry all six rating fields (overall +
  five domains) — a partial row is a non-row.
- Store the table's row index and the surrounding heading as
  provenance alongside each extracted tuple.

If no structured table is found, return zero entries and let the LLM
path do the work — do not fall through to prose-scraping.

### 6.3 Bind rating tuples to PMIDs with verifiable evidence

Every `(study_id, rating) → PMID` resolution should produce a
provenance record including:

- The reference list entry the PMID came from (raw citation string,
  authors, year, title).
- The matching strategy that succeeded (bracket, author+year,
  author-only, PubMed search).
- A confidence level derived from that strategy.

Rows produced by author-only (§3.3) or "first of many PubMed
candidates" (§3.4) should never reach a final corpus without human
or LLM adjudication against the trial's title. The comment at
`cochrane_rob.py:857` — "would need title matching for disambiguation"
— is the feature, not an afterthought.

### 6.4 Validate the PMID with PubMed before persistence

Immediately after PMID resolution, fetch that PMID's PublicationType
and ArticleTitle. Reject if PublicationType is explicitly non-trial
(`Letter`, `Editorial`, `Comment`, `News`, `Congresses`,
`Meeting Abstracts`, `Book`, `Case Reports`, `Review`, `Systematic
Review`, `Meta-Analysis`). Log and continue for ambiguous cases. The
audit we ran post-hoc (`scripts/audit_publication_types.py`) should
run *during* harvest.

### 6.5 Enforce the rating-provenance invariant in the DB

The `papers` table should enforce: a non-empty `overall_rob` implies
a non-empty `cochrane_review_pmid` and a non-empty
`rob_provenance` (JSON blob containing the extraction-table
coordinates, the resolution strategy, and the resolved reference
citation). Rows that cannot satisfy this invariant cannot be
persisted. The §3.7 footgun is not a behaviour to preserve.

### 6.6 (Meta-lesson) Validate a small sample manually before scaling

The defects in §3.2–§3.4 would have been visible on 10 spot-checked
papers. The harvest produced 328 papers in a week and was not
sanity-checked on a small sample at the author level before being
fed to downstream stages. Any future harvest that stages more than
~20 papers without a documented manual sample check is repeating
the same error.

## 7. Implications for the manuscript

The corpus is not usable as a ground truth for Cochrane-agreement
numbers as currently persisted. Two paths forward are both
defensible:

1. **Rebuild the corpus** following §6. Cost: several days of coding
   and LLM time, but produces a dataset that can be shared openly and
   reused. The manuscript's Cochrane comparison becomes the primary
   analysis again, but on a trustworthy cohort.
2. **Pivot the manuscript's headline** to the Sonnet-vs-gemma4
   inter-model agreement result (κ = +0.562 overall, +0.767 on COI),
   which does not depend on Cochrane labels. Cochrane agreement
   becomes a (caveated) secondary analysis, or moves to future work.

Path 1 is the paper the project originally intended. Path 2 is a
defensible shorter paper with a smaller claim. A combined approach —
pivot now, rebuild in parallel, submit path 2 as a preprint and path 1
as a follow-up — is also viable. The choice depends on time budget
and on how much of the existing Cochrane cohort can be salvaged after
a stricter re-harvest.

---

## Appendix — Git history used in this forensic

```
8d6f7dd 2026-03-15 Migrate from JSONL file storage to SQLite database backend
96a57be 2026-03-15 Enforce golden rules: retry, constants, docstrings, tests, file splits
22eeee8 2026-03-22 Round 2 training data pipeline: fix severity calibration
f966641 2026-03-22 Fix Cochrane RoB PMID resolution for "et al." and bracket-reference citations
d818b73 2026-03-22 Handle author initials in study ID parsing
43b9c1f 2026-03-23 Add LLM extraction cache and targeted re-resolution script
a1185c3 2026-03-23 Handle DeepSeek reasoner empty content and mixed reasoning+JSON responses
c91aff8 2026-03-23 Fix greedy JSON extraction from mixed reasoning+JSON responses
5eaa52c 2026-03-23 Batch LLM cache writes and auto-discover columns in seed export
9752b18 2026-03-23 Add expert RoB alignment report, Cochrane domain extraction, and annotated snapshot export
151ebed 2026-03-23 Skip non-dict entries in LLM RoB extraction results
886ac7f 2026-03-23 Add retry logic to LLM RoB extraction and bump max_tokens to 16K
e6a3a83 2026-03-23 Use config.deepseek_max_tokens instead of hardcoded magic number
862ff85 2026-03-23 Eliminate magic numbers and deduplicate seed_export.py
926a816 2026-03-23 Consolidate Cochrane RoB persistence and fix silent data loss
1534197 2026-03-24 Add oversized document guard and update docs for current DB state  ← guard added AFTER harvest
dbdee11 2026-03-24 Use ft_resp.content for size check to avoid re-encoding
fc51b94 2026-03-24 Fix UnboundLocalError on chunks variable for cached LLM results
2a23a16 2026-03-24 Fix AttributeError when LLM returns null for study_id or overall_rob
ffc5af3 2026-04-05 Move all packages under biasbuster/ namespace for PyPI compatibility
```
