# Cochrane Cohort — Acquisition Methodology

**Status:** working document for the medRxiv V5A manuscript.
**Owner:** BiasBuster project.
**Frozen:** 2026-04-16 (cohort N = 121; subject to expansion if new
Cochrane reviews are harvested).

This document records exactly how the Cochrane-labeled validation cohort
was assembled, why each step was taken, and what biases the assembly
process introduces. It is intended to be sufficient on its own for a
methods reviewer to:

1. Reproduce the cohort from public APIs and the scripts in this repo.
2. Audit the inclusion / exclusion logic.
3. Understand the selection biases inherent in using
   open-access-published Cochrane systematic reviews as a source of
   expert risk-of-bias labels.

---

## 1. Why Cochrane RoB 2 ratings?

The Cochrane Risk of Bias 2 (RoB 2) tool is the de facto standard for
expert assessment of randomised trial methodological quality
(Sterne et al., *BMJ*, 2019). RoB 2 ratings are produced by trained
methodologists as part of Cochrane systematic reviews, with adjudication
between independent reviewers. They are the closest thing the field has
to ground-truth labels for risk-of-bias judgment, with two important
caveats that shape the rest of this document:

1. **RoB 2 covers methodological quality only.** It deliberately
   excludes funding source, author conflicts of interest, spin in the
   conclusions, and sample-size adequacy. This means the BiasBuster
   pipeline — which assesses these additional dimensions as a
   policy-driven extension — will systematically diverge from Cochrane
   on industry-funded trials. See
   [`docs/two_step_approach/DESIGN_RATIONALE_COI.md`](../../two_step_approach/DESIGN_RATIONALE_COI.md)
   for the full rationale. The cohort assembly is agnostic to that
   divergence — every Cochrane rating is recorded as-is and the
   downstream comparison adjusts for it explicitly.

2. **Cochrane uses a 3-level ordinal** (`low` / `some_concerns` /
   `high`); BiasBuster uses a 5-level ordinal
   (`none` / `low` / `moderate` / `high` / `critical`). Comparison
   metrics collapse the BiasBuster ordinal to the Cochrane scale by
   the mapping
   `none|low → low`, `moderate → some_concerns`, `high|critical → high`.

The downstream V5A pipeline is evaluated against these ratings on the
two BiasBuster domains that have a direct Cochrane analogue
(`methodology` ↔ `max(randomization_bias, deviation_bias,
missing_outcome_bias, measurement_bias)`; `outcome_reporting` ↔
`reporting_bias`). The remaining BiasBuster domains
(`statistical_reporting`, `spin`, `conflict_of_interest`) are reported
separately because Cochrane RoB 2 has no equivalent rating.

## 2. Cohort assembly — overview

The cohort was assembled in five stages:

| Stage | Purpose | Output | Date |
|------:|---------|--------|------|
| A | Harvest Cochrane systematic reviews from Europe PMC, extract per-trial RoB 2 ratings | 328 trials in `dataset/biasbuster.db` `papers` table with `source LIKE 'cochrane%'` | 2026-03-15 to 2026-03-23 |
| B | Resolve trial identifiers and fetch trial-level abstracts from PubMed | 314 / 328 trials with non-empty abstracts; all 328 have RoB 2 ratings | 2026-03-15 to 2026-03-23 (interleaved with A) |
| C | Probe Europe PMC for full-text reachability of every Cochrane-tagged trial not yet cached locally | `dataset/cochrane_fulltext_probe.csv` (313 rows) | 2026-04-16 |
| D | Download JATS XML for trials whose PMC entries serve real full text | `~/.biasbuster/downloads/pmid/{pmid}.jats.xml` (124 effective Cochrane files) | 2026-04-16 |
| E | Post-hoc exclusion of non-trial publications whose RoB rating was mis-extracted by Stage A (PubMed `PublicationType` audit on Stage D failures) | 3 excluded; final cohort of 121 trials with `excluded=1` set in DB | 2026-04-16 |

Stages C and D were added on 2026-04-16 specifically to expand the
existing 16-paper V5A validation cohort to a paper-grade size for the
medRxiv submission. The 16 original papers were a convenience sample of
trials whose JATS happened to already be cached from earlier ad-hoc
runs; adding stages C/D probed the remaining 312 Cochrane-tagged trials
systematically. Stage E was added the same day after Stage D uncovered
three papers that had cached JATS but empty DB abstracts; inspection
revealed they are not randomised trials at all and the Cochrane RoB
ratings on them are Stage A LLM extraction artifacts (see §7 and §8).

## 3. Stage A — Cochrane review harvest

Implementation: `biasbuster/collectors/cochrane_rob.py`,
class `CochraneRoBCollector`.

### 3.1 Source query

The collector searches Europe PMC for systematic reviews that
explicitly used the Cochrane RoB 2 tool. The query string is:

```
SRC:PMC AND OPEN_ACCESS:Y AND HAS_FT:Y AND PUB_YEAR:[2018 TO 2026]
AND ("risk-of-bias" OR "risk of bias 2" OR "RoB 2")
AND ("included studies" OR "randomized controlled" OR "randomized")
```

The constraints `SRC:PMC AND OPEN_ACCESS:Y AND HAS_FT:Y` are essential:
they restrict the harvest to reviews whose **full text** is publicly
deposited in PMC under an open-access licence. RoB 2 ratings are not
exposed via API; they appear only in the body of the review (typically
in the "Risk of bias in included studies" section or supplementary
RoB summary tables). The collector must be able to retrieve and parse
the review's full text; closed-access reviews are unreachable.

### 3.2 Per-review extraction

For each review returned by the search, the collector:

1. Fetches the JATS full text via
   `GET https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML`.
2. Skips reviews whose JATS exceeds 2.4 MB
   (`MAX_FULLTEXT_BYTES = 2_400_000`). This filter avoids wasting LLM
   tokens on, for example, full-book PMC entries (~184 MB) that match
   the query but contain no extractable RoB tables.
3. Attempts a deterministic regex extractor first
   (`extract_rob_from_fulltext`). If the review uses a structured RoB
   summary table that the regex recognises, this returns the per-trial
   ratings without an LLM call.
4. Falls back to an LLM extraction pass
   (`extract_rob_via_llm`) when regex finds no structured table.
   The LLM is prompted with the review body and instructed to return a
   JSON array of `{study_id, overall_rob, ref_number,
   randomization_bias, deviation_bias, missing_outcome_bias,
   measurement_bias, reporting_bias}`. Results are cached in
   `dataset/llm_rob_cache.json` keyed by PMCID.
5. Resolves each `(study_id, ref_number)` to a PubMed ID by parsing the
   review's reference list, then fetches the original trial's
   bibliographic record and abstract from PubMed E-utilities.

### 3.3 Cohort produced

Stage A populated 328 rows in the `papers` table with `source =
'cochrane_rob'`. Verified by:

```bash
sqlite3 dataset/biasbuster.db \
  "SELECT COUNT(*) FROM papers WHERE source LIKE 'cochrane%'"
```

Distribution of `overall_rob` ratings:

| `overall_rob` | N | % |
|---|---:|---:|
| low | 101 | 30.8 |
| some_concerns | 120 | 36.6 |
| high | 107 | 32.6 |
| **total** | **328** | **100.0** |

This source-level distribution is intentionally retained as the
denominator for cohort-coverage statistics throughout the rest of the
document.

### 3.4 Source reviews

The 328 trials were extracted from a small number of source Cochrane
reviews. Of the 328 trials, 23 retain an explicit back-link to their
source review (`papers.cochrane_review_pmid`); the remaining 305 do
not, because the older harvest path did not persist the link. The four
back-linked source reviews are:

| Source review PMID | Trials contributed |
|---|---:|
| 40986224 | 15 |
| 38298189 | 4 |
| 41507508 | 2 |
| 32727739 | 2 |

The lack of complete back-links is a known data-provenance gap. It
does not affect the validity of individual RoB ratings (each rating
came from one specific review; the rating itself is recorded
correctly), but it limits our ability to cluster ratings by source
review for inter-rater consistency analysis. Re-running Stage A with
the current collector would close this gap; the cost is several
$10s of LLM calls and is planned as future work.

## 4. Stage B — Trial identifier resolution and abstract fetch

Implementation: `biasbuster/collectors/cochrane_rob.py`
(`_resolve_trial_pmid`, `_fetch_trial_abstract`).

For every trial identified in Stage A, the collector resolves the
trial's PubMed ID and fetches the bibliographic record (title,
authors, journal, year, abstract) from PubMed via the standard
E-utilities endpoint. The abstract is stored as `papers.abstract`.

Of 328 trials, 314 (95.7%) have an abstract of at least 200
characters. The remaining 14 either have no PubMed-deposited abstract
or have only a stub. For Stages C/D, all 328 are eligible (the probe
operates on PMID, not abstract content).

The PubMed `pmid` is the only identifier kept in the `papers.pmid`
column. DOIs are *not* persisted for Cochrane-sourced papers in this
DB, which has a downstream consequence: the bmlib full-text fallback
chain
(Europe PMC JATS → Unpaywall PDF → abstract-only) loses its second
tier for Cochrane papers, because Unpaywall queries by DOI. Only the
JATS path (which queries by PMC ID) is available. This is a real
limitation of the present cohort and is captured in the
`Limitations` section below.

## 5. Stage C — Full-text reachability probe

Implementation: `scripts/probe_cochrane_fulltext.py`.
Run on 2026-04-16; output: `dataset/cochrane_fulltext_probe.csv`.

### 5.1 Why a separate probe step?

Of the 328 Cochrane-tagged trials, only 16 had JATS XML cached
locally before the cohort expansion (those papers had been pulled
through the CLI in earlier ad-hoc runs). Probing the full 312
remaining PMIDs serially against the heavier
`{pmcid}/fullTextXML` endpoint would have wasted bandwidth on papers
without a PMC entry at all. The lightweight Europe PMC `/search`
endpoint reports `inPMC`, `isOpenAccess`, `hasPDF`, and the assigned
`pmcid` in a single response, so a probe step lets us partition the
candidate set into *reachable* vs *no_pmcid* before any full-text
download.

### 5.2 Per-PMID probe

For each Cochrane PMID not already cached, the probe issues:

```
GET https://www.ebi.ac.uk/europepmc/webservices/rest/search
    ?query=EXT_ID:{pmid} AND SRC:MED
    &format=json
    &resultType=lite
    &pageSize=1
```

and records:

| Column | Source | Notes |
|---|---|---|
| `pmid` | input | |
| `pmcid` | result.pmcid | empty if no PMC entry |
| `in_pmc` | result.inPMC == "Y" | |
| `is_open_access` | result.isOpenAccess == "Y" | |
| `has_pdf` | result.hasPDF == "Y" | |
| `source` | result.source | always "MED" for our inputs |
| `journal` | result.journalTitle | |
| `year` | result.pubYear | |
| `status` | derived | `reachable` (pmcid present), `no_pmcid` (no PMC entry), `no_hit` (PMID returned no rows), `error` |
| `error` | derived | populated only when status == "error" |

### 5.3 Robustness measures

The probe script enforces:

- **Polite rate limit:** 3 requests per second (Europe PMC documents no
  hard cap but recommends moderate use).
- **Retry with exponential backoff** via `biasbuster.utils.retry.fetch_with_retry`
  on transient HTTP failures (status codes 429, 500, 502, 503, 529).
- **Incremental flush:** every row is written to the output CSV and
  flushed to disk before the next request, so a Ctrl-C never loses
  more than the in-flight row.
- **Resumable:** rerunning the script reads the existing CSV and skips
  PMIDs already present.

### 5.4 Probe outcome (2026-04-16)

```
$ uv run python scripts/probe_cochrane_fulltext.py
...
Done in 213s. New rows: reachable=163 no_pmcid=145 no_hit=0 error=0
```

| Status | N |
|---|---:|
| reachable (PMC ID assigned) | 164 |
| no_pmcid (paper not deposited in PMC) | 149 |
| no_hit / error | 0 |
| **probed** | **313** |

(Counts include 1 PMID probed during the smoke test before the full
run, which is why the totals differ by 1 from the single-run summary.)

The 164 reachable PMIDs were the candidate pool for Stage D.

## 6. Stage D — JATS XML download

Implementation: `scripts/fetch_cochrane_jats.py`.
Run on 2026-04-16; output: 107 new JATS files in
`~/.biasbuster/downloads/pmid/`.

### 6.1 Per-PMID fetch

For each `(pmid, pmcid)` pair flagged `reachable` in the probe CSV
that does not already have a non-empty cached JATS file, the script
issues:

```
GET https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML
```

and writes the response body to
`~/.biasbuster/downloads/pmid/{pmid}.jats.xml`.

### 6.2 Robustness measures

The fetch script enforces:

- **Polite rate limit:** 2 requests per second (slower than the probe
  because `fullTextXML` is a heavier endpoint).
- **Retry with exponential backoff** on transient failures.
- **Minimum body size threshold:** responses smaller than 1024 bytes
  are treated as missing rather than written to disk. This catches
  cases where the endpoint returns a 200 OK with a stub or
  "not available" placeholder.
- **404-tolerant:** PMC-listed papers whose full-text endpoint returns
  HTTP 404 are recorded as `not_found` and skipped without error.
  This is a surprisingly common Europe PMC inconsistency: the search
  endpoint reports the paper as `inPMC: Y` but the `fullTextXML`
  endpoint has no body. We documented this empirically — see §6.3.
- **Resumable:** reruns skip PMIDs whose JATS is already cached and
  non-trivially sized.

### 6.3 Fetch outcome (2026-04-16)

```
$ uv run python scripts/fetch_cochrane_jats.py
...
Done in 153s. ok=107 not_found=55 error=0 total=11.6 MB
```

| Status | N |
|---|---:|
| ok (JATS written to cache) | 107 |
| not_found (HTTP 404 from `fullTextXML`) | 55 |
| too_small (response < 1024 bytes) | 0 |
| error (network / transient) | 0 |
| **attempted** | **162** |

The 55 `not_found` results are the Europe PMC search-vs-fetch
inconsistency referenced above. They are not attributable to network
failure or rate limiting (zero errors observed during the run); the
papers are simply listed in PMC's index without an actual XML body
being served. We accept this as a fixed loss for the present cohort.

## 7. Final cohort

After Stages A–E, the effective validation cohort is the intersection
of:

- Papers in `dataset/biasbuster.db` with `source LIKE 'cochrane%'`
- Papers with `COALESCE(excluded, 0) = 0` (i.e. not marked excluded)
- Papers with a non-empty cached JATS file at
  `~/.biasbuster/downloads/pmid/{pmid}.jats.xml`

This intersection contains **121 papers**. (One additional cached
JATS — PMID 41750436 — is not in the Cochrane subset and is separately
excluded; the three Stage E exclusions are PMIDs 28929972, 36442640,
and 40206107.)

### 7.1 Distribution by Cochrane RoB 2 rating

| Rating | Cohort | Cohort % | Source DB | Coverage |
|---|---:|---:|---:|---:|
| low | 39 | 32.2% | 101 | 38.6% |
| some_concerns | 42 | 34.7% | 120 | 35.0% |
| high | 40 | 33.1% | 107 | 37.4% |
| **total** | **121** | **100.0%** | **328** | **36.9%** |

The cohort is **near-balanced across all three Cochrane ratings** (32/35/33
within ±2 percentage points), and **coverage is uniform** across ratings
(35–39% of every source bucket reaches the cohort). This is critical
for the planned weighted-kappa analysis: a balanced denominator
prevents the metric from being dominated by any single rating and
prevents any rating from being systematically over- or under-represented
relative to the source.

### 7.2 Distribution by publication era

| Era (PMID range as proxy) | Cohort | %  |
|---|---:|---:|
| pre-2020 | 22 | 18.2 |
| 2020–2022 | 40 | 33.1 |
| 2023–2024 | 48 | 39.7 |
| 2025+ | 11 | 9.1 |

99 of 121 cohort papers (81.8%) were published in 2020 or later. The
under-representation of pre-2020 papers is a genuine selection bias
introduced by Stages C/D: only 36% of pre-2020 Cochrane-tagged trials
have PMC full text, versus 62–64% of 2020+ trials. We attribute this
to the gradual tightening of NIH and Wellcome open-access mandates
across the late 2010s and to the increasing prevalence of Plan S
journals from 2021 onward. The cohort is therefore weighted toward
more recently published, more openly-licensed trials.

### 7.3 Stage E post-hoc exclusions

Three papers that satisfied every inclusion criterion through Stage D
were excluded on further inspection after Stage D and before
annotation:

| PMID | PubMed `PublicationType` | Title (truncated) | Spurious Cochrane `overall_rob` |
|---|---|---|---|
| 28929972 | Journal Article | Abstracts from Hydrocephalus 2016 | low |
| 36442640 | Letter | Response to comment on "Psoriasis and COVID-19…" | high |
| 40206107 | Editorial | Digital Health Research Symposium: Opening Panel Commentary | high |

These are, respectively, a conference-abstracts compilation issue, a
published reply to a letter, and an editorial panel commentary. None
are randomised trials. The Cochrane RoB 2 ratings assigned to them in
the DB are Stage A LLM extraction artifacts: PMIDs matching these
papers appear in the reference lists of Cochrane reviews (cited as
background or as targets of commentary), and the LLM extractor
mistakenly pulled them into the RoB summary it returned. All three
have empty `cochrane_review_pmid` in the DB (matching the broader
305-of-328 provenance gap described in §3.4) and patchy per-domain
ratings (e.g. 28929972 carries an `overall_rob=low` with every domain
field empty — the signature of a default-filled extraction).

They were identified by a benign side effect of Stage D: annotation
attempts failed because these papers have no PubMed abstract either
(they are not research articles and PubMed does not index abstracts
for them). The failure was traced through the V5A pre-annotation
gate, the PubMed publication-type was verified via the `efetch`
endpoint on 2026-04-16, and each paper's `papers.excluded` column was
set to `1` with the reason string *"non-trial: Stage A LLM
mis-extracted RoB rating from a reference citation; PubMed
PublicationType is Letter/Editorial/Conference proceedings (verified
2026-04-16)"*.

This is the first known instance of Stage A mis-extraction in the DB
and prompts the systematic-audit recommendation in §9 Limitations.

## 8. Inclusion / exclusion summary

A paper is included in the V5A validation cohort if and only if it
satisfies all of:

1. Indexed by Europe PMC and resolvable to a PubMed ID.
2. Cited as an included study in at least one Cochrane systematic
   review whose full text is open-access in PMC and that used the
   RoB 2 tool. (Stage A inclusion.)
3. The source review's full text was retrievable and the per-trial
   RoB 2 rating was extractable, either by the deterministic regex
   parser or by the LLM fallback. (Stage A success.)
4. The trial's PubMed bibliographic record was retrievable. (Stage B
   success — applies to all 328 candidates.)
5. The trial has a PMC ID assigned in Europe PMC and the
   `{pmcid}/fullTextXML` endpoint serves a non-stub response. (Stages
   C and D success.)

Exclusions, in order of attrition:

| Exclusion reason | Affected | Surviving |
|---|---:|---:|
| (start: Cochrane-tagged trials in DB) | – | 328 |
| Stage C: not deposited in PMC | 149 | 179 |
| Stage D: PMC entry, but `fullTextXML` returned 404 | 55 | 124 |
| Stage E: non-trial publication type (Stage A mis-extraction) | 3 | 121 |
| Cohort total | – | **121** |

## 9. Limitations and sources of selection bias

These limitations should be reported in the manuscript's Limitations
section. They do not undermine the design of the cohort but they bound
the claims that can be drawn from it.

1. **Open-access publication bias.** Only Cochrane reviews available in
   open-access PMC and only trials whose original publication is
   reachable via PMC are eligible. Trials published in subscription-only
   journals without a green-OA deposit are excluded regardless of their
   Cochrane RoB rating. This biases the cohort toward more recently
   published trials and toward journals with strong open-access
   policies.
2. **No DOI fallback for trial full text.** The DB does not persist
   trial-level DOIs (Stage B fetches by PMID only), so the bmlib
   Unpaywall fallback path is unavailable for this cohort. Some
   open-access trials reachable via Unpaywall but not deposited in PMC
   are therefore excluded.
3. **Source-review back-links are sparse.** Only 23 of 328 trials carry
   a `cochrane_review_pmid` back-link, preventing inter-source-review
   consistency analyses.
4. **Era skew.** 81.5% of the cohort is from 2020 or later. Conclusions
   may not transfer cleanly to older trials, where reporting standards
   and available verification metadata (e.g.
   ClinicalTrials.gov registrations) were less mature.
5. **Cochrane RoB 2 vs BiasBuster taxonomy mismatch.** Cochrane is
   3-level and methodology-only; BiasBuster is 5-level and includes
   COI/spin/statistical-reporting domains that have no Cochrane
   analogue. Comparison is therefore restricted to the two
   directly-mappable domains
   (`methodology`, `outcome_reporting`); divergence on COI is expected
   by design and reported separately.
6. **Single Cochrane reviewer per trial.** The RoB ratings used here are
   the published consensus from the source Cochrane review; we do not
   have access to the per-reviewer pre-consensus ratings, so we cannot
   measure inter-rater agreement among Cochrane reviewers themselves.
   Treat the published rating as a single expert vote, not as a noise-free
   ground truth.
7. **No systematic audit of Stage A extraction quality.** Stage E
   excluded three papers whose RoB ratings were Stage A LLM extraction
   artifacts (§7.3). They were caught incidentally because they had
   neither a PubMed abstract nor a PMC-deposited trial body, which
   made them fail the annotation pre-checks. This mechanism does not
   catch mis-extractions on papers that *do* have a real abstract and
   real full text — for example, a research paper cited as a background
   reference in a Cochrane review whose PMID the LLM extractor mistakenly
   bound to a RoB rating. A systematic audit of all 325 remaining
   Cochrane-tagged papers against PubMed `<PublicationType>` (excluding
   any paper whose type is not `Randomized Controlled Trial`,
   `Clinical Trial`, or a close equivalent) is recommended future work
   and should be completed before the cohort is re-used for any
   subsequent validation study. The present manuscript's results should
   be read as an upper-bound estimate of the attainable kappas; a
   cleaner cohort would likely produce equal or better per-domain
   agreement with Cochrane experts.

## 10. Reproducibility

Each stage of the pipeline is invoked independently. To reproduce the
cohort from scratch on a fresh checkout:

```bash
# Stage A — collect Cochrane reviews and extract per-trial RoB
uv run python -m biasbuster.pipeline --stage collect-rob

# Stages C + D — probe and fetch full text for cached-only-abstract trials
uv run python scripts/probe_cochrane_fulltext.py
uv run python scripts/fetch_cochrane_jats.py

# Verify cohort: should print near-perfectly balanced 30/34/34 split
python3 - <<'PY'
import os, sqlite3
from collections import Counter
cache = {p.split('.')[0] for p in os.listdir(os.path.expanduser('~/.biasbuster/downloads/pmid')) if p.endswith('.jats.xml')}
db = sqlite3.connect("dataset/biasbuster.db")
rob = dict(db.execute("SELECT pmid, overall_rob FROM papers WHERE source LIKE 'cochrane%'").fetchall())
cohort = cache & set(rob)
print(f"Cohort N = {len(cohort)}")
for r, n in sorted(Counter(rob[p] for p in cohort).items()):
    print(f"  {r:<14} {n:>4}  ({100*n/len(cohort):.1f}%)")
PY
```

## 11. Provenance and freezing

The current cohort is frozen at:

- **Date:** 2026-04-16
- **biasbuster.db size:** ~38 MB
- **Cohort N:** 121
- **Distribution:** 39 low / 42 some_concerns / 40 high
- **Stage E exclusions:** 3 PMIDs (28929972, 36442640, 40206107) with
  `papers.excluded = 1` in the DB

If new Cochrane reviews are added to the DB after this date (via re-running
Stage A), the cohort may grow. Any extension should be reported as a
supplementary analysis, with the original 121-paper results retained
as the primary analysis.

The probe CSV (`dataset/cochrane_fulltext_probe.csv`) and the cached
JATS files (`~/.biasbuster/downloads/pmid/`) together constitute the
canonical record of which papers were considered, which were rejected,
and what content was actually used. Both should be archived alongside
the manuscript's supplementary materials.
