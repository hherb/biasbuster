# Cochrane Corpus Rebuild — Design

**Status:** design. Implementation starts after this document is signed off.
**Inputs:** lessons from [`FORENSICS.md`](./FORENSICS.md); inclusion rule
("all five RoB 2 domain ratings plus the overall rating present") from
2026-04-17 design discussion.
**Scope:** rebuild of Stage A of the BiasBuster pipeline
(`biasbuster/collectors/cochrane_rob.py`) plus the DB invariant
enforcement and the new validation gate. The V5A assessment pipeline
is out of scope — this design rebuilds ground truth only.

---

## 1. Goals and success criteria

### 1.1 Goals

1. Produce a Cochrane-derived validation corpus for the BiasBuster
   manuscript whose every row carries a verifiable provenance chain
   back to a real Cochrane RoB 2 assessment inside a real systematic
   review.
2. Make it structurally impossible for the Stage A defects catalogued
   in FORENSICS.md §3 to reappear — not by diligence, by invariants.
3. Keep the rebuild small in scope: rewrite Stage A and the DB
   invariant; reuse everything else.

### 1.2 Success criteria

Measurable at the end of rebuild Phase 4 (see §11):

- **Inclusion completeness**: every row in `papers` marked with a
  non-empty `overall_rob` must have all five RoB 2 domain ratings
  populated, a non-empty `cochrane_review_pmid`, and a non-empty
  `rob_provenance` record pointing to the specific review section
  the rating came from.
- **Manual sample pass rate**: the 20-paper pre-scaling check (§8)
  finds ≥19 of 20 rating tuples match the source review when spot-
  checked by hand.
- **Publication-type integrity**: 0 non-trial papers
  (Letter / Editorial / Comment / News / Congresses / Meeting
  Abstracts / Book / Case Reports / Review / Systematic Review /
  Meta-Analysis) in the final cohort, verified by a post-harvest
  `scripts/audit_publication_types.py` run.
- **Cohort size**: ≥80 unique trials with all five domain ratings.
  Smaller is acceptable if the manual check confirms cleanliness;
  larger is better for kappa confidence intervals.

### 1.3 Explicit non-goals

- Expanding training data volume. The earlier harvest's size came from
  breadth; this rebuild prioritises fidelity over volume.
- Producing a reusable public dataset. Publishing the corpus as a data
  artefact is a separate decision.
- Rebuilding the V5A annotation side. V5A annotations are PMID-keyed
  and independent of Cochrane label quality.

## 2. Inclusion criteria — the litmus test

A trial is admitted to the new cohort if and only if all of the
following are true:

1. The trial's RoB 2 assessment was extracted from a document that is
   a **completed systematic review published in the Cochrane Database
   of Systematic Reviews** (ISSN 1469-493X) and whose RoB table uses
   **RoB 2 domains** (not RoB 1). See §3 for the query constraint and
   §4.1 for the header-row check that enforces this.
2. The extraction produced a complete 6-field rating tuple:
   `(overall, randomization, deviations, missing_outcome,
   measurement, reporting)`. Any tuple missing one or more fields is
   rejected outright — no "partial" rows. This is the litmus test.
2a. The extraction came from a table that has a per-outcome granularity
   marker — an `Outcome` column, per-outcome row grouping, or an
   explicit primary/secondary label. Pattern-4 tables (single row per
   study with no outcome column) are rejected as methodologically
   ambiguous. See §10 Q6.
3. Each rating value is one of the three RoB 2 levels
   (`low` / `some concerns` / `high`); anything else (empty, `unclear`,
   `n/a`) is rejected.
4. The trial's PMID resolution used bracket-reference lookup or
   surname-plus-year-plus-title-match (§5). Surname-only matches and
   "first of many" PubMed search results are rejected.
5. The resolved PMID's PubMed `PublicationType` is compatible with
   "trial" (§6). Explicit non-trial types disqualify the row.
6. The paper's full-text JATS is already cached under
   `~/.biasbuster/downloads/pmid/{pmid}.jats.xml` or is fetched
   successfully via the existing `scripts/fetch_cochrane_jats.py`
   pipeline.

Any row failing any of these six rules is dropped at the point the
rule is evaluated, logged to
`dataset/cochrane_rebuild_rejects.jsonl`, and never reaches
`papers.overall_rob`.

## 3. Source document query

Replace the existing Europe PMC query at
`biasbuster/collectors/cochrane_rob.py:261-267` with:

```
JOURNAL:"Cochrane Database Syst Rev"
  AND ISSN:1469-493X
  AND SRC:PMC
  AND OPEN_ACCESS:Y
  AND HAS_FT:Y
  AND PUB_TYPE:"Systematic Review"
  AND PUB_YEAR:[2018 TO 2026]
```

Design notes:

- `JOURNAL` + `ISSN` together are defence-in-depth: if Europe PMC's
  journal normalisation lags (they do cite CDSR under slightly
  different names across years), the ISSN still matches.
- `PUB_TYPE:"Systematic Review"` is added as a third filter specifically
  because protocol papers are also published in CDSR. The Cochrane
  protocol series does *not* carry that PublicationType — a completed
  review does.
- `PUB_YEAR:[2018 TO 2026]` keeps the old bound because RoB 2 (the
  v2.0 tool) was released in 2019; pre-2018 reviews typically use
  RoB 1, which has different domains and is out of scope for this
  corpus.
- Non-CDSR sources that *also* use RoB 2 (BMJ rapid reviews, some
  JAMA systematic reviews, etc.) are deliberately excluded. They are
  a valid future expansion; for this rebuild, CDSR is the
  authoritative source.

The collector search pagination stays as-is (cursor-based, 25 per
page).

## 4. Extraction strategy

Two paths, tried in order. Every successful extraction must return
the full 6-field tuple (per §2.2); partial tuples are rejected by
both paths.

### 4.1 Structural JATS table extraction (primary)

CDSR reviews present their per-study RoB 2 ratings in a "Risk of bias
in included studies" section containing one or more JATS `<table>`
elements. The structural extractor:

1. Parses the JATS XML (`xml.etree.ElementTree` — same as elsewhere
   in the codebase).
2. Iterates all `<sec>` elements; identifies candidate sections by
   title match against any of:
   `"Risk of bias in included studies"`,
   `"Risk of bias"` (if the parent section concerns `"included
   studies"`),
   `"Assessment of risk of bias"`.
3. Within each candidate section, iterates all `<table-wrap>` elements.
4. For each table, inspects the header row (`<thead>`) and accepts the
   table only if its column headers cover all five RoB 2 domains — by
   keyword match (`"randomi[sz]ation"`, `"deviation"`, `"missing"`,
   `"measurement"`, `"reporting"`) plus an `"overall"` column **plus
   evidence of per-outcome granularity** (either an `Outcome` column,
   a per-outcome row grouping, or an explicit primary/secondary label).
   Tables with a study-id column as the only row key — pattern 4 — are
   rejected as non-RoB-2-conformant. Tables whose columns describe RoB 1
   domains (random-sequence-generation, allocation-concealment,
   blinding-of-participants, etc.) are also rejected (see §10 Q3).
5. Iterates `<tbody><tr>` rows. For each row:
   - Extracts the study identifier from column 1 (typically
     `"FirstAuthor YEAR"`).
   - Extracts rating strings from the five domain columns and the
     overall column.
   - Normalises each rating string to `low` / `some concerns` / `high`
     by keyword match (case-insensitive, tolerant of abbreviations
     like `L`, `SC`, `H`).
   - Emits the row as a candidate tuple only if all six normalised
     values are non-empty and valid.
6. Records provenance: review PMCID, section title, table index in
   the review, row index in the table. Stored as JSON in the new
   `rob_provenance` column (see §7).

Rows where any field is blank, `unclear`, `n/a`, `not applicable`, or
otherwise unparseable are dropped with a per-row log entry.

### 4.2 LLM fallback (secondary)

For reviews where the structural extractor returns zero complete rows
(section missing, unconventional table format, ratings given in prose
rather than a table), the existing LLM path
(`extract_rob_via_llm`) runs.

Changes to the LLM path:

- **Prompt tightened** to require the 6-field tuple in every row;
  partial rows instructed to be returned only with explicit `null`
  markers which are then rejected at validation.
- **Output schema validation** after parsing: reject any row where
  any of the six fields is missing, `null`, or not one of the three
  permitted values.
- **Provenance captured less precisely than the structural path** —
  the LLM fallback records the review PMCID plus the chunk index,
  since it cannot point at a specific table row. This is acceptable
  because the downstream manual check (§8) will sample LLM-fallback
  rows at a higher rate than structural rows.

Papers for which both paths fail to produce a complete tuple are
logged to `dataset/cochrane_rebuild_rejects.jsonl` and skipped.

## 5. PMID resolution

Complete rewrite of `resolve_pmids_from_refs`, `resolve_pmids_via_doi`,
`resolve_study_pmids`. Only three paths, in priority order:

### 5.1 Bracket-reference (highest confidence)

If the extracted row carries a bracketed reference number (e.g.
`[28]`), match against the review's reference list by label. If a
match is found and the reference has a `<pub-id pub-id-type="pmid">`
element, take the PMID and record `resolution_method = "bracket_ref"`.

### 5.2 Author + year + title match (medium confidence)

If no bracket number, parse the study ID for first-author surname
and year. Look up references with the same (surname, year). If
multiple match (common), perform a string-similarity check between
the review's citation text and each candidate reference's title.
Accept the candidate if similarity passes a threshold (we use
`difflib.SequenceMatcher.ratio() >= 0.70` as a starting point;
tune on the 20-paper sample). Record `resolution_method =
"author_year_title"`.

### 5.3 Cochrane `<ref>` direct PMID (auxiliary)

Some CDSR references already contain `<pub-id pub-id-type="pmid">`
directly, making the study-ID match unnecessary. If the reference
list entry the study ID points at has a PMID, use it directly.
Record `resolution_method = "direct_ref_pmid"`.

### 5.4 Explicitly removed

The following resolution strategies are **removed**:

- Surname-only matching (FORENSICS §3.3).
- "First of multiple" PubMed `esearch` results (FORENSICS §3.4).
- Any path that assigns a PMID without either a bracket number or an
  author+year+title match.

A study ID that cannot be resolved by one of §5.1–§5.3 is dropped
with a rejection log entry. No "best effort" PMIDs ever persist.

## 6. Publication-type validation

Before persisting any row, fetch the resolved PMID's `<PublicationType>`
list via PubMed `efetch` (reuse the batched fetch in
`scripts/audit_publication_types.py`; lift its constants into the
collector).

Reject (drop with log entry) if the PublicationType list contains any
of: `Letter`, `Editorial`, `Comment`, `News`, `Congresses`,
`Meeting Abstracts`, `Personal Narratives`, `Biography`, `Book`,
`Case Reports`, `Review`, `Systematic Review`, `Meta-Analysis`,
`Address`, `Lectures`.

Accept unconditionally if the list contains any of: `Randomized
Controlled Trial`, `Controlled Clinical Trial`, `Clinical Trial`
(including phase variants), `Pragmatic Clinical Trial`,
`Equivalence Trial`, `Adaptive Clinical Trial`.

Flag as `ambiguous` (accept, but record the status) if only
`Journal Article` is present — same tri-state classifier as the
existing audit script. The §8 manual check looks closely at
`ambiguous` rows.

## 7. DB changes

### 7.1 Schema

Add two columns to `papers`:

```sql
ALTER TABLE papers ADD COLUMN rob_provenance JSON;
ALTER TABLE papers ADD COLUMN rob_source_version TEXT;
```

- `rob_provenance` — JSON blob with at least:
  `{"review_pmid": "...", "review_pmcid": "...", "review_title": "...",
    "section": "...", "table_index": N, "row_index": N,
    "study_id_text": "...", "reference_citation": "...",
    "resolution_method": "bracket_ref|author_year_title|direct_ref_pmid",
    "publication_type_check": "trial|ambiguous",
    "extraction_method": "structural_table|llm_fallback",
    "extractor_version": "v2.0.0"}`
- `rob_source_version` — short string identifying which harvest run
  populated the row (e.g. `"rebuild-2026-04"`), so future rebuilds
  can co-exist with archived data.

### 7.2 Invariant (application-enforced)

A row satisfies the RoB provenance invariant iff all of:

```
overall_rob IS NOT NULL AND overall_rob != ''
randomization_bias IS NOT NULL AND randomization_bias != ''
deviation_bias IS NOT NULL AND deviation_bias != ''
missing_outcome_bias IS NOT NULL AND missing_outcome_bias != ''
measurement_bias IS NOT NULL AND measurement_bias != ''
reporting_bias IS NOT NULL AND reporting_bias != ''
cochrane_review_pmid IS NOT NULL AND cochrane_review_pmid != ''
rob_provenance IS NOT NULL
rob_source_version = 'rebuild-2026-04'
```

A new `Database.upsert_cochrane_paper_v2()` method checks the
invariant and raises `ValueError` if any condition fails; the
collector catches the error, logs a rejection, and moves on. The
check-and-insert are wrapped in a single transaction.

The old `upsert_cochrane_paper` method stays for archival writes to
the legacy rows. It is not called by the rebuild.

### 7.3 Legacy rows

All existing Cochrane-sourced rows (the 328 from the March harvest)
are marked excluded with reason pointing at the forensic document:

```sql
UPDATE papers
SET excluded = 1,
    excluded_reason = 'legacy Cochrane corpus — provenance unverifiable; see docs/papers/drafts/medrxiv_V5/FORENSICS.md'
WHERE source LIKE 'cochrane%'
  AND COALESCE(excluded, 0) = 0
  AND rob_source_version IS NULL;
```

Legacy rows stay in the DB as forensic artefact. Rebuild rows are
inserted with a fresh `source='cochrane_rob'` + `rob_source_version='rebuild-2026-04'`
and are never conflated with legacy rows in cohort queries.

## 8. Validation gate — 20-paper manual check

This is the hard stop between scope check and full harvest.

### 8.1 Procedure

After implementing §2–§7 and running on the first 3–5 Cochrane
reviews, generate a manifest at
`dataset/cochrane_rebuild_manual_check.md` with the first 20
candidate rows. Each row shows:

- Resolved PMID and PubMed-fetched title
- Source review PMID + title
- The exact table (or chunk) the rating came from, verbatim
- Extracted 6-field rating tuple
- Resolution method used

User (hherb) opens 20 rows and each referenced review, checks by eye
that the table row and the rating actually match, and signs off with
a commit message like `manual-check: 20/20 pass, proceed to full
harvest`.

If ≥2 rows fail (wrong PMID, wrong rating, phantom entry), the
extractor is adjusted and the 20-row manifest is regenerated until
it passes. Every iteration is logged so the final design is tracked.

### 8.2 Why 20

Statistically a 19/20 (95%) pass rate already gives >80% confidence
that the true population error rate is below 10%. For a larger
guarantee (true rate below 1%) we would need ~300; that's the right
number for a publication sample, but 20 is the right number for a
scaling gate. If the rebuild is good, 20 will pass; if it's subtly
broken, 20 will show it.

## 9. File changes

### 9.1 New files

- `biasbuster/collectors/cochrane_rob_v2.py` — the rewritten Stage A.
  Greenfield file alongside the legacy one so the old code remains
  runnable for forensics. Exports `CochraneRoBCollectorV2` and
  `collect_cochrane_rob_v2()`.
- `biasbuster/collectors/rob_table_extractor.py` — the structural
  JATS table extractor (§4.1), pure-function implementation that
  takes a JATS bytes blob and returns a list of candidate-tuple
  dataclasses. Unit-testable in isolation.
- `scripts/cochrane_rebuild_manual_check.py` — generates the 20-paper
  manifest markdown.
- `scripts/cochrane_rebuild_run.py` — the rebuild orchestrator:
  collects reviews → extracts → validates → writes DB.

### 9.2 Modified files

- `biasbuster/database.py` — adds `upsert_cochrane_paper_v2`,
  `migrate_add_rob_provenance_columns`. Keep the legacy upsert.
- `scripts/audit_publication_types.py` — lift its constants
  (`TRIAL_TYPES`, `NON_TRIAL_TYPES`, `classify`) into a shared
  module `biasbuster/utils/pubtype.py` so the rebuild collector
  can reuse them.

### 9.3 Unchanged

- The V5A assessment pipeline, prompts, annotator backends.
- The JATS fetch infrastructure (`scripts/fetch_cochrane_jats.py`).
- The V5A annotation batch runner.
- Anything in `docs/papers/drafts/medrxiv_V5/` — sidecars update
  after the rebuild lands.

## 10. Open questions

Questions I cannot resolve without the user's input. Defaults given
so the design can proceed, but explicit sign-off changes the answer
if needed.

1. **LLM for the fallback extractor (§4.2).** The legacy code used
   DeepSeek reasoner. The rebuild will use whatever is configured in
   `config.py`. **Default: keep DeepSeek reasoner** (it was only the
   inputs that were wrong; the LLM path itself looked fine in
   `INITIAL_FINDINGS_V3.md`). Switch if user prefers Anthropic Claude
   for consistency with V5A annotation.
2. **Per-outcome vs per-study RoB ratings.** RoB 2 supports per-outcome
   rating; some CDSR reviews publish multiple RoB rows per trial (one
   per outcome). **Decided (2026-04-17): primary-else-first-row
   collapse.** For each trial with multiple rows, take the row labelled
   as primary; if no primary marker is present, take the first row for
   that trial. Record `per_outcome_variant=true` in the provenance
   whenever a trial had more than one row. Worst-case (max across rows)
   is an acceptable alternative and may be recorded as a secondary
   provenance field for sensitivity analysis.
3. **Reviews that use RoB 1 instead of RoB 2.** Some older CDSR reviews
   still use RoB 1 (different domains: random sequence generation,
   allocation concealment, blinding of participants/personnel/assessors,
   incomplete outcome data, selective reporting, other). **Decided
   (2026-04-17): reject.** Mixing RoB 1 and RoB 2 ratings would break
   the per-domain comparison in the downstream analysis — different
   tools, non-comparable domains. The structural extractor rejects any
   table whose column headers don't cover the five RoB 2 domains.
4. **Reference lists without PMIDs.** Some CDSR reviews cite studies
   with only author/year/title and no `<pub-id>`. **Default: use the
   author+year+title resolver (§5.2) to find the PMID via PubMed
   `esearch`, confirmed by title-similarity.** This is a structured
   search, not the old "first of many" anti-pattern.
5. **Whether to stop when per-domain yield is low.** If after the first
   50 reviews only 20 trials pass the §2 litmus test, do we continue
   to 500 reviews hoping for more? **Default: continue until either
   500 reviews are processed or 200 valid trials are reached.** Cap
   exists only to bound runtime.
6. **Pattern-4 reviews (single-study rating with no outcome column).**
   Some CDSR reviews carry one RoB row per trial with no outcome
   column — effectively a pre-RoB-2 "one per study" style that the
   reviewer happened to populate with RoB 2 domain fields. **Decided
   (2026-04-17): reject.** Cleaner comparisons matter more than
   cohort size here. The structural extractor requires a column
   header match indicating per-outcome granularity; pattern-4 tables
   fail that check and are logged as `rejected: pattern_4_no_outcome_column`.

## 11. Order of operations

Target: 3–5 days of focused work, in 4 phases.

### Phase 1 — Design sign-off and schema migration (½ day)

1. User reviews this document and signs off, or requests changes.
2. Run `database.py`'s new migration to add `rob_provenance` and
   `rob_source_version` columns.
3. Mark legacy Cochrane rows as excluded (§7.3).
4. Commit: `rebuild-phase1: schema migration + legacy archival`.

### Phase 2 — Core implementation (2 days)

1. `biasbuster/collectors/rob_table_extractor.py` with unit tests on
   3 known-good JATS reviews (pick any three CDSR reviews from
   Europe PMC; save the JATS files in `tests/fixtures/`).
2. `biasbuster/collectors/cochrane_rob_v2.py` with the new query and
   resolution logic.
3. `biasbuster/database.py` adds `upsert_cochrane_paper_v2` with
   invariant enforcement.
4. `biasbuster/utils/pubtype.py` shared module; `audit_publication_types.py`
   refactored to import from it.
5. Unit tests on the invariant: try to insert an incomplete tuple,
   assert it raises.

### Phase 3 — Validation gate (½ day)

1. Run the collector against the first 3–5 CDSR reviews in dev mode
   (writes candidate rows to a scratch table or in-memory, does not
   touch `papers`).
2. Generate `dataset/cochrane_rebuild_manual_check.md`.
3. User manually reviews 20 rows.
4. Iterate on the extractor until 20/20 pass.
5. Commit: `rebuild-phase3: manual-check 20/20 pass`.

### Phase 4 — Full harvest (0.5–1 day of wall-clock, ½ day of work)

1. Run the orchestrator against the full CDSR query.
2. For each new PMID that doesn't have a cached JATS, the existing
   fetch pipeline runs (rate-limited, Europe PMC).
3. For each new PMID that doesn't yet have Sonnet/gemma4 V5A
   annotations, run the V5A batch runner scoped to just those PMIDs.
4. Regenerate the Cochrane comparison report against the new cohort.
5. Commit: `rebuild-phase4: harvest complete, N=<actual>`.

### Phase 5 — Manuscript update (post-rebuild, out of scope here)

Rewrite the relevant sections of METHODOLOGY.md with the new cohort
numbers; draft the primary results with trustworthy Cochrane κ; write
RESULTS.md.

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Structural extractor misses non-standard table layouts | Medium | LLM fallback picks them up; §8 manual check catches any that slip through |
| LLM fallback hallucinates ratings (the original failure mode was different — extraction ran on wrong docs — but the LLM path has its own failure modes too) | Medium | The 6-field complete-tuple requirement rejects most hallucinations because LLMs usually fail on 1–2 fields first; §8 manual check on LLM-fallback rows at higher rate than structural rows |
| PMID resolution via title-similarity threshold 0.70 is wrong | Medium | Tune on 20-paper sample; acceptable to reject borderline cases rather than admit wrong ones |
| Per-outcome RoB reporting in some CDSR reviews produces confusing double-rows | Low | Default to primary-outcome or first-row rule (§10.2); §8 catches mismatches |
| Corpus ends up too small for paper-grade kappa CIs | Medium | Not a design risk — just a data-availability finding. Successful rebuild at N=80 is still a publishable cohort; if yield is surprisingly low we publish the finding anyway and report widened CIs honestly |
| Rebuild takes longer than 5 days | Medium | Schedule slip is acceptable since publication is not time-gated |

---

## Appendix A — What not to do

Explicit list of anti-patterns from the forensics, cross-referenced
to the code lines that should *not* reappear:

- No catch-all regex patterns on unstructured text
  (`cochrane_rob.py:510-522` anti-pattern).
- No surname-only reference resolution
  (`cochrane_rob.py:941-945` anti-pattern).
- No "first of multiple" PubMed `esearch` results
  (`cochrane_rob.py:854-861` anti-pattern).
- No population of `cochrane_review_pmid` before the PMID resolution
  has completed and been validated.
- No persistence of partial rating tuples; reject at the extractor.
- No "we'll add the size guard later" — `MAX_FULLTEXT_BYTES` is the
  first line of the collector class, not an afterthought.

## Appendix B — Fixture papers for unit tests

Three known-good CDSR reviews to use as extraction test fixtures.
User (hherb) will hand-pick from already-downloaded JATS or fetch
specifically for this, aiming for variety in table layout:

- A clean, textbook RoB 2 per-outcome summary table (positive case).
- A CDSR review where the RoB spans multiple tables or is spread
  across narrative + table (structural-extractor stress test).
- A CDSR review whose RoB rows use per-outcome granularity with
  multiple pre-specified outcomes per trial (collapse-rule test).

Optional fourth fixture for negative-case testing: a CDSR review
using RoB 1 (must be rejected) and/or a pattern-4 table (must be
rejected). Rejection is a first-class behaviour and worth a test.

The JATS XML for these is saved to `tests/fixtures/cochrane_reviews/`
and pinned in the test suite so the extractor is regression-testable
across future changes.
