# Open-Access-First Risk-of-Bias Benchmark — Design

**Date:** 2026-07-17
**Status:** design — awaiting owner review before implementation planning.
**Objective:** 3 (Evaluations & papers). Produces a new, publishable
risk-of-bias benchmark whose every item pairs an open-access trial full text
with a human-expert RoB 2 label.
**Relationship to prior work:** generalises the `medrxiv_V5` Cochrane corpus
rebuild (`docs/papers/drafts/medrxiv_V5/REBUILD_DESIGN.md`) from "Cochrane
only" to "any RoB 2 systematic review", and adds an **open-access
trial-full-text guarantee** the rebuild does not have. It does **not** modify
the Eisele-Metzger 2025 replication (`studies/eisele_metzger_replication/`) or
its benchmark DB.

---

## 1. Why this exists

Two past efforts produced wrong results because of how trials were sourced,
and paywalls were the common denominator:

1. **EM 2025 replication.** Of the 100 benchmark RCTs, only ~41 had native
   open full text. ~50 were PMID-but-paywalled, 9 regional-journal trials were
   unrecoverable, and **RCT030 was a wrong-paper acquisition** — the fetcher
   resolved the parent Cochrane review (PMID 37131928) instead of the paywalled
   primary trial, so every model-vs-Cochrane pair for it scored a different
   document than intended (`studies/eisele_metzger_replication/exclusions.py`,
   issue #29).
2. **Cochrane corpus harvest.** The search admitted any OA document mentioning
   "RoB 2" (protocols, an overview of reviews, a 5.9 MB conference-proceedings
   book); greedy regex + fuzzy PMID matching then bound ratings to arbitrary
   papers. ~9% of rows definitely wrong, ~25% likely wrong
   (`docs/papers/drafts/medrxiv_V5/FORENSICS.md`).

**Root cause (shared).** Both start from a *fixed target trial* and then try to
*fetch it*. When the target is paywalled the pipeline either fails or silently
substitutes the wrong document. Paywalls do not merely cause misses; they cause
wrong-document substitution.

**The fix (design principle).** Invert the order. Enumerate the **open-access
population first**, and admit a trial only when its full text is genuinely in
hand *and* it carries a human-expert RoB 2 label with verified linkage. Open
access becomes a construction guarantee, not a hope, and there is no paywalled
target to substitute for.

## 2. Decisions locked with the owner (2026-07-17)

| Decision | Choice |
|---|---|
| What the papers feed | A **new** OA-first RoB benchmark (not a repair of EM 2025, not training data) |
| Ground-truth tool | **Any peer-reviewed RoB 2 assessment** — Cochrane *or* any journal systematic review; RoB 1 / ROBINS-I excluded |
| Sourcing approach | **C — hybrid, staged**: a trusted expert-labelled OA core first, then a fresh OA-first harvest to expand and de-bias |
| Redistribution | **Publishable dataset.** Every item needs a redistributable-licensed full text *and* a redistributable-or-re-derivable label |
| License strictness | **Include NC/ND, flagged per item.** Admit CC-BY-NC and CC-BY-ND trials, tagged so downstream users can filter to CC-BY/CC-BY-SA/CC0 |

## 3. Landscape — existing datasets (research, 2026-07-17)

| Dataset | RoB tool | Ground truth | OA trial text | Size | Role here |
|---|---|---|---|---|---|
| **ROBoto2** ([arXiv 2511.03048](https://arxiv.org/abs/2511.03048), [github.com/larchlab/ROBoto2](https://github.com/larchlab/ROBoto2)) | **RoB 2** | **Human expert** (5 reviewers); 245 manual gold + ~203 LLM-assisted | trials from Cochrane CENTRAL; OA not filtered by them | 245 gold, pediatric only | **Seed pool** (Stage A). Use only the 245 manual rows |
| **EM 2025 cohort** (`studies/eisele_metzger_replication/`) | RoB 2 | Human expert (Cochrane) | ~41/100 native OA | 100 RCTs | **Candidate PMID list** (Stage A). Re-derive labels; do not copy EM supplement |
| **RoBIn** ([arXiv 2410.21495](https://arxiv.org/abs/2410.21495), [github.com/phdabel/robin](https://github.com/phdabel/robin)) | RoB **1** | *Automated* distant supervision from CDSR | OA PubMed articles | ~7.3k trials | **Candidate OA-trial pool only** — not ground truth (automated + RoB 1) |
| **RobotReviewer corpus** ([github.com/ijmarshall/robotreviewer](https://github.com/ijmarshall/robotreviewer)) | RoB 1 | *Automated* (CDSR) | PDFs, OA unclear, GPL code only | ~12.8k trials | Not used — automated, RoB 1, licensing unclear |

**Load-bearing distinction.** For a *benchmark* the ground truth must be
**human-expert RoB 2**. RoBIn and RobotReviewer were built by exactly the
distant-supervision method that corrupted the earlier BiasBuster corpus; they
are usable as *candidate pools* to mine for OA trials, never as scored answers.

## 4. The benchmark item — inclusion litmus test

A candidate becomes a benchmark item **iff all four hold**. Any failure drops
the candidate at the point of evaluation and logs it to
`dataset/oa_rob_benchmark_rejects.jsonl` with the failing rule.

1. **OA full text in hand.** The trial is in the **PMC Open Access Subset** and
   its full-text JATS has been retrieved and cached. A recorded license from
   the OA-subset license map. "Free full text" that is not in the OA subset does
   **not** qualify (not redistributable). PDF-only scraping is not a path.
2. **Complete human-expert RoB 2 tuple.** All six fields present —
   `(overall, d1_randomization, d2_deviations, d3_missing_outcome,
   d4_measurement, d5_selection_of_result)` — each normalised to exactly one of
   `low` / `some concerns` / `high`. Empty, `unclear`, `n/a` → reject the row.
   Per-outcome variants collapse **primary-else-first-row**, recording
   `per_outcome_variant=true` (same rule as `medrxiv_V5` §10 Q2).
3. **Verified trial↔label linkage.** The PMID that the rating attaches to is
   resolved by **bracket-reference** or **author + year + title-similarity**
   (`difflib.SequenceMatcher.ratio() >= 0.70`, tuned on the manual sample).
   **Forbidden:** surname-only matching; "first of many" PubMed `esearch`
   results. The resolved PMID's PubMed PublicationType must be trial-compatible
   (accept `Randomized Controlled Trial`, `Controlled Clinical Trial`,
   `Clinical Trial` and phase/pragmatic/equivalence/adaptive variants; reject
   `Review`, `Systematic Review`, `Meta-Analysis`, `Letter`, `Editorial`,
   `Comment`, `News`, `Congresses`, `Meeting Abstracts`, `Book`, `Case
   Reports`; `Journal Article`-only is `ambiguous` → accept but flag for the
   manual gate).
4. **Provenance + license recorded.** Each item stores the label source
   (source-review PMID/PMCID/title, table index, row index, resolution method,
   extraction method, similarity score) and the trial's license id, so every
   published row is auditable and filterable.

## 5. Ground-truth as data, not prose (publishability)

The label stored is the **six-field rating tuple plus a citation** to the
source assessment. A rating value (`"D1 = low"`) is a fact about the trial, not
copyrightable expression, so re-recording it with attribution is clean even
when the source review is subscription-access (e.g. CDSR / Wiley). The
review's *prose justification* is copyrightable and is **not** stored (an
optional short verbatim quote ≤ the fair-use limit may be kept only for OA
CC-licensed reviews). This single rule resolves three problems at once:

- The EM 2025 supplement's unresolved redistribution rights (HANDOVER) — we
  re-derive from the primary review instead of copying EM's table.
- ROBoto2's missing data license — we re-record the tuple + citation rather
  than republish their annotation records (still: confirm the authors' terms;
  see §9 risk R1).
- Non-OA source reviews — extracting the tuple as data is permissible; copying
  the review body is not.

**Prefer OA systematic reviews** for label sourcing where a trial is assessed
in more than one: an OA CC-BY review makes both the rating *and* its supporting
table redistributable, maximising what the published benchmark can carry.

## 6. Stage A — trusted expert-labelled OA core (ship first)

Goal: a defensible, publishable core with **zero fresh PMID-resolution risk**,
because both seed pools are already human-expert RoB 2.

### 6.1 Seed pool 1 — ROBoto2 manual gold (245)
1. Ingest ROBoto2's `dataset/` records; keep only rows with a `manual_assessment`
   (drop the ~203 `roboto2_assessment` LLM-assisted rows).
2. Resolve each `paper_id` → PMID → PMCID.
3. Keep only trials in the PMC OA Subset (litmus §4.1). Fetch + cache JATS.
4. Re-record the six-field tuple + citation to ROBoto2 and the primary trial.
5. Validate the tuple (litmus §4.2) and PublicationType (litmus §4.3).

### 6.2 Seed pool 2 — EM 2025 OA subset (candidate PMIDs only)
1. Take the EM benchmark's trial PMIDs **only as a candidate list** — not the
   labels. Exclude `WRONG_PAPER_RCTS` (RCT030).
2. For each, confirm PMC OA-subset full text (litmus §4.1); drop non-OA ones.
3. **Re-derive** each RoB 2 tuple from the *primary Cochrane review's own RoB 2
   table* via the Stage B structural extractor (§7.2), storing provenance. Do
   **not** read EM's supplementary table.
4. Validate as above.

### 6.3 Output
A core cohort (rough order 100–200 items) of OA trials with human-expert RoB 2
labels, complete provenance, per-item license tags. Known limitation: pediatric
skew from ROBoto2 — Stage B corrects it. This core is independently publishable
even if Stage B is deferred.

## 7. Stage B — fresh OA-first harvest (expand + de-bias)

Attach labels to the OA population in the **inverted** order.

### 7.1 Trial pool (open-access first)
Europe PMC query, license-clean and full-text-present by construction:
```
(SRC:MED OR SRC:PMC) AND OPEN_ACCESS:Y AND HAS_FT:Y
  AND (PUB_TYPE:"Randomized Controlled Trial"
       OR PUB_TYPE:"Controlled Clinical Trial"
       OR PUB_TYPE:"Clinical Trial")
```
The OA-subset + license filter (litmus §4.1) is re-applied per trial before
admission — the query narrows, the litmus test decides.

### 7.2 Label attachment (structural, verified)
For each OA trial, find systematic reviews that assessed it and extract that
trial's RoB 2 row:
- **Discovery:** Europe PMC citation links (reviews citing the trial) **and**
  reference-list PMID match (reviews whose reference list contains the trial's
  PMID). Prefer OA reviews (§5).
- **Extraction (structural, primary):** parse the review JATS; locate RoB
  sections by heading; accept a `<table-wrap>` only if its header row covers all
  five RoB 2 domains **plus** a per-outcome granularity marker (reject
  RoB 1-domain tables and pattern-4 single-row-per-study tables). Emit a row
  only when all six normalised fields are valid. Record table/row provenance.
  (Reuse `medrxiv_V5` §4.1 verbatim in spirit; implement as a shared extractor.)
- **Extraction (LLM fallback, secondary):** only when the structural path
  returns zero complete rows; tightened prompt requiring the full tuple; schema
  validation rejects partial rows; coarser (chunk-level) provenance; sampled at
  a higher rate in the manual gate.
- **Linkage:** bracket-ref or author+year+title-similarity (litmus §4.3). No
  surname-only, no first-of-many.

### 7.3 Manual gate (hard stop before scaling)
After the first 3–5 reviews' worth of candidate rows, generate
`dataset/oa_rob_benchmark_manual_check.md` (resolved PMID + fetched title;
source review PMID + title; the exact table/chunk verbatim; the six-field
tuple; resolution method + similarity). Owner reviews 20 rows by eye. **≥19/20
must match** or the extractor is adjusted and the manifest regenerated. Only
after sign-off does the full harvest run. (FORENSICS §6.6 meta-lesson.)

## 8. Data model, storage, deliverable

### 8.1 A new, isolated store
- **`dataset/oa_rob_benchmark.db`** (new SQLite) — never touches
  `dataset/eisele_metzger_benchmark.db` or `dataset/biasbuster.db`. The
  HANDOVER gotcha stands: never re-run a DROP-and-rebuild against a DB holding
  evaluation rows. This benchmark gets its own schema and its own build script
  guarded against destructive re-runs (append/upsert, not drop).
- Per-item record fields (minimum): `trial_pmid`, `trial_pmcid`, `trial_doi`,
  `trial_title`, `trial_license` (e.g. `CC-BY`, `CC-BY-NC`, `CC-BY-ND`, `CC0`),
  `license_redistributable` (bool), `fulltext_path` (gitignored cache),
  `rob2_overall`, `rob2_d1`…`rob2_d5`, `per_outcome_variant`,
  `label_source` (`roboto2` | `cochrane_review` | `oa_sr` | …),
  `source_review_pmid`, `source_review_pmcid`, `table_index`, `row_index`,
  `resolution_method`, `similarity_score`, `pubtype_check`
  (`trial` | `ambiguous`), `extraction_method`
  (`structural_table` | `llm_fallback`), `manual_verified` (bool),
  `benchmark_version`.
- **Rejections** logged to `dataset/oa_rob_benchmark_rejects.jsonl` with the
  failing rule — a first-class artefact, not a silent skip (CLAUDE.md).

### 8.2 Incremental + resumable
Per CLAUDE.md: harvest saves incrementally (per-trial checkpoint files or
per-row DB upsert with an `on_result`-style callback), retries network calls
with exponential backoff up to `MAX_RETRIES`, and never truncates full text fed
to the extractor (chunk-and-map-reduce if a review exceeds context;
`MAX_FULLTEXT_BYTES` guard is the first line of the collector, not an
afterthought). Any step expected to run > 2 min is printed for the owner to run
in their own terminal, not run in-session.

### 8.3 Shareable artifact
A redistributable export = per-item {identifiers, license tag, RoB 2 tuple,
provenance} for **all** items, plus cached OA full text **only** for
CC-licensed trials, with NC/ND items tagged so consumers can filter to
CC-BY/CC-BY-SA/CC0. Non-redistributable specifics (source-review prose) are
never exported — only citations.

## 9. Success criteria, risks

### 9.1 Success criteria (measurable)
- Manual-gate pass rate **≥ 19/20** before any scaling.
- Final cohort: **0** non-trial rows; **0** partial tuples; **every** row
  carries a license tag and a verifiable provenance chain.
- **≥ 150** items with clinical-area spread beyond pediatrics (Stage A + B).
- Every published-export row is redistributable-or-citation-only — no copied
  non-OA prose.

### 9.2 Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| R1 — ROBoto2 has no data license; republishing their records is unclear | High | Re-record tuple + citation only; **confirm reuse terms with the authors** before publishing Stage-A ROBoto2-derived rows; if refused, keep those rows internal-eval-only and flag them |
| R2 — OA-subset trials assessed by an SR are rarer than hoped → small cohort | Medium | Not a design flaw — report the true yield; widen to "any RoB 2 SR" (already chosen) lifts the ceiling vs Cochrane-only; publish at N=150 with honest CIs |
| R3 — structural extractor misses non-standard tables | Medium | LLM fallback + manual gate catch stragglers |
| R4 — title-similarity threshold admits a wrong PMID | Medium | Tune 0.70 on the manual sample; prefer rejecting borderline over admitting wrong |
| R5 — an OA trial is assessed with conflicting RoB 2 tuples by two reviews | Medium | Record both; default to the OA review; flag disagreement for the manual gate; consider as a sensitivity signal |

## 10. Explicit non-goals
- Not repairing the EM 2025 cohort in place (its DB is frozen).
- Not building training data — this is an evaluation benchmark.
- Not using automated/distant-supervision labels (RoBIn, RobotReviewer) as
  ground truth.
- Not modifying the locked EM pre-analysis plan or the `medrxiv_V5` rebuild
  (this reuses the rebuild's *rules*, in a separate store).
- Not scraping paywalled or non-OA-subset full text by any route.

## 11. Order of operations (for the implementation plan)
1. Confirm ROBoto2 reuse terms (R1); set up the isolated DB + schema + reject log.
2. Build the shared structural RoB 2 table extractor + PMID-linkage + pubtype
   validator (unit-tested on saved JATS fixtures, incl. negative cases).
3. Stage A: ingest ROBoto2 manual gold + EM OA candidates → OA-filter →
   re-derive/validate → core cohort.
4. Manual gate on 20 rows; iterate the extractor to ≥19/20.
5. Stage B: OA-first trial pool → label attachment → incremental harvest.
6. Build the redistributable export + a `scripts/audit_oa_rob_benchmark.py`
   that re-checks all four litmus rules across the final DB.

---

## Appendix — sources consulted (2026-07-17)
- ROBoto2 — arXiv 2511.03048; github.com/larchlab/ROBoto2 (RoB 2, 245 expert +
  ~203 LLM-assisted, pediatric, no explicit data license found).
- RoBIn — arXiv 2410.21495; github.com/phdabel/robin (RoB 1 domains, automated
  distant supervision from CDSR over OA PubMed articles).
- RobotReviewer — github.com/ijmarshall/robotreviewer (RoB 1, ~12,808 automated
  PDF annotations, GPL code only).
- JMIR 2025;e70450 and ROBUST-RCT (medRxiv 2025.08.12.25333520) — recent
  LLM-RoB evaluation studies, context for the field.
- Internal: `docs/papers/drafts/medrxiv_V5/{REBUILD_DESIGN,FORENSICS}.md`;
  `studies/eisele_metzger_replication/exclusions.py`; `HANDOVER.md`.
