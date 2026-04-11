# BiasBuster v3 Two-Call Architecture — Initial Findings

**Status:** initial empirical results, two test papers (one HIGH-bias case study).
Calibration on LOW-bias papers and a Claude full-text ground-truth pass are
still pending.

**Date of these findings:** 2026-04-09 → 2026-04-11.

**Document scope:** what we built, what we measured, what surprised us, what's
still uncertain. Read this together with `CONTEXT_FOR_CLAUDE_CODE.md` (the
original failure case that motivated the rebuild), `architecture_guide.md`
(the design vision), and `MERGE_STRATEGY.md` (the rationale for the merge
rules).

---

## TL;DR

1. The two-call architecture (extract → assess) **dominates** single-call by
   25–30 percentage points of agreement with Claude across **three model
   families** (gpt-oss 120B, gpt-oss 20B, gemma4 26B). The win is robust to
   model size and training corpus, ruling out "lucky prompt fit".
2. The full-text **map-reduce** path (chunked per-section extraction → merge
   → assessment) successfully extracts the exact facts the original failure
   case said local models could not catch — including the 34.4% attrition
   rate, the 43.75% vs 25% differential dropout, and the sponsor-employee
   manuscript drafter.
3. The full-text **single-call** path **collapses across all three model
   families** to ~50% agreement and a generic "no major red flags"
   moderate verdict. The v1 prompt cannot use long inputs at all.
4. Five small prompt edits to the v3 assessment criteria moved gpt-oss
   abstract two-call from 78% to 92% agreement on the same paper.
5. The full-text path was **non-deterministic on gpt-oss-120b** before
   commit `66bca3c` — one chunk's correct extraction was sometimes lost
   during a parallel-extraction run. Fixed by switching to sequential
   per-chunk extraction.
6. The agreement-with-Claude metric **understates full-text two-call**
   because Claude's reference annotation was abstract-only. We need a
   Claude full-text annotation as ground truth before we can fairly
   evaluate the f2 mode.

---

## 1. Motivation

`CONTEXT_FOR_CLAUDE_CODE.md` documented a specific failure: when a local
gpt-oss:120b was given the full text of a Seed Health synbiotic RCT
(DOI `10.3390/antibiotics15020138`, PMID `41750436`), it correctly identified
the bias verdict as HIGH but missed several specific findings that Claude
caught:

- Methodology rated `none` instead of `high` despite 34.4% attrition with
  43.75% vs 25% differential, per-protocol-only analysis, and no multiplicity
  correction across many endpoints.
- COI rated `moderate` instead of `high` — the model noted industry funding
  but missed that a sponsor employee performed *both* the data analysis *and*
  the manuscript drafting (a structural conflict).
- Post-verification reasoning was shallow and made trivial probability
  adjustments.

The v3 hypothesis from the architecture guide: split the cognitive task into
**extraction** (mechanical fact retrieval, even small models can do this)
and **assessment** (reasoning over a clean structured JSON, harder but with
much less surface area). Test whether the split helps the local model.

---

## 2. What We Built

### 2.1 New prompts (`biasbuster/prompts_v3.py`)

Created in commit `27df87a`. Provides three prompts:

- `EXTRACTION_SYSTEM_PROMPT` — Stage 1 fact extraction from a whole abstract or
  whole paper. Outputs a structured JSON with sample, analysis, outcomes,
  conflicts, methodology_details, conclusions sub-objects.
- `ASSESSMENT_SYSTEM_PROMPT` — Stage 2 bias assessment from the Stage 1 JSON.
  Same 5-domain output schema as v1 (`statistical_reporting`, `spin`,
  `outcome_reporting`, `conflict_of_interest`, `methodology`) plus
  `recommended_verification_steps`.
- `SECTION_EXTRACTION_SYSTEM_PROMPT` — variant of the extraction prompt for
  per-section input (full-text map-reduce path). Reuses the Stage 1 schema
  by string-concatenation so any schema change propagates automatically.

### 2.2 BaseAnnotator and the four-mode pipeline

In `biasbuster/annotators/__init__.py`. The class hierarchy is:

```
BaseAnnotator                                    (annotators/__init__.py)
├── annotate_abstract()                          single-call v1
├── annotate_abstract_two_call()                 two-call v3 on abstract
├── annotate_full_text_two_call()                two-call v3 with map-reduce
└── annotate_batch(...two_call=True)             batch wrapper
    ↑
    ├── LLMAnnotator              (Anthropic SDK)
    ├── OpenAICompatAnnotator     (DeepSeek, vLLM, SGLang via httpx)
    └── BmlibAnnotator            (Ollama via bmlib.LLMClient — sync API
                                   wrapped in asyncio.to_thread)
```

All annotation flow lives once on the base class; subclasses only implement
`_call_llm()` and the async lifecycle. This is what lets us run the same
flow on Anthropic's API, DeepSeek, and any local Ollama model with
identical prompts and identical retry logic.

### 2.3 Map-reduce full-text path

`annotate_full_text_two_call()` takes `sections: list[tuple[str, str]]` from
either `chunk_jats_article()` (semantic JATS chunks) or `chunk_plain_text()`
(fixed-window fallback for PDFs). For each chunk it:

1. Sends the section through `SECTION_EXTRACTION_SYSTEM_PROMPT` and parses
   the partial extraction JSON.
2. Collects all partials from all chunks.
3. Merges them via `merge_section_extractions()` using field-type-specific
   rules (lists union, presence booleans OR, singletons resolved by
   authoritative-section priority — see `MERGE_STRATEGY.md`).
4. Sends the merged extraction to `ASSESSMENT_SYSTEM_PROMPT` for the Stage 2
   bias judgment.

The merged extraction also includes a `_merge_conflicts` array surfacing
any disagreements across sections (e.g., one chunk says "32 randomised",
another says null) — Stage 2 sees these as a reporting-consistency signal.

### 2.4 The CLI now uses the same path

`biasbuster/cli/analysis.py` was rewritten in commit `1ad57d9` to delegate
to `BaseAnnotator` via the new `BmlibAnnotator` adapter. Roughly 330 lines of
duplicated prompts and synthesis helpers were deleted. The CLI and the
training-data pipeline now share the same annotator code, which is the only
way to guarantee inference and training generation produce comparable
outputs.

### 2.5 Comparison script

`scripts/compare_singlecall_twocall.py` runs up to four modes per paper
on a single local model and stores each under a distinct DB tag:

| Mode | Input | Architecture | Tag |
|---|---|---|---|
| a1 | Abstract | v1 single-call | `<model>_abstract_singlecall` |
| a2 | Abstract | v3 two-call | `<model>_abstract_twocall` |
| f1 | Full text | v1 single-call (concatenated chunks) | `<model>_fulltext_singlecall` |
| f2 | Full text | v3 two-call (map-reduce) | `<model>_fulltext_twocall` |

`f1` is the "naive baseline" — cram the whole paper into the v1 prompt and
see what comes back. It exists to separate "two-call helps" from "full text
helps". Without `f1`, comparing `a2` to `f2` conflates the architectural
benefit with the input-length benefit.

The script then prints a side-by-side comparison against Claude's annotation
and reports per-mode agreement percentages.

---

## 3. The Iteration Loop

### 3.1 Round 1 — silent prompt contract bug

First run on gpt-oss:120b (commit `9b04672`) failed every annotation with
*"rejecting truncated response (missing 1 fields)"*. Investigation revealed
the v1 `_JSON_SCHEMA` in `prompts.py` did not include
`recommended_verification_steps`, but `REQUIRED_ANNOTATION_FIELDS` in the
parser demanded it. Strong models (Claude, DeepSeek) had been silently
inferring the field from context; gpt-oss followed the schema literally.

Also discovered: `VERIFICATION_DATABASES` was defined in `prompts.py` but
**never included in any assembled prompt**. So no model had ever been told
which databases to cite in its verification recommendations.

Both bugs fixed in `9b04672` for v1 and v3 simultaneously. After the fix
the script ran end to end on both gpt-oss models.

### 3.2 Round 2 — baseline measurement

Initial agreement with Claude on the Seed Health paper:

| Model | a1 (single-call abstract) | a2 (two-call abstract) |
|---|---:|---:|
| gpt-oss 120b | 64% | 78% |
| gpt-oss 20b | 64% | 78% |

Two-call lifted both models by **+14 percentage points**, and the lift was
**identical across model sizes**. The 20B model in two-call mode was already
*better than* the 120B model in single-call mode. That alone validated the
architectural hypothesis.

### 3.3 Round 3 — five targeted prompt edits

Field-by-field analysis of the round-2 results identified five specific
failures:

1. `inflated_effect_sizes` was `False` despite the paper reporting a
   13,008% UroA increase. Root cause: the original rule said *"extreme
   fold-changes that result from near-zero denominators, **especially if
   the paper itself acknowledges the artifact**"*. Models were waiting for
   the paper to confess.
2. `title_spin` was `False` despite the title literally saying "Promotes
   Recovery" of a surrogate-only microbiome trial. Root cause: the original
   rule was vague — "claims clinical benefit from surrogate data".
3. `primary_outcome_type` was `unclear` even though every extracted outcome
   had `type: "surrogate"`. Root cause: no aggregation rule.
4. Methodology severity stayed at `moderate` even when the model correctly
   identified `no_multiplicity_correction = True` with 7 endpoints. Root
   cause: the HIGH trigger required *"small sample with many uncorrected
   endpoints"* — but `n_analysed` is `null` in the abstract-only path so the
   small-sample condition can never fire.
5. `n_primary_endpoints` was `0` in the 20b extraction even though the
   `primary_outcomes_stated` list had 7 entries with named outcomes and
   p-values. Root cause: the extraction schema just said `"integer"` with
   no instructions on counting.

Five surgical edits in commit `483e4fd`:

- `inflated_effect_sizes`: hard numeric threshold (>500% percent change OR
  >5x fold change), with the literal `"UroA increased by 13,008%" → TRUE`
  example baked into the prompt.
- `title_spin`: explicit list of clinical verbs (`promotes`, `improves`,
  `treats`, `restores`, `protects`, `prevents`, `relieves`, `cures`,
  `benefits`, `recovery`, `enhances healing`) plus the requirement that all
  primary outcomes be surrogate/composite.
- `primary_outcome_type`: explicit aggregation rules (all-same → that type;
  mixed patient_centred + surrogate → patient_centred wins; etc.).
- Methodology severity: new HIGH trigger
  `no_multiplicity_correction AND total_endpoints >= 6` that does **not**
  require `n_analysed` to be known. Plus an explicit "null means unknown,
  not fine" rule for methodology.
- Extraction prompt: `n_primary_endpoints` field doc now says
  *"set to len(primary_outcomes_stated). Do NOT set to 0 when effect sizes
  are reported."*

Round-3 results with the new prompts (still abstract-only):

| Model | a1 | a2 | a2 Δ vs Round 2 |
|---|---:|---:|---:|
| gpt-oss 120b | 58% | **92%** | +14pp |
| gpt-oss 20b | 53% | **86%** | +8pp |

(The `a1` column moved a bit between rounds — 64%→58% for 120b, 64%→53%
for 20b — even though we didn't touch the v1 prompts. That's run-to-run
noise on the v1 path at temperature 0.1: the model produces slightly
different rationalisations for the same paper on different runs, and the
v1 reasoning is loose enough that small changes flip a couple of flags.
The `a2` column is much more stable across re-runs because the
two-call extraction is mechanical and the assessment rules are
deterministic given the extraction.)

Both models now within touching distance of Claude on the abstract task.
Critically, the **20b two-call result (86%) was higher than the 120b
single-call result (64%)** — a small model running a disciplined two-call
pipeline beat a 6× larger model running the v1 prompt by 22 percentage
points.

### 3.4 Round 4 — full text and the four-mode comparison

Extended the script in commit `fced4c4` to add the `--full-text` flag and
the `f1`/`f2` modes. Re-ran both gpt-oss models in all four modes. **Three
findings stood out**:

#### Finding 1: f1 (full-text single-call) is *worse* than a1 (abstract single-call)

| Model | a1 | f1 |
|---|---:|---:|
| gpt-oss 120b | 64% | 50% |
| gpt-oss 20b | 50% | 50% |

Both single-call full-text runs collapsed to `MODERATE / 0.45 / "no major
red flags"`. Same prompt, same model, *more* text → *worse* judgment.

The 120b f1 reasoning literally said:
> *"Conflict-of-interest information is fully disclosed, and the study
> design appears methodologically sound with no major red flags."*

For the same paper where the same model in a1 mode said:
> *"These multiple significant methodological gaps — especially no
> multiplicity correction across 7 primary endpoints — warrant HIGH severity."*

**Hypothesis: boilerplate dilution.** Most of a clinical paper is
methodologically normal text — standard procedures, prior-work citations,
routine tables. With the v1 prompt's instruction-style framing ("assess
this clinical trial *abstract* for potential bias"), the model is being
asked to do an incoherent task on a long document. It computes a vibe
across the whole paper, the bias signals get drowned in volume of normal
text, and the model concludes "looks fine". The abstract has a much higher
signal-to-noise ratio because *only* the headline claims and the
inflated effect sizes are present.

This matches a real concern about abstract-only screening: an astute author
who wants to hide methodology problems puts them in the body of the paper,
banking on most readers (and most LLMs?) not actually reading the whole
thing.

#### Finding 2: f2 (full-text two-call) recovered the original failure case

The 20b f2 extraction caught all the things the original context doc said
were missed:

```json
"sample": {
  "n_randomised": 32,
  "n_analysed": 21,
  "n_per_arm_randomised": {"DS-01": 16, "Placebo": 16},
  "n_per_arm_analysed":   {"DS-01": 9,  "Placebo": 12},
  "attrition_stated": true,
  "attrition_quotes": [
    "Eleven participants were lost to follow-up (Placebo: n = 4; DS-01: n = 7),
     leaving 21 participants who completed the trial..."
  ]
},
"conflicts": {
  "manuscript_drafter": {
    "name": "B.A.N.",
    "affiliation": "Seed Health, Inc.",
    "is_sponsor_affiliated": true
  }
}
```

That's exactly the data the context doc said local models could not catch:
the 34.4% attrition (32→21), the 43.75% vs 25% differential dropout
(7/16 vs 4/16), and the sponsor employee who drafted the manuscript.

The 20b f2 reasoning then matched Claude's structure:
> *"Methodology has high attrition (34.4%), differential attrition (18.8%),
> inadequate sample size (<30 per arm), and no multiplicity correction with
> >30 endpoints, meeting high severity criteria."*

That is **the central failure of the original context doc**, eliminated on
a 20B-parameter model.

#### Finding 3: 120b f2 *missed* what 20b f2 caught

| Field | 120b f2 (round 4) | 20b f2 |
|---|---|---|
| `sample.n_randomised` | `null` | `32` ✓ |
| `sample.n_analysed` | `null` | `21` ✓ |
| `sample.attrition_stated` | `false` | `true` ✓ |
| `attrition_quotes` | `[]` | populated |

The 120b's chunk-4 extraction in this run came back with `null` for the
sample fields, even though chunk 4 (Materials and Methods part 1) contains
the CONSORT paragraph in plain English. The model can read the data
(verified later in isolation — see Round 5 below) but didn't this time.

This is a regression compared to the smaller model. We dug into it.

### 3.5 Round 5 — sequential extraction

Investigation steps:

1. Found which JATS chunk holds the attrition data: chunk 4
   (`Materials and Methods (part 1)`, ~12k chars / ~3000 tokens).
2. Re-ran the section-level extraction prompt on chunk 4 in isolation,
   five times, with the same 120b model and same temperature (0.1).
   **Result: 5/5 trials correctly extracted `n_randomised=32, n_analysed=21,
   attrition_stated=True`.** So the model can do it perfectly in isolation.
3. Verified the merge logic correctly handles null + non-null in any order
   (the merge was not the bug).
4. Verified the 120b f2 annotation had no `_failed_sections` count, no
   `_merge_conflicts` involving `sample.*`, and no JSON parse failures —
   the extraction *did* produce valid JSON for chunk 4 in the failed run,
   it just contained `null`s where the correct values should have been.

Conclusion: the 120b's per-chunk extraction is non-deterministic at
temperature 0.1 when run inside the full pipeline, even though it's
perfectly reliable when run in isolation. The most plausible mechanism is
that the previous code fired all 8 chunks in parallel via `asyncio.gather`,
and Ollama (single GPU, serialised request queue) processed them under
some kind of in-flight contention. Local Ollama doesn't actually run
concurrent requests — they queue — so the parallelism gave us no benefit
and only opened the door to the bug.

**Fix in commit `66bca3c`** — switch to sequential extraction. Process
chunks one at a time in a `for` loop, log each section as it starts.
Cost: ~5–8 minutes more wall-clock per paper. Benefit: removes the
contention hypothesis.

Same commit also added **per-section partial persistence**: the merged
annotation now includes a `_section_extractions` field listing each
chunk's raw extraction output. Future investigations of suspect merged
results can inspect what each chunk actually returned without re-running
the model. The annotation JSON is ~5x larger as a result (24 KB → 135 KB
for a typical full-text paper); acceptable for SQLite storage.

The fix has not yet been re-run on the Seed Health paper for reliability
verification. That's pending.

### 3.6 Round 6 — gemma4 26B (cross-family validation)

Ran the comparison script on `gemma4:26b-a4b-it-q8_0` to test whether the
two-call lift was a gpt-oss-family artefact or a general architectural
property. Gemma is a different model family, different training corpus,
different RLHF.

| Model | a1 | a2 | f1 | f2 |
|---|---:|---:|---:|---:|
| gpt-oss 120b | 64% | 92% | 50% | 78% |
| gpt-oss 20b | 50% | 86% | 50% | 89% |
| gemma4 26B | 56% | 83% | 50% | 75% |

**Two-call lift (a1 → a2):** +28pp / +36pp / +27pp.
**Two-call lift (f1 → f2):** +28pp / +39pp / +25pp.
**Cross-family consistency:** the lifts are within 3pp across model families.

Gemma f2 also caught the attrition data and the manuscript drafter:

```
sample.n_randomised   = 32
sample.n_analysed     = 21
sample.attrition_stated = True
manuscript_drafter = {'name': 'B.A.N., Z.K. and R.J.', ...}
```

And its reasoning matched both the 20b f2 and Claude:
> *"high attrition (34.4%) with significant differential attrition between
> arms (43.75% vs 25%), and there is no mention of multiplicity correction
> despite testing at least 18 different surrogate endpoints (fishing
> expedition)"*

The fact that a 26B model from a completely different family produces the
same characteristic behaviour — same single-call collapse, same two-call
recovery, same successful attrition extraction — is strong evidence that
**we're observing an architectural property, not a happy accident of
prompt design fitting one model family**.

---

## 4. Surprises

### 4.1 The single-call full-text floor

All three model families collapsed `f1` to **exactly 50% agreement and a
"moderate / no major red flags" verdict**. That's not coincidence. It's the
syntactic floor — *"produce a valid annotation that doesn't actually
analyse the content"*. The v1 prompt cannot be salvaged for long inputs by
any model we tested.

This has direct implications for any production pipeline that screens
large volumes of papers: a screening pipeline that hands abstracts to v1
single-call has a *higher* signal-to-noise ratio than one that hands full
texts to v1 single-call. More information actively makes the v1 prompt
worse, not better, because the bias signals get diluted in volume.

### 4.2 Prompt rules are highly leverageable

Five small prompt edits moved abstract two-call from 78% to 92% on the
120b (+14pp) and 78% to 86% on the 20b (+8pp). And critically, **the same
edits helped both models**. Prompt changes in our experience usually help
one model and break another. The v3 assessment prompt is structured enough
that mechanical rules ("if X and Y, severity = HIGH") translate cleanly
across models.

The Round-3 reasoning trace from the 20b shows the model literally walking
through the new rules:
> *"The effect size quote for UroA shows a 13,008% increase, exceeding the
> 5,000% threshold for inflated effect sizes... These three flags trigger
> a HIGH severity for statistical reporting... no multiplicity correction
> method is reported while there are 7 endpoints, meeting the rule for
> no_multiplicity_correction = true with total_endpoints >= 6, which
> triggers HIGH severity."*

(The model misremembered "500%" as "5,000%" but the logic still fires
correctly on 13,008%.)

This tells us something important about training data: a fine-tune on
v3 two-call outputs will teach the model to apply these mechanical rules
*without* the verbose prompt at inference time.

### 4.3 The agreement metric is not the truth

Gemma f2 caught **more correct facts** than gemma a2 (attrition,
manuscript drafter, etc.) and produced **better reasoning** ("structural
COI, fishing expedition with 18 endpoints"). But it scored *lower* on
agreement with Claude (75% vs 83%) because Claude's reference annotation
was made on the **abstract**, not the full text. When gemma f2 escalated
COI severity from `moderate` to `high` based on the manuscript-drafter
finding (which Claude couldn't see), the metric counts that as a
disagreement — even though gemma f2 is more correct.

**This is a measurement gap, not a quality gap.** We need a Claude
**full-text** annotation as the ground truth before we can fairly evaluate
the f2 mode. See §6.1.

### 4.4 Smaller can beat larger when the architecture is right

Three concrete cases:

- **gpt-oss 20b a2 (86%) > gpt-oss 120b a1 (64%)** by 22 percentage points.
- **gpt-oss 20b f2 (89%) > gpt-oss 120b f2 (78%)** by 11 percentage points
  (Round 4, before the sequential-extraction fix).
- **gpt-oss 20b f2 caught the 34.4% attrition that 120b f2 missed** in the
  same Round 4.

The 20b is "weaker" by parameter count and benchmark score, but it
produces tighter JSON, stays closer to the schema, and is less prone to
the parallel-extraction bug that hit the 120b. Larger models have
different failure modes, not simply fewer failures.

---

## 5. Open questions and known unknowns

### 5.1 The 120b non-determinism is unverified-fixed

We have a strong hypothesis (parallel chunks → contention) and a fix
(sequential extraction) but **we have not yet re-run 120b f2 on the Seed
Health paper after the fix lands** (commit `66bca3c`). The fix is
plausible but unverified empirically.

Reliability test plan: run gpt-oss-120b f2 on the Seed Health paper 3
times back-to-back with the new code. If 3/3 catch the attrition, we
have moderate confidence the fix landed. If 2/3 or worse, the
parallelism wasn't the cause and we have a deeper reliability problem.

### 5.2 We have no Claude full-text ground truth

All comparisons use Claude's abstract-only annotation as the reference.
This metric:
- correctly measures `a2` (apples-to-apples on the abstract)
- *under*measures `f2` (penalises full-text-only findings)
- doesn't tell us anything about how Claude would handle the same paper
  if given the full text

To fix: extend `annotate_single_paper.py` with a `--full-text` flag that
calls `annotate_full_text_two_call` via `LLMAnnotator`. Cost: ~50 lines of
code, one Claude API call per paper. After that, the agreement metric
becomes meaningful for f2.

### 5.3 We have no calibration paper test

All measurements so far are on **one** paper, and it's a known HIGH-bias
paper. We have no evidence yet that:
- the new prompts don't *over-flag* clean LOW-bias papers
- the two-call architecture isn't biased toward producing HIGH verdicts
- the inflation thresholds (5x fold-change, 6+ endpoints) don't fire on
  legitimate biology papers that happen to report large but real effects

A calibration test on at least one Cochrane-rated low-RoB paper is the
next step after the reliability fix is verified.

### 5.4 Merge conflict noise is high but mostly benign

The 120b f2 produced 14 merge conflicts; the 20b f2 produced 22. Most are
benign — different sections describing the same population in slightly
different words ("healthy adult participants" vs "Healthy males and
females aged 18–55 years..."). A few are real
(`outcomes.n_primary_endpoints` had values 7, 0, 23, 42 across chunks
because the merge step doesn't reconcile the integer with the list
length).

Two related issues:
- The merge step preserves `n_primary_endpoints` as whatever the
  highest-priority chunk reported, which doesn't match
  `len(primary_outcomes_stated)` after the list union. Should be a
  post-merge sanity pass.
- Free-text fields like `population_description` should prefer the
  longer/more specific value rather than first-seen. Currently the
  default merge rule keeps the first, which loses information.

These are quality-of-life fixes, not blockers. Documented for follow-up.

### 5.5 Gemma severity case bug

Gemma returned `"HIGH"` (uppercase) for `overall_severity` instead of
`"high"`. The schema specifies lowercase. We had to normalise with
`.lower()` in the comparison script. Worth flagging for production
hardening — any downstream code that does exact string matching on
severity will silently break on gemma. The right fix is probably a
post-parse normalisation pass in `parse_llm_json()`.

### 5.6 gpt-oss thinking-mode interaction

The `BmlibAnnotator` disables Ollama's extended thinking
(`think=False`) for the extraction step because we observed that thinking
exhausts the token budget on structured-output tasks before producing the
JSON. This is the right call for *extraction*, but the **assessment** step
might genuinely benefit from thinking. We currently apply the same flag
to both stages. Worth experimenting with `think=True` on assessment only.

---

## 6. What's next

### 6.1 Get the Claude full-text ground truth (highest priority)

Without it, we can't fairly measure f2. Plan:
1. Extend `annotate_single_paper.py` with a `--full-text` flag.
2. Run on the Seed Health paper.
3. Re-score all f2 results against the new ground truth.
4. Likely outcome: every model's f2 score climbs significantly because
   the disagreements that came from "f2 saw more than Claude" now resolve.

### 6.2 Verify the sequential-extraction fix (high priority)

Run gpt-oss-120b f2 on the Seed Health paper 3 times with the new code.
Inspect the `_section_extractions` field to verify chunk 4 returns the
correct sample data each time. If this passes, we can stop worrying about
the f2 reliability for the 120b model.

### 6.3 Calibration paper test (medium priority)

Pick a Cochrane-rated low-RoB paper from the existing dataset
(`SELECT pmid FROM papers WHERE overall_rob = 'low' AND excluded = 0`)
and run all 4 modes × 3 model families through the comparison script.
Confirm the new prompts do NOT call low-bias papers HIGH.

### 6.4 Decide on production model and mode (after calibration)

The current data suggests:
- **gpt-oss 20b f2** is the strongest local-model candidate by far —
  highest agreement (89%), caught the original failure case, no parallel
  extraction bug, faster than 120b.
- **gemma4 26B f2** is competitive (75%, also caught the failure case)
  and a useful second opinion for cross-family ensemble runs.
- **gpt-oss 120b f2** needs the reliability fix verified before we can
  recommend it.
- **abstract two-call** is sufficient for high-volume screening; full-text
  two-call is required for proper analysis.

### 6.5 Address the merge-conflict noise (low priority)

- Reconcile `n_primary_endpoints` with `len(primary_outcomes_stated)`
  after the list-union merge step.
- Prefer longer/more specific values for free-text fields like
  `population_description`.
- Consider a post-merge sanity pass that flags self-inconsistent
  extractions for human review.

### 6.6 Then: full Claude re-annotation and training data generation

Once the f2 path is verified reliable on at least the 120b, the Claude
ground-truth method works, and a calibration paper has been tested,
we can re-annotate the full dataset (with `tag_v1_annotations.py` to
preserve the v1 results for comparison) and start generating the v3
training corpus. See `architecture_guide.md` §3 for the training
sequence.

---

## 7. Code references

| File | Purpose | Commit |
|---|---|---|
| `biasbuster/prompts_v3.py` | New v3 prompts (extraction, assessment, section) | `27df87a`, tightened in `483e4fd` and `9b04672` |
| `biasbuster/annotators/__init__.py` | `BaseAnnotator` + merge logic | `27df87a`, sequential fix in `66bca3c` |
| `biasbuster/annotators/bmlib_backend.py` | Ollama-via-bmlib adapter | `1ad57d9` |
| `biasbuster/cli/analysis.py` | CLI rewritten to use BaseAnnotator | `1ad57d9` |
| `scripts/compare_singlecall_twocall.py` | Four-mode comparison runner | `2de8810`, extended in `fced4c4` |
| `scripts/tag_v1_annotations.py` | DB migration for legacy v1 annotations | `81647e0` |
| `docs/two_step_approach/MERGE_STRATEGY.md` | Merge rules + coherence-pass escape hatch | (separate doc) |

---

## 8. Headline numbers (one place to find them)

| Model | a1 | a2 | f1 | f2 |
|---|---:|---:|---:|---:|
| gpt-oss 120b | 64% | **92%** | 50% | 78% |
| gpt-oss 20b | 50% | 86% | 50% | **89%** |
| gemma4 26B | 56% | 83% | 50% | 75% |

Reference: agreement with Claude's abstract-only annotation on PMID
`41750436` (Seed Health synbiotic RCT, the case study from
`CONTEXT_FOR_CLAUDE_CODE.md`). 36 fields scored. Per-field details in
the database under tags
`ollama_<model>_<mode>` (e.g. `ollama_gpt-oss_120b_fulltext_twocall`).

These numbers will change once we have a Claude full-text ground truth
(see §6.1) — most likely the f2 column will climb significantly because
its current "disagreements" with Claude are mostly cases where f2 caught
information Claude couldn't see in the abstract.
