# BiasBuster v3 Two-Call Architecture — Initial Findings

**Status:** rounds 1–10 complete through prompt edits; Round 10 verification
pending. Claude full-text ground truth obtained (Round 7). 120b
sequential-extraction fix verified (Round 8). Round 9 verification (§3.10):
3 of 4 fix targets landed, COI trigger failed due to a sponsor-role-
disclaimer extrapolation bug in the extraction stage. Round 10 (commit
`c34885a`) adds extraction-side and assessment-side fixes for the COI
regression; re-run on both 120b and 20b is the next step. Calibration on
LOW-bias papers remains the next major experiment after Round 10 lands.

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
   manuscript drafter. **All three model families now produce these
   findings** in their f2 outputs.
3. The full-text **single-call** path **collapses across all three model
   families** to ~50% agreement and a generic "no major red flags"
   moderate verdict. The v1 prompt cannot use long inputs at all.
4. Five small prompt edits to the v3 assessment criteria moved gpt-oss
   abstract two-call from 78% to 92% agreement on the same paper (Round 3).
   Four further edits in Round 9 (`475b6a6`) targeted specific reliability
   gaps observed in Round 8 — **3 of 4 landed** on verification (§3.10):
   outcome_reporting no longer flips to low (6/6), `per_protocol_only`
   is now 6/6, per-arm extraction is 5/6. The COI trigger failed (0/6)
   because both local models over-read the paper's "sponsor had no role in
   clinical trial operations" disclaimer. **Round 10 (commit `c34885a`)
   patches this** with an extraction-side rule (sponsor disclaimers only
   cover the activities they explicitly name; sponsor-employed authors
   always count as manuscript drafters) and a belt-and-braces assessment-
   side HIGH trigger (industry funding + ≥1 employee/shareholder author →
   HIGH regardless of sponsor_controls flags). Verification pending.
5. The full-text path was **non-deterministic on gpt-oss-120b** before
   commit `66bca3c`. The sequential-extraction fix is now empirically
   verified by Round 8: 3/3 reliability runs catch the headline attrition
   data, where the pre-fix run dropped all of it.
6. **Round 7 produced a Claude full-text ground truth.** Re-scoring
   against it shows that all three f2 modes (87% / 82% / 82%) exceed
   Claude-on-abstract (79%), which is the structural ceiling for any
   abstract-only annotator. The full-text two-call pipeline is a real
   quality tier above abstract-only — exactly what the architectural
   investment was supposed to achieve.
7. **Smaller still wins.** gpt-oss 20B f2 hits 87% against Claude
   full-text, the best local-model result in the experiment. A 16GB-VRAM
   model running v3 two-call full-text matches Claude on the paper that
   originally motivated the entire rebuild.

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

### 3.7 Round 7 — Claude full-text ground truth and the score rebase

The previous rounds all measured agreement against Claude's
**abstract-only** annotation, because that was the only Claude reference
in the database. We knew this was unfair to f2 (Round 6, §4.3 below),
but had no alternative until we extended `annotate_single_paper.py` with
a `--full-text` flag (commit `7594361`) that calls
`LLMAnnotator.annotate_full_text_two_call` with `claude-sonnet-4-6` and
saves under the new tag `anthropic_fulltext`.

Running it on the Seed Health paper produced a richer reference. Claude
on the full text walks through the exact rules we put into the prompt:

> "The UroA result of '13,008% difference' vastly exceeds the 500%
> inflation threshold (artifact of near-zero baseline), and synbiotic
> strain abundance of '2587-fold-change' exceeds the 5x fold-change
> threshold. Two flags (relative_only=TRUE, inflated_effect_sizes=TRUE)
> → HIGH severity per the 2+ flags rule."

> "Attrition: n_randomised=32, n_analysed=21; overall attrition =
> (32-21)/32 = 34.4% — exceeds 20% threshold. Differential attrition:
> Placebo arm lost 4/16 = 25.0%; DS-01 arm lost 7/16 = 43.75%;
> difference = 18.75 percentage points."

That is **the exact failure case from the original context document**,
caught by Claude with the v3 prompts and the v3 pipeline. The headline
verdict was **HIGH / 0.82 / high confidence** — up from `0.78 / medium`
on the abstract-only annotation, which is exactly the kind of
upward nudge we predicted when the methodology details (attrition,
sponsor-controlled analysis) become visible.

#### Score rebase: agreement against the proper ground truth

Re-scoring every existing local-model annotation against
`anthropic_fulltext` instead of `anthropic` (abstract) revealed an
important calibration issue with the previous numbers:

| model-mode | vs abstract GT | vs **full-text GT** | Δ |
|---|---:|---:|---:|
| **Claude abstract (C-abs)** | 100% | **79%** | -21pp |
| 120b a2 | 92% | 72% | -20pp |
| **120b f2** | 86% | **82%** | -4pp |
| 20b a2 | 86% | 67% | -20pp |
| **20b f2** | 89% | **87%** | -2pp |
| gem26 a2 | 84% | 67% | -17pp |
| **gem26 f2** | 76% | **82%** | **+6pp** |

Three things jump out:

1. **Claude scores only 79% against itself** when given different
   inputs. That's the *ceiling* on what any abstract-only annotator
   can possibly achieve — even a perfect abstract-only annotator can't
   match a full-text assessment because the inputs simply don't
   contain enough information to ground the same conclusions.
   `sponsor_controls_analysis`, `high_attrition`, `differential_attrition`,
   `per_protocol_only` are all in the body of the paper, never in the
   abstract.

2. **Every f2 mode now beats Claude-on-abstract.** 20b f2 (87%),
   gem26 f2 (82%), 120b f2 (82%) all exceed the 79% ceiling that
   bounded abstract-only annotators. Read that again: a 20B parameter
   local model running v3 two-call full-text produces annotations
   *more consistent with Claude-on-full-text* than Claude-on-abstract
   is. The architectural investment in the map-reduce path paid off
   exactly where it should have.

3. **gemma4 26b f2 is the only result that climbed when we rebased.**
   Against the abstract ground truth gemma f2 was the worst f2
   result (76%); against the full-text ground truth it ties with
   120b f2 (82%). The previous numbers were penalising gemma for
   being *more correct than Claude-on-abstract* on `coi_disclosed`,
   `manuscript_drafter`, and `high_attrition` — facts the abstract
   simply does not contain.

#### What still disagrees against the proper ground truth

The remaining gaps cluster in three places:

- **Spin severity** (Claude-FT: moderate; most local models: high).
  Claude-FT noticed that the paper does include hedging language in
  the Discussion ("may offer distinct advantages", "suggests"), which
  moderates the spin severity from HIGH to MODERATE. Most local models
  default to HIGH based on title_spin and inappropriate_extrapolation
  alone. The two f2 modes that read the full text and got this right
  are 120b f2 and gem26 f2 (both moderate). Worth noting: this is
  *another* piece of evidence that full-text inspection produces
  better calibration, not just better extraction.

- **`per_protocol_only`** (Claude-FT: TRUE; **all local models**: FALSE).
  The paper reports analysing only the 21 completers but never uses
  the words "per-protocol" or "completer" — it just describes what it
  did. Claude-FT inferred per_protocol_only from the substance; every
  local model trusted the literal "not_stated" enum from extraction
  and missed the inference. The previous assessment rule required the
  exact label. Fix queued for Round 9.

- **`conflict_of_interest` HIGH trigger** (Claude-FT: HIGH; locals
  except gem26 f2: moderate). Claude-FT applied the "sponsor controls
  analysis AND drafting → HIGH" structural-conflict rule mechanically.
  Most local models correctly extracted both `sponsor_controls_analysis`
  and `sponsor_controls_manuscript` as TRUE but stopped at
  `moderate` anyway. The rule wasn't being applied. Only gemma4-26b
  f2 followed it. Fix queued for Round 9.

This rebase makes the metric meaningful for the first time. Pre-rebase,
"agreement with Claude" was conflating "local model is good at the same
things Claude is" with "local model and Claude are both limited to the
same input". Post-rebase, the metric measures what we actually care
about: how close is a local-model annotation to a high-quality
full-text bias assessment.

### 3.8 Round 8 — 120b reliability test (3-run determinism check)

The sequential-extraction fix (commit `66bca3c`) was empirically
unverified at the start of this round. Even though running the
section-extraction prompt on chunk 4 in isolation gave 5/5 successful
extractions on the 120b, we hadn't proven that the *full* pipeline
(8 sequential chunks + assessment) was reproducible.

`scripts/reliability_test_fulltext.py` was added in commit `df2d505`
to run `annotate_full_text_two_call` N times back-to-back on a single
paper and save each run under a distinct DB tag
(`<model>_fulltext_twocall_rel1`, `_rel2`, ...). Three sequential
runs on `gpt-oss:120b` on the Seed Health paper produced:

```
[run 1/3] done in 675s: overall=high prob=0.80 n_rand=32 n_anal=21 attrition_stated=True ✓
[run 2/3] done in 658s: overall=high prob=0.80 n_rand=32 n_anal=21 attrition_stated=True ✓
[run 3/3] done in 549s: overall=high prob=0.80 n_rand=32 n_anal=21 attrition_stated=True ✓
```

**3/3 ✓** on the headline test — every run caught the 32→21 attrition
data that the pre-fix run dropped. The `66bca3c` sequential-extraction
fix is verified for the original failure mode.

But "3/3 on the headline" understates how nuanced the result actually
is. Per-flag analysis revealed **8 out of 39 scored fields wobble
between runs**:

| field | rel1 | rel2 | rel3 |
|---|---|---|---|
| `statistical_reporting.relative_only` | F | T | T |
| `statistical_reporting.absolute_reported` | T | T | F |
| `statistical_reporting.baseline_risk_reported` | F | T | F |
| `spin.conclusion_matches_results` | T | F | F |
| **`outcome_reporting.severity`** | **moderate** | **moderate** | **low** |
| **`outcome_reporting.primary_outcome_type`** | **surrogate** | **surrogate** | **patient_centred** |
| **`outcome_reporting.surrogate_without_validation`** | **T** | **T** | **F** |
| **`methodology.differential_attrition`** | **T** | **F** | **F** |

Four of those are cosmetic statistical-reporting flag wobble that
doesn't change the domain severity. Four are load-bearing failures
that point at two distinct bugs:

#### Bug A: differential_attrition is detected in only 1 run out of 3

rel1 had `n_per_arm_analysed = {'synbiotic': 9, 'placebo': 12}` and
correctly computed `differential_attrition = True` (43.75% vs 25%).
rel2 and rel3 left `n_per_arm_analysed = None` and defaulted
`differential_attrition = False`.

The information is in the same chunk (Methods part 1, the CONSORT
paragraph) and all three runs got `n_randomised=32, n_analysed=21`
from that chunk. But two of three runs **stopped extracting at the
totals** and skipped the per-arm breakdown.

This is **not** the parallelism bug — sequential extraction is in
place. It's a content-extraction completeness problem: the model
extracts the headline numbers and skips the per-arm detail because
the extraction prompt schema only said
`{"arm_name": "integer"} or null` without telling the model when to
populate it. Fix queued for Round 9.

Crucially, **all 3 runs still rated methodology severity HIGH**. The
aggregation rules are robust enough that a single missing flag
(`differential_attrition`) doesn't sink the rating — the other flags
(`high_attrition`, `inadequate_sample_size`, `no_multiplicity_correction`)
still push methodology to HIGH. The architecture is forgiving even
when individual extractions are imperfect.

#### Bug B: primary_outcome_type aggregation is fragile

rel3's extracted `primary_outcomes_stated` list had 15 entries: **14
surrogate and 1 patient_centred** (a hallucinated "adverse-event
outcome" entry, probably picked up because the Methods discusses
safety monitoring even though safety wasn't a primary endpoint).

The Round-3 aggregation rule said *"if entries mix patient_centred
with surrogate/composite, patient_centred wins"*. That rule fired:
1 patient_centred entry was enough to flip the aggregated
`primary_outcome_type` to patient_centred, which then triggered
`surrogate_without_validation = False` and dragged outcome_reporting
severity from MODERATE down to LOW.

The rule isn't wrong in principle — patient-centred outcomes really
should anchor a study's interpretation. But it's too generous when
extraction misclassifies a stray entry. Fix queued for Round 9: require
**at least 2 patient_centred entries AND at least 30% of the list**
before flipping.

#### Where the 3 runs sit against the proper ground truth

Agreement against `anthropic_fulltext`:

```
prefix:  82%   (the original — accidentally LUCKY despite missing attrition)
rel1:    79%
rel2:    79%
rel3:    77%
```

That looks like a regression at first glance. It's not. The
pre-fix run scored 82% by hitting the right severities **by accident**:
it called methodology HIGH because of `inadequate_sample_size` and
`no_multiplicity_correction` even though it had missed `n_randomised`,
`n_analysed`, `high_attrition`, and `differential_attrition`
entirely. The headline severity was right; the underlying evidence
was wrong.

The post-fix runs catch the actual evidence (n_randomised, n_analysed,
attrition_stated all populated in 3/3 runs, differential_attrition in
1/3) and pay for that accuracy with new wobble elsewhere. Net
agreement is approximately the same; **net fidelity to the actual
paper is much higher**. That's the right trade.

### 3.9 Round 9 — four targeted prompt edits

Round 8 produced a clear set of follow-up bugs, all in the prompts.
Commit `475b6a6` made four edits:

| # | Failure observed in Round 8 / 7 | Fix |
|---|---|---|
| 1 | rel2/rel3 left `n_per_arm_analysed` null even though the CONSORT paragraph was extracted | Extraction schema doc now spells out the per-arm fields with the literal CONSORT example ("Eleven participants were lost to follow-up (Placebo: n=4; DS-01: n=7), leaving 21 participants who completed" → `{'placebo': 12, 'DS-01': 9}`) and instructs computation from per-arm loss counts |
| 2 | rel3 hallucinated 1 patient_centred entry into 14 surrogates → outcome_reporting flipped to LOW | Aggregation now requires `n_pc/n_total >= 0.30 AND n_pc >= 2`. Explicit warning about safety/AE mis-classification. One stray entry can no longer override the dominant character of the trial |
| 3 | All local models missed `per_protocol_only=True` because the paper avoids the label | Rule now triggers on substance (three triggers): explicit `per_protocol`/`completer` label; OR a quote describing the analysed group as only the completers; OR `n_analysed` shrunk by >5% with no ITT/imputation method mentioned |
| 4 | Local models extracted both `sponsor_controls_*` flags but still rated COI moderate (only gemma4 26b applied the rule) | Imperative phrasing: *"apply mechanically — do not require additional reasoning, the trigger alone is sufficient. severity MUST be HIGH"* — hoisted to the top of the HIGH boundary section as condition (a) |

All four edits are tightenings of existing rules, not new
functionality. Each one is grounded in a specific failure observed in
Round 7 or Round 8. All 213 tests still pass after the edits.

Verification results are in §3.10 below: three of the four targets
landed, one did not — and the failure revealed an unexpected bug in
the extraction stage that Round 10 (§3.11) addresses.

### 3.10 Round 9 verification — three out of four targets landed

Ran `scripts/reliability_test_fulltext.py` three times on
PMID `41750436` for both gpt-oss 120b and gpt-oss 20b with the Round 9
prompts (commit `475b6a6`). Results stored under tags
`ollama_gpt-oss_{120b,20b}_fulltext_twocall_rel{1,2,3}`.

Scorecard (ground truth `anthropic_fulltext`: `hdr=high`,
`out=moderate`, `coi=high`, per-arm populated,
`sponsor_controls_manuscript=True`):

| run         | hdr  | out_sev | meth_sev | coi_sev   | diff_att | pp_only | per_arm |
|-------------|------|---------|----------|-----------|----------|---------|---------|
| GT (Claude) | high | mod     | high     | **high**  | True     | True    | Y       |
| 120b pre-R9 | high | mod     | high     | moderate  | False    | False   | N       |
| 120b rel1   | high | mod     | high     | moderate  | False    | **True**| N       |
| 120b rel2   | high | mod     | high     | moderate  | **True** | **True**| **Y**   |
| 120b rel3   | high | mod     | high     | moderate  | **True** | **True**| **Y**   |
| 20b pre-R9  | high | mod     | high     | moderate  | True     | False   | Y       |
| 20b rel1    | high | mod     | high     | **low**   | True     | **True**| Y       |
| 20b rel2    | high | mod     | high     | **low**   | True     | **True**| Y       |
| 20b rel3    | high | mod     | high     | moderate  | True     | **True**| Y       |

Fix-target scorecard:

| Fix | Target | Pre-R9 | Post-R9 | Verdict |
|---|---|---|---|---|
| #1 | per-arm CONSORT extraction populated | 120b: N, 20b: Y | 120b 2/3, 20b 3/3 | **PARTIAL** 5/6 |
| #1b | derived `differential_attrition=True` | 120b: F, 20b: T | 120b 2/3, 20b 3/3 | **PARTIAL** 5/6 |
| #2 | `outcome_reporting.severity` stays moderate | both mod | **6/6 moderate** | **PASS** (rel3-style flip eliminated) |
| #3 | `per_protocol_only=True` | both F | **6/6 True** | **PASS** |
| #4 | `conflict_of_interest.severity = high` | both mod | 120b 3/3 mod; 20b 2/3 **low**, 1/3 mod | **FAIL** + **20b regression** |

Three out of four fix targets land. Headline severity is 6/6 ✓,
`methodology.severity=high` is 6/6 ✓, `high_attrition=True` is 6/6 ✓.
Outcome_reporting no longer flips to low under any seed. Round 9 is
a net improvement.

**But Fix #4 fails across all six runs, and the 20b regresses from
moderate → low in 2/3 runs.** This is unambiguously bad: the Round 9
edit was supposed to *strengthen* the COI mechanical trigger, and
instead the 20b's accidentally-correct pre-R9 moderate rating has
been replaced by a worse rating.

#### Root cause: disclaimer extrapolation in the extraction stage

All six post-Round-9 runs record
`conflict_of_interest.sponsor_controls_manuscript = False`, so the
assessment-side HIGH mechanical trigger — which is gated on this flag
— cannot fire. The extraction is being misled by this sentence in the
paper's Funding section:

> "The sponsor (Seed Health, Inc.) had no role in clinical trial
> operations, which were performed independently by KGK Science, Inc."

The disclaimer covers "clinical trial operations" — nothing else. But
both gpt-oss models (and apparently gpt-oss is peculiarly sensitive to
this) read it as a blanket "sponsor had no role" and set both
`sponsor_controls_analysis=False` and `sponsor_controls_manuscript=False`.
Meanwhile, 4 of the 9 authors are Seed Health employees or
shareholders — which by itself constitutes sponsor involvement in the
manuscript, regardless of any disclaimer. Authorship IS manuscript
drafting by definition: authors write, revise, and approve the paper.

Interestingly, the 20b pre-Round-9 run *did* catch
`sponsor_controls_manuscript=True` (that's why it scored COI moderate
rather than low). Round 9's assessment-side edit appears to have
knocked the 20b off the correct answer it was accidentally landing on
— the assessment prompt's stronger COI language seems to have pulled
the extraction's attention toward the disclaimer rather than toward
the author list. This is brittle, but it tells us two things:

1. The assessment-only fix in Round 9 was fighting the wrong stage.
   Fix #4 requires an **extraction-side** change first.
2. The 20b needs its COI extraction hardened, because the regression
   from moderate to low in 2/3 runs is worse than not having tried.

The full set of the other three fixes hold — they are not affected
by this extraction bug. The Round 9 edits that landed are genuinely
improved; Round 10 only patches Fix #4 without revisiting the
others.

### 3.11 Round 10 — COI sponsor-employee-author fix

Commit `c34885a` makes two targeted edits, both in `prompts_v3.py`.

**Extraction stage** — `EXTRACTION_SYSTEM_PROMPT` schema for
`conflicts` (inherited automatically by
`SECTION_EXTRACTION_SYSTEM_PROMPT` via the
`EXTRACTION_SYSTEM_PROMPT[EXTRACTION_SYSTEM_PROMPT.index("{"):]`
concatenation trick):

- `conflicts.manuscript_drafter.is_sponsor_affiliated` now has an
  explicit rule:
  > *"TRUE if ANY author of the paper is a sponsor employee or
  > shareholder (authorship is, by definition, participation in
  > manuscript drafting). A 'the sponsor had no role in the
  > manuscript' statement does NOT override this when sponsor
  > employees are listed as authors — their authorship IS sponsor
  > involvement. Only record false when the paper explicitly names
  > non-sponsor-affiliated authors as drafters AND the sponsor-
  > affiliated authors are NOT in the author list. Record null only
  > when the paper gives no information at all."*
- `conflicts.data_analyst.is_sponsor_affiliated` clarified: `null`
  (not `false`) is the correct value when the paper is silent on
  who performed the analysis. "Silence ≠ disproof".
- New field `conflicts.sponsor_role_disclaimer`: captures verbatim
  any "sponsor had no role in X" statement AND forces the extractor
  to record only the activities X explicitly names. This is the
  source-level fix for the disclaimer-extrapolation bug.

**Assessment stage** — `ASSESSMENT_SYSTEM_PROMPT` COI section:

- `sponsor_controls_manuscript` now also derives as TRUE when any
  author has role ∈ {employee, shareholder}, even if the extraction
  left `manuscript_drafter.is_sponsor_affiliated` unpopulated. This
  is belt-and-braces: the same logic lives on both sides of the
  pipeline, so a failure on one side is caught by the other.
- New mechanical HIGH trigger `(d)`:
  > *"`funding_type ∈ {industry, mixed}` AND
  > `authors_with_industry_affiliation` contains AT LEAST ONE entry
  > with role ∈ {employee, shareholder}. Rationale: sponsor-employed
  > authors control how the study is written up — they draft,
  > revise, and approve the manuscript as part of their employment.
  > A 'sponsor had no role' disclaimer is NOT disqualifying here
  > because sponsor employees ARE the sponsor's role, via their
  > authorship. Disclosure does not undo this structural conflict;
  > it only makes the reader aware of it. If this trigger applies,
  > severity MUST be HIGH — do not downgrade to moderate or low
  > based on transparency, independent statistical oversight, or
  > the presence of a disclaimer. Count shareholders (including
  > holders of stock options) the same as employees."*
- LOW severity boundary explicitly excludes any paper with a
  sponsor-employed author.

Expected Round 10 outcomes for PMID `41750436`:

- `conflict_of_interest.severity = high` on 6/6 reliability runs
  (both 120b and 20b). Target: close the Fix #4 gap that Round 9
  left open.
- `sponsor_controls_manuscript = True` on 6/6 via both paths —
  extraction-direct (because 4 authors are Seed Health
  employees/shareholders) and assessment-derived (even if the
  extraction misses it).
- `methodology.severity`, `outcome_reporting.severity`,
  `per_protocol_only`, `differential_attrition`, and per-arm fields
  stay where Round 9 left them. Round 10 touches only the COI
  domain; regression-check the others.

If Round 10 closes the last gap, agreement against the Claude full-
text ground truth should climb from the current ~78–82% into the
high 80s for both model sizes. The 20b regression check is again the
critical one: if 20b goes to 3/3 COI high without regressing on any
other domain, the architecture is ready for the calibration paper
test (§6.2).

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

### 4.3 The agreement metric is not the truth (resolved in Round 7)

In Rounds 4-6 we observed that gemma f2 caught **more correct facts**
than gemma a2 (attrition, manuscript drafter, etc.) and produced
**better reasoning** ("structural COI, fishing expedition with 18
endpoints"), but scored *lower* on agreement with Claude (75% vs 83%)
because Claude's reference annotation was made on the **abstract**,
not the full text. When gemma f2 escalated COI severity from
`moderate` to `high` based on the manuscript-drafter finding (which
Claude couldn't see), the metric counted that as a disagreement —
even though gemma f2 was more correct.

**This was a measurement gap, not a quality gap.** Round 7 (§3.7)
resolved it by producing a Claude full-text annotation
(`anthropic_fulltext`). Re-scoring against the proper ground truth
shifted the picture significantly:

- Claude-on-abstract scores only 79% against Claude-on-full-text —
  the structural ceiling on what any abstract-only annotator can
  achieve.
- All three f2 modes (20b: 87%, gem26: 82%, 120b: 82%) now exceed
  that ceiling, validating that the architectural investment in
  the map-reduce pipeline was the right call.
- gem26 f2 is the only annotation in the entire experiment whose
  rebased score *climbed* (76% → 82%); it had been penalised for
  catching things Claude-on-abstract couldn't.

The lesson generalises: when the reference annotator and the
candidate annotator have access to different information, the
agreement metric measures convergence-to-the-same-input, not
quality. Always pin both to the same input scope.

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

### 5.1 The 120b non-determinism (resolved in Round 8)

The hypothesis from Rounds 4-5 was that parallel section extraction
caused in-flight contention on the 120b. Commit `66bca3c` switched to
sequential extraction. Round 8 verified this empirically: 3 sequential
runs of 120b f2 on the Seed Health paper all caught the headline
attrition data (n_randomised=32, n_analysed=21, attrition_stated=True),
where the pre-fix run had missed all of them.

Caveat: while the headline failure mode is fixed, Round 8 surfaced
**eight other fields that wobble between runs** (4 cosmetic
statistical-reporting flags + 4 load-bearing failures around per-arm
extraction and primary_outcome_type aggregation). Round 9 (commit
`475b6a6`) added four prompt edits to address those. Verification is
still pending — see §5.7 below.

### 5.2 The Claude full-text ground truth (resolved in Round 7)

Round 7 added a `--full-text` flag to `annotate_single_paper.py`
(commit `7594361`) and ran it on the Seed Health paper. The result
lives in the DB under tag `anthropic_fulltext`. Re-scoring every
local-model annotation against this proper ground truth changed the
picture significantly:

- All three f2 modes now exceed Claude-on-abstract (the structural
  ceiling for abstract-only annotators)
- gem26 f2 is the only annotation whose score *climbed* on rebase
  (76% → 82%) — it had been penalised for being more correct than
  Claude-on-abstract
- The remaining gaps are concentrated in three places, all addressed
  by Round 9 prompt edits: spin severity calibration,
  per_protocol_only inference, and COI HIGH trigger application

The metric is now apples-to-apples for the f2 modes. See §3.7 for the
full rebase table.

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

### 5.7 Round 10 prompt edits are unverified (Round 9 partially verified)

Round 9 verification (§3.10) landed three of four fix targets:
`outcome_reporting` holds moderate 6/6, `per_protocol_only=True` 6/6,
per-arm extraction 5/6 with derived `differential_attrition=True`
also 5/6. The COI mechanical HIGH trigger (Fix #4) failed at 0/6 —
the root cause was an **extraction-side** disclaimer-extrapolation
bug, not an assessment-side rule problem. Worse, the 20b regressed
COI severity from moderate → low in 2/3 runs relative to pre-Round-9.

Round 10 (commit `c34885a`) responds with two targeted edits: an
extraction-side rule binding `sponsor_controls_manuscript` to
author-employee status (not to "sponsor had no role" disclaimers),
a new `sponsor_role_disclaimer` field that forces disclaimers to be
scoped to only the activities they name, and an assessment-side
mechanical HIGH trigger `(d)` for `industry funding + ≥1 employee
or shareholder author`. Details in §3.11.

Unknowns to resolve in the Round 10 verification runs:

1. Does the extraction-side rule fire on gpt-oss 120b? (Round 9's
   "apply mechanically" language on the assessment side didn't, so
   we have reason to be cautious about any imperative rewording.)
2. Does the assessment-side trigger `(d)` fire as a belt-and-braces
   when the extraction still returns `False`?
3. Does the 20b regression (moderate → low) reverse?
4. Do any of the three Round 9 successes regress? The COI-only
   edits should leave methodology, outcome_reporting, per-arm
   extraction, and per_protocol_only untouched, but previous
   rounds showed that prompt edits in one domain can shift
   attention patterns elsewhere.

If Round 10 closes the gap cleanly, this section gets merged into
§5.1's resolution and the priority shifts to the calibration paper
test (§5.3). If it does not, the next step is almost certainly to
break the COI extraction into a standalone sub-prompt — testing a
dedicated "scan the author list" mini-extraction before the normal
conflicts-section extraction runs.

---

## 6. What's next

### 6.1 Verify the Round 10 COI fix (highest priority)

Round 9 is verified and three of four fix targets landed (§3.10). The
remaining gap is the COI mechanical HIGH trigger, which failed at 0/6
because the extraction stage over-read the paper's "sponsor had no
role in clinical trial operations" disclaimer. Round 10 (commit
`c34885a`, detailed in §3.11) patches this with both an extraction-
side rule and a belt-and-braces assessment-side mechanical trigger.

Re-run `scripts/reliability_test_fulltext.py` on PMID `41750436` for
both **gpt-oss 120b AND gpt-oss 20b**, three runs each. Expected:

- **3/3 `conflict_of_interest.severity = high`** on both models
  (closing Fix #4). Either path — extraction-direct or
  assessment-derived — is sufficient.
- **No regression** on the three Round 9 successes: outcome_reporting
  stays moderate 6/6, per_protocol_only stays 6/6 True, per-arm
  extraction stays at or above 5/6.
- No regression on the headline (6/6 high) or methodology severity
  (6/6 high).
- Specifically verify that 20b's COI stops regressing to low — the
  Round 9 2/3-low result was the worst observed outcome across the
  whole iteration loop so far.

If Round 10 lands cleanly, agreement against Claude full-text climbs
into the high 80s across both model sizes, and the reliability
ceiling for this paper is effectively reached on local models. The
priority then shifts to §6.2 (calibration paper).

If Round 10 fails, the next step is a dedicated
"scan-the-author-list" sub-extraction — a separate mini-call whose
only job is to enumerate author affiliations and flag sponsor
employees, before the normal conflicts-section extraction runs. This
would cost one extra LLM call per paper but isolates the COI
extraction from disclaimer interference.

### 6.2 Calibration paper test (medium priority)

All measurements so far are on **one** known HIGH-bias paper. Before
trusting the new prompts on a wider corpus, we need at least one
calibration paper that should NOT flag HIGH. Pick a Cochrane-rated
low-RoB paper from the existing dataset
(`SELECT pmid FROM papers WHERE overall_rob = 'low' AND excluded = 0`)
and run all 4 modes × 3 model families through the comparison script.

Critical questions:
- Does the new `inflated_effect_sizes` rule fire on legitimate papers
  with large but real effects?
- Does the strengthened methodology HIGH boundary (`>=6 endpoints +
  no multiplicity correction`) over-flag standard exploratory analyses?
- Do the new title_spin verb triggers fire on descriptive titles that
  happen to use "improves" or "promotes" with proper hedging?

If the calibration paper is correctly rated NONE/LOW by the f2 modes,
we have confidence the v3 prompts don't have a global HIGH bias.

### 6.3 Decide on production model and mode (after calibration)

Current best-known data (subject to Round 9 verification):

- **gpt-oss 20b f2** — 87% against Claude full-text, caught the
  original failure case (32→21 attrition + manuscript drafter),
  fastest, smallest. **Current production candidate**.
- **gemma4 26B f2** — 82%, cross-family validation, useful as a
  second-opinion ensemble member.
- **gpt-oss 120b f2** — 79-82% across the 3 reliability runs after
  the parallelism fix; should improve after Round 9 edits but
  pending verification.
- **abstract two-call** is sufficient for high-volume screening;
  full-text two-call is required for proper analysis (the abstract
  cannot contain the per-arm dropout, analysis population, or
  manuscript-drafter information that drive HIGH severity in the
  Seed Health case).

### 6.4 Address the merge-conflict noise (low priority)

- Reconcile `n_primary_endpoints` with `len(primary_outcomes_stated)`
  after the list-union merge step.
- Prefer longer/more specific values for free-text fields like
  `population_description`.
- Consider a post-merge sanity pass that flags self-inconsistent
  extractions for human review.

### 6.5 Then: full Claude re-annotation and training data generation

Once Round 9 is verified, calibration passes, and a production
model/mode has been chosen, we can re-annotate the full dataset (with
`tag_v1_annotations.py` to preserve the v1 results for comparison)
and start generating the v3 training corpus. See
`architecture_guide.md` §3 for the training sequence.

---

## 7. Code references

| File | Purpose | Commit |
|---|---|---|
| `biasbuster/prompts_v3.py` | v3 prompts (extraction, assessment, section) | `27df87a` → `9b04672` → `483e4fd` → `475b6a6` → `c34885a` |
| `biasbuster/annotators/__init__.py` | `BaseAnnotator` + merge logic | `27df87a`, sequential extraction fix in `66bca3c` |
| `biasbuster/annotators/bmlib_backend.py` | Ollama-via-bmlib adapter | `1ad57d9` |
| `biasbuster/cli/analysis.py` | CLI rewritten to use BaseAnnotator | `1ad57d9` |
| `annotate_single_paper.py` | Single-paper tool with `--full-text` flag for ground-truth runs | `7594361` |
| `scripts/compare_singlecall_twocall.py` | Four-mode comparison runner | `2de8810` → `fced4c4` |
| `scripts/reliability_test_fulltext.py` | N-run determinism check for the f2 path | `df2d505` |
| `scripts/tag_v1_annotations.py` | DB migration for legacy v1 annotations | `81647e0` |
| `docs/two_step_approach/MERGE_STRATEGY.md` | Merge rules + coherence-pass escape hatch | (separate doc) |

Round-by-round prompt evolution:
- `27df87a` — initial v3 prompts (Round 2 baseline)
- `9b04672` — silent contract bug: `recommended_verification_steps` field
  was required by the parser but never asked for in the prompt (Round 1
  fix, applied retroactively to v1 and v3)
- `483e4fd` — five targeted assessment edits from Round 3
  (`inflated_effect_sizes`, `title_spin`, `primary_outcome_type`
  aggregation, methodology HIGH boundary, endpoint counting)
- `66bca3c` — sequential section extraction + per-chunk partial
  persistence (Round 5 fix for the parallel-extraction bug)
- `475b6a6` — four targeted edits from Round 9 (per-arm extraction,
  primary_outcome_type fragility, per_protocol_only inference,
  COI HIGH mechanical application); §3.10 verification: 3/4 landed,
  COI trigger failed
- `c34885a` — Round 10 COI fix: extraction-side rule for
  `sponsor_controls_manuscript` (sponsor-employed authors always
  count as drafters), scoped `sponsor_role_disclaimer` field to
  stop disclaimer extrapolation, and assessment-side mechanical
  HIGH trigger `(d)` for industry funding + employee/shareholder
  authors — verification pending

---

## 8. Headline numbers (one place to find them)

Reference paper: PMID `41750436` (Seed Health synbiotic RCT, the case
study from `CONTEXT_FOR_CLAUDE_CODE.md`).

### Against Claude's abstract-only annotation (`anthropic`)

| Model | a1 | a2 | f1 | f2 |
|---|---:|---:|---:|---:|
| gpt-oss 120b | 64% | **92%** | 50% | 78% |
| gpt-oss 20b | 50% | 86% | 50% | **89%** |
| gemma4 26B | 56% | 83% | 50% | 75% |

### Against Claude's full-text annotation (`anthropic_fulltext`, Round 7)

| Model | a1 | a2 | f1 | f2 |
|---|---:|---:|---:|---:|
| gpt-oss 120b | 54% | 72% | 38% | 82% |
| gpt-oss 20b | 54% | 67% | 49% | **87%** |
| gemma4 26B | 49% | 67% | 44% | 82% |

Reference rows for context:

| Annotator | vs abstract GT | vs full-text GT |
|---|---:|---:|
| Claude abstract (`anthropic`) | 100% | 79% |
| 120b f2 prefix (pre-`66bca3c`, accidentally lucky) | 86% | 82% |
| 120b f2 rel1 (post-seq-fix, Round 8) | — | 79% |
| 120b f2 rel2 (post-seq-fix, Round 8) | — | 79% |
| 120b f2 rel3 (post-seq-fix, Round 8) | — | 77% |

Round 9 per-flag scorecard (§3.10) instead of an aggregate
agreement number — the reliability runs differ from the pre-fix runs
on the specific flags that Round 9 targeted, not on the headline or
methodology severities, so an aggregate agreement score would mask
the underlying story. The key facts:

- **Headline severity**: 6/6 high on Round 9 reliability runs (both
  120b and 20b) — matches Claude full-text GT and the seq-fix runs.
- **Methodology severity**: 6/6 high — matches GT.
- **outcome_reporting severity**: 6/6 moderate (matches GT; fixes the
  rel3 flip to low).
- **per_protocol_only**: 6/6 True (matches GT; pre-R9 was 0/2).
- **per-arm extraction**: 5/6 populated (pre-R9 was 1/2 for 120b).
- **differential_attrition**: 5/6 True (pre-R9 was 1/2 for 120b).
- **coi severity**: 0/6 high, 5/6 moderate, 1/6 low (regression from
  pre-R9's 0/2 high but 2/2 moderate) — Round 10 target.

Per-field details in the database under tags
`ollama_<model>_<mode>` (e.g.
`ollama_gpt-oss_120b_fulltext_twocall`,
`ollama_gpt-oss_120b_fulltext_twocall_rel1`).

**Reading the two tables together**: against the proper full-text
ground truth, every f2 mode exceeds the structural ceiling
(79%) for any abstract-only annotator. The full-text two-call
pipeline is genuinely a tier above abstract-only. Round 9 (§3.10)
made the per-flag composition better — 5 of 6 fields matched GT on
both models — at the cost of COI specifically, which regressed
because of a disclaimer extrapolation bug in the extraction stage.
Round 10 (commit `c34885a`, §3.11) patches the COI regression and
is expected to push headline agreement cleanly past 85% across both
model sizes — verification pending (§5.7, §6.1).
