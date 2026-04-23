"""QUADAS-2 prompts — seed text for Step 10 iteration.

Same orchestration shape as cochrane_rob2: one focused LLM call per
domain, each self-contained so the stateless call can't rely on
context from previous calls.

Key differences from RoB 2:

- Vocabulary: QUADAS-2 uses Yes/No/Unclear for signalling-question
  answers and Low/High/Unclear for the per-domain rating. The prompts
  spell out both so the LLM doesn't conflate the two.
- 2-D output: the first three domains require BOTH a bias rating and
  an applicability-concern rating. The fourth (flow and timing) has
  bias only. The prompt JSON-shape suffix accordingly varies.
- Extraction: we reuse the biasbuster ``SECTION_EXTRACTION_SYSTEM_PROMPT``
  for Stage 1 so the pull-facts-from-methods-and-results call stays
  methodology-agnostic.

Every constant in this module is a plain string (CLAUDE.md:
"prompts.py as single source of truth"). The stage dispatcher in
``__init__.py`` maps stage names to these constants.
"""

from __future__ import annotations

from biasbuster.prompts_v3 import SECTION_EXTRACTION_SYSTEM_PROMPT

# JSON-shape suffix for the three applicability-carrying domains
# (patient selection, index test, reference standard). The LLM must
# emit both ratings.
_JSON_SHAPE_APPLICABILITY = """

CRITICAL OUTPUT RULE: Respond with ONLY the JSON object below. No
prose, no markdown headings, no commentary, no thinking-aloud. Your
entire reply must start with `{` and end with `}` and parse as JSON.

Return JSON with exactly this shape:

{{
  "domain": "<DOMAIN_SLUG>",
  "signalling_answers": {{ "<question_id>": "<yes|no|unclear>", ... }},
  "bias_rating": "<low|high|unclear>",
  "applicability": "<low|high|unclear>",
  "justification": "<one or two sentences>",
  "evidence_quotes": [
    {{"text": "<verbatim excerpt>", "section": "<Methods|Results|...>"}}
  ]
}}
"""

# JSON-shape suffix for flow-and-timing — no applicability dimension.
_JSON_SHAPE_BIAS_ONLY = """

Return JSON with exactly this shape (flow and timing has no
applicability dimension — omit the applicability field):

{{
  "domain": "<DOMAIN_SLUG>",
  "signalling_answers": {{ "<question_id>": "<yes|no|unclear>", ... }},
  "bias_rating": "<low|high|unclear>",
  "justification": "<one or two sentences>",
  "evidence_quotes": [
    {{"text": "<verbatim excerpt>", "section": "<Methods|Results|...>"}}
  ]
}}
"""


def _with_applicability_shape(body: str, domain_slug: str) -> str:
    return body + _JSON_SHAPE_APPLICABILITY.replace(
        "<DOMAIN_SLUG>", domain_slug,
    )


def _with_bias_only_shape(body: str, domain_slug: str) -> str:
    return body + _JSON_SHAPE_BIAS_ONLY.replace(
        "<DOMAIN_SLUG>", domain_slug,
    )


# ---- Domain 1: Patient Selection -------------------------------------

_QUADAS_PATIENT_SELECTION_PROMPT = _with_applicability_shape("""\
You are applying QUADAS-2 to Domain 1 of 4: Patient Selection. Answer
each signalling question Yes / No / Unclear using evidence from the
study's methods section.

Signalling questions (bias):
1.1 Was a consecutive or random sample of patients enrolled?
1.2 Was a case-control design avoided?
1.3 Did the study avoid inappropriate exclusions?

Guidance:
- A case-control design (e.g. a deliberate mix of "diseased" cases and
  "healthy" controls) typically inflates apparent accuracy; answering
  "no" to Q1.2 means bias is almost certainly "high".
- Convenience sampling or post-hoc exclusion of ambiguous cases maps
  to "no" / "unclear" on Q1.1 / Q1.3 and raises bias concerns.

Bias rating guidance:
  - "low": all three signalling questions answered "yes".
  - "high": any signalling question answered "no" in a way that could
    plausibly inflate or deflate estimated accuracy.
  - "unclear": key information is missing and no-risk assessment is
    not defensible.

Applicability concern (separate from bias):
Did the included patients match the review question (target population,
setting, disease spectrum)? Rate:
  - "low" if patients closely match a primary-care / typical-clinical use;
  - "high" if patients deliberately differ (e.g. only severe disease,
    only a single sub-population);
  - "unclear" if patient spectrum can't be determined from the text.
""", "patient_selection")

# ---- Domain 2: Index Test --------------------------------------------

_QUADAS_INDEX_TEST_PROMPT = _with_applicability_shape("""\
You are applying QUADAS-2 to Domain 2 of 4: Index Test.

Signalling questions (bias):
2.1 Were the index test results interpreted without knowledge of the
    results of the reference standard?
2.2 If a threshold was used, was it pre-specified?

Guidance:
- "No" to Q2.1 indicates potential incorporation bias — the index-test
  result has been biased by knowing the reference-standard answer.
- "No" to Q2.2 flags threshold optimisation (picking the cutoff post-hoc
  to maximise reported accuracy). Common bias source in test-development
  studies.

Bias rating:
  - "low": both signalling questions "yes".
  - "high": any "no".
  - "unclear" otherwise.

Applicability concern:
Was the index test, its conduct, or its interpretation different from
the review question (e.g. a different device, different technician
training level)? Rate "low" / "high" / "unclear".
""", "index_test")

# ---- Domain 3: Reference Standard ------------------------------------

_QUADAS_REFERENCE_STANDARD_PROMPT = _with_applicability_shape("""\
You are applying QUADAS-2 to Domain 3 of 4: Reference Standard.

Signalling questions (bias):
3.1 Is the reference standard likely to correctly classify the target
    condition?
3.2 Were the reference-standard results interpreted without knowledge
    of the results of the index test?

Guidance:
- "No" on Q3.1 means the comparator isn't a proper gold standard (e.g.
  a noisy proxy like ICD code for a clinical diagnosis). A low-quality
  reference standard compresses true sensitivity/specificity.
- "No" on Q3.2 means review bias — the reference interpreter was
  influenced by knowing the index-test result.

Bias rating:
  - "low": both "yes".
  - "high": any "no".
  - "unclear" otherwise.

Applicability concern:
Did the reference standard target the same condition as the review
question (e.g. biopsy-confirmed cancer vs. imaging-suspicion)? Rate
"low" / "high" / "unclear".
""", "reference_standard")

# ---- Domain 4: Flow and Timing ---------------------------------------
# No applicability dimension — this domain is process-quality only.

_QUADAS_FLOW_AND_TIMING_PROMPT = _with_bias_only_shape("""\
You are applying QUADAS-2 to Domain 4 of 4: Flow and Timing. This
domain has NO applicability dimension — it is a process-quality check.

Signalling questions (bias):
4.1 Was there an appropriate interval between the index test and the
    reference standard?
4.2 Did all patients receive the same reference standard?
4.3 Were all patients included in the analysis?

Guidance:
- "No" on Q4.1 flags differential-verification-by-time concerns (the
  disease state may have changed between tests).
- "No" on Q4.2 flags differential verification (different reference
  standards for different subgroups; typically inflates accuracy).
- "No" on Q4.3 flags dropout / missing-data bias; sizeable withdrawals
  after enrollment are a classic flow-and-timing problem.

Bias rating:
  - "low": all three "yes".
  - "high": any "no" in a way that plausibly biases the estimated
    accuracy.
  - "unclear" otherwise.

Do NOT emit an applicability field for this domain.
""", "flow_and_timing")


# ---- Stage-name → prompt dispatch ------------------------------------

_STAGE_PROMPTS: dict[str, str] = {
    "extract_section": SECTION_EXTRACTION_SYSTEM_PROMPT,
    "domain_patient_selection": _QUADAS_PATIENT_SELECTION_PROMPT,
    "domain_index_test": _QUADAS_INDEX_TEST_PROMPT,
    "domain_reference_standard": _QUADAS_REFERENCE_STANDARD_PROMPT,
    "domain_flow_and_timing": _QUADAS_FLOW_AND_TIMING_PROMPT,
}


def build_system_prompt(stage: str) -> str:
    """Dispatch a stage name to the corresponding QUADAS-2 prompt."""
    try:
        return _STAGE_PROMPTS[stage]
    except KeyError as exc:
        raise KeyError(
            f"quadas_2 methodology has no prompt for stage {stage!r}. "
            f"Known stages: {sorted(_STAGE_PROMPTS)}"
        ) from exc


def domain_stage_name(domain_slug: str) -> str:
    """Return the stage name to feed ``build_system_prompt`` for a domain."""
    return f"domain_{domain_slug}"
