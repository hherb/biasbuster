"""Cochrane RoB 2 prompts — seed text for Step 8 iteration.

The prompts here implement the Cochrane RoB 2 flow described in
docs/ASSESSING_RISK_OF_BIAS.md and the Cochrane Handbook chapter 8:

- Stage 1 is shared with biasbuster: per-section extraction via
  ``prompts_v3.SECTION_EXTRACTION_SYSTEM_PROMPT``. We reuse it because
  the extraction *task* (pull randomization procedure, allocation
  concealment, blinding details, outcome measurement, etc. out of the
  methods text) is the same regardless of which risk-of-bias tool is
  applied downstream.
- Stage 2 has one prompt per RoB 2 domain. Each prompt lists the
  signalling questions from the Handbook and asks the LLM to answer
  each one Y/PY/PN/N/NI with an evidence quote, then emit a per-domain
  judgement (``low``/``some_concerns``/``high``) with a short rationale.

v1 prompts are intentionally minimal — just enough for the pipeline to
run end-to-end. Step 8 is where the prompts get tightened against real
Cochrane-reviewed papers with expert RoB 2 ground truth.

Every prompt is a module-level constant (CLAUDE.md: "Never inline prompt
text" / "prompts.py as single source of truth"). The stage dispatcher in
``__init__.py`` maps stage-name strings to these constants.
"""

from __future__ import annotations

# Reused from biasbuster's v3 path — extraction is methodology-agnostic
# (extract facts from text; let the assessor layer interpret them).
from biasbuster.prompts_v3 import SECTION_EXTRACTION_SYSTEM_PROMPT

# Per-domain assessment prompts. One LLM call per domain per outcome,
# so each prompt is self-contained (no cross-domain dependencies).
# The five domains and their signalling questions follow the Cochrane
# Handbook chapter 8 and the RoB 2 guidance document.

# ---- Domain 1: Randomization process -----------------------------------

_ROB2_RANDOMIZATION_PROMPT = """\
You are applying Cochrane RoB 2 to Domain 1 of 5: bias arising from the
randomization process. Answer each signalling question with one of:
Y (Yes), PY (Probably Yes), PN (Probably No), N (No), NI (No Information).

Signalling questions for this domain:
1.1 Was the allocation sequence random?
1.2 Was the allocation sequence concealed until participants were
    enrolled and assigned to interventions?
1.3 Did baseline differences between intervention groups suggest a
    problem with the randomization process?

For each question, provide the answer and a short evidence quote from
the paper (methods, results, supplement, or protocol). Then emit an
overall domain judgement using Cochrane's algorithm:

  - "low": Q1.1 yes/PY AND Q1.2 yes/PY AND Q1.3 no/PN.
  - "high": Q1.1 no/PN OR Q1.2 no/PN OR Q1.3 yes/PY.
  - "some_concerns": everything else (including NI in key questions).

Return JSON with exactly this shape:

{
  "domain": "randomization",
  "signalling_answers": {
    "1.1": "<Y|PY|PN|N|NI>",
    "1.2": "<Y|PY|PN|N|NI>",
    "1.3": "<Y|PY|PN|N|NI>"
  },
  "judgement": "<low|some_concerns|high>",
  "justification": "<one or two sentences>",
  "evidence_quotes": [
    {"text": "<verbatim excerpt>", "section": "<Methods|Results|...>"}
  ]
}
"""

# Each domain prompt below embeds the output-shape reminder via this
# suffix so the LLM sees a complete JSON-shape spec on every call — the
# calls are stateless, so cross-referencing "same shape as domain 1"
# would be unreliable. The ``<DOMAIN_SLUG>`` placeholder is substituted
# at module-load time.
_JSON_SHAPE_SUFFIX_TEMPLATE = """

Return JSON with exactly this shape:

{{
  "domain": "<DOMAIN_SLUG>",
  "signalling_answers": {{ "<question_id>": "<Y|PY|PN|N|NI>", ... }},
  "judgement": "<low|some_concerns|high>",
  "justification": "<one or two sentences>",
  "evidence_quotes": [
    {{"text": "<verbatim excerpt>", "section": "<Methods|Results|...>"}}
  ]
}}
"""


def _with_shape(body: str, domain_slug: str) -> str:
    """Append the JSON-shape reminder with the correct domain slug substituted."""
    return body + _JSON_SHAPE_SUFFIX_TEMPLATE.replace(
        "<DOMAIN_SLUG>", domain_slug,
    )


# ---- Domain 2: Deviations from intended interventions -----------------

_ROB2_DEVIATIONS_PROMPT = _with_shape("""\
You are applying Cochrane RoB 2 to Domain 2 of 5: bias due to deviations
from intended interventions. Answer each signalling question Y/PY/PN/N/NI.

This domain has two variants depending on the effect of interest:
- "effect of assignment to intervention" (intention-to-treat)
- "effect of adhering to intervention" (per-protocol)

For v1, answer the ITT variant. If the paper does not make the
distinction explicit, assume the primary analysis is ITT.

Signalling questions:
2.1 Were participants aware of their assigned intervention during the trial?
2.2 Were carers and people delivering the interventions aware of
    participants' assigned intervention during the trial?
2.3 If Y/PY/NI to 2.1 or 2.2: were there deviations from the intended
    intervention that arose because of the trial context?
2.4 If Y/PY to 2.3: were these deviations likely to have affected the outcome?
2.5 If Y/PY/NI to 2.4: were these deviations balanced between groups?
2.6 Was an appropriate analysis used to estimate the effect of
    assignment to intervention?
2.7 If N/PN/NI to 2.6: was there potential for a substantial impact on
    the result due to the failure to analyse participants in the group
    to which they were randomized?

Algorithm:
  - "low": 2.1/2.2 both N/PN AND 2.6 Y/PY.
  - "high": 2.5 N/PN OR 2.7 Y/PY.
  - "some_concerns": otherwise.
""", "deviations_from_interventions")

# ---- Domain 3: Missing outcome data -----------------------------------

_ROB2_MISSING_DATA_PROMPT = _with_shape("""\
You are applying Cochrane RoB 2 to Domain 3 of 5: bias due to missing
outcome data.

Signalling questions:
3.1 Were data for this outcome available for all, or nearly all,
    participants randomized?
3.2 If N/PN/NI: is there evidence that the result was not biased by
    missing outcome data?
3.3 If N/PN/NI to 3.2: could missingness in the outcome depend on its
    true value?
3.4 If Y/PY/NI to 3.3: is it likely that missingness depended on its
    true value?

Algorithm:
  - "low": 3.1 Y/PY OR (3.2 Y/PY).
  - "high": 3.4 Y/PY.
  - "some_concerns": otherwise.
""", "missing_outcome_data")

# ---- Domain 4: Measurement of the outcome -----------------------------

_ROB2_MEASUREMENT_PROMPT = _with_shape("""\
You are applying Cochrane RoB 2 to Domain 4 of 5: bias in measurement
of the outcome.

Signalling questions:
4.1 Was the method of measuring the outcome inappropriate?
4.2 Could measurement or ascertainment of the outcome have differed
    between intervention groups?
4.3 Were outcome assessors aware of the intervention received?
4.4 If Y/PY/NI to 4.3: could assessment of the outcome have been
    influenced by knowledge of the intervention received?
4.5 If Y/PY/NI to 4.4: is it likely that assessment was influenced?

Algorithm:
  - "low": 4.1 N/PN AND 4.2 N/PN AND 4.3 N/PN.
  - "high": 4.1 Y/PY OR 4.2 Y/PY OR 4.5 Y/PY.
  - "some_concerns": otherwise.
""", "outcome_measurement")

# ---- Domain 5: Selection of the reported result -----------------------

_ROB2_REPORTING_PROMPT = _with_shape("""\
You are applying Cochrane RoB 2 to Domain 5 of 5: bias in selection of
the reported result.

Signalling questions:
5.1 Were the data that produced this result analysed in accordance with
    a pre-specified analysis plan that was finalised before unblinded
    outcome data were available?
5.2 Is the numerical result likely to have been selected, on the basis
    of the results, from multiple eligible outcome measurements
    (scales, time points, analyses)?
5.3 Is the numerical result likely to have been selected, on the basis
    of the results, from multiple eligible analyses of the data?

Algorithm:
  - "low": 5.1 Y/PY AND 5.2 N/PN AND 5.3 N/PN.
  - "high": 5.2 Y/PY OR 5.3 Y/PY.
  - "some_concerns": otherwise.
""", "selection_of_reported_result")

# ---- Overall rationale synthesis --------------------------------------

_ROB2_OVERALL_SYNTHESIS_PROMPT = """\
You have completed RoB 2 assessment for all five domains of a single
outcome. The per-domain judgements follow Cochrane's worst-wins rule
for the outcome-level overall: "high" if any domain is "high", else
"some_concerns" if any domain is "some_concerns", else "low".

Given the domain judgements, write a 2-3 sentence rationale for the
overall judgement that names the domains driving the result.

Return JSON:
{
  "overall_judgement": "<low|some_concerns|high>",
  "overall_rationale": "<2-3 sentences>"
}
"""


# ---- Stage-name → prompt constant dispatch ----------------------------

# Keys match the convention used by the biasbuster pass-through adapter:
# stages are free-form strings the orchestration layer passes into
# ``methodology.build_system_prompt(stage)``.
_STAGE_PROMPTS: dict[str, str] = {
    "extract_section": SECTION_EXTRACTION_SYSTEM_PROMPT,
    "domain_randomization": _ROB2_RANDOMIZATION_PROMPT,
    "domain_deviations_from_interventions": _ROB2_DEVIATIONS_PROMPT,
    "domain_missing_outcome_data": _ROB2_MISSING_DATA_PROMPT,
    "domain_outcome_measurement": _ROB2_MEASUREMENT_PROMPT,
    "domain_selection_of_reported_result": _ROB2_REPORTING_PROMPT,
    "synthesize": _ROB2_OVERALL_SYNTHESIS_PROMPT,
}


def build_system_prompt(stage: str) -> str:
    """Dispatch a stage name to the corresponding Cochrane RoB 2 prompt.

    The dispatcher is a module-level dict lookup so mis-typed stage
    names fail loudly at call time with the list of valid stages.
    """
    try:
        return _STAGE_PROMPTS[stage]
    except KeyError as exc:
        raise KeyError(
            f"cochrane_rob2 methodology has no prompt for stage "
            f"{stage!r}. Known stages: {sorted(_STAGE_PROMPTS)}"
        ) from exc


def domain_stage_name(domain_slug: str) -> str:
    """Return the stage name to feed into :func:`build_system_prompt` for a domain."""
    return f"domain_{domain_slug}"
