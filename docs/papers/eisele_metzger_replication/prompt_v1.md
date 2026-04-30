# Prompt Specification v1 — RoB 2 Assessment

**Status:** LOCKED 2026-04-30 18:19:54 +0800.
**Companion to:** `preanalysis_plan.md` in the same folder.
**Source of truth for the running code:** `biasbuster/methodologies/cochrane_rob2/prompts.py` at the commit hash recorded below.
**Commit hash (lock):** `7854a1caefee7b5412f3be1e903e5bdc0ada9382`.

---

## 1. Architecture

Per RoB 2 outcome assessed, **6 LLM calls** are made:
1. Domain 1 — Randomization process
2. Domain 2 — Deviations from intended interventions (ITT variant)
3. Domain 3 — Missing outcome data
4. Domain 4 — Measurement of the outcome
5. Domain 5 — Selection of the reported result
6. Synthesis — Overall judgment from the 5 domain results

Total LLM calls per pass per RCT: **6**.
Total LLM calls in this study: **6 × 100 RCTs × 3 passes × 4 models × 2 input protocols = 14,400**.

This per-domain architecture mirrors the procedure followed by Eisele-Metzger 2025 (whose supplementary data contains per-domain Claude 2 rationale text for each of D1–D5) and matches the structure of the Cochrane RoB 2 tool itself. Single-shot all-domains prompting was rejected to avoid cross-domain rationale carryover within a single context window.

## 2. Pass independence

Each of the 3 passes per (model × RCT × protocol) is a fresh API/Ollama call with no shared context. Within a single pass, the 6 calls proceed sequentially and *do not share context* — each domain prompt is self-contained per the existing `prompts.py` design (line 33 comment: *"each prompt is self-contained (no cross-domain dependencies)"*). Synthesis (call 6) takes the structured outputs of calls 1–5 as its input but receives no domain rationale text; only the final domain judgments.

This means the synthesis call is **deterministic given the 5 domain judgments** and follows Cochrane's worst-wins rule. We retain it as a step (rather than computing in code) to capture the model's overall rationale prose — but the judgment itself is algorithmic.

## 3. System prompts (verbatim, frozen at lock hash)

The prompts below are the verbatim text emitted by `biasbuster/methodologies/cochrane_rob2/prompts.py::build_system_prompt(stage)` at the lock commit hash. All four evaluated models receive identical system prompts. Any subsequent edit to `prompts.py` invalidates this specification and requires a numbered amendment (`prompt_v2.md`) with re-locking.

### 3.1 Domain 1 — Randomization process (`stage="domain_randomization"`)

```
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

CRITICAL OUTPUT RULE: Respond with ONLY the JSON object below. No
prose, no markdown headings, no commentary, no thinking-aloud. Your
entire reply must start with `{` and end with `}` and parse as JSON.

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
```

### 3.2 Domain 2 — Deviations from intended interventions (`stage="domain_deviations_from_interventions"`)

(See `prompts.py` lines 114–143. ITT variant. JSON shape per Section 3.7 below with `domain: "deviations_from_interventions"`.)

### 3.3 Domain 3 — Missing outcome data (`stage="domain_missing_outcome_data"`)

(See `prompts.py` lines 147–165. JSON shape per Section 3.7 below with `domain: "missing_outcome_data"`.)

### 3.4 Domain 4 — Measurement of the outcome (`stage="domain_outcome_measurement"`)

(See `prompts.py` lines 169–186. JSON shape per Section 3.7 below with `domain: "outcome_measurement"`.)

### 3.5 Domain 5 — Selection of the reported result (`stage="domain_selection_of_reported_result"`)

(See `prompts.py` lines 190–208. JSON shape per Section 3.7 below with `domain: "selection_of_reported_result"`.)

### 3.6 Synthesis (`stage="synthesize"`)

```
You have completed RoB 2 assessment for all five domains of a single
outcome. The per-domain judgements follow Cochrane's worst-wins rule
for the outcome-level overall: "high" if any domain is "high", else
"some_concerns" if any domain is "some_concerns", else "low".

Given the domain judgements, write a 2-3 sentence rationale for the
overall judgement that names the domains driving the result.

CRITICAL OUTPUT RULE: Respond with ONLY the JSON object below. No
prose, no markdown headings, no commentary, no thinking-aloud. Your
entire reply must start with `{` and end with `}` and parse as JSON.

Return JSON:
{
  "overall_judgement": "<low|some_concerns|high>",
  "overall_rationale": "<2-3 sentences>"
}
```

### 3.7 Common JSON shape (per-domain calls)

```json
{
  "domain": "<DOMAIN_SLUG>",
  "signalling_answers": { "<question_id>": "<Y|PY|PN|N|NI>", ... },
  "judgement": "<low|some_concerns|high>",
  "justification": "<one or two sentences>",
  "evidence_quotes": [
    {"text": "<verbatim excerpt>", "section": "<Methods|Results|...>"}
  ]
}
```

## 4. User message construction (the "context" carrier)

The user message is constructed identically for both input protocols, with the same NI guidance applied to both. Differential framing between protocols is explicitly *not* used, since that would confound the protocol comparison (§6.5 of the pre-analysis plan).

### 4.1 User message template (verbatim, frozen)

```
You are assessing one outcome of a single RCT for risk of bias.

RCT identification:
  Study label: {rct_label}
  Outcome under assessment: {cr_outcome}

Available source materials follow. If a signalling question cannot be
answered from the available text, return "NI" (No Information) for that
question. Do not infer details that are not present in the text.

--- BEGIN SOURCE MATERIALS ---
{materials_block}
--- END SOURCE MATERIALS ---

Apply the RoB 2 procedure described in your system prompt and return
the JSON object specified.
```

### 4.2 `materials_block` for ABSTRACT-ONLY protocol

```
[ABSTRACT — main paper]
{abstract_text}
```

### 4.3 `materials_block` for FULL-TEXT protocol

```
[MAIN PAPER — full text]
{main_paper_text}

[PROTOCOL]
{protocol_text_or_NOT_AVAILABLE}

[TRIAL REGISTRATION]
{registration_text_or_NOT_AVAILABLE}
```

The strings `NOT AVAILABLE` are used verbatim when the corresponding source is missing. The model is told via the system prompt's NI rule to score signalling questions accordingly.

## 5. Decoding parameters (frozen)

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | model default | Each model's vendor-published default. We do not tune this; we want each model evaluated as a user would naively encounter it. |
| `max_tokens` | 2000 per call | Empirically sufficient for the JSON shape; 3-sigma above expected length. |
| `top_p` | model default | As temperature. |
| `response_format` | `json_object` if supported by the SDK; otherwise `text` with our own parser | Anthropic and OpenAI-compat SDKs may differ; the parser is canonical. |
| `seed` | not set | Deliberately not setting a fixed seed: between-pass variation is part of what we are measuring (run-to-run κ, §6.3). |
| Conversation history | none | Each of the 6 calls per pass is a fresh, stateless call. |

## 6. Output parsing

JSON is parsed with `annotators.__init__.parse_llm_json` (existing project utility, handles markdown-fence strip and minor whitespace tolerance). On parse failure the call is retried up to 2 times with the same prompt; on third failure the result is recorded as `parse_failure` with raw output preserved for audit. Per pre-analysis plan §8, if `parse_failure` rate exceeds 20% for any model, we halt and revise prompts before proceeding.

## 7. Mapping LLM judgments → Cochrane labels

| Our prompt label | Cochrane CSV label | Eisele-Metzger CSV label |
|---|---|---|
| `low` | "low risk of bias" | `low risk` |
| `some_concerns` | "some concerns" | `some concerns` |
| `high` | "high risk of bias" | `high risk` |

Mapping is exact and lossless. The benchmark database stores the canonical `low` / `some_concerns` / `high` form throughout.

## 8. What is NOT in this specification

- **Few-shot examples.** None included. We are evaluating zero-shot capability, matching Eisele-Metzger's setup.
- **Chain-of-thought prefixes.** The prompts demand pure-JSON output; no explicit "think step by step" preface. Models that emit reasoning before JSON have it stripped by the parser, not by prompt design.
- **Domain-2 per-protocol variant.** ITT only (per `prompts.py` line 122–124). Per-protocol assessment is out of scope.
- **Model-specific prompting tweaks.** All four models receive identical text. Model-specific optimization is rejected on transparency grounds.
