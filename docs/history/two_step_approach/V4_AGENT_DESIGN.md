# BiasBuster v4 — Tool-Calling Assessment Agent — Design

**Status:** design committed 2026-04-12. Phase 0 (this document). Phases 1–3 in flight.

**Audience:** anyone who will modify the assessment pipeline, anyone interpreting v4 output, and anyone trying to understand why the project pivoted from prompt engineering to agentic tool calling.

**Read before this:**
- [`INITIAL_FINDINGS_V3.md`](./INITIAL_FINDINGS_V3.md) — full empirical history of v3 (10 rounds of prompt engineering + reliability testing + the calibration test that motivates v4)
- [`DESIGN_RATIONALE_COI.md`](./DESIGN_RATIONALE_COI.md) — why the COI domain is intentionally more aggressive than Cochrane RoB 2; the hard-rule policy that v4 must preserve
- [`architecture_guide.md`](./architecture_guide.md) — the original two-call architecture rationale that v3 implemented

---

## §1 Why v4

### 1.1 What v3 + Round 10 achieved

The v3 two-call architecture (extraction → assessment) was the right move
on the original problem. It moved local-model agreement against Claude's
full-text annotation from ~50% (v1 single-call) to 79–87% on the
motivating Seed Health failure case, across three model families.
Round 10's Conflict-of-Interest mechanical trigger (d) closed the last
single-paper gap with all three families producing matching results
(see §3.12 in `INITIAL_FINDINGS_V3.md`).

### 1.2 What the calibration test revealed

The §6.2 calibration test (5 papers × 4 modes × 3 local models, scored
against Claude's full-text annotation) showed something the single-paper
reliability test could not: **the v3 prompt-side rule logic does not
generalise across the RoB spectrum**. Detailed in §3.13. The ranking
that Round 10 produced (gemma4 winning on the Seed Health paper)
inverted on the calibration set (120b winning on multi-paper headline
matches). And the cells where models failed all had the same shape:

- **Lidocaine paper (Cochrane HIGH)** — 20b f2 returned `low/0.25` and
  gemma4 f2 returned `moderate/0.45`. Both extracted the actual primary
  and secondary outcome lists correctly (1 + 7+ = 8+ entries), but the
  in-prompt assessment trusted the paper's broken `n_primary_endpoints=0`
  count field instead of counting the lists. Or, in gemma4's case,
  counted them correctly but failed to apply the
  `total_endpoints≥6 AND no_multiplicity_correction → HIGH` rule. The
  120b f2 only got the right answer because its reasoning text recovered
  past its own broken extraction ("at least six distinct outcome tests")
  and applied the rule manually.
- **rTMS paper (Cochrane LOW)** — 20b over-called COI from `moderate`
  to `high` on the strength of consulting-only ties (Brainsway, Magventure,
  Janssen, Lundbeck) — none of those authors are *employees*, so trigger
  (d) should not have fired. The 20b applied the consulting-tie signal
  to the trigger anyway. gemma4 went the other direction and rated
  everything `low/0.25`, missing the consulting-tie COI signal that
  Claude correctly flagged as `moderate`.
- **Both anchor failures and the 20b rel2 critical over-escalation in
  §3.12** — every single one of these is an arithmetic or boolean-logic
  failure. None of them is a text-reasoning failure.

Even Claude itself shows the symptom on the Seed Health paper: it sets
`n_primary_endpoints = 0` and `n_secondary_endpoints = 0` on the
extraction even though `primary_outcomes_stated` is a list of 10
items and `secondary_outcomes_stated` has 8+. Claude's reasoning text
recovers and applies the rules correctly because Claude can re-read
its own output, but the structured count fields are wrong.

### 1.3 The Option B experiment

On 2026-04-12, after seeing the calibration test results, I built a
proof-of-concept Python aggregator (`biasbuster/assessment/`) that
translates the v3 ASSESSMENT_DOMAIN_CRITERIA prompt (the 240 lines of
mechanical rules from `prompts_v3.py`) into pure Python. The
aggregator consumes the existing v3 extraction JSON, ignores the
unreliable `n_*_endpoints` count fields, counts the outcome lists
directly, and applies every rule and severity cascade
deterministically. Full provenance trace for each rating.

The validation script (`scripts/validate_v4_aggregator.py`) ran the
aggregator against the existing extractions in the database for the
5 calibration papers + Seed Health. Two findings:

**Hypothesis confirmed.** The aggregator fixes the lidocaine
catastrophic under-call on every model. Per-paper results:

| Model | v3 stored | v4 algorithmic | Δ on lidocaine paper |
|---|---|---|---|
| 120b f2 | high/0.68 ✓ | high/0.72 ✓ | (already correct) |
| 20b f2 | **low/0.25 ✗** | **high/0.72 ✓** | **fixed** |
| gemma4 f2 | **moderate/0.45 ✗** | **high/0.72 ✓** | **fixed** |

The v3 ranking on multi-paper calibration was 120b 4/5, 20b 3/5,
gemma4 3/5. With Python aggregation on the SAME extractions: 120b 3/5,
20b 4/5, gemma4 3/5. Aggregate looks like a wash, but **the wins and
losses are in different categories** — and one of the categories
matters much more than the other.

**Hypothesis exposed its limit.** The aggregator over-fires on the
rTMS paper (Cochrane LOW). Claude rates `moderate/0.35` on the same
paper, and Claude's own reasoning explains why:

> *"The HIGH boundary triggers: no_multiplicity_correction = TRUE
> AND total_endpoints ≥ 6. **However, calibrating against Cochrane
> RoB2 = LOW overall and the fact this is an explicitly exploratory
> secondary analysis of an RCT** (lower standard for multiplicity
> correction is common in this context, and the paper itself
> acknowledges exploratory/overfitting limitations), I rate
> MODERATE rather than HIGH for methodology."*

Claude **knows the rule fires**. Claude **chooses to override it**
based on contextual judgment that the paper is an exploratory
secondary analysis where the rule shouldn't apply mechanically.
The Python aggregator has no concept of "exploratory secondary
analysis" and no way to override its own rules — it just sees
`no_multiplicity_correction=True AND total_endpoints=11 ≥ 6` and
fires HIGH every time.

**This is the architectural floor.** There is a category of decisions
that genuinely requires reading the paper for context, not just
applying boolean logic to extracted fields. Pure Python cannot
replicate it. Pure LLM-with-rules-in-the-prompt cannot do the
arithmetic reliably (the v3 calibration failures). Both halves are
necessary.

### 1.4 The tool-calling reframing

What I initially proposed for v4 was a three-stage pipeline:

```
extraction (LLM) → mechanical aggregation (Python)
                 → context gate (LLM mini-call)
                 → final assessment (Python with overrides)
```

The user proposed a much better architecture: **make the Python
aggregator a tool the LLM can call**, and let the LLM be the agent.

```
extraction (LLM) → assessment agent (LLM + tools)
                       │
                       ├─ run_mechanical_assessment(extraction)  ← Python
                       ├─ check_clinicaltrials(nct)              ← API
                       ├─ check_open_payments(authors)           ← API
                       ├─ ... (more tools)
                       └─ produces final assessment with overrides
```

This is simpler in three ways:

1. **One agent loop instead of three pipeline stages.** The "should
   I override the mechanical rule?" decision and the "should I gather
   more verification data?" decision are the same kind of decision —
   "given what I know, should I do more before answering?". Forcing
   them into separate stages was artificial.

2. **The smaller policy prompt replaces the 20 KB rules prompt.**
   v3's `ASSESSMENT_SYSTEM_PROMPT` is 877 lines and ~43 KB; ~240 of
   those lines are mechanical rules. v4's
   `ASSESSMENT_AGENT_SYSTEM_PROMPT` is ~80 lines and ~2 KB —
   framing, tool descriptions, override policy, and the output schema.
   Everything else is in Python where it belongs.

3. **New tools cost almost nothing.** Database lookups, API queries,
   future Python checkers all fit naturally as tool wrappers. The
   agent loop doesn't change. The prompt doesn't grow. The only
   thing that has to be added is the tool definition and the
   dispatch case. See §3 for the catalogue and the upcoming
   tool list.

The price is one prerequisite: **bmlib does not yet support tool
calling.** It declares `supports_function_calling=True` on three
provider capability classes but the implementation is missing.
Phase 1 of the v4 plan extends bmlib for tool calling — work that
is valuable on its own merits regardless of v4 (any future tool
that uses bmlib benefits) and should land before v4 wires it in.

---

## §2 Architectural decisions

The numbered decisions below are referenced from the implementation
plan at `~/.claude/plans/serialized-snacking-dragonfly.md`. Each one
records the choice, the reason, and (where relevant) the alternatives
considered.

### D1 — Architecture: extraction (LLM) → assessment agent (LLM + tools)

Two stages, not three. The previously-considered three-stage pipeline
(Python aggregate → LLM gate → Python final) collapses into a single
agent loop because the override decision and the verification decision
are the same kind of decision. The `run_mechanical_assessment` tool
consumes the extraction and returns draft severities + provenance;
the LLM agent reviews them, optionally calls verification tools, and
emits a final assessment with explicit overrides for any mechanical
rule the contextual judgment deemed inapplicable.

### D2 — bmlib gets tool calling first (Phase 1), as a prerequisite

Implemented as a feature branch in `/Users/hherb/src/bmlib/`
(`feature/tool-calling`), merged to main when biasbuster v4 has
validated against it (end of Phase 2). Three providers in scope:
Ollama (native tool API via ollama-python ≥0.3), Anthropic
(`anthropic` SDK `tools=`), OpenAI-compatible (`openai` SDK `tools=`).

Why bmlib and not raw HTTP? Because three model families means three
HTTP protocols, and we already have a clean adapter pattern in
`bmlib/llm/providers/`. Adding tool calling there once is the same
work as adding it three times in biasbuster.

### D3 — Public API: explicit `tools` parameter on `LLMClient.chat()`

Following the `json_mode` precedent. Not buried in `**kwargs`. New
types `LLMToolDefinition` / `LLMToolCall` in `bmlib/llm/data_types.py`.
`LLMResponse` extended with optional `tool_calls` field. Provider
capability flag (`supports_function_calling`) becomes load-bearing —
clients check it before passing tools, and `LLMClient.chat()`
raises a clear error if the resolved provider doesn't support tools.

Alternative considered: pass `tools` via `**kwargs` like the existing
`think=` parameter does on Ollama. Rejected because it's not
discoverable in the API surface and because validation would have
to happen inside each provider rather than once in `LLMClient`.

### D4 — Tool definition format: OpenAI-style

Single shared format because (a) ollama-python ≥0.3 accepts it,
(b) openai SDK requires it, (c) the Anthropic provider does the
small format conversion internally. Saves callers from learning
three formats. The conversion to Anthropic's format
(`{"name": ..., "description": ..., "input_schema": ...}`) lives
in the Anthropic provider, not the caller.

### D5 — v4 lives in `biasbuster/assessment_agent.py` (single file ~400 lines)

Stays under the 500-line guideline. Imports the existing
`biasbuster/assessment/` aggregator and the existing
`biasbuster/agent/tools.py` wrappers. No new package needed.

### D6 — Legacy `biasbuster/agent/` is NOT touched in this work

It stays in place behind the existing `--verify` flag as the
regex-routed verification path. v4 is additive. Once v4 is validated
and stable in production, a separate retirement commit can remove
the legacy planner and runner. Tools wrappers in `tools.py` are
shared and remain useful for both.

### D7 — New annotator method `BaseAnnotator.annotate_full_text_agentic()`

Runs the existing extraction stage and then hands the result to the
assessment agent. Existing `annotate_full_text_two_call()` is
untouched. No backend-specific overrides — the agent loop is generic
and uses the annotator's LLM client via a new `_call_llm_with_tools()`
method that all three concrete annotators implement.

### D8 — Phase 2 validation uses Anthropic Claude only

Cleanest tool API, best documented, gives a known-good baseline
that isolates "is the architecture sound?" from "does the local
model handle tool calls well?". Local models (gemma4 26B,
gpt-oss 20B/120B) join in Phase 3 once Claude validation passes.

### D9 — Hard-rule enforcement: post-hoc Python check

After the LLM agent finishes, a Python pass verifies the LLM didn't
downgrade any rule marked `overridable=False` in
`biasbuster/assessment/rules.py`. If it did, the Python pass forces
the severity back up and appends an audit note like
`model attempted to override trigger (d) — rejected per policy`.
This requires adding an `overridable: bool` metadata flag to each
`Rule` instance. Trigger (d) and trigger (a) are non-overridable;
the multiplicity / per-protocol / spin rules ARE overridable, as
the rTMS case showed they should be.

Alternative considered: trust the prompt instruction alone. Rejected
because we have evidence (Round 9 mechanical-trigger language failing
to fire on local models) that prompt instructions are not enforceable
contracts. The post-hoc check is 20 lines of Python and closes the
loophole completely.

### D10 — Documentation is the first deliverable

This document is Phase 0. The design doc gets committed and pushed
BEFORE any code is written, so the design is reviewable and
decoupled from implementation. If the design is wrong, the cost
of finding out is reading a markdown file, not unwinding hundreds
of lines of code.

### D11 — New v4 prompt at `biasbuster/prompts_v4.py` is short and policy-focused (~2 KB)

It contains the framing (risk of bias not proof), the tool inventory
with descriptions, the override policy, and the output schema. The
240 lines of mechanical rules from `prompts_v3.py`
ASSESSMENT_DOMAIN_CRITERIA are deleted because they now live in
Python and the LLM accesses them by calling `run_mechanical_assessment`.

### D12 — Maximum agent iterations capped at 5

Guards against runaway tool-calling loops. Each iteration: model
emits either `tool_calls` or final JSON. If iteration 5 has not
produced final JSON, the agent forces a final-answer call with
`tool_choice="none"`. Empirical expectation based on the workload:
1 call to `run_mechanical_assessment` + 0–3 verification tool calls
+ final answer = 2–5 iterations typical. The cap is generous enough
that real workloads should never hit it.

---

## §3 The tool catalogue

Tools the v4 agent can call, organised by category. Each entry shows
the function signature, what it does, the data source, and whether
it's available in Phase 2 vs added later.

### 3.1 Computation tools

These are pure-Python functions with no I/O. Deterministic, fast,
unit-testable in isolation.

| Tool | Phase | What it does |
|---|---|---|
| `run_mechanical_assessment(extraction)` | **Phase 2** | Wraps `biasbuster.assessment.assess_extraction`. Returns the full draft assessment with `_provenance` showing every rule that fired, what inputs it read, and the resulting per-domain severities. **Always called first** by the agent. |

The agent's first action on every paper is to call this tool. Its
output is the substrate the agent reasons over: "given these
mechanically-derived flags and these pre-computed severities, do
I need to override anything based on context I see in the
extraction?".

### 3.2 Existing verification tools (already in `biasbuster/agent/tools.py`)

The legacy verification agent (regex-routed) already wraps these.
The v4 agent reuses them as-is.

| Tool | Phase | What it does | Backend |
|---|---|---|---|
| `check_clinicaltrials(nct_id, abstract, title, pmid)` | **Phase 2** | Looks up trial registration on ClinicalTrials.gov. Detects outcome switching by comparing registered outcomes to the paper. Auto-extracts NCT ID from abstract or title if not provided. | ClinicalTrials.gov v2 API |
| `check_open_payments(authors)` | **Phase 2** | Searches CMS Open Payments for industry payment records to the first 3 authors. | CMS Open Payments API |
| `check_orcid(authors)` | **Phase 2** | Fetches author affiliation histories. Flags undisclosed industry ties. | ORCID API |
| `check_europmc_funding(pmid, doi)` | **Phase 2** | Queries Europe PMC for grant/funder metadata that the paper may not have disclosed. | Europe PMC API |
| `check_retraction_status(pmid, doi)` | **Phase 2** | Checks PubMed title and Crossref `updated-by` for retraction notices issued after publication. | PubMed + Crossref |
| `run_effect_size_audit(pmid, title, abstract)` | **Phase 2** | Local heuristic. Re-runs the effect-size auditor from `enrichers/effect_size_auditor.py`. No network. | Pure Python |

These tools were designed for the legacy verification flow but are
backend-agnostic. The v4 agent calls them via the same dispatcher.
Their `ToolResult` shape (defined in `biasbuster/agent/tools.py`)
fits cleanly into the agent's tool-result protocol.

### 3.3 Future tools (Phase 3+)

These are not built in v4 but the architecture explicitly supports
adding them. Listed here so the design doc covers extensibility.
Each is a small wrapper, ~50 lines.

| Tool | Backend | Why it's interesting |
|---|---|---|
| `lookup_cochrane_rob(pmid)` | SQL — biasbuster DB | We already have Cochrane RoB 2 ratings for hundreds of papers in our `papers` table. The agent can cross-check its judgment against expert ratings on papers we've seen before. |
| `query_prior_annotations(pmid, model_name)` | SQL — biasbuster DB | "Have we already annotated this paper with a stronger model?" Saves cost on incremental updates and lets the agent compare its own judgment to a prior reference. |
| `check_effect_size_artifact(quote)` | Python | Given an effect size quote like "13,008% increase", computes whether the denominator was near zero and flags the result as a measurement artefact rather than a real effect. Currently buried inside the v3 prompt's `inflated_effect_sizes` rule; making it a tool lets the agent explicitly verify and explain. |
| `check_protocol_registration_divergence(nct_id, paper_outcomes)` | API + diff | Fetches the registered protocol from ClinicalTrials.gov and computes a structural diff against the paper's reported outcomes. Catches outcome switching without the LLM needing to parse the registry response. |
| `lookup_author_affiliation_history(orcid_or_name)` | ORCID | Returns the author's full employment history. Stronger COI signal than what the paper discloses. |
| `check_protocol_amendments(nct_id)` | API | Returns the list of protocol amendments to a registered trial with timestamps. Late amendments to primary outcomes are a strong signal of post-hoc design changes. |
| `get_paper_citations_by_authors(author_list, journal)` | API | Returns same-journal prior publications by the same authors. Pattern detection: "this group has published 5 trials with the same sponsor in the last 3 years, all positive". |
| `lookup_funder_history(funder_name)` | DB or API | Returns the funder's history of supported trials and their outcomes. Pattern detection: "this funder has 12 published trials, all reporting favourable results — possible publication-selection bias". |

The agent doesn't need to be reprogrammed to use new tools. New
tool = new wrapper + tool definition + dispatch case. The prompt
inventory grows by one bullet point.

---

## §4 The new prompt

The full text of `biasbuster/prompts_v4.py` is below, annotated.
Total size: ~80 lines / ~2 KB. This replaces v3's 877-line / ~43 KB
`ASSESSMENT_SYSTEM_PROMPT`.

```python
"""
BiasBuster v4 — Assessment Agent system prompt.

Replaces v3's 240-line ASSESSMENT_DOMAIN_CRITERIA prompt with a short
policy prompt that instructs the LLM to use tools for the mechanical
parts.
"""

ASSESSMENT_AGENT_SYSTEM_PROMPT = """\
You are a clinical-trial bias assessor. You have just been given the
structured facts extracted from a randomised trial (sample sizes,
outcomes, funding, author affiliations, etc.). Your job is to produce
a calibrated risk-of-bias assessment.

CRITICAL FRAMING — risk of bias, NOT proof of bias

Your output is a *risk* assessment, not a forensic finding. A HIGH
rating means "a reader should independently verify these findings
before accepting them", not "this paper is wrong". The categorical
severity and the numeric overall_bias_probability work together: a
paper rated `high/0.68` carries structural risk that warrants
verification, while a paper rated `high/0.85` has multiple
co-occurring concerns. Use both axes to express what you see.

TOOLS YOU CAN CALL

You MUST call run_mechanical_assessment FIRST on every paper. It
returns the draft per-domain severities and a full provenance trace
showing exactly which rules fired on which extracted values. Treat
its output as the starting point.

After reviewing the mechanical assessment:

1. For each domain, ask: does the rule that fired actually apply to
   this specific paper, or is this paper a legitimate exception? For
   example:
     - The multiplicity correction rule fires on an explicitly
       EXPLORATORY secondary analysis → override to lower severity
       and explain.
     - The differential_attrition rule fires but the paper has
       robust ITT with multiple-imputation → override and explain.
     - The title_spin rule fires on a descriptive title that
       contains a listed verb by coincidence → override and explain.

2. For borderline cases, optionally call verification tools:
     - check_clinicaltrials(nct_id) — verify registered vs reported
       outcomes when you suspect selective reporting.
     - check_open_payments(authors) — industry payment records
       when COI concerns exceed what the disclosure mentions.
     - check_orcid(authors) — author affiliation histories for
       undisclosed industry ties.
     - check_europmc_funding(pmid) — funder metadata.
     - check_retraction_status(pmid) — post-publication notices.
     - run_effect_size_audit(pmid, title, abstract) — re-runs the
       effect-size auditor when you want a second opinion on
       inflated_effect_sizes.

3. Produce the final assessment as a JSON object matching the
   schema below. For EVERY override of a mechanical rating, list
   the override in `_overrides` with the domain, the original
   severity, the new severity, and the contextual reason.

HARD RULES (you may NOT override these)

The Conflict-of-Interest mechanical triggers (a)–(d) are STRUCTURAL
risk signals and are not subject to your contextual judgment. If
run_mechanical_assessment returns `conflict_of_interest.severity = high`
because trigger (d) fired (industry funding + ≥1 sponsor-employed or
shareholder author), you MUST keep that rating at high. You MAY,
however, position the overall_bias_probability at the lower edge of
the HIGH anchor range (0.65–0.72) when the methodology domains are
otherwise clean — that's how you express "structural risk present,
methodology fine". See DESIGN_RATIONALE_COI.md for the policy.

A post-hoc enforcement check will reject any output that downgrades
a non-overridable rule. If you try, your downgrade will be silently
reverted and an audit note will be added saying you tried.

OUTPUT SCHEMA

Return ONLY a JSON object. No preamble, no markdown fences. Schema:

{
  "statistical_reporting": { ... },     // same shape as v3
  "spin": { ... },
  "outcome_reporting": { ... },
  "conflict_of_interest": { ... },
  "methodology": { ... },
  "overall_severity": "none|low|moderate|high|critical",
  "overall_bias_probability": 0.00-1.00,
  "recommended_verification_steps": [ "..." ],
  "reasoning": "step-by-step text",
  "confidence": "low|medium|high",
  "_overrides": [
    {
      "domain": "methodology",
      "mechanical_severity": "high",
      "final_severity": "moderate",
      "reason": "Exploratory secondary analysis where multiplicity
                  correction is not customarily required..."
    }
  ]
}

Domain severity boundaries are encoded in run_mechanical_assessment.
You do not need to know them — you just need to decide when to
override what the tool returned.
"""
```

The interesting design choice is that **the prompt does not contain
the severity boundary rules**. The LLM never sees "moderate is when
exactly one of relative_only / selective_p_values / subgroup_emphasis
is true". It only sees the *output* of those rules and decides
whether to override. This is a much smaller surface area for the
prompt to get wrong, and it means rule changes happen in Python
(version-controlled, unit-testable) rather than in prompts
(invisible until something breaks).

---

## §5 Override policy

The `overridable` flag on each rule in `biasbuster/assessment/rules.py`
controls whether the LLM agent is allowed to downgrade a domain
rating below what the rule produced. This section documents which
rules are overridable and which are not, with worked examples for
both ends of the policy.

### 5.1 Non-overridable rules

These are structural signals that the LLM cannot downgrade no
matter what context it sees. The post-hoc enforcement check
(`AssessmentAgent._enforce_hard_rules`) rejects any final output
where one of these severities has been downgraded.

| Rule | Domain | Why non-overridable |
|---|---|---|
| **Trigger (a)** — `sponsor_controls_analysis AND sponsor_controls_manuscript` | conflict_of_interest | Sponsor employees controlling both analysis AND manuscript drafting compromises the independence of every claim in the paper. Disclosure does not undo the conflict. |
| **Trigger (b)** — industry funding AND undisclosed COI AND industry author affiliations | conflict_of_interest | Three concurrent risk factors. The combination is the signal. |
| **Trigger (c)** — `sponsor_controls_analysis` AND surrogate primary outcomes | conflict_of_interest | The sponsor controls how surrogate measurements are interpreted, which is the highest-leverage intervention point for selective reporting. |
| **Trigger (d)** — industry/mixed funding AND ≥1 author with role employee or shareholder | conflict_of_interest | Authorship by a sponsor employee IS the sponsor's role in the manuscript, regardless of any "sponsor had no role" disclaimer. The Round 10 design decision documented in `DESIGN_RATIONALE_COI.md`. |

Critically: the LLM CAN still adjust `overall_bias_probability`
within the HIGH anchor band (0.65–0.85) on these papers. A trigger-(d)
paper with otherwise-clean methodology should land at the bottom of
the HIGH range (~0.68), expressing "structural risk present, methodology
fine". A trigger-(d) paper with multiple co-occurring methodology
concerns should land near the top (~0.85). The categorical severity
is fixed; the probability captures the gradation.

### 5.2 Overridable rules

Everything else. The methodology cascade (multiplicity / per-protocol /
attrition / sample size), the spin cascade (title_spin / extrapolation),
the statistical reporting cascade — all of these can be overridden
when the LLM judges that the rule fires technically but the context
makes it inappropriate.

#### Worked example: rTMS paper (PMID 32382720) — multiplicity override

The rule that fires:
```
no_multiplicity_correction = TRUE
total_endpoints = 11 (8 primary + 3 secondary in Claude's extraction)
→ methodology.severity = HIGH
```

What Claude actually rated: `methodology = moderate`.

Claude's reasoning text:
> *"The HIGH boundary triggers: no_multiplicity_correction = TRUE
> AND total_endpoints ≥ 6. However, calibrating against Cochrane
> RoB2 = LOW overall and the fact this is an explicitly exploratory
> secondary analysis of an RCT (lower standard for multiplicity
> correction is common in this context, and the paper itself
> acknowledges exploratory/overfitting limitations), I rate
> MODERATE rather than HIGH for methodology. Two concerns present
> (per_protocol_only + no multiplicity correction) support
> MODERATE."*

This is a legitimate override. The mechanical rule fires correctly,
but the context (exploratory secondary analysis, paper acknowledges
its own limitations, Cochrane gave it LOW overall) means the rule
should not produce HIGH on this specific paper.

In v4, the agent's output for this paper would include:
```json
"_overrides": [
  {
    "domain": "methodology",
    "mechanical_severity": "high",
    "final_severity": "moderate",
    "reason": "Explicitly exploratory secondary analysis of an RCT. Lower
               standard for multiplicity correction is customary in this
               context, and the paper itself acknowledges
               exploratory/overfitting limitations. Cochrane RoB 2 rating
               is LOW overall."
  }
]
```

The Phase 2 validation script explicitly tests for this override
behaviour on PMID 32382720.

#### Worked example: tapinarof paper (PMID 39777610) — non-overridable trigger (d)

The rule that fires:
```
funding_type = "industry" (Dermavant Sciences)
authors_with_industry_affiliation = [
  Brown (employee), Piscitelli (employee), Tallman (employee),
  Rubenstein (employee), Simpson (consultant), ...
]
→ trigger (d) fires → conflict_of_interest.severity = HIGH
```

The mechanical assessment returns COI=high. The LLM might be
tempted to override on the basis that:
- The trial is well-conducted (Cochrane LOW)
- COI is fully disclosed
- Statistical reporting is good

These are all valid observations, but they don't lift the structural
COI risk — `DESIGN_RATIONALE_COI.md` is clear that authorship by a
sponsor employee is itself the sponsor's role in the manuscript and
that disclosure does not undo it.

If the LLM tries to downgrade `coi.severity` to `moderate`, the
post-hoc enforcement check forces it back to `high` and appends:
```json
"_overrides": [
  {
    "domain": "conflict_of_interest",
    "attempted_severity": "moderate",
    "final_severity": "high",
    "policy": "trigger_d_non_overridable",
    "reason": "Mechanical trigger (d) is structural — see
               DESIGN_RATIONALE_COI.md. Severity reverted to mechanical
               value. The LLM may adjust overall_bias_probability within
               the HIGH band but cannot downgrade the categorical rating."
  }
]
```

The same paper would have its `overall_bias_probability` set by the
LLM to ~0.68 (bottom of HIGH band) reflecting that the methodology
itself is sound — that adjustment IS allowed.

### 5.3 Adding new non-overridable rules

If future calibration work reveals that another rule needs to be
hard-enforced, adding it is a one-line change in
`biasbuster/assessment/rules.py`: set `overridable=False` on the
relevant `Rule` constructor. The post-hoc enforcement check picks
it up automatically. No prompt edits, no agent retraining.

The bar for non-overridability is high. By default, every rule is
overridable. The four COI triggers above are the only initial
non-overridable rules because they encode an explicit policy
documented at length in `DESIGN_RATIONALE_COI.md`. Other rules
should not become non-overridable without similar explicit
documentation of why context can never excuse them.

---

## §6 Validation plan

The Phase 2 validation answers one question: **does the v4 agent,
running on Claude, fix the v3 calibration failures without
introducing new ones?**

### 6.1 Test set

Same 5 calibration papers as `INITIAL_FINDINGS_V3.md` §6.2:

| PMID | Cochrane RoB | Role |
|---|---|---|
| 32382720 | low (5/5) | Academic LOW anchor (rTMS depression). Tests the override-on-exploratory case. |
| 39777610 | low | Industry LOW (Dermavant tapinarof). Tests trigger (d) non-override enforcement. |
| 39905419 | some_concerns | Public-funded middle-of-scale. Tests calibration on the moderate end. |
| 39691748 | high | Academic HIGH (lidocaine patch). Tests that v4 reproduces v3's methodology HIGH on a paper where v3 was correct. |
| 41750436 | n/a | Seed Health synbiotic — the motivating failure case. Tests the headline severity holds. |

Comparing against:
1. **Claude's v3 full-text annotation** (`anthropic_fulltext` DB tag) — the
   reference. Same model, different architecture.
2. **The v4 algorithmic baseline** from Option B (`assess_extraction`
   on the same extraction). Tells us what Python alone produces.
3. The legacy v3 local-model results from §6.2 — the failure cases
   v4 is supposed to fix. Phase 2 doesn't run local models, but the
   comparison numbers are stored and the new validation script
   prints them for context.

### 6.2 Pass criteria for Phase 2

| Paper | Pass criterion |
|---|---|
| 32382720 (rTMS, LOW) | v4 produces `overall_severity = moderate` AND has at least one entry in `_overrides` for `methodology` (mechanical=high → final=moderate) with a reason mentioning "exploratory" or "secondary analysis". |
| 39777610 (tapinarof, LOW industry) | v4 produces `overall_severity = high`. The post-hoc enforcement check should NOT need to fire (the LLM should leave trigger (d) alone). If a downgrade attempt is detected, the audit note must be present. |
| 39905419 (balneotherapy, some_concerns) | v4 produces `overall_severity = moderate` with `bias_probability` in the 0.40–0.65 range. |
| 39691748 (lidocaine, HIGH) | v4 produces `overall_severity = high` AND `methodology.severity = high`. (This is the v3 catastrophic-failure case for 20b f2; the algorithmic aggregator already fixes it; v4 must not regress.) |
| 41750436 (Seed Health, motivating) | v4 produces `overall_severity = high` with `bias_probability ≥ 0.78`. |

### 6.3 Operational pass criteria

- Agent loop terminates within 5 iterations on every paper
- No tool dispatch errors in the logs
- Token usage recorded per paper for Phase 3 cost comparison
- All 5 papers stored in the DB under the new
  `anthropic_fulltext_agentic` tag (preserves the v3 results for A/B)

### 6.4 Phase 3 validation (local models)

Once Phase 2 passes, Phase 3 reruns the same matrix on gemma4 26B,
gpt-oss 20B, and gpt-oss 120B. The pass criterion shifts from
"matches Claude" to "matches Claude v4 on at least 4/5 papers per
local model". The interesting observations Phase 3 will produce:

- Does the algorithmic Python aggregator + LLM override structure
  hold across model families, or is it Claude-specific?
- Do local models correctly identify the rTMS-style override case
  ("this is exploratory, override the multiplicity rule"), or do
  they over-override and produce false negatives?
- Does trigger (d) enforcement fire on local models — i.e. do they
  attempt to downgrade COI ratings that Claude leaves alone?
- Token cost per paper relative to v3 two-call

---

## §7 Migration path

v4 is **additive** for the entire roll-out. v3 stays the default
behaviour throughout Phase 2 and most of Phase 3.

### 7.1 Phase 2 — opt-in `--agentic` flag

`annotate_single_paper.py --agentic` (combined with `--full-text`)
runs the v4 agent. Without the flag, v3 two-call is unchanged.
Results land in the DB under `anthropic_fulltext_agentic` so they
do not overwrite the v3 `anthropic_fulltext` annotations. Anyone
running the existing pipeline sees no change.

### 7.2 Phase 3 — local models opt in

When `BmlibAnnotator._call_llm_with_tools` is implemented (after
bmlib's tool calling lands), the same `--agentic` flag works with
local models. Still opt-in, still parallel storage tags.

### 7.3 End of Phase 3 — v4 becomes the default

`annotate_single_paper.py --full-text` and the pipeline's
`--stage annotate` switch to v4 agentic by default. v3 two-call
remains accessible behind a `--legacy-twocall` flag for direct
comparison. The README's "Quick Start" updates to recommend the
v4 path.

### 7.4 Future cleanup (after v4 is the default and proven)

The legacy `biasbuster/agent/` regex-based verification path is
retired in a separate cleanup commit. `tools.py` survives (the
wrappers are still useful). `runner.py`, `tool_router.py`, and
`verification_planner.py` are deleted.

The v3 `prompts_v3.py` ASSESSMENT_DOMAIN_CRITERIA stays in place
as the canonical English-language version of the rules — useful as
documentation of what `biasbuster/assessment/rules.py` is supposed
to do, even after the prompt itself is no longer sent to any LLM.

---

## §8 Open questions

### 8.1 bmlib streaming + tools

bmlib does not currently support streaming. v4 doesn't need it
either — the agent loop is request-response. But streaming and
tool calling interact (the tool call may arrive partway through a
streamed response). If we add streaming to bmlib later, the tool
calling implementation will need to handle interleaved content
chunks and tool_use blocks. **Defer until streaming is actually
required** — there's no biasbuster use case that needs it today.

### 8.2 Token cost vs v3

Open empirical question. v3 makes 1 extraction call + 1 assessment
call per paper. v4 makes 1 extraction call + 1–5 assessment-loop
calls (typically 2–3). Per paper, v4 should be ~1.5–2× the LLM
calls. But each v4 call uses a much smaller prompt (~2 KB vs ~43 KB
for v3 assessment), so total token volume may be comparable or
even lower. **Phase 2 records token usage per paper; the
calibration matrix compares directly.**

### 8.3 Retiring the legacy verification path

When? The right answer is "when v4 is the default for at least
one full release cycle and we're confident no one is using the
`--verify` flag". Defer the decision until after Phase 3.

### 8.4 Cross-model consistency under tool calling

Local models claim tool support in Ollama metadata but the
quality varies. If gemma4 produces well-formed tool calls 90% of
the time and refuses 10%, v4 needs a fallback. Options:
- Retry with stronger tool_choice forcing
- Fall back to v3 two-call after N retries
- Reject the paper and return an error

Not a Phase 1 or 2 concern; Phase 3 will measure and decide.

### 8.5 Should `_provenance` from the mechanical assessment be stored separately?

Currently the agent calls `run_mechanical_assessment` and gets back
a JSON blob. The agent's final output includes the agent's own
reasoning and overrides, but the mechanical provenance (which rules
fired, what inputs they read) is in the agent's tool-call history,
not directly in the final annotation. For audit purposes it would
be valuable to store both.

**Tentative answer:** the agent's final annotation includes a
`_mechanical_provenance` key copied from the first
`run_mechanical_assessment` tool call result. Costs minimal storage,
preserves the audit trail. Phase 2 implementation should do this.

### 8.6 What happens if the LLM never calls `run_mechanical_assessment`?

The prompt says "you MUST call run_mechanical_assessment FIRST".
If the LLM ignores that instruction and produces a final answer
on its first turn, the agent loop should detect this and retry
once with `tool_choice="required"` forcing a tool call. If it
still doesn't call the tool, the agent treats the response as
invalid and falls back to running the mechanical assessment in
Python and returning its result with `_overrides=[]` and an audit
note. **Defensive behaviour, not a hot path.**

---

## Cross-references

| Document | Role |
|---|---|
| [`INITIAL_FINDINGS_V3.md`](./INITIAL_FINDINGS_V3.md) | Empirical history. §3.13 calibration test is the immediate motivation; §6.2 has the calibration paper selection. |
| [`DESIGN_RATIONALE_COI.md`](./DESIGN_RATIONALE_COI.md) | The COI policy that v4's hard-enforcement check preserves. Read before modifying any of the trigger (a)–(d) rules. |
| [`architecture_guide.md`](./architecture_guide.md) | Original two-call architecture rationale. v4 keeps the extraction stage from this design. |
| [`MERGE_STRATEGY.md`](./MERGE_STRATEGY.md) | Per-section extraction merge rules — used unchanged in v4's extraction stage. |
| [`CONTEXT_FOR_CLAUDE_CODE.md`](./CONTEXT_FOR_CLAUDE_CODE.md) | The original Seed Health failure case that motivated everything. |
| `~/.claude/plans/serialized-snacking-dragonfly.md` | The implementation plan that this document is the design for. Phases, files, verification steps. |
| `biasbuster/assessment/rules.py` | The Python translation of the v3 rules. The single source of truth that `run_mechanical_assessment` wraps. |
| `biasbuster/assessment/aggregate.py` | The orchestration that produces the draft assessment + provenance. |
| `biasbuster/agent/tools.py` | Existing verification tool wrappers. Reused by v4. |
