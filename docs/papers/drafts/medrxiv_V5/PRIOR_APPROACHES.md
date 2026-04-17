# Prior approaches and their outcomes

**Status:** working document for the medRxiv V5A manuscript.
**Scope:** what we tried before V5A, what each attempt achieved, and
why we moved on. This document feeds the manuscript's Introduction
(paragraphs on "previous approaches" and "why fine-tuning was set
aside") and the Discussion's generalisability argument.

The short version: we built and tested four architectures — single-
call, two-call, fine-tuned, and agentic-with-tools — before landing
on the decomposed V5A pipeline. Each was the natural next step given
the evidence at the time; each hit a specific, reproducible failure
mode that was only obvious in hindsight. Documenting the failures
matters because the obvious architectures are the ones other groups
are most likely to attempt.

## 1. Motivating problem

The original specification for BiasBuster asked for a risk-of-bias
assessor that:

1. **Matches Claude-class quality** on published randomised trials,
   not abstract-only model-card benchmarks.
2. **Runs locally** on 20–30 B-class hardware (Apple M-series or the
   DGX Spark / single-RTX-6000-class GPU), so assessments can be
   produced at scale without per-paper API cost and without sending
   full manuscripts to a third-party cloud.
3. **Is deterministic enough for research use.** Not necessarily
   reproducible bit-for-bit, but stable enough that a repeated run
   produces the same severity category for the same paper.
4. **Assesses more than methodology.** Extends beyond Cochrane RoB 2
   to cover conflict-of-interest structural signals, statistical
   reporting patterns (relative vs absolute effect sizes), and spin
   in the conclusions — see the companion
   [`COI_RATIONALE.md`](./COI_RATIONALE.md) (and
   [`../../two_step_approach/DESIGN_RATIONALE_COI.md`](../../two_step_approach/DESIGN_RATIONALE_COI.md))
   for the normative argument.

The second and third of these are the binding constraints. Cloud
frontier models can do the task; local models under 30 B parameters
have been the open question.

## 2. v1 — single-call prompt (abandoned 2026-02)

The first implementation put the entire task in one prompt: present
the abstract (or full text), ask the model to extract the key facts
and produce a structured RoB assessment in a single continuation.
This is the architecture most published auto-RoB systems use
(RobotReviewer and its successors) and was the obvious starting
point.

**What worked:** On abstracts, v1 produced passable output for
Claude and the 120 B-class local models. The structural category of
the bias (low / moderate / high) was often right.

**What failed:** When we moved to full text, v1 collapsed across all
three local model families we tested (gpt-oss 120 B, gpt-oss 20 B,
gemma4 26 B). In Round 4 of our calibration testing, the full-text
single-call path reverted to a generic *"no major red flags,
moderate"* verdict on almost every paper, regardless of input. The
long input pushed the model past the budget it needed to both (a)
extract facts reliably and (b) apply the bias rules to those facts
in one shot. Agreement with Claude's full-text annotation dropped
from ~80% (abstracts) to ~50% (full text). The model produced an
answer; the answer was not a function of the paper.

Documented in
[`INITIAL_FINDINGS_V3.md`](../../two_step_approach/INITIAL_FINDINGS_V3.md)
§4.1. v1 has since been kept in the codebase only as an explicit
fallback for abstract-only cases.

## 3. v3 — two-call (extract → assess) prompt engineering (2026-02 to 2026-04)

The v1 failure pointed at the instruction-following budget. v3 split
the task into two calls: Stage 1 extracts structured facts from the
paper (Stage 1 output is the paper's "numbers, populations, and
structural signals"; no bias judgement), and Stage 2 takes only the
extraction object and emits the bias assessment.

This architecture implements a map-reduce variant on full-text
input: Stage 1 runs per-section and the per-section extractions are
merged before Stage 2. The assessment prompt itself grew to roughly
240 lines across ten rounds of iteration, with explicit severity
boundary definitions, domain-specific criteria, and a curated list
of verification databases (ClinicalTrials.gov, CMS Open Payments,
ORCID, Europe PMC) the model was taught to cite.

**What worked:** v3 two-call on full text lifted local-model
agreement with Claude from ~50% (v1) to 79–87% across three model
families on the paper that had originally motivated the entire
rebuild. Abstract-mode agreement climbed to ~92% for the
120-B-class model. All three families correctly identified the
34.4% attrition, the 43.75% vs 25% differential dropout, and the
sponsor-employee manuscript drafter that v1 had missed. The
structural hard-rule triggers for conflict of interest —
sponsor-controls-analysis-and-manuscript, sponsor-employed-authors
— worked deterministically.

**What failed:** The Round-10 calibration test (five papers with
expert Cochrane RoB 2 ground truth, four modes per paper, three
local models) revealed that v3 did not generalise. On the
*lidocaine paper* (Cochrane HIGH), gpt-oss-20 B returned
`low/0.25` and gemma4-26 B returned `moderate/0.45`. Inspection
showed the extraction had correctly produced the outcome lists but
the assessment prompt was relying on an extraction count field
(`n_primary_endpoints`) that was silently wrong — the model had
emitted `0` for the count while listing eight entries in the array
field. The downstream severity rule
(`total_endpoints ≥ 6 AND no_multiplicity_correction → HIGH`)
therefore never fired. On the *rTMS paper* (Cochrane LOW), gpt-oss-
20 B over-called COI from moderate to high on consulting-tie authors
who are not employees (trigger (d) should not fire on consulting
relationships). gemma4-26 B made the inverse mistake, missing the
consulting-tie signal entirely.

Every single one of these calibration failures was an **arithmetic
or boolean-logic error**, not a text-reasoning error. The models
could read the papers; they could not reliably count across their
own extraction output while simultaneously applying a rule stack in
the same continuation. This diagnosis motivated v4.

Full history:
[`INITIAL_FINDINGS_V3.md`](../../two_step_approach/INITIAL_FINDINGS_V3.md)
§3.13 (the Round-10 calibration test),
[`CONTEXT_FOR_CLAUDE_CODE.md`](../../two_step_approach/CONTEXT_FOR_CLAUDE_CODE.md)
(the Seed Health probiotic case that originally motivated the
rebuild).

## 4. Fine-tuning (multiple rounds; formally deprioritised 2026-04)

Running in parallel with v3 prompt engineering, we attempted LoRA
fine-tuning as a separate path to the same goal: instead of making
the prompt smarter, teach a smaller model the behaviour directly.

### 4.1 What was built

A full training infrastructure was committed:

- **Dataset export**: the curated annotation DB exports to `alpaca`
  and `sharegpt` JSONL with explicit `<think>` chain-of-thought
  reasoning and 80/10/10 train/val/test splits
  (`biasbuster/export.py`).
- **Two training backends**: TRL/`SFTTrainer` + PEFT LoRA on
  DGX Spark under NGC Docker (`training/train_lora.py`), and
  `mlx_lm.tuner` QLoRA on Apple Silicon (`training/train_lora_mlx.py`).
  Both write identical `metrics.jsonl` for live monitoring.
- **End-to-end orchestrator**: `train_and_evaluate.sh` auto-versions
  runs, merges adapters, exports to Ollama, and evaluates against a
  held-out test set in one command.
- **A NiceGUI fine-tuning workbench** (`biasbuster/gui/`) exposing
  settings, training, evaluation, and export as four tabs.

All of the above remains in the codebase and is functional; the
decision to de-prioritise is about outcomes on *this* task, not
about the infrastructure.

### 4.2 What was tried (V1 through V7)

Seven fine-tuning runs are documented in
[`docs/papers/`](../../papers/) under `FIRST_RUN.md` through
`SIXTH_RUN.md` (the `FIRST` through `SEVENTH_RUN` series), plus an
eighth run under `gpt-oss-20b-biasbusterV7`. The bases evaluated
were `qwen3.5-9b`, `qwen3.5-27b`, `olmo-3.1-32b`, and
`openai/gpt-oss-20b`, on ~1 347 SFT examples distilled from Claude's
v3 assessments. The two best results are representative:

1. **qwen3.5-9b-biasbuster (Fifth Run).** A fine-tuned 9 B model
   ran comfortably on a 16 GB-VRAM laptop and, on a 157-example
   test set, sat at binary F1 0.89. It was eventually outperformed
   *on the same test set* by unmodified `gpt-oss:20b` at F1 0.918,
   which reframed the entire fine-tuning question — the baseline
   had improved faster than our adapters could close the gap.

2. **gpt-oss-20b-biasbuster (Sixth Run).** Fine-tuned on the same
   1 347 examples, 1 epoch, LR 5 × 10⁻⁶, LoRA rank 16, attention-only
   targets (skipping expert FFNs and the router to avoid expert
   collapse on the MoE). On the 157-example test set:

   | Metric | `gpt-oss:20b` baseline | `gpt-oss-20b-biasbuster` | Δ |
   |---|---:|---:|---:|
   | Binary F1 | 0.918 | **0.938** | +0.020 |
   | Recall | 0.941 | **1.000** | +0.059 |
   | **Ordinal κ** | **0.158** | 0.042 | **−0.116** |
   | Mean latency (s) | **76.7** | 123.6 | +46.9 |

   The fine-tuned model improved at *binary* flag detection (does
   this paper carry a COI/spin/stat-reporting flag at all?) but
   **severely degraded** at *ordinal* severity judgement (is the
   flag low / moderate / high / critical?). The ordinal κ — the
   metric the manuscript cares about most, because it corresponds
   to Cochrane's 3-level RoB rating — dropped from 0.158 to 0.042.
   A fine-tuning regimen that improves recall at the expense of
   severity calibration makes the assessor more alarmist but less
   discriminating, which is the wrong trade.

### 4.3 Why fine-tuning was set aside

Three observations, in order of weight:

1. **The metric that matters got worse.** The Sixth Run's ordinal-κ
   regression was not an artefact of one hyperparameter choice —
   revised hyperparameters (1 epoch, LR 5 × 10⁻⁶, dropout 0.1,
   weight-decay 0.02) produced the same pattern. The underlying
   cause appears to be that distilling a 5-level ordinal from a
   single teacher with ~1.3 k examples is a weak signal for the
   graded judgement task; the model over-fits to the binary
   "is-this-flagged" signal that dominates the loss. Scaling the
   training corpus could close this, but the sample-efficiency
   argument for small models starts to evaporate once we need
   5 k–10 k curated traces.

2. **The base models kept improving.** Between the Fifth Run
   (qwen3.5-9b, F1 = 0.89) and the Sixth Run (gpt-oss-20b baseline,
   F1 = 0.918) the un-fine-tuned zero-shot ceiling shifted under
   us. gemma4-26B — the eventual V5A winner — had not yet been
   released when the fine-tuning programme started. Each six-month
   base-model refresh compresses the gain available from distilling
   a frontier teacher.

3. **A purely-prompt path became available.** V5A (§5 below) closed
   the v3 calibration gap with no training data at all. Once V5A's
   κ-vs-Cochrane results landed, the argument for spending ~100 h of
   labelled-trace collection, ~$100 in API teacher cost, and ~8 h of
   GPU training per model to achieve the same result weakened
   considerably.

The planned "V5B" fine-tuning path described in
[`../../three_step_approach/V5B_FINE_TUNING.md`](../../three_step_approach/V5B_FINE_TUNING.md)
— distilling Claude's v4 agent traces into gemma4 and gpt-oss — was
therefore not executed. Fine-tuning remains a viable path if a
future application needs sub-10 B deployment, a domain-specific
extraction profile, or self-hostability in environments where even
a 26 B model is too heavy; for general risk-of-bias assessment on
20–30 B-class hardware, V5A supersedes it.

### 4.4 What we keep from this work

Two things. First, the training pipeline itself remains in the
codebase and is usable for any future distillation run; the
infrastructure is not wasted. Second, the ordinal-vs-binary metric
split identified in 4.2 is a useful lens on any fine-tuned RoB
assessor: always report both, because binary F1 can mask a
regression on graded severity.

## 5. v4 — tool-calling agentic assessment (2026-04)

v3's diagnosis — that small local models can extract facts and can
reason about single rules but fail when asked to arithmetic-check
their own extractions while simultaneously applying a rule stack —
suggested a structural remedy. Move the arithmetic out of the prompt.

v4 implements the full v3 rule library as pure Python
(`biasbuster/assessment/`) and exposes a single tool,
`run_mechanical_assessment`, to the LLM. The agent loop is:

1. LLM reads the merged extraction.
2. LLM calls `run_mechanical_assessment` (zero arguments — the tool
   reads the extraction from the agent context).
3. The Python code applies every severity rule deterministically,
   returns a draft per-domain assessment with full provenance
   (which rule fired, why, which extraction fields it consumed).
4. LLM reviews the provenance, writes structured REVIEW blocks for
   any domain it believes warrants an override, then emits a final
   30-field JSON with an `_overrides` array.

The v4 design intended to keep the LLM's role pure — read the
mechanical output, decide where the rules genuinely apply to the
specific paper, emit overrides — while making every arithmetic
judgement deterministic and auditable.

**What worked for Claude.** On a 5-paper calibration test, Claude
v4 matched Claude v3 on 4/5 papers and *improved* on the fifth (the
lidocaine HIGH paper v3 had under-called). The REVIEW scaffold
worked as designed: Claude produced coherent override arguments on
the one paper that needed one. Full provenance traces were clean.

**What failed for local models.** On the same 5-paper test:

| Model | v4 severity κ vs Claude | REVIEW blocks produced | Overrides applied |
|---|---:|---:|---:|
| gemma4-26 B | **−0.154** (no agreement) | 0 / 5 papers | 0 |
| gpt-oss-20 B | **−0.250** (worse than chance) | 0 / 5 papers | 1 |

Both local models **rubber-stamped** the mechanical output. They
invoked `run_mechanical_assessment`, read the provenance, and then
emitted the final JSON as a mechanical restatement of the tool
output — ignoring the REVIEW-blocks-and-overrides scaffold that the
system prompt asked for.

We tested the two obvious fixes:

- **Explicit mandatory REVIEW-block scaffold** inserted into the
  system prompt. Produced **identical gemma4 κ** (−0.154 → −0.154).
- **Just-in-time STOP nudge** injected as the first line of the
  tool-result message. Produced **slightly worse** gpt-oss κ
  (−0.200 → −0.250). Zero REVIEW blocks across both runs.

The diagnosis:
v4 asks the LLM to perform a 6-step procedure in one continuation —
call tool, read 5-domain provenance, decide which rules genuinely
apply, write 5 × REVIEW blocks, optionally call verification tools,
emit a 30-field JSON with a populated `_overrides` array. Claude
performs this routinely. For a 20–26 B parameter model, the
instruction-following budget runs out long before the scaffold is
complete, and the model falls back to the simpler generative
pattern it is most confident about: summarise the tool output,
emit the JSON verbatim.

This is not a failure of reasoning. The same models, given the
same provenance in a more constrained setting (§5 below), produced
valid contextual overrides. The failure is of instruction
*following*, specifically at the scale of multi-step stacked
instructions within a single continuation.

v4 is documented in
[`V4_AGENT_DESIGN.md`](../../two_step_approach/V4_AGENT_DESIGN.md)
and the calibration numbers are in
[`../../three_step_approach/OVERVIEW.md`](../../three_step_approach/OVERVIEW.md).

## 6. V5A — decomposition as the structural fix

The v4 failure diagnosis pointed at a specific, fixable property:
the instruction-following budget is a function of *continuation
length and step count*, not of model-scale alone. If the per-call
task is small enough that a 20–26 B model can reliably carry it to
completion, the model should be able to perform the same judgement
v4 asked it to perform — just not all at once.

V5A implements exactly this decomposition:

- Extraction → **one call per section**, same as v3/v4 Stage 1.
- **Mechanical assessment → pure Python**, same as v4 but
  uncomposed with the LLM call — the assessment runs to completion
  before the LLM sees anything.
- **Override review → one focused LLM call per elevated domain**,
  each taking only (a) the domain name, (b) the mechanical severity
  and which rule fired, (c) the extracted facts that triggered it,
  and (d) a short list of legitimate vs illegitimate override
  reasons. The call emits a 3-field JSON:
  `{decision, target_severity, reason}`. Parallel via
  `asyncio.gather`.
- **Synthesis → pure Python** assembles the final assessment from
  the mechanical draft plus per-domain decisions, runs hard-rule
  enforcement (e.g. the non-overridable structural-COI triggers from
  [`COI_RATIONALE.md`](./COI_RATIONALE.md)), and optionally
  produces a short reasoning summary via one small LLM call.

Each LLM call is small, narrow, and has one job with a 3-field
output schema. The instruction-following load that broke v4 is
removed. Architecture details and the mapping from v3/v4 rules to
V5A's per-domain overrides are in
[`PIPELINE_V5A.md`](./PIPELINE_V5A.md) and the authoritative
design document
[`../../three_step_approach/V5A_DECOMPOSED.md`](../../three_step_approach/V5A_DECOMPOSED.md).

The empirical outcome — and the headline result of this
manuscript — is that this decomposition restores local-model
judgement to expert-level accuracy on the two directly-comparable
Cochrane RoB 2 domains (methodology and outcome reporting). See
[`RESULTS.md`](./RESULTS.md) (populated once the N=121 annotation
run completes).

## 7. Summary

| Approach | Small-model κ vs Claude (5-paper calibration) | Small-model κ vs Cochrane (where available) | Failure mode |
|---|---:|---:|---|
| v1 single-call (full text) | ~−0.1 to 0 | — | Long-input collapse to generic "moderate" |
| v3 two-call | +0.4 to +0.6 on matching papers, catastrophic on others | — | Arithmetic-logic errors inside the assessment prompt |
| Fine-tuned gpt-oss-20b-biasbuster | n/a (tested on internal test set) | Ordinal κ 0.042 vs baseline 0.158 | Ordinal severity degradation; F1 improved |
| v4 agentic | gemma4 −0.154, gpt-oss −0.250 | — | Rubber-stamping; no REVIEW blocks or overrides produced |
| **V5A decomposed** | **+0.429 gemma4 on 16-paper validation** | **+1.000 gemma4 on methodology, outcome reporting (N=15 Cochrane)** | Residual calibration drift within Cochrane categories |

The table reads left-to-right as an informative sequence of
negative results, each one localising the problem more narrowly
than the one before. v1 said "full text is too long"; v3 said "the
arithmetic fails under instruction load"; fine-tuning said "the
ordinal severity signal is hard to distil cheaply"; v4 said
"separating arithmetic from judgement isn't enough, the *number of
stacked instructions* is the binding constraint". V5A is the fix
that respects all four lessons: short inputs per call, no
arithmetic in the LLM's hands, no fine-tuning dependency, and
exactly one instruction per continuation.

## 8. What this history means for the manuscript

Two points belong in the Introduction and one in the Discussion.

**Introduction.** The local-model risk-of-bias problem was not
obvious in advance. Every obvious architecture — single-call,
two-call-with-better-prompting, fine-tuning, agentic-with-tools —
was tried and has its own published-grade negative result. The V5A
contribution is not "another architecture on the pile", it is a
specific structural fix for a specific diagnosed failure mode, with
the prior attempts as the diagnostic chain.

**Introduction.** Fine-tuning is the first thing an external reader
will suggest. It was tried seven times. It works to improve
binary flag detection and it can hit 0.89–0.94 binary F1 on 9–20 B
bases. It *also* produced a concrete regression on ordinal severity
κ (0.158 → 0.042 in the Sixth Run), which is the metric that maps
onto the Cochrane 3-level rating the manuscript is validated
against. Fine-tuning is not off the table for future work — it is
off the table for *this* paper.

**Discussion.** The instruction-following-budget argument
generalises beyond RoB. Any multi-domain, multi-rule, multi-step
assessment task with structured JSON output and a small local
model as the target deployment is likely to hit the same v4-style
ceiling and benefit from the same V5A-style decomposition. The
RoB case is one instance; the pattern is the contribution.
