# V5 — Closing the Small-Model Contextual-Judgment Gap

## Context

The biasbuster project produces *risk-of-bias* assessments of clinical trials. The pipeline relies on LLMs for two jobs that cannot be done deterministically:

1. **Extraction** — pulling structured facts out of free-text full-text papers
2. **Contextual judgment** — deciding whether a mechanically-fired bias rule genuinely applies to *this specific* paper

The v3 architecture put both jobs in a single 240-line LLM prompt (see [`docs/two_step_approach/`](../two_step_approach/)). The v4 architecture decomposed it: extraction (LLM) → mechanical rules (Python) → "review and override" (LLM agent with tools). See [`V4_AGENT_DESIGN.md`](../two_step_approach/V4_AGENT_DESIGN.md).

## The problem v5 addresses

v4 works well for Claude. It does **not** work for smaller local models (gemma4-26B, gpt-oss-20B). After running the same 5 papers through both:

| Model      | Severity κ vs Claude | REVIEW blocks produced | Overrides applied |
|------------|----------------------|------------------------|-------------------|
| gemma4-26B | -0.154 (no agreement) | 0 / 5                 | 0                 |
| gpt-oss-20B| -0.250 (worse than chance) | 0 / 5            | 1                 |

A second run after adding an explicit mandatory REVIEW-block scaffold to the system prompt **plus** a just-in-time "STOP. You must review" nudge injected directly into the `run_mechanical_assessment` tool result produced **identical** gemma4 numbers (-0.154 → -0.154) and slightly *worse* gpt-oss numbers (-0.200 → -0.250). Zero REVIEW blocks were produced across either run.

The smaller models do not have the instruction-following fidelity to engage with structured intermediate reasoning artifacts when those are one of many things in the prompt. The v4 agentic loop asks them to:

1. Call `run_mechanical_assessment`
2. Read the provenance across 5 domains
3. Decide which rules genuinely apply
4. Write structured REVIEW blocks for elevated domains
5. Optionally call verification tools
6. Emit a ~30-field final JSON with a populated `_overrides` array

…all in one continuation. For Claude this is routine. For a 20–26B local model it collapses to "summarize the mechanical output in a short paragraph and emit the JSON verbatim."

## Two strategies

### V5A — Decomposed pipeline (try first)

Replace the agentic loop with a sequence of focused LLM calls, each with one narrow task and a 3-field output schema. See [`V5A_DECOMPOSED.md`](./V5A_DECOMPOSED.md).

**Hypothesis:** small models *can* reason about a single, narrowly-scoped question; they can't handle "review 5 domains and emit a 30-field JSON" in one shot.

**Cost:** ~1 day of implementation. Most pieces already exist; this is orchestration.

**Risk:** low — if it fails we learn something useful about the ceiling of small-model capability on this task.

### V5B — Fine-tuning (fallback)

Fine-tune gemma4-26B and gpt-oss-20B on Claude's v4 reasoning traces. Keep the v4 agent architecture; the change is upstream (training data) not downstream (inference). See [`V5B_FINE_TUNING.md`](./V5B_FINE_TUNING.md).

**Hypothesis:** if even the focused per-domain task is too hard for small models, the missing capability is *learned*, not promptable.

**Cost:** $50–100 Anthropic API spend for 100–200 teacher traces, 4–8 hours LoRA training per model, ~1–2 days implementation.

**Risk:** medium — fine-tuning works, but learning agent-style multi-turn reasoning from 100–200 examples is at the lower end of what's feasible.

## Decision tree

```
V5A passes (κ ≥ 0.30)? ─┬─ YES → ship V5A, V5B becomes optional polish
                        └─ NO  → V5B is the path
                                  ├─ V5B passes (κ ≥ 0.50)? → ship fine-tuned models
                                  └─ V5B fails              → small local models can't do this
                                                              task; recommend Claude-only or
                                                              hybrid (Claude on hard cases,
                                                              small model on easy ones)
```

## Success criteria recap

| Strategy | Baseline κ vs Claude | Pass threshold | Stretch goal |
|----------|----------------------|----------------|--------------|
| V5A      | gemma4: -0.154, gpt-oss: -0.250 | ≥ +0.30 for both | ≥ +0.50 |
| V5B      | same                 | ≥ +0.50 for both | ≥ +0.70 |

κ = Cohen's weighted kappa on severity agreement. Measured on the existing 5-paper test set (PMIDs 32382720, 39691748, 39777610, 39905419, 41750436), with a follow-up 25-paper validation if initial results pass.

## Note on the "three_step_approach" naming

Despite the directory name, V5A is structurally a *pipeline* of five stages (extraction → mechanical → per-domain overrides → optional verification → synthesis). The naming comes from the user-visible decomposition: extraction, judgment, synthesis. V5B is architecturally identical to v4 (so still "two-step": extraction + agentic assessment) but with fine-tuned small models in the assessment seat.
