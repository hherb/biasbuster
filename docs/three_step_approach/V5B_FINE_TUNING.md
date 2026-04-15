# V5B — Fine-tuning Design

## When to reach for this

Only if [V5A](./V5A_DECOMPOSED.md) fails its success criteria (gemma4 / gpt-oss severity κ < +0.30 after decomposition). See [`OVERVIEW.md`](./OVERVIEW.md#decision-tree) for the decision tree.

## Hypothesis

If even the focused per-domain task is too hard for small models, the missing capability is **learned**, not promptable. Fine-tune gemma4-26B and gpt-oss-20B on Claude's v4 reasoning traces. Keep the v4 agent architecture unchanged at inference time; just give the small models the right behaviour as a learned habit.

## Architecture

No architectural change to inference — fine-tuned models drop into the existing v4 agent loop (see [`V4_AGENT_DESIGN.md`](../two_step_approach/V4_AGENT_DESIGN.md)). The change is upstream: produce SFT training data from Claude's traces, run LoRA, export to Ollama, evaluate.

## Data collection

### Paper selection

100–200 papers from the existing dataset, biased for diversity:

- ~30 industry-funded RCTs (will trigger COI HIGH)
- ~20 explicitly exploratory secondary analyses (legitimate override candidates)
- ~30 multi-endpoint trials with no multiplicity correction (HIGH methodology, sometimes overridable)
- ~20 Cochrane RoB papers (have expert ground truth as anchors)
- ~30 low-bias papers (must learn to *not* override moderate→low gratuitously)
- ~40 random

### Teacher run

Run Claude (`anthropic_fulltext_agentic`) on all selected papers with the **current** v4 prompts. The REVIEW scaffold actually works for Claude, so that becomes the teacher signal. For each paper capture the full agent-loop trace:

- Stage 1 extraction (input + output)
- `run_mechanical_assessment` tool call + result
- Optional verification tool calls + results
- Final assistant turn (REVIEW blocks + final JSON with `_overrides`)

## Training format

Convert each Claude trace into a multi-turn SFT conversation in the alpaca/sharegpt format the existing training code already consumes:

```
Turn 1 — system: ASSESSMENT_AGENT_SYSTEM_PROMPT (verbatim from prompts_v4)
Turn 2 — user: build_agent_user_message(pmid, title, extraction_json)
Turn 3 — assistant: tool_call(run_mechanical_assessment, {})
Turn 4 — tool: <mechanical assessment output>
[optional turns 5-N for verification tool calls]
Turn N+1 — assistant: REVIEW <domain>: ... + final JSON with _overrides
```

The model learns by example to:

- Always call `run_mechanical_assessment` first
- Read `_provenance.domain_rationales` and identify which rules fired
- Write structured REVIEW blocks for elevated domains
- Apply contextual override criteria correctly
- Emit final JSON with a populated `_overrides` array

## Files to create / modify

### New files

- `training/collect_v4_traces.py` — driver that runs Claude on a list of PMIDs and saves the full conversation trace per paper (to JSONL)
- `training/format_v4_for_sft.py` — converts traces to alpaca/sharegpt JSONL ready for `train_lora.py` / `train_lora_mlx.py`

### Modified files

- `training/configs.py` and `training/configs_mlx.py` — add presets for the v4-tuned variants (likely just lower LR + slightly higher rank than the existing v3 configs, since the agent format is more demanding than v3 single-call)

### Reused unchanged

- `training/train_lora.py` (DGX Spark) and `training/train_lora_mlx.py` (Mac MLX)
- `train_and_evaluate.sh` (auto-versioning orchestrator)
- `training/export_to_ollama.sh` (Ollama deploy)
- `training/data_utils.py` (alpaca → chat format conversion)

## Training targets

- `gemma4:26b-a4b-it-q8_0` — MLX 4-bit LoRA on Mac (proven path)
- `gpt-oss:20b` — MLX 4-bit/8-bit (MoE — already handled in `configs_mlx.py` per [`CLAUDE.md`](../../CLAUDE.md))

## Verification

1. Train one adapter per model, export to Ollama as `gemma4-26b-biasbusterV5B` and `gpt-oss-20b-biasbusterV5B`
2. Run the unchanged v4 agent loop with the fine-tuned models on the same 5-paper test set
3. Compare against:
   - Original v4 results (the -0.154 / -0.250 baseline)
   - V5A results if V5A produced anything usable
   - The held-out Cochrane RoB ground truth (25-paper held-out set)
4. **Success criterion:** severity κ ≥ 0.50 vs Claude on the 5-paper set; meaningful improvement on Cochrane RoB held-out

## Cost / effort

- ~$50–100 in Anthropic API spend for 100–200 full-text agentic traces
- ~4–8 hours of LoRA training per model on Mac M-series
- ~1–2 days of implementation + iteration on hyperparameters
- Risk: medium — fine-tuning works, but learning agent-style multi-turn reasoning from 100–200 examples is at the lower end of what's feasible. May need 500+ examples if first attempt underperforms.

## Open questions

- **Multi-turn tool-calling format vs flattened "cheat" format**: the cleanest training signal is the full multi-turn transcript including the tool call and tool result. Some smaller base models (especially non-instruction-tuned or older chat-tuned ones) struggle with multi-turn tool-use at training time. Fallback: flatten to a single-turn format where the "tool result" is inlined into the user message, at the cost of the model no longer learning to *call* the tool — just to produce the right output given the tool's output.
- **Mixed-teacher supervision**: for Cochrane papers we have both Claude's v4 output AND the human expert RoB rating. Training on the human rating where available (with Claude as the reasoning text teacher) may outperform pure Claude-distillation. Worth A/B testing.
