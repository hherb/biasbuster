# Preparing Round 4: GPT-OSS 20B Fine-Tuning

**Date:** 2026-03-25
**Status:** V6 re-evaluation in progress with fixed harness
**Prerequisite:** Read `docs/MISTAKES_TO_ROUND_3.md` for context

---

## What Changed Before Round 4

### Infrastructure Fixes (evaluation/harness.py)

1. **Harmony thinking capture**: Ollama's native API returns a `thinking`
   field for GPT-OSS (analysis channel). The harness now captures it and
   combines it with `content` (final channel) via `ModelOutput.full_output`.

2. **Auto-detect Ollama endpoints**: The harness now uses the Ollama native
   API (`/api/chat`) when the endpoint is on port 11434, instead of requiring
   `--num-ctx` to trigger it. The native API is required to receive the
   `thinking` field.

3. **Token budget increased**: `max_tokens` default raised from 4000 to 8000
   to prevent the analysis channel from exhausting the token budget before
   the final channel is generated.

4. **Clean data model**: `ModelOutput` stores `thinking` and `raw_output`
   as orthogonal fields. `full_output` property reconstructs `<think>` blocks
   for the scorer. No redundant state.

### Data Pipeline (unchanged from Round 2/3)

- 942 training / 116 validation / 123 test examples
- Unified prompts via `prompts.py` (severity boundaries in both annotation
  and training prompts)
- Retraction severity floors from Retraction Watch CSV
- No oversampling (natural severity distribution)
- `<think>` blocks split into Harmony `thinking` field in `data_utils.py`

---

## V6 Re-evaluation Results (2026-03-25)

The re-evaluation with the fixed harness revealed that V6's poor scores
were NOT caused by the harness discarding output. The model genuinely
fails to produce the expected format:

1. **`<think>` tags are literal text**, not Harmony channel tokens —
   the model didn't learn the channel structure
2. **No JSON output** — after `</think>`, the model produces markdown prose
3. **3/8 outputs hit 8000-token ceiling** — thinking is extremely verbose
4. **Degenerate repetition** in some outputs

### Diagnosis

Attention-only LoRA (0.04% trainable params) at 1 epoch (236 steps) cannot
learn the Harmony channel token structure. Channel switching likely requires
FFN layers, which are completely frozen. The model falls back to producing
`<think>` as literal text — a simpler pattern that attention layers can learn.

This is the same failure as V4/Run 8 (markdown instead of JSON), now with
literal `<think>` tags added.

### Decision: Single-Channel Training

Since the model naturally produces literal `<think>` tags, lean into that
instead of fighting the Harmony channel structure:

- Keep `<think>` as literal text in `content` (single channel)
- Focus training on the JSON output format after `</think>`
- Train for 2 epochs to reinforce format adherence
- The scorer's existing `<think>` regex extraction handles this format

---

## Round 4 Training Plan

### Data format change

`training/data_utils.py` now skips the Harmony channel split for GPT-OSS.
The `make_formatting_func()` function detects Harmony tokenizers and keeps
`<think>` blocks as literal text in the `content` field instead of splitting
into the `thinking` field. This matches what the model naturally produces.

The training data token sequence becomes:
```
<|start|>assistant<|channel|>final<|message|><think>reasoning</think>{json}<|return|>
```

Instead of the previous (broken) dual-channel:
```
<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>
<|start|>assistant<|channel|>final<|message|>{json}<|return|>
```

### Hyperparameters

| Parameter | V6 (Run 9) | Round 4 (Run 10) | Rationale |
|-----------|:----------:|:----------------:|-----------|
| Learning rate | 5e-6 | 5e-6 | Proven stable; 8e-6 caused collapse |
| Epochs | 1 | **2** | Loss was still descending at step 236 |
| LoRA rank / alpha | 32 / 64 | 32 / 64 | Keep capacity for format learning |
| LoRA dropout | 0.1 | 0.1 | Adequate regularization |
| Weight decay | 0.02 | 0.02 | Adequate regularization |
| Label smoothing | 0.0 | 0.0 | Not needed; caused issues in Run 8 |
| LoRA targets | q,k,v,o | q,k,v,o | Attention-only (MoE safety) |
| **Data format** | Harmony dual-channel | **Single-channel** | Model can't learn channel tokens |

---

## Evaluation Checklist

Before running evaluation on any new model version:

### Pre-flight

- [ ] `curl` a test prompt to the model via Ollama native API and inspect
      the full JSON response — verify `thinking` and `content` fields are
      both present
- [ ] Confirm the evaluation harness uses the Ollama native path (check
      for `:11434` in the endpoint URL)
- [ ] Set `--max-tokens 8000` (or higher) to avoid token budget exhaustion

### Post-evaluation

- [ ] Check `thinking_present_rate` — should be >0% for Harmony models
- [ ] Check that `output_tokens < max_tokens` for most examples (if many
      examples hit the max, increase the budget)
- [ ] Inspect 5-10 raw outputs manually to verify format correctness
- [ ] Compare F1, recall, kappa, and verification_score against baseline

### Evaluation commands

```bash
# Full evaluation with fixed harness
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gpt-oss-20b-biasbusterV7 \
    --endpoint-a http://localhost:11434 \
    --mode fine-tuned \
    --max-tokens 8000 \
    --output eval_results/

# Compare against baseline
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a gpt-oss-20b-biasbusterV7 \
    --endpoint-a http://localhost:11434 \
    --model-b gpt-oss:20b \
    --endpoint-b http://localhost:11434 \
    --mode fine-tuned \
    --max-tokens 8000 \
    --sequential \
    --output eval_results/
```

---

## Success Criteria for Round 4

| Metric | Baseline (zero-shot) | Target |
|--------|:-------------------:|:------:|
| Binary F1 | 0.925 | >= 0.92 |
| Recall | 87.5% | >= 85% |
| Weighted kappa | 0.048 | >= 0.10 |
| Verification score | 0.460 | >= 0.40 |
| Thinking present | 0% | >= 80% (as literal `<think>` in content) |
| Calibration error | 0.976 | < 0.90 |
| JSON parse success | — | >= 95% (structured output after `</think>`) |

---

## Key Insights

### 1. Fix the measurement first

The evaluation harness was silently discarding Harmony thinking output.
This was fixed but revealed the model wasn't using Harmony channels anyway.
Lesson: **always `curl` the model and inspect the raw response before
running evaluation.**

### 2. Attention-only LoRA has format learning limits

With only 0.04% trainable parameters (attention layers only), the model
can learn reasoning patterns but struggles with novel output structures
(Harmony channels, JSON format). Format adherence likely requires FFN
involvement, which is frozen in MoE attention-only LoRA.

Lesson: **for MoE models, keep the output format as close to the
pretrained format as possible.** Don't try to teach new structural
tokens (channel switching) — use text patterns (`<think>`) that
attention layers can learn.

### 3. Single-channel is the pragmatic choice

The model naturally produces literal `<think>` tags and the scorer
already handles them. Fighting the Harmony channel structure cost
two training rounds. Lean into what works.

---

## File Reference

| File | Purpose |
|------|---------|
| `evaluation/harness.py` | Fixed: Harmony thinking capture, Ollama auto-detect, token budget |
| `evaluation/run.py` | Fixed: uses `full_output` for scoring and DB persistence |
| `training/data_utils.py` | Harmony channel-aware formatting (unchanged from Round 3) |
| `training/configs.py` | `_MOE_OVERRIDES` — hyperparameters for Round 4 |
| `docs/ROUND_3.md` | Hyperparameter history and Harmony template fix details |
| `docs/MISTAKES_ROUND_1_AND_FIXES.md` | Data quality issues from Round 1 |
| `docs/MISTAKES_TO_ROUND_3.md` | Evaluation infrastructure issues (this round) |
