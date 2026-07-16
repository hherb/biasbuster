# Mistakes Leading Into Round 3: Evaluation Infrastructure Blind Spots

**Date:** 2026-03-25
**Scope:** GPT-OSS 20B fine-tuning Runs 6-9 (V1 through V6)
**Prior art:** `docs/MISTAKES_ROUND_1_AND_FIXES.md` (data quality issues),
`docs/ROUND_3.md` (hyperparameter tuning and Harmony template fix)

---

## Summary

After fixing the Round 1 data quality issues (prompt divergence, retraction
contamination, oversampling) and the Round 3 Harmony template mismatch,
V6 scored **worse** than every prior version — F1 dropped from 0.925 (baseline)
to 0.652. Investigation revealed that the model was likely working correctly
but the **evaluation harness couldn't see its output**.

This document records the infrastructure-level mistakes that caused misleading
evaluation results across all GPT-OSS fine-tuning rounds.

---

## Mistake 1: Evaluation Harness Discarded Harmony Thinking Output

### What happened

The evaluation harness (`evaluation/harness.py`) extracted model output from
the Ollama response using:

```python
raw_output = data.get("message", {}).get("content", "")
```

For GPT-OSS with the Harmony response format, Ollama returns a multi-field
response:

```json
{
  "message": {
    "role": "assistant",
    "content": "4",
    "thinking": "The user asks 2+2. The answer is 4..."
  }
}
```

The `thinking` field (Harmony's `analysis` channel) was completely ignored.
After the Run 9 template fix correctly trained the model to put reasoning
in the `analysis` channel and structured output in the `final` channel, the
harness captured only the `final` channel content — which was often empty
because the model exhausted its token budget on the analysis channel.

### Impact

- V6 outputs: many examples had `raw_output=""` with `output_tokens=4000-8000`
- Empty outputs were scored as `severity=none` → massive false-negative rate
- V6 recall collapsed from 87.5% (baseline) to 49.2%
- V6 F1 of 0.652 was **not measuring model quality** — it was measuring
  what the scorer could recover from empty/truncated outputs
- This issue was present since Run 6 but was masked because pre-Harmony-fix
  models put everything in the `content` field (single channel)

### Fix

```python
# evaluation/harness.py — capture both Harmony channels
msg = data.get("message", {})
raw_output = msg.get("content", "")
thinking = msg.get("thinking", "")
```

Added `thinking` field to `ModelOutput` with a `full_output` property that
combines thinking + content as `<think>...</think>\n{content}` for the
scorer's existing extraction logic.

---

## Mistake 2: Ollama Native API Gate Was Wrong

### What happened

The harness used the Ollama native API (`/api/chat`) only when `num_ctx`
was explicitly set via `--num-ctx`. Otherwise, it used the OpenAI-compatible
endpoint (`/v1/chat/completions`), even for Ollama endpoints.

```python
# OLD: only used native API when num_ctx was set
use_ollama_native = self.config.num_ctx is not None
```

The OpenAI-compatible endpoint does **not** return the `thinking` field.
So even after adding thinking capture code, it would only work if the user
happened to pass `--num-ctx`.

### Impact

All V6 evaluation was done via the OpenAI-compatible path, which silently
dropped the Harmony thinking output.

### Fix

Auto-detect Ollama endpoints by parsing the URL port:

```python
from urllib.parse import urlparse
parsed_url = urlparse(endpoint)
is_ollama_endpoint = parsed_url.port == 11434
use_ollama_native = self.config.num_ctx is not None or is_ollama_endpoint
```

---

## Mistake 3: Token Budget Too Small for Two-Channel Output

### What happened

`max_tokens` was set to 4000 (evaluation harness default). Training data
averages ~2,700 chars of reasoning + ~1,400 chars of assessment. But at
inference, the model generates more verbose reasoning than the training
examples, especially in the analysis channel.

With the Harmony template fix, the model correctly uses the analysis channel
for extended reasoning — but with a 4000-token budget, it often exhausted
all tokens on analysis before reaching the final channel.

### Impact

Many V6 outputs hit exactly 4000 tokens with empty `content` (final channel).
Even with the harness fix capturing `thinking`, the final channel was still
empty/truncated because there were no tokens left for it.

### Fix

Increased `max_tokens` default from 4000 to 8000. The CLI `--max-tokens`
flag overrides the default, so existing scripts still work.

---

## Mistake 4: Redundant State in ModelOutput

### What happened (during the fix)

The initial fix prepended thinking into `raw_output` as a `<think>` block
AND stored it separately in `ModelOutput.thinking`. This meant thinking
content was stored three times: in `thinking`, embedded in `raw_output`,
and in the JSONL file.

### Fix

Kept `thinking` and `raw_output` as orthogonal fields. Added a `full_output`
property that combines them on demand. The JSONL stores both fields separately;
the DB callback and scorer use `full_output`.

---

## The Recurring Pattern

Round 1 and V6 are the **same fundamental failure** from opposite sides:

| | Round 1 | V6 |
|---|---|---|
| Symptom | Over-predicted HIGH | Over-predicted NONE |
| Root cause | Data quality (prompt mismatch, oversampling) | Measurement (harness ignores output) |
| Misled by | Model appeared to work, metrics looked plausible | Model appeared broken, metrics looked terrible |

The pattern: **each round fixed the model or data without validating the
measurement infrastructure**. The evaluation harness should have been tested
with a known-good Harmony model output before any fine-tuning run.

### Lesson: Fix the Measurement First

Before evaluating any model:

1. **Manual sanity check**: `curl` a test prompt to the model and inspect
   the full JSON response, including all fields
2. **Verify the harness captures what you expect**: check that `raw_output`
   is non-empty and contains the model's actual response
3. **Check token budget**: if `output_tokens` equals `max_tokens` on many
   examples, the budget is too small

---

## Mistake 5: Harmony Dual-Channel Training Was Unlearnable

### What happened

After fixing the harness (Mistakes 1-3) and re-evaluating V6, the results
showed that the model was NOT using Harmony channels at all:

- `<think>` tags appear as **literal text** in `content` (the `final` channel),
  NOT in Ollama's `thinking` field (the `analysis` channel)
- After `</think>`, the model produces markdown prose, NOT the expected JSON
- 3 of 8 initial outputs hit the 8000-token ceiling due to verbose thinking
- Some outputs show degenerate repetition (e.g., `&recrs=&recrs=&recrs=...`)

### Root cause

The Harmony template fix correctly formatted training data with channel
tokens (`<|channel|>analysis`, `<|channel|>final`). But with attention-only
LoRA (0.04% trainable params, targeting only q/k/v/o) and 1 epoch (236 steps),
the model could not internalize the channel token structure. Channel switching
likely requires FFN layer involvement (token generation patterns), which is
completely frozen in attention-only LoRA.

The model fell back to a simpler pattern it could learn with attention alone:
literal `<think>` text tags instead of Harmony channel tokens. This is the
**same failure mode as Run 8/V4** (markdown instead of JSON), just now with
`<think>` tags.

### Fix

Abandon the Harmony dual-channel approach for GPT-OSS training. Use
single-channel training with `<think>` as literal text:

- `data_utils.py`: GPT-OSS models keep everything in `content` (no split)
- The scorer's existing `<think>` extraction regex handles this format
- The model naturally produces literal `<think>` tags (proven by V6 outputs)
- Focus training signal on the JSON output format after `</think>`

See `docs/PREPARING_ROUND_4.md` for the updated training plan.

---

## Files Changed

| File | Change |
|------|--------|
| `evaluation/harness.py` | Capture Harmony `thinking` field; auto-detect Ollama endpoints; increase `max_tokens` to 8000; add `full_output` property to `ModelOutput` |
| `evaluation/run.py` | Use `full_output` for scorer and DB persistence |
| `training/data_utils.py` | Skip Harmony channel split for GPT-OSS (single-channel mode) |

---

## Timeline

| Date | Event |
|------|-------|
| 2026-03-18 | Run 6 (V1): Saturated, kappa destroyed, but F1 0.938 — looks OK |
| 2026-03-19 | Run 7: Conservative config, undertrained, not evaluated |
| 2026-03-20 | Run 8 (V4): Gradient collapse epoch 2, markdown output — format wrong |
| 2026-03-21 | Harmony template mismatch discovered and fixed in data_utils.py |
| 2026-03-24 | Run 9 (V6): Template fix applied, F1 drops to 0.652 — **worse than baseline** |
| 2026-03-25 | Investigation reveals harness was discarding Harmony thinking output |
| 2026-03-25 | Harness fixed, V6 re-evaluation shows model uses literal `<think>` tags, not Harmony channels |
| 2026-03-25 | Decision: abandon Harmony dual-channel, use single-channel with literal `<think>` |
