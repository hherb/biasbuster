# Round 3: GPT-OSS 20B Hyperparameter Optimization

**Date:** 2026-03-24
**Model:** openai/gpt-oss-20b (21B total params, 3.6B active, 32 experts, top-4 routing)
**Hardware:** DGX Spark (ARM/Blackwell/GB10, 128 GB unified memory) via NGC Docker
**Training framework:** TRL SFTTrainer + PEFT LoRA

---

## 1. Context

Round 2 (documented in `SIXTH_RUN.md`) established two key findings:

1. **Run 6** (LR 1e-5, 3 epochs): Rapid saturation — loss collapsed to ~0 within
   200 steps, gradient norms collapsed, 2/3 of compute wasted. The fine-tuned model
   improved binary detection (F1 0.938, perfect recall) but **destroyed severity
   calibration** (kappa 0.158 → 0.042) and **failed to produce thinking chains**.

2. **Revised config** (LR 5e-6, 1 epoch, dropout 0.1, weight_decay 0.02): Applied
   as a corrective to Run 6's saturation. This is what Round 3 begins with.

The open questions entering Round 3:
- Can we train longer without saturating, now that LR is halved?
- Can increased LoRA capacity (rank 32) teach thinking chains where rank 16 failed?
- Can label smoothing preserve severity calibration through multi-epoch training?

---

## 2. Run 7: Conservative Baseline (LR 5e-6, 1 Epoch)

### 2.1 Configuration

| Parameter | Value | Change from Run 6 |
|-----------|-------|-------------------|
| Learning rate | 5e-6 | Halved from 1e-5 |
| Epochs | 1 | Reduced from 3 |
| LoRA rank / alpha | 16 / 32 | Unchanged |
| LoRA dropout | 0.1 | Doubled from 0.05 |
| Weight decay | 0.02 | Doubled from 0.01 |
| Batch size | 1 (effective 4) | Unchanged |
| Training examples | 942 | Re-exported dataset (was 1,347) |
| Total steps | 236 | 942 / (1 × 4) = 236 |

### 2.2 Training Curves

**Training loss:**
- Started at ~5.0
- Smooth, steady descent throughout all 236 steps
- Ended at ~1.67 — **still clearly decreasing** at the final step
- No plateau, no saturation, no collapse

**Eval loss:**
- Tracked training loss closely throughout
- No divergence — zero overfitting signal
- Final eval loss ~1.8

**Gradient norms:**
- Started at ~4.5
- Settled to ~0.7 by step 150
- Remained stable through the end — healthy, non-collapsed gradients

**Learning rate:**
- Cosine schedule peaked around step 25
- Decayed smoothly to near zero by step 236

**GPU memory:**
- Allocated: ~40 GiB (stable)
- Max allocated: ~50 GiB (peak)
- Comfortable headroom in 128 GB unified memory

### 2.3 Analysis

Run 7 is the **opposite** of Run 6:

| Aspect | Run 6 (LR 1e-5, 3 epochs) | Run 7 (LR 5e-6, 1 epoch) |
|--------|---------------------------|--------------------------|
| Convergence | Saturated at step ~200 | Still descending at step 236 |
| Final train loss | ~0.0 | ~1.67 |
| Gradient norms | Collapsed to ~0 | Stable at ~0.7 |
| Compute utilization | 1/3 useful, 2/3 wasted | 100% useful but ended too early |

**Diagnosis:** The Run 6 corrections (halved LR, 1 epoch) successfully eliminated
saturation, but overcorrected. The model is clearly undertrained — it has not
extracted the full learning signal from the dataset.

Key evidence:
- Loss still descending with no inflection point
- Eval loss tracking train loss (no overfitting)
- Healthy gradient norms (model is still learning, not memorizing)
- The 5e-6 LR at 1 epoch leaves significant capacity on the table

---

## 3. Run 8: Optimized Configuration

Based on Run 7's undertrained curves, we make three targeted changes:

### 3.1 Changes

| Parameter | Run 7 | Run 8 | Rationale |
|-----------|:-----:|:-----:|-----------|
| Learning rate | 5e-6 | **8e-6** | Split the difference between 5e-6 (undertrained) and 1e-5 (saturated). Modest bump to steepen the loss curve |
| Epochs | 1 | **2** | Loss still descending at epoch end; a second pass should extract remaining signal. Watch eval loss for divergence |
| LoRA rank / alpha | 16 / 32 | **32 / 64** | Doubles adapter capacity from 7.9M to ~15.8M trainable params. Attention-only LoRA at rank 16 was identified as too restrictive in Run 6 analysis — may be why thinking chains never emerged |
| Label smoothing | 0.0 | **0.05** | Softens targets for better calibration. Acts as mild regularization in epoch 2 to prevent the memorization that plagued Run 6. Already validated on 9B models |

### 3.2 Unchanged Parameters

| Parameter | Value | Why keep |
|-----------|-------|---------|
| LoRA targets | q,k,v,o (attention-only) | Expert collapse risk is real; attention-only is the safe MoE strategy |
| LoRA dropout | 0.1 | Still needed as regularization |
| Weight decay | 0.02 | Still needed as regularization |
| Batch size | 1 (effective 4) | GPU memory headroom exists but gradient norms are healthy at this batch size |
| Warmup ratio | 0.1 | Standard, works well |

### 3.3 Expected Training Dynamics

- **~472 steps** (942 examples × 2 epochs / effective batch 4)
- With 8e-6 LR, expect faster initial descent than Run 7
- Epoch 1 should reach loss ~1.0-1.2 (vs Run 7's 1.67 at same point)
- Epoch 2 should push loss down further, potentially to ~0.5-0.8
- Label smoothing prevents loss from collapsing to 0 (theoretical minimum is ~0.05)
- **Key signal to watch:** eval loss diverging from train loss mid-epoch 2 = stop

### 3.4 What We Hope to Gain

1. **Thinking chains:** The rank 32 adapter has 2x the capacity of rank 16. Run 6
   analysis (§9.2) identified insufficient adapter capacity as a likely cause of
   the missing `<think>` chains. More capacity + more training steps = better chance
   of learning the new output structure.

2. **Better severity calibration:** Label smoothing prevents the model from becoming
   overconfident on modal severity classes. Combined with more training (vs Run 7's
   undertrained single epoch), the model should learn the severity scale rather than
   defaulting to MODERATE.

3. **Maintained detection performance:** Run 7's healthy training dynamics (no
   saturation, no overfitting) suggest the regularization stack (dropout 0.1,
   weight_decay 0.02, attention-only targets) is working. Adding label smoothing
   and modest LR/rank increases should not destabilize this.

---

## 4. Run 8 Results (LR 8e-6, 2 Epochs, Rank 32)

### 4.1 Training Curves

**Training loss:**
- Started very high (~22) due to rank 32 LoRA's random initialization creating
  larger perturbations than rank 16 (Run 7 started at ~5)
- Rapid recovery: dropped to ~3 by step 50, ~2 by step 100
- Plateaued at ~1.5 around step 200-250 (roughly end of epoch 1)
- Epoch 2 (steps 236-472) showed minimal further decrease — loss stayed ~1.0-1.5
- Final loss: ~1.5

**Eval loss:**
- First eval (step 50) at ~3-4
- Subsequent evals dropped to ~0.5-1.0 range
- Tracked train loss throughout — no overfitting divergence

**Gradient norms:**
- Started very high (~25-27) — expected with 2x more trainable parameters
- Settled to ~2-3 by step 100
- **Collapsed to near 0 after step ~250** (start of epoch 2)
- This echoes Run 6's gradient collapse pattern, though less severe

**Learning rate:**
- Cosine schedule peaked at 8e-6 around step 50
- Decayed smoothly to near zero by step 472

**GPU memory:**
- Allocated: ~40 GiB (stable)
- Max allocated: ~60 GiB (peak, +10 GiB vs Run 7 due to rank 32)
- Still comfortable within 128 GB unified memory

### 4.2 Analysis

Run 8 falls between the two extremes:

| Aspect | Run 6 (saturated) | Run 7 (undertrained) | Run 8 |
|--------|:-:|:-:|:-:|
| LR | 1e-5 | 5e-6 | 8e-6 |
| Rank | 16 | 16 | 32 |
| Final loss | ~0.0 | ~1.67 | ~1.5 |
| Gradient collapse | Step ~200 | Never | Step ~250 |
| Epoch utilization | 1/3 useful | 100% useful | Epoch 1 useful, epoch 2 diminishing |

**Key observations:**

1. **Epoch 2 was mostly wasted.** The gradient collapse after step ~250 indicates
   the model extracted most learnable signal during epoch 1. The second epoch added
   diminishing returns — gradients near zero mean the adapter weights barely changed.

2. **Label smoothing worked as intended.** Unlike Run 6 (loss → 0.0), Run 8's loss
   plateaued at ~1.5 — the label smoothing floor prevented complete memorization.
   This is a healthy sign for calibration.

3. **Rank 32 initialization shock.** The ~22 starting loss (vs ~5 for rank 16) is
   purely an artifact of the larger random LoRA matrices creating bigger initial
   perturbations. The model recovered within ~50 steps. Not harmful, but worth noting
   — the first ~50 steps are essentially "undoing the initialization" rather than
   learning the task.

4. **Higher LR (8e-6) + higher rank (32) converged faster.** The combination reached
   the ~1.5 plateau in ~200 steps, similar to Run 7's trajectory despite the
   initialization handicap. This suggests the LR/rank combination is well-matched.

5. **The core question remains unanswered:** Did rank 32 produce thinking chains?
   Did label smoothing improve calibration? Only evaluation can tell.

### 4.3 Comparison to Decision Framework

From §6, pre-run predictions vs actual outcomes:

| Predicted scenario | Actual? | Notes |
|-------------------|:---:|-------|
| Still undertrained at step 472 | No | Loss plateaued; gradient norms collapsed |
| Saturates early (loss ~0 before 300) | **Partial** | Loss didn't hit 0 (label smoothing), but gradients collapsed ~step 250 |
| Eval loss diverges mid-epoch 2 | No | Eval tracked train loss throughout |
| Thinking chains emerge | **TBD** | Requires evaluation |
| Severity calibration improves | **TBD** | Requires evaluation |

### 4.4 Evaluation: No Thinking Chains, No JSON Output

Run 8 was merged, exported to Ollama as `gpt-oss-20b-biasbusterV4`, and evaluated
against 123 test examples using `--mode fine-tuned`.

**Results:**
- **0 errors** out of 123 — the model consistently produced parseable output
- **Mean latency: 142.8s** per example (~2.4 minutes) — much slower than zero-shot,
  suggesting the model was producing *something* lengthy
- **0/123 outputs contained `<think>` blocks** — no thinking chains
- **123/123 outputs failed JSON parsing** — the model produced free-text markdown
  analysis instead of the structured JSON format

**Example output (first ~500 chars):**
```
**1. Statistical Reporting**
- **What we see:**
  - No relative-risk, odds-ratio, or "percentage" improvements → no relative-risk-like numbers.
  - η² (explained variance) is not a relative-risk metric → not interpretable for clinical decision-making.
  ...
- **Severity:** **HIGH** – multiple concerns (endpoint multiplicity, clinical-significance gap)
```

The model learned *how to analyze bias* (the content is substantive and domain-relevant)
but did **not learn the output format**. This is a different failure mode from Run 6:

| Aspect | Run 6 | Run 8 |
|--------|:-----:|:-----:|
| Output format | JSON (correct) | Markdown (wrong) |
| Thinking chains | Missing | Missing |
| Content quality | Degraded calibration | Substantive analysis |
| Severity ratings | Present but miscalibrated | In prose, not parseable |

### 4.5 Root Cause: Harmony Template Mismatch

Investigation of the GPT-OSS chat template (`chat_template.jinja`) revealed a
fundamental **training/inference format mismatch**.

**The Harmony response format** uses multi-channel output:
```
<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
<|start|>assistant<|channel|>final<|message|>{answer}<|return|>
```

The HF tokenizer's `apply_chat_template()` maps message fields to channels:
- `message["thinking"]` → `analysis` channel (reasoning)
- `message["content"]` → `final` channel (answer)

**What happened during training (Runs 6-8):**

`data_utils.py` put the entire training output (including `<think>` blocks) into
the `content` field:

```python
# OLD — everything goes into content → final channel
messages = [
    {"role": "assistant", "content": "<think>reasoning</think>{json}"},
]
```

The template rendered this as:
```
<|start|>assistant<|channel|>final<|message|><think>reasoning</think>{json}<|return|>
```

The model learned to produce output in the `final` channel only. The `<think>` tags
were treated as literal text within the final channel, not as a structural signal.

**What happened during inference (Ollama):**

Ollama's Harmony template generates:
```
<|start|>assistant<|channel|>analysis<|message|>
```

The model was prompted to start in the `analysis` channel — a token sequence it
**never saw during training**. It fell back to its pretrained prose style (Harmony's
analysis channel is designed for free-text reasoning), producing the markdown output
we observed.

### 4.6 The Fix: Split Think Blocks into Harmony Channels

Updated `training/data_utils.py` to split `<think>` blocks into the `thinking`
message field:

```python
# NEW — thinking goes to analysis channel, JSON to final channel
content, thinking = _split_think_block(example["output"])
assistant_msg = {"role": "assistant", "content": content}
if thinking:
    assistant_msg["thinking"] = thinking
```

This produces the correct Harmony channel tokens during training:
```
<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
<|start|>assistant<|channel|>final<|message|>{json}<|return|>
```

Now the training format matches the inference format. The model will learn:
1. Use the `analysis` channel for reasoning (replaces `<think>` blocks)
2. Switch to the `final` channel for the JSON answer

**Validation:** All 942 training examples have `<think>` blocks and split cleanly.
The `_split_think_block()` function uses a regex to extract the think block prefix
and returns `(content, thinking)` — non-Harmony models silently ignore the `thinking`
field, so this change is backward-compatible.

---

## 5. Run 9: Harmony-Correct Training

### 5.1 Configuration

The primary change is the template fix in `data_utils.py`. Hyperparameters are
reverted to conservative values — the template fix is the high-impact change, and
we don't want confounding variables:

| Parameter | Run 8 | Run 9 | Rationale |
|-----------|:-----:|:-----:|-----------|
| Learning rate | 8e-6 | **5e-6** | Proven stable in Run 7; 8e-6 caused gradient collapse in epoch 2 |
| Epochs | 2 | **1** | Epoch 2 showed gradient collapse in Run 8; 1 epoch is sufficient |
| LoRA rank / alpha | 32 / 64 | 32 / 64 | Keep rank 32 — more capacity may help with two-channel output |
| Label smoothing | 0.05 | **0.0** | Removed — may have weakened format adherence; not needed with 1 epoch |
| LoRA dropout | 0.1 | 0.1 | Unchanged |
| Weight decay | 0.02 | 0.02 | Unchanged |
| **Template fix** | No | **Yes** | `<think>` blocks split into Harmony `thinking` field |

### 5.2 What We Expect

The template fix changes *what* the model sees during training at the token level.
Previously, the model learned:
```
...<|channel|>final<|message|><think>reasoning</think>{json}<|return|>
```

Now it will learn:
```
...<|channel|>analysis<|message|>reasoning<|end|>
<|start|>assistant<|channel|>final<|message|>{json}<|return|>
```

This aligns with Harmony's native structure and with how Ollama prompts the model
at inference time. The model should:

1. **Produce reasoning in the analysis channel** — this replaces `<think>` blocks.
   The analysis channel output may or may not be visible to the user depending on
   the Ollama client, but the model will use it for chain-of-thought.

2. **Produce JSON in the final channel** — this is what the evaluation harness parses.

3. **Converge differently** — the initial loss may be higher or lower than Run 8
   depending on how well the pretrained model already understands the two-channel
   structure. If the pretrained Harmony training already used analysis+final channels,
   our fine-tuning data now *reinforces* this structure instead of fighting it.

### 5.3 Running

```bash
sudo rm -rf training_output/gpt-oss-20b-lora/
./run_training.sh gpt-oss-20b
```

---

## 6. Config Diffs

### 6.1 Run 7 → Run 8

```python
# training/configs.py _MOE_OVERRIDES
"learning_rate": 5e-6  → 8e-6    # +60%
"lora_r": 16           → 32      # 2x
"lora_alpha": 32       → 64      # maintain alpha/r = 2
"num_train_epochs": 1  → 2       # 2x
"label_smoothing_factor": 0.0 → 0.05  # new
```

### 6.2 Run 8 → Run 9

```python
# training/configs.py _MOE_OVERRIDES
"learning_rate": 8e-6  → 5e-6    # reverted to proven stable rate
"num_train_epochs": 2  → 1       # epoch 2 was wasted
"label_smoothing_factor": 0.05 → removed  # not needed with 1 epoch

# training/data_utils.py (new: Harmony channel-aware formatting)
# <think> blocks split into message["thinking"] field
# → renders as <|channel|>analysis instead of inline in <|channel|>final
```

---

## 7. Hyperparameter History

| Run | LR | Epochs | Rank | Dropout | Weight Decay | Label Smooth | Steps | Final Loss | Template Fix | Outcome |
|-----|:--:|:------:|:----:|:-------:|:------------:|:------------:|:-----:|:----------:|:---:|---------|
| 6 | 1e-5 | 3 | 16 | 0.05 | 0.01 | 0.0 | ~1,200 | ~0.0 | No | Saturated; severity destroyed; no thinking chains |
| 7 | 5e-6 | 1 | 16 | 0.1 | 0.02 | 0.0 | 236 | ~1.67 | No | Undertrained; healthy dynamics; not evaluated |
| 8 | 8e-6 | 2 | 32 | 0.1 | 0.02 | 0.05 | 472 | ~1.5 | No | Gradient collapse epoch 2; no JSON output; Harmony mismatch discovered |
| 9 | 5e-6 | 1 | 32 | 0.1 | 0.02 | 0.0 | 236 | TBD | **Yes** | Pending |

---

## 8. Decision Framework for Run 10 (If Needed)

Based on Run 9 outcomes:

| If Run 9 shows... | Then for Run 10... |
|-------------------|-------------------|
| JSON output + analysis channel reasoning | **Success** — evaluate metrics vs baseline |
| JSON output but no reasoning | Model uses final channel correctly but skips analysis; may need `add_generation_prompt` tuning |
| Still markdown/prose output | Template fix didn't propagate; check tokenizer cache, verify formatted output |
| Loss much higher than Run 8 | Two-channel structure is harder to learn; try 2 epochs or LR 8e-6 |
| Loss much lower than Run 8 | Two-channel structure aligns with pretraining; model is learning faster |
| Thinking chains but degraded calibration | Template fix worked; add label smoothing back (0.05) for Run 10 |

---

## 9. Key Lessons

### 9.1 Chat Template Alignment Is Critical for MoE Models

GPT-OSS's Harmony format is fundamentally different from ChatML. The multi-channel
response structure (`analysis` for reasoning, `final` for answers) means that
training data must be formatted with separate `thinking` and `content` fields.
Putting everything in `content` trains the model to use only the `final` channel,
which creates a mismatch at inference time when Ollama prompts the model to start
in the `analysis` channel.

**This is not documented in the OpenAI fine-tuning cookbook** or anywhere else we
found. It only becomes apparent when:
1. You fine-tune with `<think>` style reasoning chains, AND
2. You serve via Ollama (which uses the full Harmony template), AND
3. You notice the model produces prose instead of JSON

### 9.2 Hyperparameter Tuning Was a Red Herring

Runs 7 and 8 systematically explored LR, rank, epochs, and label smoothing. The
training curves looked reasonable — loss descended, no catastrophic overfitting.
But the output format was wrong for a structural reason (template mismatch) that
no amount of hyperparameter tuning could fix.

**Lesson:** When a fine-tuned model produces output in a fundamentally different
format than expected, investigate the tokenization/template pipeline before
adjusting hyperparameters.

---

## 10. File Reference

| File | Change |
|------|--------|
| `training/data_utils.py` | Added `_split_think_block()` to extract `<think>` blocks into Harmony `thinking` field |
| `training/configs.py` | Updated `_MOE_OVERRIDES`: LR 5e-6, rank 32, 1 epoch, removed label smoothing |
| `docs/ROUND_3.md` | This document |
| `docs/papers/SIXTH_RUN.md` | Run 6 analysis (prior round, referenced for context) |
| `docs/ROUND_2_PREPARATIONS.md` | Data pipeline changes between Round 1 and Round 2 |
