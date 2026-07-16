# Training Monitor — Interpretation Guide

A step-by-step guide to every metric and parameter displayed on the BiasBuster Training Monitor dashboard. For each item we cover: what it means, what healthy training looks like, and what to do when things go wrong.

---

## Progress Bar

**What it shows:** `Step X/Y | Epoch Z | ETA`

- **Step** — one optimiser update. With `gradient_accumulation_steps=4` and `per_device_train_batch_size=1`, each step processes 4 examples before updating weights.
- **Epoch** — one full pass through the training data. Fractional values (e.g. 1.41) mean you are 41% through the second pass.
- **ETA** — wall-clock estimate based on elapsed time and current progress. Unreliable in the first ~5% of training.

**What to hope for:** Steady progress with no stalls. A sudden stall could indicate an OOM crash, a hung GPU, or a data loading bottleneck.

---

## Charts

### 1. Training Loss (`train_loss`)

**What it means:** The cross-entropy loss computed on each training batch. It measures how well the model predicts the next token in the training data. Lower is better.

**What healthy training looks like:**
- Starts high (2.0–3.0 for a pre-trained model being fine-tuned) and drops steeply in the first ~10–20% of training.
- Settles into a gradual decline or plateau (0.4–0.8 range for LoRA fine-tuning on structured output tasks like ours).
- Small step-to-step fluctuations are normal — the line should trend downward, not be perfectly smooth.

**Warning signs and remedies:**

| Symptom | Likely cause | What to do |
|---|---|---|
| Loss stays flat from the start | Learning rate too low, or LoRA rank too small | Increase `learning_rate` by 2–5x, or increase `lora_r` |
| Loss oscillates wildly | Learning rate too high | Reduce `learning_rate` by 2–5x |
| Loss spikes suddenly | Corrupted/outlier training example, or numerical instability | Check the training data around that step; ensure `bf16=True` is set |
| Loss drops to near zero (<0.1) | Severe overfitting or data leakage | Check dataset for duplicates; reduce `num_train_epochs` |
| Loss increases after initial drop | Learning rate too high during decay, or the model is diverging | Reduce `learning_rate`; check if `max_grad_norm` is being hit frequently |

---

### 2. Eval Loss (`eval_loss`)

**What it means:** The same cross-entropy loss, but computed on the held-out validation set (no gradient updates). This is the ground truth measure of generalisation. Plotted as red dots at every `eval_steps` interval.

**What healthy training looks like:**
- Tracks training loss, staying close to it or slightly above.
- Should decrease and then plateau. The checkpoint where eval loss is lowest is your best model.

**Warning signs and remedies:**

| Symptom | Likely cause | What to do |
|---|---|---|
| Eval loss decreases then increases while train loss keeps falling | **Overfitting** — the model is memorising training data | Stop training at the eval loss minimum (the trainer does this automatically with `load_best_model_at_end=True`). For future runs: reduce epochs, increase dropout, or add more training data |
| Eval loss is always much higher than train loss | Underfitting on eval distribution, or train/val data mismatch | Check that train and val splits come from the same distribution; ensure no data leakage in train |
| Eval loss is lower than train loss | Dropout is active during training but not eval (normal and expected for small gaps); or the val set is easier | No action needed if the gap is small. If val loss is dramatically lower, check val set composition |
| Eval loss is flat from the start | Model is not learning anything useful for the val set | Increase model capacity (`lora_r`), increase learning rate, or check that val data matches the task |

---

### 3. Learning Rate

**What it means:** The current step size for the optimiser. Controls how aggressively weights are updated. The chart shows how the learning rate changes over the course of training according to the schedule.

**With our cosine schedule (`lr_scheduler_type=cosine`):**
1. **Warmup phase** (first 10% of steps, controlled by `warmup_ratio=0.1`): LR ramps linearly from ~0 up to the peak value (`learning_rate=0.0002`). This prevents large, destructive updates when the model hasn't adjusted to the new data yet.
2. **Decay phase** (remaining 90%): LR follows a cosine curve back down toward zero. This allows the model to make large adjustments early and fine-grained adjustments later.

**What healthy training looks like:**
- A smooth ramp up, a brief plateau near peak, then a smooth cosine decay.
- The shape should be a smooth curve with no discontinuities.

**Warning signs and remedies:**

| Symptom | Likely cause | What to do |
|---|---|---|
| LR is flat at zero | Scheduler misconfiguration or training hasn't started | Check `learning_rate` and `warmup_ratio` values |
| LR spikes or has discontinuities | Checkpoint resume with mismatched scheduler state | Delete stale checkpoints and restart, or set `--resume` correctly |

**Tuning guidance:**
- If loss oscillates: reduce peak `learning_rate`.
- If loss barely moves: increase peak `learning_rate`.
- If loss spikes during warmup: increase `warmup_ratio` (e.g. 0.1 → 0.15).
- The cosine schedule is a good default for LoRA fine-tuning; rarely needs changing.

---

### 4. GPU Memory (GiB)

**What it means:** Two lines track CUDA memory on the training GPU:
- **allocated** (solid blue) — memory currently in use by tensors (model weights, activations, gradients, optimiser states).
- **max_allocated** (dashed red) — the peak memory ever allocated during training. This is your high-water mark.

**What healthy training looks like:**
- Both lines should be flat and stable after the first few steps.
- `allocated` should stay well below total GPU VRAM (96 GiB on DGX Spark's B200). A comfortable margin is 10–15% headroom.
- `max_allocated` should be only slightly above `allocated` — the gap represents transient spikes during forward/backward passes.

**Warning signs and remedies:**

| Symptom | Likely cause | What to do |
|---|---|---|
| Memory climbs steadily over time | Memory leak (unlikely with standard Trainer, but possible with custom callbacks) | Check for tensors being accumulated in lists without `.detach()` |
| OOM crash (training stops) | Model + batch + gradients exceed VRAM | Reduce `per_device_train_batch_size`, reduce `max_seq_length`, enable `gradient_checkpointing`, or reduce `lora_r` |
| Memory usage is very low (<30% of VRAM) | Underutilising the GPU | Consider increasing `per_device_train_batch_size` or `max_seq_length` for better throughput |
| Periodic spikes in max_allocated | Eval steps allocate extra memory for the val forward pass | Normal — just ensure spikes don't approach VRAM limit |

---

### 5. Gradient Norm

**What it means:** The L2 norm of the gradient vector across all trainable parameters before clipping. It tells you how large the proposed weight update is at each step.

**What healthy training looks like:**
- Starts moderate (0.05–0.5 for LoRA fine-tuning), may dip as the model settles, then remains relatively stable.
- Gentle fluctuations are normal. The trend should be flat or slowly decreasing.
- Should stay well below `max_grad_norm` (1.0 in our config) — that's the clipping threshold.

**Warning signs and remedies:**

| Symptom | Likely cause | What to do |
|---|---|---|
| Grad norm frequently hits `max_grad_norm` (1.0) | Learning rate too high, or the data has very hard/noisy examples | Reduce `learning_rate`; check for corrupted data |
| Grad norm spikes sharply | Outlier batch with unusual loss, or numerical instability | Isolated spikes are fine if loss doesn't spike too. Persistent spikes → reduce `learning_rate` |
| Grad norm near zero | Model has stopped learning, or vanishing gradients | Increase `learning_rate`; check that LoRA layers are actually attached to the right modules |
| Grad norm gradually increases each epoch | Mild overfitting — the model is trying harder to fit training examples | Expected behaviour in later epochs. If it accelerates sharply, consider early stopping |
| Grad norm is very noisy/jagged | Effective batch size too small | Increase `gradient_accumulation_steps` to smooth gradients |

---

## Hyperparameters Table

### Core Training Parameters

#### `learning_rate` (0.0002)
The peak learning rate for the optimiser (AdamW). This is the single most impactful hyperparameter.
- **Too high:** loss oscillates or diverges.
- **Too low:** loss decreases very slowly or plateaus early.
- **Good range for LoRA fine-tuning:** 1e-4 to 5e-4 for most 7B–32B models.

#### `lr_scheduler_type` (cosine)
How the learning rate changes over training. Cosine decay is the standard choice for fine-tuning — it provides aggressive learning early and gentle refinement later. Other options: `linear` (simpler decay), `constant` (no decay — rarely used for fine-tuning).

#### `warmup_ratio` (0.1)
Fraction of total steps spent linearly ramping up from zero to `learning_rate`. Prevents destructive updates before the model has adjusted to the new data distribution.
- **Higher (0.15–0.2):** safer, but wastes more steps on warmup.
- **Lower (0.03–0.05):** faster convergence, but risks early instability.
- **0.1 is a good default** for most LoRA runs.

#### `per_device_train_batch_size` (1)
Number of examples processed per GPU per forward pass. Set to 1 because 32B models are large — a single example at 4096 tokens already consumes significant VRAM.
- Effective batch size = `per_device_train_batch_size` x `gradient_accumulation_steps` = 1 x 4 = **4 examples per optimiser step**.

#### `gradient_accumulation_steps` (4)
Number of forward/backward passes before performing an optimiser step. Simulates a larger batch size without the memory cost.
- **Higher values** (8, 16): smoother gradients, more stable training, but slower convergence per wall-clock time.
- **Lower values** (1, 2): noisier gradients, faster iteration, may need lower learning rate.
- **4 is a good balance** for fine-tuning with batch size 1.

#### `bf16` (True)
Use bfloat16 mixed precision. Halves memory for activations and speeds up compute on modern GPUs (A100, H100, B200). bfloat16 has the same exponent range as float32, so it's more stable than float16 for training.
- Should always be True on hardware that supports it.

#### `max_grad_norm` (1.0)
Gradient clipping threshold. If the gradient norm exceeds this, gradients are scaled down proportionally. Prevents catastrophic updates from outlier batches.
- **1.0 is the standard default.** Rarely needs changing.
- If you see frequent clipping (gradient norm chart hitting 1.0 repeatedly), reduce `learning_rate` rather than raising this.

#### `num_train_epochs` (3)
Number of full passes through the training data. For fine-tuning:
- **1 epoch:** minimal overfitting risk, but may underfit.
- **2–3 epochs:** standard for instruction fine-tuning with LoRA.
- **>5 epochs:** high overfitting risk, especially with small datasets.
- With `load_best_model_at_end=True`, the trainer picks the best checkpoint regardless of which epoch it's from.

#### `logging_steps` (10)
How often (in steps) to log training metrics (loss, learning rate, grad norm, GPU memory). Lower values give smoother charts but add slight overhead.

#### `save_steps` (50)
How often to save a checkpoint to disk. Each checkpoint includes the LoRA adapter weights, optimiser state, and scheduler state — enabling resume if training crashes.
- Lower values: more checkpoints, more disk usage, finer granularity for picking the best model.
- Higher values: less disk usage, but you lose more work on crash.

#### `eval_steps` (50)
How often to run evaluation on the validation set. Each eval computes loss over the full val set.
- Should generally match `save_steps` so every saved checkpoint has an eval_loss score.
- Lower values give more eval data points but slow down training (eval is pure forward pass, no gradients).

### LoRA Parameters

#### `model_name_or_path` (allenai/OLMo-3.1-32B-Instruct)
The base pre-trained model being fine-tuned. LoRA adds small trainable adapter matrices on top of the frozen base weights.

#### `lora_r` (16)
The rank of the low-rank adapter matrices. Controls how many parameters LoRA adds.
- **Higher rank** (32, 64): more capacity, can learn more complex adaptations, but uses more memory and risks overfitting.
- **Lower rank** (4, 8): fewer parameters, faster training, less overfitting risk, but may underfit on complex tasks.
- **16 is a solid default** for structured output tasks on 32B models.

#### `lora_alpha` (32)
Scaling factor for LoRA updates. The effective learning rate for LoRA layers is proportional to `lora_alpha / lora_r`.
- **Rule of thumb:** set `lora_alpha = 2 * lora_r` (which is what we have: 32 = 2 x 16).
- Higher ratio → more aggressive LoRA updates. Lower ratio → more conservative.
- If training is unstable, try reducing `lora_alpha` to match `lora_r` (i.e. 16/16).

#### `lora_dropout` (0.05)
Dropout applied to LoRA layers during training. Randomly zeros out 5% of LoRA activations per forward pass to reduce overfitting.
- **0.0:** no regularisation. Fine for large datasets.
- **0.05–0.1:** light regularisation. Good default for small/medium datasets.
- **>0.1:** aggressive regularisation. Use if overfitting is severe.

#### `max_seq_length` (4096)
Maximum number of tokens per training example. Sequences longer than this are truncated.
- Directly impacts memory: longer sequences = more activations stored for backprop.
- Must be long enough to contain the full prompt + response for your task. BiasBuster annotations with `<think>` chains can be long — 4096 is a reasonable choice.
- If you see many truncation warnings in logs, consider increasing this (at the cost of memory).

---

## Putting It All Together: Reading the Dashboard

### Healthy training run checklist

1. **Train loss** drops quickly in the first epoch, then gradually decreases or plateaus.
2. **Eval loss** tracks train loss closely, with at most a small gap.
3. **Learning rate** shows a clean warmup → cosine decay shape.
4. **GPU memory** is flat and stable with headroom.
5. **Gradient norm** is stable or slowly decreasing, well below the clipping threshold.
6. **Progress** advances steadily with a reasonable ETA.

### When to stop early

- Eval loss has been increasing for 2+ consecutive eval points while train loss keeps dropping → overfitting. The trainer will select the best checkpoint automatically, but you're wasting compute.
- Gradient norms spike above 0.5 and loss becomes unstable → the run may be diverging. Consider restarting with a lower learning rate.
- Loss has plateaued for a full epoch with no improvement → further training is unlikely to help.

### After training completes

The trainer automatically loads the checkpoint with the lowest eval_loss (thanks to `load_best_model_at_end=True`) and saves it as `final_adapter/`. To verify which checkpoint was selected:

```bash
cat <output_dir>/trainer_state.json | python3 -c "
import json, sys
state = json.load(sys.stdin)
print(f'Best checkpoint: {state.get(\"best_model_checkpoint\")}')
print(f'Best eval_loss:  {state.get(\"best_metric\")}')
"
```
