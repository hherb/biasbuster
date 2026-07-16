# Fine-Tuning OpenAI's GPT-OSS 20B: A Practitioner's Guide to LoRA on MoE Models

*Everything we learned the hard way so you don't have to.*

OpenAI's GPT-OSS 20B is an impressive model. It's their first open-weight release, it uses a Mixture-of-Experts architecture that activates only 3.6B of its 21B parameters per token, and it ships in a novel MXFP4 quantization format that fits the whole model in about 14 GB. If you've pulled it from Ollama and run it, you've probably been impressed.

But then you tried to fine-tune it.

And that's where things got interesting.

We spent several days getting LoRA fine-tuning working end-to-end on an NVIDIA DGX Spark (ARM/Blackwell, 128 GB unified memory). The process exposed a chain of issues that aren't documented anywhere — not in the OpenAI cookbook, not in the HuggingFace docs, not in the Ollama guides. This article is the write-up we wish had existed when we started.

## The Model You're Actually Working With

Before touching any training code, it's worth understanding what makes GPT-OSS different from every other model you've fine-tuned.

**Mixture-of-Experts.** The model has 24 transformer layers, each containing 32 expert FFN subnetworks. For each token, a learned router selects 4 of these 32 experts. The attention layers (Q, K, V, O projections) are shared across all tokens — they're dense, standard transformer attention. The expert FFN layers are where the MoE magic happens.

**MXFP4 quantization.** The expert FFN weights are stored in MXFP4 — a 4-bit microscaling floating-point format standardised by the Open Compute Project. Each value is 4 bits (E2M1: 1 sign, 2 exponent, 1 mantissa) with a shared 8-bit scaling exponent per block of 32 elements. In the safetensors files, these appear as:

- `model.layers.*.mlp.experts.gate_up_proj_blocks` — uint8 tensors (packed FP4 values)
- `model.layers.*.mlp.experts.gate_up_proj_scales` — uint8 tensors (E8M0 block exponents)

Everything else — attention layers, router weights, layer norms, embeddings — is standard BF16.

**Harmony response format.** GPT-OSS doesn't use ChatML (`<|im_start|>` / `<|im_end|>`). It uses OpenAI's Harmony format with `<|start|>` / `<|end|>` tokens and multi-channel output. Importantly, we found that **attention-only LoRA cannot teach the model to use Harmony's channel structure** (analysis vs final channels). The model falls back to producing literal `<think>` tags as text instead of using channel tokens. Training data must therefore use single-channel format with `<think>` as literal text in `content`.

## Before You Start

Training runs inside an NGC Docker container. Make sure your user is in the `docker` group so `./run_training.sh` can launch containers without `sudo`:

```bash
groups                           # should list 'docker'
sudo usermod -aG docker $USER   # if not, add yourself and re-login
```

You'll also need exported training data — see [Chapter 6: Exporting Training Data](../manual/06_export.md) or run `uv run python pipeline.py --stage export`.

## Step 1: Training Configuration

### What to Target with LoRA

For dense models like Qwen or LLaMA, you typically target all linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. For GPT-OSS, **target only the attention layers**:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Why? The expert FFN layers are the core of the MoE routing mechanism. Modifying them with LoRA risks "expert collapse" — a failure mode where all tokens start routing to the same experts, effectively turning your 21B MoE into a broken 3.6B dense model. The router weights should also be left frozen for the same reason.

With attention-only LoRA at rank 16, you'll see about 7.9M trainable parameters out of 20.9B total — that's 0.04%. This sounds absurdly small, but it works. The model's pretrained representations are strong enough that adjusting how it *attends* is sufficient to teach it new tasks.

### Hyperparameters That Actually Work

We learned these through trial and error:

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 5e-6 | 40x lower than the typical 2e-4. Higher rates cause the model to memorise training data within ~200 steps |
| Epochs | 2 | Loss still descending at 1 epoch; gradient collapse in Run 8 was from LR 8e-6, not epoch count |
| LoRA dropout | 0.1 | Higher than the typical 0.05 to combat rapid memorisation |
| Weight decay | 0.02 | Additional regularisation |
| LoRA rank | 32 | Doubled from 16 — more capacity needed for Harmony dual-channel output |
| LoRA alpha | 64 | Alpha/rank = 2 (standard ratio) |

The key insight: **GPT-OSS needs less training, not more.** Its strong baseline performance (we measured F1 0.918 on our task with zero-shot prompting) means the model already knows most of what you're teaching it. Fine-tuning is about shaping its output format and adding domain-specific reasoning patterns, not teaching it new knowledge.

If you see your training loss hitting zero within the first 200 steps and gradient norms collapsing, you're not doing something wrong — you're just training too long. Cut your epochs and lower your learning rate.

## Step 2: Loading the Model (The MXFP4 Problem)

Here's where the first non-obvious issue appears. You write the standard loading code:

```python
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
```

The model loads fine. Training starts. Then on the first backward pass:

```
RuntimeError: Function MmBackward0 returned an invalid gradient at index 1
  - expected device meta but got cuda:0
```

**What happened?** The MXFP4 expert weights don't support backward passes. PyTorch can do the forward computation (multiply inputs by MXFP4 weights), but it can't compute gradients through them. Even though the expert weights are frozen (no LoRA applied), the chain rule still needs to propagate gradients *through* these layers to reach the attention LoRA adapters in earlier layers.

**The fix:** Dequantize the MXFP4 weights to BF16 on load:

```python
from transformers import Mxfp4Config

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    attn_implementation="eager",  # required for GPT-OSS attention pattern
    quantization_config=Mxfp4Config(dequantize=True),
)
```

Two things to note:

1. `Mxfp4Config(dequantize=True)` converts the 4-bit expert weights to BF16 in memory. This increases memory from ~14 GB to ~42 GB. On a DGX Spark with 128 GB unified memory, this is fine. On a consumer GPU, you'll need to plan for it.

2. `attn_implementation="eager"` is required because GPT-OSS uses alternating dense and locally banded sparse attention patterns that PyTorch's SDPA doesn't handle correctly.

### The device_map Trap

If you use `device_map="auto"` (as the official OpenAI cookbook suggests), accelerate may decide your model doesn't fit and offload some parameters to the meta device. This produces the same `MmBackward0` error but for a different reason — meta tensors can't participate in gradient computation.

On single-GPU systems, use `device_map={"": 0}`. On multi-GPU systems, `device_map="auto"` with explicit `max_memory` constraints.

## Step 3: The Merge Problem

Training completes. You have a LoRA adapter in `final_adapter/`. Now you want to merge it into the base model and deploy.

The standard approach:

```python
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", ...)
model = PeftModel.from_pretrained(model, "path/to/adapter")
model = model.merge_and_unload()
model.save_pretrained("merged_output/")
```

This fails:

```
NotImplementedError  (in revert_weight_conversion → reverse_transform)
```

**What happened?** When `save_pretrained()` detects that the model was loaded with a quantization config, it tries to re-quantize the weights back to their original format. The MXFP4→BF16 dequantization was implemented. The BF16→MXFP4 re-quantization was not. `NotImplementedError`.

This happens regardless of whether you loaded on CPU or GPU, and regardless of whether you clear the `quantization_config` from the model config. The `revert_weight_conversion` hook is triggered by metadata stored during the forward weight conversion.

### The Solution: Surgical Byte-Level Merge

Since LoRA only modifies attention layers (which are BF16 in the original model) and the expert weights (MXFP4) are frozen, we don't need to load the model through transformers at all. We can operate directly on the safetensors files:

1. **Copy** each shard file byte-for-byte from the base model to the output directory
2. **Parse** the safetensors JSON header to locate attention tensor byte offsets
3. **Read** attention tensors as raw BF16 bytes, convert to float32 via bit shifting
4. **Compute** the LoRA delta: `merged = base + (lora_B @ lora_A) * (alpha / r)`
5. **Write** the merged float32 back as BF16 bytes at the same offset

Everything that isn't an attention tensor — all the MXFP4 expert blocks and scales, the router weights, layer norms, embeddings — passes through completely untouched. The output is byte-for-byte identical to the original model except for the 96 attention weight tensors (4 projections × 24 layers) that LoRA modified.

The BF16 ↔ float32 conversion is trivial because BF16 is just the upper 16 bits of a float32:

```python
def bf16_bytes_to_f32(raw_bytes, shape):
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16_bytes(arr):
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
```

The entire merge runs in seconds, uses minimal memory (only one tensor in RAM at a time), requires no GPU, no Docker, no torch — just numpy and the safetensors file format spec.

Result: a ~12.8 GB merged model with native MXFP4 expert weights preserved.

## Step 4: Ollama Export (The Template Trap)

You have a merged model. Time to import it into Ollama:

```bash
ollama create my-model -f Modelfile
```

Your Modelfile points to the merged safetensors directory. The import succeeds. You run the model. And you get... word salad.

```
>>> who are you?
".

So the answer is "Who are you?" The hint says the last line is the answer.
So it's "Who are you?" The title: "Answer the question in the title"...
```

**What happened?** Ollama did not pick up the Harmony chat template from the model's `tokenizer_config.json` or `chat_template.jinja`. Instead, it defaulted to `TEMPLATE {{ .Prompt }}` — a bare passthrough that feeds raw text to a model expecting structured Harmony format tokens.

This is a silent failure. No error, no warning. The model just produces incoherent output.

**The fix:** You need to explicitly provide the Harmony template in the Modelfile. The problem is that the Harmony template is 378 lines long — it handles tool calling, reasoning levels, multi-channel output, and response formatting. You're not going to write it by hand.

The practical solution: extract it from the base `gpt-oss:20b` Ollama model (which ships with the correct template) and inject it into your Modelfile:

```bash
# Extract everything after the FROM line from the base model
echo "FROM /path/to/merged/model/" > Modelfile
ollama show gpt-oss:20b --modelfile | sed '1,/^FROM /d' >> Modelfile
ollama create my-model -f Modelfile
```

This gives your fine-tuned model the exact same template, stop tokens, and parameters as the base model — but with your merged weights.

### The Adapter Path That Doesn't Work (Yet)

You might think: skip the merge entirely, use Ollama's `ADAPTER` directive to apply the LoRA adapter on top of the base model at runtime:

```
FROM gpt-oss:20b
ADAPTER /path/to/adapter/
```

This would be ideal — MXFP4 preserved, no merge needed. And Ollama will happily accept the Modelfile and start "converting adapter." Then:

```
Error: unsupported architecture
```

Ollama supports LoRA adapters for LLaMA, Mistral, Gemma, and a few others. GPT-OSS is not on the list yet. Similarly, if you convert the adapter to GGUF format and try to load it at runtime:

```
Error: 500 Internal Server Error: failed to initialize model: loras are not yet implemented
```

This will presumably be fixed in a future Ollama release. Until then, the merge + import path is the only option.

## The Complete Pipeline

Putting it all together:

```bash
# 1. Train with TRL/PEFT (in NGC Docker or any CUDA environment)
pip install trl peft datasets 'transformers>=4.57'

python your_train_script.py \
    --model openai/gpt-oss-20b \
    --target-modules q_proj k_proj v_proj o_proj \
    --lr 5e-6 --epochs 1 --lora-rank 16

# 2. Surgical merge (no GPU needed, just numpy)
python surgical_merge.py \
    --base-model openai/gpt-oss-20b \
    --adapter-path output/final_adapter \
    --output-dir output/merged

# 3. Import into Ollama with correct Harmony template
ollama pull gpt-oss:20b  # need the base model for its template

echo "FROM output/merged/" > /tmp/Modelfile
ollama show gpt-oss:20b --modelfile | sed '1,/^FROM /d' >> /tmp/Modelfile
ollama create my-gpt-oss-finetuned -f /tmp/Modelfile

# 4. Verify the template was applied correctly
ollama show my-gpt-oss-finetuned --modelfile | head -5
# Should NOT show "TEMPLATE {{ .Prompt }}"

# 5. Test
ollama run my-gpt-oss-finetuned
```

Total output size: ~12.8 GB with native MXFP4, same inference speed as the base model (~31 tokens/sec on Apple M3, faster on GPU).

The surgical merge script referenced above is conceptually simple — the full
implementation (about 250 lines of Python with no torch dependency) is
available in the [BiasBuster repository](https://github.com/hherb/biasbuster/blob/main/training/merge_adapter_surgical.py).

## Step 5: Evaluation (The Thinking Trap)

You run the evaluation harness and get terrible scores. F1 drops from 0.92 to 0.65. Recall collapses. The model appears broken.

**What happened?** The evaluation harness used the OpenAI-compatible API (`/v1/chat/completions`), which does not return the Harmony `thinking` field. Your fine-tuned model correctly puts reasoning in the `analysis` channel and the structured answer in the `final` channel — but the harness only sees the `final` channel. With a 4000-token budget, the model often exhausts all tokens on analysis-channel reasoning before producing any final-channel output. Empty output → scored as "no bias" → recall collapses.

This is particularly insidious because:

1. The **base model** (zero-shot) puts everything in one channel, so the harness works fine for baselines
2. The **fine-tuned model** (correctly) uses two channels, but the harness can't see the second one
3. The result looks like fine-tuning made the model worse, when it may have made it better

**The fix:**

1. Use the Ollama **native API** (`/api/chat`), which returns both `content` and `thinking` fields. The harness auto-detects Ollama endpoints on port 11434.

2. Set `--max-tokens 8000` to prevent token budget exhaustion.

3. **Always do a pre-flight sanity check** before evaluation:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "my-gpt-oss-finetuned",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": false
}' | python3 -m json.tool | grep -A2 '"thinking"'
```

If `thinking` is present, the model is working correctly and the harness will capture it.

4. After evaluation, check `thinking_present_rate` in the results — should be >0% for fine-tuned Harmony models.

## What We'd Do Differently

**Start with 2 epochs at low LR.** Our first run used 3 epochs and the model saturated in 200 steps out of 1,200. Two-thirds of the training compute was wasted on near-zero gradients.

**Don't fight the MXFP4.** We spent considerable time trying to make `save_pretrained()` work, trying `device_map="auto"` with `max_memory`, trying to clear `quantization_config` — all dead ends caused by an unimplemented reverse transform in the transformers library. The surgical byte-level merge was the right approach from the start.

**Check the Ollama template immediately.** We only discovered the Harmony template issue after the full pipeline was complete and the model was producing garbage. A quick `ollama show my-model --modelfile | grep TEMPLATE` would have caught it in seconds.

**Don't assume adapter overlay will work.** We built an elegant adapter export script before discovering that Ollama doesn't support LoRA adapters for the GPT-OSS architecture. Check the supported architecture list first.

## Summary of Issues and Fixes

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| MXFP4 backward pass | `MmBackward0` gradient error | No backward pass for MXFP4 ops | `Mxfp4Config(dequantize=True)` |
| Device offloading | Same gradient error | accelerate offloads to meta device | `device_map={"": 0}` (single GPU) |
| Attention implementation | Silent wrong results | SDPA can't handle GPT-OSS sparse attention | `attn_implementation="eager"` |
| save_pretrained() | `NotImplementedError` | BF16→MXFP4 reverse transform missing | Surgical byte-level merge |
| Ollama template | Incoherent output | Harmony template not auto-detected | Extract from base `gpt-oss:20b` model |
| Ollama adapter | "unsupported architecture" | LoRA not implemented for GPT-OSS arch | Must merge before import |
| Rapid saturation | Loss→0 in 200 steps | Model too capable for small dataset + high LR | 2 epochs, LR 5e-6, dropout 0.1, rank 32 |
| Harmony channel mismatch | Model produces prose, not JSON | Training split output into Harmony channels that attention-only LoRA can't learn | Keep everything in single `content` channel with literal `<think>` tags |
| Evaluation shows terrible F1 | Recall collapses to ~50% | Harness uses OpenAI-compat API, misses `thinking` field | Use Ollama native API (auto-detected on port 11434) |
| Token budget exhaustion | Empty final-channel output | Analysis channel consumes all 4000 tokens | Set `--max-tokens 8000` |

The GPT-OSS 20B is a remarkable model — fast, capable, and genuinely open. Fine-tuning it just requires navigating a few landmines that aren't documented yet. Hopefully this guide helps you avoid them.

---

*This guide was developed during the BiasBuster project — building fine-tuned models for detecting bias in biomedical research abstracts. The code is open source at [github.com/hherb/biasbuster](https://github.com/hherb/biasbuster).*
