# 9. Merging, Quantizing & Deploying

**What you'll do:** Merge the LoRA adapter into the base model, optionally quantize it, and deploy to Ollama for inference.

## Quick Path: lora2ollama.sh

For an end-to-end merge + export in one command:

```bash
# Cross-platform: auto-detects Linux (Docker) vs macOS (MLX)
./lora2ollama.sh qwen3.5-27b
./lora2ollama.sh qwen3.5-27b --quantize Q4_K_M
./lora2ollama.sh qwen3.5-9b-4bit --quantize Q4_K_M  # MLX preset

# Custom Ollama model name
./lora2ollama.sh qwen3.5-27b --ollama-model-name my-bias-detector
```

The script automatically detects your platform:
- **Linux**: Uses `run_merge.sh` (Docker-based dense merge or surgical MoE merge)
- **macOS**: Uses `run_merge_mlx.sh` (MLX fuse with de-quantize)

Then exports to Ollama via `training/export_to_ollama.sh`.

## Step 1: Merge the Adapter

LoRA training produces a small adapter (~50-100 MB) that must be merged back into the base model for efficient inference.

### Dense Models (Qwen, OLMo)

```bash
./run_merge.sh qwen3.5-27b
./run_merge.sh olmo-3.1-32b
./run_merge.sh qwen3.5-9b
```

The merge runs inside the NGC Docker container:
1. The base model is loaded on CPU (no GPU needed -- merge is arithmetic on weight tensors)
2. The LoRA adapter from `training_output/{model}-lora/final_adapter/` is loaded
3. Adapter weights are merged into the base model via `model.merge_and_unload()`
4. The merged model is saved in 2 GB shards to manage memory
5. The tokenizer is saved from the base model (avoids broken tokenizer configs from the adapter)

### GPT-OSS MoE (Surgical Merge)

GPT-OSS uses a special surgical merge that preserves the native MXFP4 expert weights:

```bash
./run_merge.sh gpt-oss-20b
```

Unlike the standard merge, the surgical merge:
- Runs on the host (no Docker needed) -- operates directly on safetensors files
- Copies each shard file byte-for-byte to the output directory
- Reads the safetensors header to locate attention tensors (q/k/v/o_proj)
- Applies the LoRA delta in float32, converts back to BF16, writes in place
- All other tensors (MXFP4 experts, router, norms, embeddings) are untouched -- bit-for-bit identical to the original

The result is a ~14 GB model that can be imported directly into Ollama, which serves MXFP4 natively.

### Merge + Quantize + Deploy (All-in-One)

```bash
./run_merge.sh olmo-3.1-32b --quantize q8_0
./run_merge.sh qwen3.5-27b --quantize Q4_K_M
```

With `--quantize`, the script will merge the adapter, convert to GGUF format, and import into Ollama in one step.

**Output:**
```
training_output/qwen3.5-27b-merged/
├── model.safetensors.00001-of-00006
├── model.safetensors.00002-of-00006
├── ...
├── model.safetensors.index.json
├── config.json
├── generation_config.json
└── tokenizer files
```

## Step 2: Quantize (if not using --quantize)

If you didn't use `--quantize` during merge, export to GGUF separately:

```bash
# 8-bit quantization (matches baseline evaluation)
bash training/export_to_ollama.sh \
    training_output/olmo-3.1-32b-merged \
    olmo-3.1-32b-biasbuster \
    --gguf q8_0

# 4-bit quantization (smaller, faster, slight quality loss)
bash training/export_to_ollama.sh \
    training_output/qwen3.5-27b-merged \
    qwen3.5-27b-biasbuster \
    --gguf Q4_K_M
```

### Quantization Types

| Type | Size (27B model) | Quality | Method |
|------|-------------------|---------|--------|
| `q8_0` | ~28 GB | Near-lossless | Single-pass (fast) |
| `Q4_K_M` | ~15 GB | Good balance | Two-pass (fp16 then quantize) |
| `Q3_K_L` | ~12 GB | Slight loss | Two-pass |
| `f16` | ~54 GB | Full precision | Single-pass |

Single-pass types (`q8_0`, `f16`, `bf16`) are converted directly by `convert_hf_to_gguf.py`, avoiding the ~64 GB fp16 intermediate file. Two-pass types (`Q4_K_M`, `Q5_K_M`, etc.) use `llama-quantize` and keep the intermediate file.

### Requirements

- **llama.cpp** cloned in the project root: `git clone https://github.com/ggml-org/llama.cpp.git`
- For two-pass quantization, build llama.cpp: `cd llama.cpp && cmake -B build && cmake --build build`
- A dedicated Python venv is created automatically in `llama.cpp/.venv/` on first run

## Step 3: Deploy to Ollama

If you used `--gguf` in the export step, the model is already imported into Ollama. Without `--gguf`, the export script imports safetensors directly (full precision).

### Chat Templates

The export script automatically includes the appropriate chat template in the Modelfile:
- **Qwen, OLMo**: ChatML template
- **GPT-OSS**: Harmony chat template (per OpenAI specification)

### Import Without Quantization (Full Precision)

```bash
bash training/export_to_ollama.sh \
    training_output/qwen3.5-27b-merged \
    qwen3.5-27b-biasbuster
```

### Verify Deployment

```bash
# Check the model is listed
ollama list | grep biasbuster

# Quick test
ollama run olmo-3.1-32b-biasbuster "who are you?"

# Check the API
curl -s http://localhost:11434/v1/models | python3 -m json.tool
```

## Troubleshooting

### Permission Errors

If you see "Permission denied" errors, it's likely because Docker ran as root and created files owned by root in your project directory:

```bash
sudo chown -R $(whoami):$(whoami) .venv/ training_output/
```

### Tokenizer Errors During GGUF Conversion

If `convert_hf_to_gguf.py` fails with `TokenizersBackend` or similar tokenizer errors:

```bash
# Fix the tokenizer config in the merged model
sudo sed -i 's/"tokenizer_class": "TokenizersBackend"/"tokenizer_class": "PreTrainedTokenizerFast"/' \
    training_output/olmo-3.1-32b-merged/tokenizer_config.json
```

This is already handled in `merge_adapter.py` for new merges, but may affect previously merged models.

### Model Outputs Garbage After Ollama Import

If the deployed model produces function-calling markup or other unexpected output, the Modelfile is missing a chat template. The export script now includes a chat template automatically. If using an older export, re-run the export command.

## Next Step

[Evaluating Fine-Tuned Models](10_evaluation.md) -- compare your fine-tuned model against the zero-shot baseline.
