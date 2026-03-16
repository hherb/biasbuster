# Running BiasBuster on NVIDIA DGX Spark

## Hardware

- **GPU**: NVIDIA GB10 (Blackwell architecture, compute capability 12.1 / SM121)
- **Memory**: 128GB unified memory
- **CUDA Toolkit**: 13.0
- **Platform**: ARM64 (aarch64), Linux

## Model Serving

### SGLang: Does NOT work

SGLang fails because `sgl_kernel` only ships prebuilt CUDA kernels for SM90 and SM100.
There is no SM121 support as of March 2026.

```
ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library!
GPU Info:
- Compute capability: 121
- Expected variant: SM121 (precise math for compatibility)
```

PyTorch also warns that max supported CUDA capability is 12.0, but this alone doesn't block execution.

### vLLM: Not tested

Not installed. Likely has the same custom CUDA kernel compatibility issues as SGLang.

### Ollama: Works (with workaround for HF cache)

Ollama compiles/selects kernels at runtime and supports the GB10.
It exposes an OpenAI-compatible API at `http://localhost:11434/v1/chat/completions`.

**Problem**: Ollama refuses to import directly from the HuggingFace cache directory because HF stores model files as symlinks pointing to `../../blobs/...`, which Ollama rejects as "insecure path" (path traversal).

```
Error: insecure path: ../../blobs/9019228d172c87d5603266c2d56672d119e838facffa164de803a1ebf0d716d2
```

**Attempted fixes**:
1. Absolute symlinks pointing to resolved blob paths -- FAILED (Ollama still detects traversal)
2. Hard links to resolved blob paths (zero extra disk usage, same inodes) -- WORKS

**Working approach**:

Create a temp directory with hard links to the resolved HF blob files:
```bash
SNAP=~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<hash>
RESOLVED=/tmp/<model>-resolved
mkdir -p "$RESOLVED"
for f in "$SNAP"/*; do
    ln "$(realpath "$f")" "$RESOLVED/$(basename "$f")"
done
```

Then create a Modelfile and import:
```bash
echo "FROM $RESOLVED/" > /tmp/Modelfile.mymodel
ollama create my-model -f /tmp/Modelfile.mymodel
```

The hard link directory can be deleted after `ollama create` completes — Ollama copies the
data into its own blob store (`~/.ollama/models/`).

### SGLang Docker Container: Partially works (outdated image)

The `lmsysorg/sglang:spark` Docker image is purpose-built for DGX Spark but has limitations:

**Problem 1: `python` is a directory, not a binary**

```bash
# FAILS — "python: cannot execute: Is a directory"
sudo docker run --gpus all -p 8000:8000 lmsysorg/sglang:spark \
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-27B --port 8000
```

Fix: use `python3` explicitly.

**Problem 2: Outdated transformers library**

The image ships an older `transformers` that does not recognize `qwen3_5` model type.
Qwen3.5 was released after the image was built.

```
KeyError: 'qwen3_5'
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

Fix: upgrade transformers inside the container with `--break-system-packages`
(safe inside a container):

```bash
sudo docker run --gpus all -p 8000:8000 lmsysorg/sglang:spark \
  bash -c "pip install --break-system-packages --upgrade transformers && \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-27B --port 8000 --reasoning-parser qwen3"
```

Note: there is no `lmsysorg/sglang:latest-spark` tag — only `:spark` exists.

**Problem 3: PyTorch CUDA capability warning**

The image's PyTorch (2.10) warns that GB10's compute capability 12.1 exceeds the
maximum supported (12.0). This is a warning, not a hard failure, but may cause
issues with certain CUDA kernels.

**Status**: Not yet confirmed working end-to-end. Older models supported by the
bundled transformers (e.g., `Qwen/Qwen3-32B`) may work without the upgrade step.

### Native SGLang (pip install): Does NOT work

Installing sglang directly into a Python venv fails because PyTorch on PyPI does
not ship aarch64 + CUDA wheels. Only the CPU-only build (`torch 2.9.1+cpu`) is
installed, causing:

```
RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA) is available.
```

Triton also falls back to CPU:
```
UserWarning: Triton is not supported on current platform, roll back to CPU.
```

There are no official PyTorch aarch64 CUDA wheels on PyPI or the PyTorch download
indices (`cu126`, `cu128`, `cu130`) as of March 2026. The only source of
aarch64 + CUDA PyTorch is the NVIDIA NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.11-py3`).

## Evaluation Run Log

### 2026-03-16: First eval attempt — total failure (no servers running)

Ran the evaluation harness expecting SGLang servers on ports 8000 and 8001:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-27b --endpoint-a http://localhost:8000 \
    --model-b olmo-3.1-32b --endpoint-b http://localhost:8001 \
    --mode zero-shot \
    --output eval_results/zero_shot/
```

**Result**: 89/89 errors for both models — `"All connection attempts failed"`. No servers
were actually running. The harness handled the errors gracefully (no crash), but produced
all-zero scores and a meaningless comparison report. Mean latency showed 0.0s because
connection refusal is near-instant.

**Root cause**: SGLang native install doesn't work on DGX Spark (no aarch64 CUDA wheels),
and the SGLang Docker container hadn't been started. The eval runner originally required
all model endpoints to be live simultaneously.

**Fix**: Added `--sequential` flag to `evaluation/run.py` so models can be evaluated one
at a time on a single GPU. The runner now polls each endpoint until it responds before
starting inference, then waits for the next server to appear before continuing.

**Lesson**: Always verify the serving backend is up before running evals:
```bash
curl http://localhost:8000/v1/models  # should return model info
```

## Running Evaluations with Ollama

Since Ollama manages model loading/unloading automatically, both models can share the same endpoint.
Use `--sequential` mode so only one model is loaded at a time:

```bash
uv run python -m evaluation.run \
    --test-set dataset/export/alpaca/test.jsonl \
    --model-a qwen3.5-27b --endpoint-a http://localhost:11434 \
    --model-b olmo-3.1-32b --endpoint-b http://localhost:11434 \
    --mode zero-shot \
    --output eval_results/zero_shot/ \
    --sequential
```

Note: The `--model-a` / `--model-b` names must match the Ollama model names (as shown by `ollama list`).

## Models Downloaded (HF cache)

Located in `~/.cache/huggingface/hub/`:

- `Qwen/Qwen3.5-27B` — 27B params, safetensors format (11 shards)
- `allenai/Olmo-3.1-32B-Instruct` — 32B params
- `allenai/Olmo-3.1-32B-Instruct-SFT` — 32B params (SFT variant)

All three fit comfortably in 128GB unified memory.
