"""
Surgical LoRA merge that preserves MXFP4 expert weights.

Instead of loading the full model through transformers (which dequantizes
MXFP4 and can't re-quantize on save), this script operates directly on
the safetensors files:

1. Copies all non-attention tensors (experts, router, norms, embeddings)
   byte-for-byte from the original model shards — MXFP4 stays MXFP4.
2. Loads only the attention tensors (q/k/v/o_proj) as BF16.
3. Applies the LoRA delta: merged = base + (lora_B @ lora_A) * (alpha / r)
4. Writes new shards with merged attention weights + original everything else.

Result: ~14 GB model identical to gpt-oss:20b except for merged attention
weights.  Can be imported directly into Ollama which serves MXFP4 natively.

Dependencies: safetensors, numpy, huggingface_hub (all in the project venv).
Does NOT require torch.

Usage:
    uv run python -m training.merge_adapter_surgical --model gpt-oss-20b

    uv run python -m training.merge_adapter_surgical \\
        --base-model openai/gpt-oss-20b \\
        --adapter-path training_output/gpt-oss-20b-lora/final_adapter \\
        --output-dir training_output/gpt-oss-20b-merged
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from training.configs import MODEL_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_base_model_path(model_name_or_path: str) -> Path:
    """Resolve a HuggingFace model ID to its local cache path.

    Args:
        model_name_or_path: HF model ID (e.g. "openai/gpt-oss-20b") or local path.

    Returns:
        Path to the directory containing model safetensors and config.
    """
    path = Path(model_name_or_path)
    if path.exists() and path.is_dir():
        return path

    # Resolve from HuggingFace cache
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(model_name_or_path)
    return Path(cache_dir)


def _bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert a bfloat16 numpy array to float32 for arithmetic.

    numpy doesn't natively support bfloat16, so safetensors returns BF16
    tensors as uint16 views.  We convert via the BF16 bit layout:
    BF16 is the upper 16 bits of a float32, so shift left by 16.
    """
    if arr.dtype == np.uint16:
        # BF16 stored as uint16 — convert to float32
        f32_bits = arr.astype(np.uint32) << 16
        return f32_bits.view(np.float32)
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float32)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    raise ValueError(f"Unexpected dtype {arr.dtype} for BF16 conversion")


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert a float32 numpy array back to bfloat16 (stored as uint16).

    Truncates the lower 16 bits of each float32 value (round-to-nearest-even
    would be more precise but truncation matches the common fast path).
    """
    f32 = arr.astype(np.float32)
    # View as uint32 and take upper 16 bits
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def load_lora_adapter(adapter_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load LoRA adapter weights and config.

    Args:
        adapter_path: Directory containing adapter_config.json and
            adapter_model.safetensors.

    Returns:
        Tuple of (adapter_weights, adapter_config).
    """
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path}")

    with open(config_path) as f:
        config = json.load(f)

    weights = load_file(str(weights_path))
    return weights, config


def compute_lora_deltas(
    adapter_weights: dict[str, np.ndarray],
    adapter_config: dict,
) -> dict[str, np.ndarray]:
    """Compute the weight deltas for each LoRA-adapted layer.

    LoRA stores two low-rank matrices per target: lora_A (r x in) and
    lora_B (out x r).  The delta is: (lora_B @ lora_A) * (alpha / r).

    PEFT names follow the pattern:
        base_model.model.model.layers.{N}.self_attn.{target}.lora_A.weight
        base_model.model.model.layers.{N}.self_attn.{target}.lora_B.weight

    We return deltas keyed by the original model tensor name:
        model.layers.{N}.self_attn.{target}.weight

    Args:
        adapter_weights: Raw adapter tensors from safetensors.
        adapter_config: Adapter config dict (contains r, lora_alpha).

    Returns:
        Dict mapping base model tensor names to their LoRA deltas (as BF16 uint16).
    """
    r = adapter_config["r"]
    alpha = adapter_config["lora_alpha"]
    scaling = alpha / r

    # Group lora_A and lora_B by their base layer name
    lora_pairs: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in adapter_weights.items():
        # Strip PEFT prefix: base_model.model.{base_name}.lora_{A|B}.weight
        if ".lora_A." in key:
            base_name = key.replace("base_model.model.", "").replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_name, {})["A"] = tensor
        elif ".lora_B." in key:
            base_name = key.replace("base_model.model.", "").replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_name, {})["B"] = tensor

    deltas = {}
    for base_name, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            logger.warning("Incomplete LoRA pair for %s, skipping", base_name)
            continue

        # Compute delta in float32 for precision, then convert back to BF16
        a_f32 = _bf16_to_f32(pair["A"])
        b_f32 = _bf16_to_f32(pair["B"])
        delta_f32 = (b_f32 @ a_f32) * scaling

        weight_key = f"{base_name}.weight"
        deltas[weight_key] = _f32_to_bf16(delta_f32)
        logger.debug("  Computed delta for %s: shape %s", weight_key, delta_f32.shape)

    return deltas


def surgical_merge(
    base_model_path: Path,
    adapter_path: Path,
    output_dir: Path,
) -> None:
    """Perform a surgical LoRA merge preserving MXFP4 expert weights.

    Args:
        base_model_path: Path to the base model directory (with safetensors).
        adapter_path: Path to the LoRA adapter directory.
        output_dir: Path to write the merged model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load LoRA adapter ---------------------------------------------------
    logger.info("Loading LoRA adapter from %s", adapter_path)
    adapter_weights, adapter_config = load_lora_adapter(adapter_path)
    deltas = compute_lora_deltas(adapter_weights, adapter_config)
    logger.info("  Computed %d LoRA deltas (r=%d, alpha=%d)",
                len(deltas), adapter_config["r"], adapter_config["lora_alpha"])

    # --- Load shard index ----------------------------------------------------
    index_path = base_model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"model.safetensors.index.json not found in {base_model_path}"
        )

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    logger.info("  Base model has %d tensors across %d shards",
                len(weight_map), len(shard_files))

    # --- Process each shard --------------------------------------------------
    merged_count = 0
    copied_count = 0
    applied_deltas: set[str] = set()

    for shard_file in shard_files:
        src_path = base_model_path / shard_file
        dst_path = output_dir / shard_file
        logger.info("Processing shard: %s", shard_file)

        # Load the entire shard
        shard_tensors = load_file(str(src_path))

        # Apply LoRA deltas to attention tensors in this shard
        for tensor_name in list(shard_tensors.keys()):
            if tensor_name in deltas:
                base_tensor = shard_tensors[tensor_name]
                delta = deltas[tensor_name]

                if base_tensor.shape != delta.shape:
                    raise ValueError(
                        f"Shape mismatch for {tensor_name}: "
                        f"base={base_tensor.shape}, delta={delta.shape}"
                    )

                # Both are BF16 stored as uint16.  Convert to f32, add, back to BF16.
                base_f32 = _bf16_to_f32(base_tensor)
                delta_f32 = _bf16_to_f32(delta)
                merged_f32 = base_f32 + delta_f32
                shard_tensors[tensor_name] = _f32_to_bf16(merged_f32)

                merged_count += 1
                applied_deltas.add(tensor_name)
                logger.info("    Merged: %s", tensor_name)
            else:
                copied_count += 1

        # Save the shard — non-attention tensors (MXFP4 experts, router,
        # norms, etc.) pass through unchanged
        save_file(shard_tensors, str(dst_path))
        logger.info("  Saved: %s", dst_path.name)

    logger.info("  %d tensors merged, %d tensors copied unchanged",
                merged_count, copied_count)

    # Verify all deltas were applied
    unapplied = set(deltas.keys()) - applied_deltas
    if unapplied:
        logger.warning("  WARNING: %d LoRA deltas not found in base model!",
                        len(unapplied))
        for name in sorted(unapplied):
            logger.warning("    Missing: %s", name)

    # --- Copy non-weight files -----------------------------------------------
    for fname in [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "special_tokens_map.json", "chat_template.jinja",
        "model.safetensors.index.json",
    ]:
        src = base_model_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))
            logger.info("  Copied: %s", fname)

    total_size = sum(f.stat().st_size for f in output_dir.glob("*.safetensors"))
    logger.info("Done! Merged model saved to: %s", output_dir)
    logger.info("  Total size: %.1f GB (MXFP4 preserved)", total_size / 1024**3)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Surgical LoRA merge preserving MXFP4 expert weights"
    )
    parser.add_argument(
        "--model",
        help=f"Model preset key ({', '.join(MODEL_PRESETS)}). "
             "Sets paths automatically.",
    )
    parser.add_argument(
        "--base-model",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the merged model",
    )
    args = parser.parse_args()

    # Resolve from preset
    if args.model:
        if args.model not in MODEL_PRESETS:
            parser.error(
                f"Unknown model key {args.model!r}. "
                f"Available: {', '.join(MODEL_PRESETS)}"
            )
        args.base_model = args.base_model or MODEL_PRESETS[args.model]
        args.adapter_path = (
            args.adapter_path or f"training_output/{args.model}-lora/final_adapter"
        )
        args.output_dir = args.output_dir or f"training_output/{args.model}-merged"
    elif not all([args.base_model, args.adapter_path, args.output_dir]):
        parser.error(
            "Provide either --model or all of "
            "--base-model, --adapter-path, --output-dir"
        )

    base_path = resolve_base_model_path(args.base_model)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    logger.info("Base model: %s", base_path)
    logger.info("Adapter: %s", adapter_path)
    logger.info("Output: %s", output_dir)

    surgical_merge(base_path, adapter_path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
