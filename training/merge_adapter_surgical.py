"""
Surgical LoRA merge that preserves MXFP4 expert weights.

Instead of loading the full model through transformers (which dequantizes
MXFP4 and can't re-quantize on save), this script operates directly on
the safetensors files at the byte level:

1. Copies each shard file byte-for-byte to the output directory.
2. Reads the safetensors header to locate attention tensors (q/k/v/o_proj).
3. Reads those tensors as raw BF16 bytes, applies the LoRA delta in float32,
   converts back to BF16, and writes the modified bytes in place.
4. All other tensors (MXFP4 experts, router, norms, embeddings) are
   untouched — bit-for-bit identical to the original.

Result: ~14 GB model identical to gpt-oss:20b except for merged attention
weights.  Can be imported directly into Ollama which serves MXFP4 natively.

Dependencies: numpy, huggingface_hub, safetensors (header parsing only).
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
import struct
import sys
from pathlib import Path

import numpy as np

from training.configs import MODEL_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BF16 ↔ F32 conversion via bit manipulation
# ---------------------------------------------------------------------------

def bf16_bytes_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    """Convert raw BF16 bytes to a float32 numpy array.

    BF16 is the upper 16 bits of float32: shift left by 16 to reconstruct.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    f32_bits = u16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Convert a float32 numpy array to raw BF16 bytes.

    Truncates the lower 16 bits of each float32 (matching the common fast path
    used by most BF16 implementations).
    """
    f32 = arr.astype(np.float32)
    u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    return u16.tobytes()


# ---------------------------------------------------------------------------
# Safetensors raw I/O
# ---------------------------------------------------------------------------

def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Read the safetensors header and return (metadata_dict, data_offset).

    The data_offset is the byte position where tensor data begins
    (8 bytes for header size + header_size bytes for JSON header).
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    meta = json.loads(header_json)
    data_start = 8 + header_size
    return meta, data_start


def read_tensor_bytes(path: Path, data_start: int, offset_begin: int,
                      offset_end: int) -> bytes:
    """Read raw tensor bytes from a safetensors file."""
    with open(path, "rb") as f:
        f.seek(data_start + offset_begin)
        return f.read(offset_end - offset_begin)


def write_tensor_bytes(path: Path, data_start: int, offset_begin: int,
                       data: bytes) -> None:
    """Write raw tensor bytes into a safetensors file at the correct offset."""
    with open(path, "r+b") as f:
        f.seek(data_start + offset_begin)
        f.write(data)


# ---------------------------------------------------------------------------
# LoRA adapter loading
# ---------------------------------------------------------------------------

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

    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(model_name_or_path)
    return Path(cache_dir)


def load_lora_adapter(adapter_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load LoRA adapter weights and config.

    The adapter safetensors contains BF16 tensors. We parse the header
    and read each tensor as raw BF16 bytes, converting to float32 numpy
    arrays for arithmetic.

    Returns:
        Tuple of (adapter_weights_f32, adapter_config).
    """
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors not found in {adapter_path}"
        )

    with open(config_path) as f:
        config = json.load(f)

    # Parse adapter safetensors and read all tensors as float32
    meta, data_start = read_safetensors_header(weights_path)
    weights: dict[str, np.ndarray] = {}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        offsets = info["data_offsets"]
        raw = read_tensor_bytes(weights_path, data_start, offsets[0], offsets[1])
        weights[name] = bf16_bytes_to_f32(raw, info["shape"])

    return weights, config


def compute_lora_deltas(
    adapter_weights: dict[str, np.ndarray],
    adapter_config: dict,
) -> dict[str, np.ndarray]:
    """Compute the weight deltas for each LoRA-adapted layer.

    LoRA stores two low-rank matrices per target: lora_A (r x in) and
    lora_B (out x r).  The delta is: (lora_B @ lora_A) * (alpha / r).

    PEFT key pattern:
        base_model.model.model.layers.{N}.self_attn.{target}.lora_A.weight

    Returns deltas keyed by the original model tensor name:
        model.layers.{N}.self_attn.{target}.weight

    All deltas are float32 numpy arrays.
    """
    r = adapter_config["r"]
    alpha = adapter_config["lora_alpha"]
    scaling = alpha / r

    # Group lora_A and lora_B by their base layer name
    lora_pairs: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in adapter_weights.items():
        if ".lora_A." in key:
            base = key.replace("base_model.model.", "").replace(
                ".lora_A.weight", ""
            )
            lora_pairs.setdefault(base, {})["A"] = tensor
        elif ".lora_B." in key:
            base = key.replace("base_model.model.", "").replace(
                ".lora_B.weight", ""
            )
            lora_pairs.setdefault(base, {})["B"] = tensor

    deltas: dict[str, np.ndarray] = {}
    for base_name, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            logger.warning("Incomplete LoRA pair for %s, skipping", base_name)
            continue
        delta = (pair["B"] @ pair["A"]) * scaling
        weight_key = f"{base_name}.weight"
        deltas[weight_key] = delta
        logger.debug("  Delta %s: shape %s", weight_key, delta.shape)

    return deltas


# ---------------------------------------------------------------------------
# Surgical merge
# ---------------------------------------------------------------------------

def surgical_merge(
    base_model_path: Path,
    adapter_path: Path,
    output_dir: Path,
) -> None:
    """Perform a surgical LoRA merge preserving MXFP4 expert weights.

    Strategy:
    1. Copy each shard file byte-for-byte to output_dir.
    2. For each attention tensor that has a LoRA delta:
       a. Read the raw BF16 bytes from the copied shard.
       b. Convert to float32, add the delta, convert back to BF16.
       c. Write the modified bytes back to the copied shard in place.
    3. All non-attention tensors (MXFP4 experts, etc.) remain untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load LoRA adapter ---------------------------------------------------
    logger.info("Loading LoRA adapter from %s", adapter_path)
    adapter_weights, adapter_config = load_lora_adapter(adapter_path)
    deltas = compute_lora_deltas(adapter_weights, adapter_config)
    logger.info(
        "  Computed %d LoRA deltas (r=%d, alpha=%d)",
        len(deltas), adapter_config["r"], adapter_config["lora_alpha"],
    )

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
    logger.info(
        "  Base model has %d tensors across %d shards",
        len(weight_map), len(shard_files),
    )

    # --- Process each shard --------------------------------------------------
    merged_count = 0
    applied_deltas: set[str] = set()

    for shard_file in shard_files:
        src_path = base_model_path / shard_file
        dst_path = output_dir / shard_file
        logger.info("Processing shard: %s", shard_file)

        # Step 1: Copy the entire shard file byte-for-byte
        shutil.copy2(str(src_path), str(dst_path))

        # Step 2: Parse header to find attention tensors in this shard
        meta, data_start = read_safetensors_header(dst_path)

        for tensor_name, info in meta.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name not in deltas:
                continue
            if info["dtype"] != "BF16":
                logger.warning(
                    "    Skipping %s: expected BF16, got %s",
                    tensor_name, info["dtype"],
                )
                continue

            delta = deltas[tensor_name]
            shape = info["shape"]
            offsets = info["data_offsets"]

            # Verify shapes match
            if list(delta.shape) != shape:
                raise ValueError(
                    f"Shape mismatch for {tensor_name}: "
                    f"delta={list(delta.shape)}, base={shape}"
                )

            # Step 3: Read base tensor, merge, write back
            raw = read_tensor_bytes(dst_path, data_start, offsets[0], offsets[1])
            base_f32 = bf16_bytes_to_f32(raw, shape)
            merged_f32 = base_f32 + delta
            merged_bytes = f32_to_bf16_bytes(merged_f32)

            write_tensor_bytes(dst_path, data_start, offsets[0], merged_bytes)

            merged_count += 1
            applied_deltas.add(tensor_name)
            logger.info("    Merged: %s", tensor_name)

    logger.info(
        "  %d tensors merged, all others copied byte-for-byte", merged_count,
    )

    # Verify all deltas were applied
    unapplied = set(deltas.keys()) - applied_deltas
    if unapplied:
        logger.warning(
            "  WARNING: %d LoRA deltas not found in base model!", len(unapplied),
        )
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
            args.adapter_path
            or f"training_output/{args.model}-lora/final_adapter"
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
