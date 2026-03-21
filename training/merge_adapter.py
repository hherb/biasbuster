"""
Merge a LoRA adapter into its base model and save as full-weight safetensors.

Run inside the NGC container:

    python -m training.merge_adapter --model qwen3.5-27b
    python -m training.merge_adapter --model olmo-3.1-32b

Or with explicit paths:

    python -m training.merge_adapter \\
        --base-model Qwen/Qwen3.5-27B \\
        --adapter-path training_output/qwen3.5-27b-lora/final_adapter \\
        --output-dir training_output/qwen3.5-27b-merged
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.configs import MODEL_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--model",
        help=f"Model preset key ({', '.join(MODEL_PRESETS)}). "
             "Sets --base-model, --adapter-path, --output-dir automatically.",
    )
    parser.add_argument(
        "--base-model",
        help="HuggingFace model name or path (e.g. Qwen/Qwen3.5-27B)",
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to the saved LoRA adapter (final_adapter/ directory)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the merged model",
    )
    args = parser.parse_args()

    # Resolve from preset if --model is given
    if args.model:
        if args.model not in MODEL_PRESETS:
            parser.error(
                f"Unknown model key {args.model!r}. "
                f"Available: {', '.join(MODEL_PRESETS)}"
            )
        args.base_model = args.base_model or MODEL_PRESETS[args.model]
        args.adapter_path = args.adapter_path or f"training_output/{args.model}-lora/final_adapter"
        args.output_dir = args.output_dir or f"training_output/{args.model}-merged"
    elif not all([args.base_model, args.adapter_path, args.output_dir]):
        parser.error(
            "Provide either --model or all of --base-model, --adapter-path, --output-dir"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect MXFP4 models (e.g. GPT-OSS).  On GPU with Triton the weights
    # stay in native MXFP4 — no dequantization, so save_pretrained() has
    # nothing to revert and can write MXFP4 directly.  On CPU (no Triton)
    # the weights are auto-dequantized to BF16 and we must clear
    # quantization_config to avoid the unimplemented reverse transform.
    is_mxfp4 = "gpt-oss" in (args.model or "").lower()
    use_gpu = is_mxfp4 and torch.cuda.is_available()

    if use_gpu:
        logger.info("Loading base model on GPU (preserving native MXFP4)")
        device_map = "auto"
    else:
        logger.info("Loading base model on CPU")
        device_map = "cpu"

    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load and merge adapter
    logger.info(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(
        model, args.adapter_path, device_map=device_map
    )

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # If loaded on CPU, MXFP4 was auto-dequantized to BF16.  Clear the
    # quantization_config so save_pretrained() doesn't try the unimplemented
    # BF16→MXFP4 reverse transform.  When loaded on GPU with Triton, the
    # weights are still native MXFP4 — keep quantization_config intact so
    # the saved model preserves the MXFP4 format.
    if not use_gpu and hasattr(model.config, "quantization_config"):
        logger.info("  Clearing quantization_config (CPU dequantized to BF16)")
        del model.config.quantization_config

    # Save merged model in 2 GB shards to avoid memory spikes from
    # serialising the entire state dict at once.
    logger.info(f"Saving merged model to {output_dir}")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    # Save tokenizer from the base model — the adapter copy may have a
    # broken tokenizer_class (e.g. "TokenizersBackend" for OLMo).
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Done! Merged model saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
