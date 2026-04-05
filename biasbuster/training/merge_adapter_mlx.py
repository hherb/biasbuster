"""
Fuse an MLX LoRA adapter into the base model for deployment.

Runs directly on macOS — no Docker required. After fusing, the output can
be exported to Ollama via training/export_to_ollama.sh.

Usage:
    uv run python -m training.merge_adapter_mlx --model qwen3.5-27b-4bit
    uv run python -m training.merge_adapter_mlx --model qwen3.5-9b-4bit --de-quantize

With explicit paths:
    uv run python -m training.merge_adapter_mlx \\
        --base-model mlx-community/Qwen3.5-9B-4bit \\
        --adapter-path training_output/qwen3.5-9b-4bit-mlx-lora \\
        --output-dir training_output/qwen3.5-9b-4bit-merged \\
        --de-quantize
"""

import argparse
import logging
import sys
from pathlib import Path

from mlx_lm import fuse

from biasbuster.training.configs_mlx import MLX_MODEL_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for MLX adapter fusion."""
    parser = argparse.ArgumentParser(
        description="Fuse MLX LoRA adapter into base model (Apple Silicon)"
    )
    parser.add_argument(
        "--model",
        help=f"Model preset key ({', '.join(MLX_MODEL_PRESETS)}). "
             "Sets paths automatically.",
    )
    parser.add_argument(
        "--base-model",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to the directory containing adapters.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the fused model",
    )
    parser.add_argument(
        "--de-quantize",
        action="store_true",
        help="De-quantize to full precision (required for GGUF/Ollama export)",
    )
    args = parser.parse_args()

    # Resolve from preset if --model is given
    if args.model:
        if args.model not in MLX_MODEL_PRESETS:
            parser.error(
                f"Unknown model key {args.model!r}. "
                f"Available: {', '.join(MLX_MODEL_PRESETS)}"
            )
        preset = MLX_MODEL_PRESETS[args.model]
        args.base_model = args.base_model or preset.hf_repo
        args.adapter_path = args.adapter_path or f"training_output/{args.model}-mlx-lora"
        args.output_dir = args.output_dir or f"training_output/{args.model}-merged"
    elif not all([args.base_model, args.adapter_path, args.output_dir]):
        parser.error(
            "Provide either --model or all of --base-model, --adapter-path, --output-dir"
        )

    # Verify adapter exists
    adapter_dir = Path(args.adapter_path)
    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.exists():
        logger.error("Adapter not found: %s", adapter_file)
        logger.error("Run training first, or check --adapter-path")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Base model: %s", args.base_model)
    logger.info("Adapter: %s", args.adapter_path)
    logger.info("Output: %s", args.output_dir)
    logger.info("De-quantize: %s", args.de_quantize)

    # mlx_lm.fuse handles: load base → load adapter → merge → save
    logger.info("Fusing adapter into base model...")
    fuse(
        model=args.base_model,
        adapter_path=str(adapter_dir),
        save_path=str(output_dir),
        de_quantize=args.de_quantize,
    )

    logger.info("Done! Fused model saved to: %s", output_dir)
    if args.de_quantize:
        logger.info(
            "  Full-precision output is ready for GGUF conversion / Ollama export."
        )
        logger.info(
            "  Next: bash training/export_to_ollama.sh %s %s-biasbuster",
            output_dir, args.model or "model",
        )
    else:
        logger.info(
            "  Quantized fused model can be used directly with mlx_lm.generate()."
        )
        logger.info(
            "  For Ollama export, re-run with --de-quantize."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
