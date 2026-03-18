"""
MLX LoRA/QLoRA fine-tuning for BiasBuster on Apple Silicon.

Runs directly on macOS — no Docker required. Uses pre-quantized models from
the mlx-community on HuggingFace for memory-efficient QLoRA fine-tuning.

Usage:
    uv run python -m training.train_lora_mlx --model qwen3.5-27b-4bit
    uv run python -m training.train_lora_mlx --model qwen3.5-9b-4bit --max-iters 5
    uv run python -m training.train_lora_mlx --model qwen3.5-9b-4bit --resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner import train, TrainingArgs
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers

from training.callbacks_mlx import MLXMetricsLoggerCallback
from training.configs_mlx import MLXLoRAConfig, get_mlx_config
from training.data_utils import ensure_chat_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default alpaca export directory
DEFAULT_ALPACA_DIR = "dataset/export/alpaca"


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def compute_total_iters(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
) -> int:
    """Compute total training iterations from dataset size and config.

    Args:
        dataset_size: Number of training examples.
        batch_size: Per-step batch size.
        num_epochs: Number of training epochs.

    Returns:
        Total iterations.
    """
    iters_per_epoch = max(dataset_size // batch_size, 1)
    return iters_per_epoch * num_epochs


def build_lora_config(cfg: MLXLoRAConfig) -> dict:
    """Build the LoRA config dict expected by mlx_lm.tuner.utils.linear_to_lora_layers.

    Returns:
        Dict with keys: num_layers, lora_parameters.
    """
    return {
        "num_layers": cfg.lora_num_layers,
        "lora_parameters": {
            "rank": cfg.lora_rank,
            "scale": cfg.lora_scale,
            "dropout": cfg.lora_dropout,
        },
    }


def find_latest_adapter(output_dir: str) -> str | None:
    """Return the path to the latest adapter file in output_dir, or None."""
    out = Path(output_dir)
    if not out.exists():
        return None
    # MLX-lm saves adapters as adapters.safetensors (or numbered checkpoints)
    adapter_file = out / "adapters.safetensors"
    if adapter_file.exists():
        return str(adapter_file)
    # Check for checkpoint directories (checkpoint-NNN/)
    numeric_checkpoints = []
    for p in out.glob("checkpoint-*/adapters.safetensors"):
        suffix = p.parent.name.split("-")[-1]
        try:
            numeric_checkpoints.append((int(suffix), p))
        except ValueError:
            continue  # skip non-numeric checkpoint dirs
    if numeric_checkpoints:
        numeric_checkpoints.sort(key=lambda x: x[0])
        return str(numeric_checkpoints[-1][1])
    return None


def main() -> int:
    """Main entry point for MLX LoRA fine-tuning."""
    parser = argparse.ArgumentParser(
        description="MLX LoRA/QLoRA fine-tuning for BiasBuster (Apple Silicon)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model preset key (e.g. qwen3.5-27b-4bit, qwen3.5-9b-4bit)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_ALPACA_DIR,
        help="Path to alpaca JSONL directory (default: dataset/export/alpaca)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest adapter checkpoint in output_dir",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=-1,
        help="Cap training iterations (for smoke testing; -1 = compute from epochs)",
    )
    args = parser.parse_args()

    # --- Config --------------------------------------------------------------
    cfg = get_mlx_config(args.model, output_dir=args.output_dir)
    if args.max_iters > 0:
        cfg.max_iters = args.max_iters

    logger.info("Model: %s", cfg.model_name_or_path)
    logger.info("Output: %s", cfg.output_dir)
    logger.info(
        "LoRA rank=%d, scale=%.1f, num_layers=%d",
        cfg.lora_rank, cfg.lora_scale, cfg.lora_num_layers,
    )

    # --- Data preparation: alpaca → chat format ------------------------------
    alpaca_dir = Path(args.data_dir)
    chat_dir = Path(cfg.data_dir)
    logger.info("Ensuring chat-format data in %s", chat_dir)
    ensure_chat_data(alpaca_dir, chat_dir)

    # Count training examples for iteration calculation
    train_file = chat_dir / "train.jsonl"
    if not train_file.exists():
        logger.error("Training data not found: %s", train_file)
        return 1
    dataset_size = count_jsonl_lines(train_file)
    logger.info("Training examples: %d", dataset_size)

    # --- Compute iterations --------------------------------------------------
    # MLX-lm uses iterations (not epochs). We compute total iters from epochs.
    # Note: MLX-lm TrainingArgs has no grad_accumulation — batch_size is the
    # effective batch size per iteration.
    total_iters = compute_total_iters(
        dataset_size, cfg.batch_size, cfg.num_train_epochs,
    )
    if cfg.max_iters > 0:
        total_iters = min(total_iters, cfg.max_iters)
    iters_per_epoch = max(dataset_size // cfg.batch_size, 1)
    logger.info(
        "Total iterations: %d (%d epochs × %d iters/epoch)",
        total_iters, cfg.num_train_epochs, iters_per_epoch,
    )

    # --- Load model + tokenizer ----------------------------------------------
    logger.info("Loading model: %s", cfg.model_name_or_path)
    model, tokenizer = load(cfg.model_name_or_path)

    # --- Apply LoRA ----------------------------------------------------------
    lora_cfg = build_lora_config(cfg)
    model.freeze()
    linear_to_lora_layers(model, lora_cfg["num_layers"], lora_cfg["lora_parameters"])
    num_train_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    pct = 100 * num_train_params / total_params if total_params > 0 else 0
    logger.info(
        "Trainable: %s / %s (%.2f%%) parameters",
        f"{num_train_params:,}", f"{total_params:,}", pct,
    )
    model.train()

    # --- Save adapter config -------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_config_path = out_dir / "adapter_config.json"
    with open(adapter_config_path, "w") as f:
        json.dump(lora_cfg, f, indent=2)
    logger.info("Adapter config saved to %s", adapter_config_path)

    # --- Load datasets via mlx_lm.tuner.datasets -----------------------------
    # load_dataset accepts a local dir path or HF repo ID + tokenizer keyword
    train_set, valid_set, _ = load_dataset(
        str(chat_dir), tokenizer=tokenizer
    )

    # --- Training args -------------------------------------------------------
    adapter_file = str(out_dir / "adapters.safetensors")
    training_args = TrainingArgs(
        adapter_file=adapter_file,
        iters=total_iters,
        batch_size=cfg.batch_size,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        steps_per_save=cfg.steps_per_save,
        max_seq_length=cfg.max_seq_length,
        grad_checkpoint=True,
    )

    # --- Optimizer with cosine schedule + warmup -----------------------------
    warmup_steps = int(total_iters * cfg.warmup_ratio)
    decay_steps = max(total_iters - warmup_steps, 1)
    lr_schedule = optim.cosine_decay(cfg.learning_rate, decay_steps)
    if warmup_steps > 0:
        lr_schedule = optim.join_schedules(
            [optim.linear_schedule(0.0, cfg.learning_rate, warmup_steps), lr_schedule],
            [warmup_steps],
        )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    # --- Metrics callback ----------------------------------------------------
    metrics_cb = MLXMetricsLoggerCallback(
        output_dir=cfg.output_dir,
        total_iters=total_iters,
        num_epochs=cfg.num_train_epochs,
        iters_per_epoch=iters_per_epoch,
        extra_config={
            "model_name_or_path": cfg.model_name_or_path,
            "lora_rank": cfg.lora_rank,
            "lora_scale": cfg.lora_scale,
            "lora_num_layers": cfg.lora_num_layers,
            "lora_dropout": cfg.lora_dropout,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "max_seq_length": cfg.max_seq_length,
            "mask_prompt": cfg.mask_prompt,
            "platform": "apple_silicon_mlx",
        },
    )

    # --- Resume from checkpoint? ---------------------------------------------
    if args.resume:
        existing = find_latest_adapter(cfg.output_dir)
        if existing:
            logger.info("Resuming from adapter: %s", existing)
            model.load_weights(existing, strict=False)
        else:
            logger.info("No existing adapter found — starting fresh")

    # --- Train ---------------------------------------------------------------
    logger.info("Starting MLX LoRA training...")
    try:
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=optimizer,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=metrics_cb,
        )
    finally:
        # Ensure metrics.jsonl gets a completion record even on crash/interrupt
        metrics_cb.finalize(total_iters)

    # --- Summary -------------------------------------------------------------
    logger.info("Training complete!")
    logger.info("  Adapter saved to: %s", adapter_file)
    logger.info("  Metrics log: %s", Path(cfg.output_dir) / "metrics.jsonl")
    logger.info(
        "  Next: merge adapter with  "
        "uv run python -m training.merge_adapter_mlx --model %s",
        args.model,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
