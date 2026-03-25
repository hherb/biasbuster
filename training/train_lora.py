"""
LoRA fine-tuning script for BiasBuster bias detection models.

Designed to run inside the NGC PyTorch container on DGX Spark:

    sudo docker run --gpus all --rm -it \\
        -v /home/hherb/src/biasbuster:/workspace/biasbuster \\
        -v /home/hherb/.cache/huggingface:/root/.cache/huggingface \\
        -w /workspace/biasbuster \\
        nvcr.io/nvidia/pytorch:25.11-py3 \\
        bash -c "pip install -q trl peft datasets 'transformers>=4.57' && \\
                 python -m training.train_lora --model qwen3.5-27b"

Supports checkpoint/resume: if the output directory already contains checkpoints,
pass --resume to continue training from the latest one.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

try:
    from transformers import Mxfp4Config  # available in transformers ≥ 4.57
except ImportError:
    Mxfp4Config = None  # type: ignore[assignment,misc]

from training.callbacks import MetricsLoggerCallback
from training.configs import LoRATrainingConfig, get_config
from training.data_utils import (
    load_alpaca_jsonl,
    make_formatting_func,
    validate_harmony_channels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path to the latest checkpoint-* directory, or None."""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        ckpt_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def print_trainable_params(model) -> None:
    """Log the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total > 0 else 0
    logger.info(
        f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%) parameters"
    )


def check_sequence_lengths(
    cfg: LoRATrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
) -> None:
    """Check if any examples exceed max_seq_length and prompt the user.

    Tokenizes every example with the model's tokenizer using the same chat
    template that SFTTrainer will apply.  If any exceed max_seq_length, the
    user is asked whether to increase the limit, continue with truncation,
    or abort.

    Args:
        cfg: Training configuration (provides model name, max_seq_length).
        train_dataset: Training split.
        val_dataset: Validation split (may be None).
    """
    logger.info(
        f"Pre-flight token length check (max_seq_length={cfg.max_seq_length})..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    formatting_func = make_formatting_func(tokenizer)

    # Verify Harmony channel rendering if applicable (GPT-OSS)
    validate_harmony_channels(tokenizer, train_dataset[0])
    logger.info("  Harmony channel validation passed (or not applicable)")

    over_limit: list[tuple[str, int, int]] = []  # (split, index, length)
    max_seen = 0

    for split_name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        if dataset is None:
            continue
        for i in range(len(dataset)):
            text = formatting_func(dataset[i])
            token_ids = tokenizer.encode(text)
            n_tokens = len(token_ids)
            if n_tokens > max_seen:
                max_seen = n_tokens
            if n_tokens > cfg.max_seq_length:
                over_limit.append((split_name, i, n_tokens))

    logger.info(
        f"  Longest sequence: {max_seen} tokens "
        f"(limit: {cfg.max_seq_length})"
    )

    if not over_limit:
        logger.info("  All examples fit within max_seq_length. No truncation needed.")
        return

    # --- Examples would be truncated — ask the user --------------------------
    logger.warning(
        f"  {len(over_limit)} example(s) exceed max_seq_length "
        f"({cfg.max_seq_length} tokens):"
    )
    # Show up to 5 worst offenders
    over_limit.sort(key=lambda x: x[2], reverse=True)
    for split_name, idx, length in over_limit[:5]:
        logger.warning(f"    {split_name}[{idx}]: {length} tokens")
    if len(over_limit) > 5:
        logger.warning(f"    ... and {len(over_limit) - 5} more")

    suggested = ((max_seen // 256) + 1) * 256  # round up to next 256
    print(
        f"\n{'='*60}\n"
        f"  {len(over_limit)} example(s) will be TRUNCATED.\n"
        f"  Longest: {max_seen} tokens, current limit: {cfg.max_seq_length}\n"
        f"\n"
        f"  Options:\n"
        f"    [i] Increase max_seq_length to {suggested} and continue\n"
        f"    [t] Truncate (continue with current limit — data WILL be lost)\n"
        f"    [a] Abort training\n"
        f"{'='*60}"
    )

    while True:
        choice = input("  Your choice [i/t/a]: ").strip().lower()
        if choice == "i":
            cfg.max_seq_length = suggested
            logger.info(f"  max_seq_length increased to {suggested}")
            return
        elif choice == "t":
            logger.warning(
                f"  Continuing with truncation — {len(over_limit)} examples "
                f"will lose data beyond token {cfg.max_seq_length}"
            )
            return
        elif choice == "a":
            logger.info("  Aborted by user.")
            sys.exit(0)
        else:
            print("  Please enter 'i', 't', or 'a'.")


def build_trainer(
    cfg: LoRATrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    resume: bool = False,
) -> SFTTrainer:
    """Load model, apply LoRA, and return a configured SFTTrainer."""

    # --- Tokenizer -----------------------------------------------------------
    logger.info(f"Loading tokenizer: {cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = cfg.max_seq_length

    # --- Base model ----------------------------------------------------------
    logger.info(f"Loading model: {cfg.model_name_or_path} (bf16)")

    load_kwargs: dict = dict(
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
    )

    # GPT-OSS MoE: expert weights are stored in MXFP4 format. The backward
    # pass is not implemented for MXFP4, so we dequantize to BF16 on load to
    # allow gradient flow through frozen expert layers during LoRA training.
    # Ref: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
    #
    # Mxfp4Config requires device_map="auto"; we constrain it to GPU 0 via
    # max_memory to prevent accelerate from offloading to meta/CPU (which
    # breaks backward pass with "expected device meta but got cuda:0").
    if cfg.mxfp4_dequantize:
        if Mxfp4Config is None:
            raise RuntimeError(
                "Mxfp4Config requires transformers >= 4.57. "
                "Upgrade: pip install 'transformers>=4.57'"
            )
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: f"{torch.cuda.get_device_properties(0).total_memory // (1024**3)}GiB"}
        logger.info("  MXFP4 dequantize=True (MoE expert weights → BF16)")
        logger.info("  max_memory={0: %s}", load_kwargs["max_memory"][0])

    logger.info(f"  attn_implementation={cfg.attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, **load_kwargs
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # --- LoRA ----------------------------------------------------------------
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # --- Training args -------------------------------------------------------
    # Compute warmup_steps from warmup_ratio (warmup_ratio is deprecated in
    # TRL v5.2+).  total_steps = ceil(len(train) / batch / grad_accum) * epochs.
    import math
    effective_batch = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    )
    total_steps = math.ceil(len(train_dataset) / effective_batch) * cfg.num_train_epochs
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    logger.info(
        f"Warmup: {warmup_steps} steps "
        f"({cfg.warmup_ratio:.0%} of {total_steps} total steps)"
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        label_smoothing_factor=cfg.label_smoothing_factor,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=cfg.eval_steps if val_dataset is not None else None,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        report_to="none",
        dataloader_num_workers=0,
        max_steps=cfg.max_steps,
        remove_unused_columns=False,
        packing=False,
    )

    # --- Formatting func for SFTTrainer --------------------------------------
    formatting_func = make_formatting_func(tokenizer)

    # --- Metrics callback ----------------------------------------------------
    metrics_callback = MetricsLoggerCallback(
        output_dir=cfg.output_dir,
        resume=resume,
        extra_config={
            "model_name_or_path": cfg.model_name_or_path,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "max_seq_length": cfg.max_seq_length,
            "weight_decay": cfg.weight_decay,
            "label_smoothing_factor": cfg.label_smoothing_factor,
        },
    )

    # --- Trainer -------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=[metrics_callback],
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for BiasBuster bias detection"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model preset key (e.g. qwen3.5-27b, olmo-3.1-32b)",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Path to training JSONL (default: from config)",
    )
    parser.add_argument(
        "--val-file",
        default=None,
        help="Path to validation JSONL (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap training steps (for smoke testing; -1 = unlimited)",
    )
    # Optional hyperparameter overrides (used by the GUI workbench)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank (r)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length")
    args = parser.parse_args()

    # --- Config --------------------------------------------------------------
    cfg = get_config(args.model, output_dir=args.output_dir)
    if args.train_file:
        cfg.train_file = args.train_file
    if args.val_file:
        cfg.val_file = args.val_file
    if args.max_steps > 0:
        cfg.max_steps = args.max_steps
    # Apply optional hyperparameter overrides
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lora_rank is not None:
        cfg.lora_r = args.lora_rank
        cfg.lora_alpha = args.lora_rank * 2  # maintain alpha/r = 2
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.max_seq_len is not None:
        cfg.max_seq_length = args.max_seq_len

    logger.info(f"Model: {cfg.model_name_or_path}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"LoRA r={cfg.lora_r}, alpha={cfg.lora_alpha}")

    # --- Data ----------------------------------------------------------------
    logger.info(f"Loading training data: {cfg.train_file}")
    train_dataset = load_alpaca_jsonl(cfg.train_file)
    logger.info(f"  {len(train_dataset)} training examples")

    val_dataset = None
    if cfg.val_file and Path(cfg.val_file).exists():
        logger.info(f"Loading validation data: {cfg.val_file}")
        val_dataset = load_alpaca_jsonl(cfg.val_file)
        logger.info(f"  {len(val_dataset)} validation examples")

    # --- Pre-flight truncation check -----------------------------------------
    check_sequence_lengths(cfg, train_dataset, val_dataset)

    # --- Build trainer -------------------------------------------------------
    trainer = build_trainer(cfg, train_dataset, val_dataset, resume=args.resume)

    # --- Resume from checkpoint? ---------------------------------------------
    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(cfg.output_dir)
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
        else:
            logger.info("No existing checkpoint found — starting fresh")

    # --- Train ---------------------------------------------------------------
    logger.info("Starting training...")
    result = trainer.train(resume_from_checkpoint=resume_from)

    # --- Save final adapter --------------------------------------------------
    final_dir = Path(cfg.output_dir) / "final_adapter"
    logger.info(f"Saving final adapter to {final_dir}")
    trainer.model.save_pretrained(str(final_dir))
    trainer.processing_class.save_pretrained(str(final_dir))

    # --- Summary -------------------------------------------------------------
    metrics = result.metrics
    train_loss = metrics.get("train_loss")
    train_runtime = metrics.get("train_runtime")
    logger.info("Training complete!")
    logger.info(f"  Train loss: {train_loss:.4f}" if train_loss is not None else "  Train loss: N/A")
    logger.info(f"  Train runtime: {train_runtime:.0f}s" if train_runtime is not None else "  Train runtime: N/A")
    logger.info(f"  Final adapter saved to: {final_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
