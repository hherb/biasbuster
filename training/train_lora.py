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
    TrainingArguments,
)
from trl import SFTTrainer

from training.callbacks import MetricsLoggerCallback
from training.configs import LoRATrainingConfig, get_config
from training.data_utils import load_alpaca_jsonl, make_formatting_func

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


def build_trainer(
    cfg: LoRATrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
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

    # --- Base model ----------------------------------------------------------
    logger.info(f"Loading model: {cfg.model_name_or_path} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # native PyTorch SDPA; works on SM121 via NGC
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
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
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
    )

    # --- Formatting func for SFTTrainer --------------------------------------
    formatting_func = make_formatting_func(tokenizer)

    # --- Metrics callback ----------------------------------------------------
    metrics_callback = MetricsLoggerCallback(
        output_dir=cfg.output_dir,
        extra_config={
            "model_name_or_path": cfg.model_name_or_path,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "max_seq_length": cfg.max_seq_length,
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
        max_seq_length=cfg.max_seq_length,
        packing=False,
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
    args = parser.parse_args()

    # --- Config --------------------------------------------------------------
    cfg = get_config(args.model, output_dir=args.output_dir)
    if args.train_file:
        cfg.train_file = args.train_file
    if args.val_file:
        cfg.val_file = args.val_file
    if args.max_steps > 0:
        cfg.max_steps = args.max_steps

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

    # --- Build trainer -------------------------------------------------------
    trainer = build_trainer(cfg, train_dataset, val_dataset)

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
