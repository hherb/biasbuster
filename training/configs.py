"""
Training configuration for LoRA fine-tuning.

All hyperparameters are centralised here so both models use identical settings
for a controlled comparison — only the base model differs.
"""

from dataclasses import dataclass, field


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MODEL_PRESETS = {
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "olmo-3.1-32b": "allenai/OLMo-3.1-32B-Instruct",
}


@dataclass
class LoRATrainingConfig:
    """Complete configuration for a LoRA fine-tuning run."""

    # Model
    model_name_or_path: str = ""
    output_dir: str = ""

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 10

    # Data
    train_file: str = "dataset/export/alpaca/train.jsonl"
    val_file: str = "dataset/export/alpaca/val.jsonl"

    # Optional: cap training steps for smoke testing
    max_steps: int = -1


def get_config(model_key: str, output_dir: str | None = None) -> LoRATrainingConfig:
    """Return a config preset for the given model key.

    Args:
        model_key: One of the keys in MODEL_PRESETS (e.g. "qwen3.5-27b").
        output_dir: Override the default output directory.
    """
    if model_key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model key {model_key!r}. "
            f"Available: {', '.join(MODEL_PRESETS)}"
        )
    out = output_dir or f"training_output/{model_key}-lora"
    return LoRATrainingConfig(
        model_name_or_path=MODEL_PRESETS[model_key],
        output_dir=out,
    )
