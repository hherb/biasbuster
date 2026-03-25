"""
Training configuration for LoRA fine-tuning.

Hyperparameters are centralised here with model-size-aware defaults.
The base config targets 27-32B models; 9B models get optimised overrides
applied automatically in ``get_config()``.
"""

from dataclasses import dataclass, field


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MODEL_PRESETS = {
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "olmo-3.1-32b": "allenai/OLMo-3.1-32B-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

# Overrides for gpt-oss MoE models.  Conservative LR to avoid expert collapse;
# target only attention layers (skip expert FFNs and router) for stable training.
# MXFP4 expert weights must be dequantized for gradient flow (backward pass not
# implemented for MXFP4); attn_implementation="eager" per OpenAI cookbook.
_MOE_OVERRIDES: dict = {
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 5e-6,               # proven stable in Run 7; 8e-6 caused gradient collapse in Run 8 epoch 2
    "lora_r": 32,                        # doubled from 16 — more capacity needed for analysis+final channel output
    "lora_alpha": 64,                    # maintain alpha/r = 2
    "lora_dropout": 0.1,                 # combat rapid memorisation
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,               # 2 epochs; Run 8 collapse was due to 8e-6 LR, not epoch count
    "warmup_ratio": 0.1,
    "save_total_limit": 3,
    "weight_decay": 0.02,                # additional regularisation
    "mxfp4_dequantize": True,
    "attn_implementation": "eager",
}

# Overrides for 9B-class models.  Rationale documented in SECOND_RUN.md §6.4.
_9B_OVERRIDES: dict = {
    "lora_r": 32,                       # more LoRA capacity for smaller model
    "lora_alpha": 64,                   # maintain alpha/r = 2
    "learning_rate": 2e-4,              # match 27B default — 4e-4 caused early saturation
    "gradient_accumulation_steps": 4,   # match 27B default — smoother gradients
    "num_train_epochs": 3,              # match 27B default — 5 epochs wasted compute
    "lora_dropout": 0.08,               # slightly higher to combat overfitting
    "warmup_ratio": 0.1,               # match 27B default
    "save_total_limit": 3,             # match 27B default
    "weight_decay": 0.02,              # regularisation for small model
    "label_smoothing_factor": 0.05,    # soften targets for better calibration
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
    weight_decay: float = 0.0
    label_smoothing_factor: float = 0.0

    # MoE / MXFP4 — set automatically for gpt-oss models
    mxfp4_dequantize: bool = False
    attn_implementation: str = "sdpa"

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

    Applies model-size-aware overrides for 9B models automatically.

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
    config = LoRATrainingConfig(
        model_name_or_path=MODEL_PRESETS[model_key],
        output_dir=out,
    )

    # Apply model-size/architecture-specific overrides
    if "gpt-oss" in model_key.lower():
        for key, value in _MOE_OVERRIDES.items():
            setattr(config, key, value)
    elif "9b" in model_key.lower():
        for key, value in _9B_OVERRIDES.items():
            setattr(config, key, value)

    return config
