"""
MLX LoRA/QLoRA training configuration for Apple Silicon.

Mirrors training/configs.py but with MLX-specific parameters. Uses pre-quantized
models from the mlx-community on HuggingFace for QLoRA fine-tuning.
"""

from dataclasses import dataclass


@dataclass
class MLXModelPreset:
    """Metadata for an MLX-community model preset."""

    hf_repo: str
    description: str
    recommended_num_layers: int = 16
    recommended_batch_size: int = 1


MLX_MODEL_PRESETS: dict[str, MLXModelPreset] = {
    # 64GB Mac: 4-bit quantized models
    "qwen3.5-9b-4bit": MLXModelPreset(
        hf_repo="mlx-community/Qwen3.5-9B-4bit",
        description="Qwen3.5-9B 4-bit — 64GB Mac (comfortable, ~10GB)",
        recommended_num_layers=16,
        recommended_batch_size=2,
    ),
    "qwen3.5-9b-8bit": MLXModelPreset(
        hf_repo="mlx-community/Qwen3.5-9B-8bit",
        description="Qwen3.5-9B 8-bit — 64GB Mac (comfortable, ~15GB)",
        recommended_num_layers=24,
        recommended_batch_size=2,
    ),
    # 64GB (tight) or 128GB Mac
    "qwen3.5-27b-4bit": MLXModelPreset(
        hf_repo="mlx-community/Qwen3.5-27B-4bit",
        description="Qwen3.5-27B 4-bit QLoRA — 64GB (tight) / 128GB (~25GB)",
        recommended_num_layers=16,
        recommended_batch_size=1,
    ),
    # 128GB Mac only
    "qwen3.5-27b-8bit": MLXModelPreset(
        hf_repo="mlx-community/Qwen3.5-27B-8bit",
        description="Qwen3.5-27B 8-bit QLoRA — 128GB Mac only (~38GB)",
        recommended_num_layers=32,
        recommended_batch_size=1,
    ),
    # gpt-oss MoE: 21B total / 3.6B active — 64GB+ Mac
    "gpt-oss-20b-4bit": MLXModelPreset(
        hf_repo="mlx-community/gpt-oss-20b-MXFP4-Q4",
        description="gpt-oss-20b 4-bit MoE (32 experts, top-4) — 64GB+ (~10GB model)",
        recommended_num_layers=16,
        recommended_batch_size=1,
    ),
    "gpt-oss-20b-8bit": MLXModelPreset(
        hf_repo="mlx-community/gpt-oss-20b-MXFP4-Q8",
        description="gpt-oss-20b 8-bit MoE (32 experts, top-4) — 128GB Mac (~20GB model)",
        recommended_num_layers=24,
        recommended_batch_size=1,
    ),
}


@dataclass
class MLXLoRAConfig:
    """Complete configuration for an MLX LoRA/QLoRA fine-tuning run."""

    # Model
    model_name_or_path: str = ""
    output_dir: str = ""

    # LoRA — matches training/configs.py defaults where applicable
    lora_num_layers: int = 16
    lora_rank: int = 16
    lora_scale: float = 32.0       # MLX "scale" = PEFT "alpha"
    lora_dropout: float = 0.05

    # Training
    num_train_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    mask_prompt: bool = True       # only compute loss on assistant output

    # Reporting / checkpointing
    steps_per_report: int = 10
    steps_per_eval: int = 50
    steps_per_save: int = 50

    # Data (chat-format JSONL directory — generated from alpaca exports)
    data_dir: str = "dataset/export/chat"

    # Optional: cap training iterations for smoke testing (-1 = compute from epochs)
    max_iters: int = -1


def get_mlx_config(
    model_key: str,
    output_dir: str | None = None,
) -> MLXLoRAConfig:
    """Return an MLXLoRAConfig preset for the given model key.

    Args:
        model_key: One of the keys in MLX_MODEL_PRESETS.
        output_dir: Override the default output directory.
    """
    if model_key not in MLX_MODEL_PRESETS:
        available = ", ".join(MLX_MODEL_PRESETS)
        raise ValueError(
            f"Unknown MLX model key {model_key!r}. Available: {available}"
        )
    preset = MLX_MODEL_PRESETS[model_key]
    out = output_dir or f"training_output/{model_key}-mlx-lora"
    config = MLXLoRAConfig(
        model_name_or_path=preset.hf_repo,
        output_dir=out,
        lora_num_layers=preset.recommended_num_layers,
        batch_size=preset.recommended_batch_size,
    )

    # MoE models: conservative LR to avoid expert collapse
    if "gpt-oss" in model_key.lower():
        config.learning_rate = 1e-5
        config.lora_rank = 16
        config.lora_scale = 32.0

    return config
