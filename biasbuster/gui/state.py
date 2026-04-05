"""Shared application state, defaults, platform detection, and persistence.

Extracted to its own module to avoid circular imports between ``app.py``
and the individual tab modules.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".biasbuster" / "gui_settings.json"

# Keys that should be persisted between sessions.
_PERSISTABLE_KEYS = {
    "model_key", "backend", "learning_rate", "num_epochs", "lora_rank",
    "batch_size", "gradient_accumulation", "max_seq_length",
    "train_file", "val_file", "test_file",
    "eval_endpoint_a", "eval_model_a",
    "eval_endpoint_b", "eval_model_b",
    "eval_mode", "ollama_model_name", "quantization",
}


def default_state() -> dict:
    """Return fresh application state with sensible defaults."""
    return {
        # Settings (persisted)
        "model_key": "",
        "backend": "",
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "lora_rank": 16,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "max_seq_length": 8192,
        "train_file": "dataset/export/alpaca/train.jsonl",
        "val_file": "dataset/export/alpaca/val.jsonl",
        "test_file": "dataset/export/alpaca/test.jsonl",
        "eval_endpoint_a": "http://localhost:11434",
        "eval_model_a": "",
        "eval_endpoint_b": "",
        "eval_model_b": "",
        "eval_mode": "zero-shot",
        "ollama_model_name": "",
        "quantization": "Q4_K_M",
        # Runtime (not persisted)
        "platform": {},
        "training_output_dir": "",
        "project_dir": "",
    }


def detect_platform() -> dict:
    """Detect OS, architecture, and available tools."""
    system = platform.system()
    machine = platform.machine()
    is_macos = system == "Darwin"
    is_linux = system == "Linux"
    has_docker = shutil.which("docker") is not None
    has_ollama = shutil.which("ollama") is not None

    backends: list[str] = []
    if is_macos:
        backends.append("mlx")
    if is_linux:
        backends.append("trl")

    return {
        "system": system,
        "machine": machine,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_windows": system == "Windows",
        "has_docker": has_docker,
        "has_ollama": has_ollama,
        "backends": backends,
    }


def save_settings(state: dict) -> None:
    """Persist user-configurable settings to disk."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in state.items() if k in _PERSISTABLE_KEYS}
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    logger.debug("Settings saved to %s", SETTINGS_PATH)


def load_settings() -> dict:
    """Load previously saved settings, returning an empty dict on failure."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text())
        return {k: v for k, v in data.items() if k in _PERSISTABLE_KEYS}
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not load settings from %s", SETTINGS_PATH)
        return {}
