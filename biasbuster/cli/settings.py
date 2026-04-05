"""Configuration loading for the BiasBuster CLI.

Reads settings from a TOML config file (~/.biasbuster/config.toml by default),
with environment variable overrides and CLI flag overrides on top.

Resolution order (highest priority first): CLI flag > env var > config file > default.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".biasbuster" / "config.toml"
DEFAULT_MODEL = "ollama:qwen3.5-9b-biasbuster"
DEFAULT_DB_PATH = "dataset/biasbuster.db"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"


@dataclass
class CLIConfig:
    """Resolved configuration for a BiasBuster CLI invocation."""

    # Model selection
    model: str = DEFAULT_MODEL

    # API keys
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    ncbi_api_key: str = ""

    # Endpoints
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    deepseek_base: str = DEFAULT_DEEPSEEK_BASE

    # General
    email: str = ""
    db_path: str = DEFAULT_DB_PATH

    # Verification pipeline
    crossref_mailto: str = ""

    # LLM parameters
    temperature: float = 0.1
    max_tokens: int = 4000

    # Known LLM provider names — used to distinguish "provider:model" from
    # Ollama model names that contain colons (e.g. "gpt-oss:20b").
    KNOWN_PROVIDERS = frozenset({
        "anthropic", "ollama", "openai", "deepseek", "mistral", "gemini",
    })

    @property
    def provider(self) -> str:
        """Extract provider name from model string.

        Returns the part before the first colon if it's a known provider,
        otherwise defaults to "ollama" (Ollama model names often contain
        colons for tags, e.g. "gpt-oss:20b").
        """
        if ":" in self.model:
            prefix = self.model.split(":", 1)[0]
            if prefix in self.KNOWN_PROVIDERS:
                return prefix
        return "ollama"

    @property
    def model_name(self) -> str:
        """Extract model name from model string.

        Strips the provider prefix if present and recognised, otherwise
        returns the full string (which is the Ollama model name/tag).
        """
        if ":" in self.model:
            prefix = self.model.split(":", 1)[0]
            if prefix in self.KNOWN_PROVIDERS:
                return self.model.split(":", 1)[1]
        return self.model


def _deep_get(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)  # type: ignore[assignment]
    return d


def load_config(
    config_path: str | Path | None = None,
    *,
    cli_model: str | None = None,
    cli_email: str | None = None,
    cli_db_path: str | None = None,
) -> CLIConfig:
    """Load configuration with layered overrides.

    Args:
        config_path: Path to TOML config file. None uses the default path.
        cli_model: Model override from CLI --model flag.
        cli_email: Email override from CLI --email flag.
        cli_db_path: Database path override from CLI --db flag.

    Returns:
        Fully resolved CLIConfig.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    file_data: dict[str, Any] = {}

    if config_path.exists():
        try:
            file_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            logger.debug("Loaded config from %s", config_path)
        except Exception:
            logger.warning("Failed to parse config file %s, using defaults", config_path)
    else:
        logger.debug("No config file at %s, using defaults", config_path)

    cfg = CLIConfig()

    # Layer 1: config file values
    cfg.model = _deep_get(file_data, "model", "default", default=DEFAULT_MODEL)
    cfg.anthropic_api_key = _deep_get(file_data, "api_keys", "anthropic")
    cfg.deepseek_api_key = _deep_get(file_data, "api_keys", "deepseek")
    cfg.ncbi_api_key = _deep_get(file_data, "api_keys", "ncbi")
    cfg.ollama_endpoint = _deep_get(
        file_data, "endpoints", "ollama", default=DEFAULT_OLLAMA_ENDPOINT
    )
    cfg.deepseek_base = _deep_get(
        file_data, "endpoints", "deepseek_base", default=DEFAULT_DEEPSEEK_BASE
    )
    cfg.email = _deep_get(file_data, "general", "email")
    cfg.db_path = _deep_get(file_data, "general", "db_path", default=DEFAULT_DB_PATH)
    cfg.crossref_mailto = _deep_get(file_data, "verify", "crossref_mailto")

    # Layer 2: environment variable overrides
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", cfg.anthropic_api_key)
    cfg.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", cfg.deepseek_api_key)
    cfg.ncbi_api_key = os.environ.get("NCBI_API_KEY", cfg.ncbi_api_key)
    cfg.email = os.environ.get("BIASBUSTER_EMAIL", cfg.email)

    # Layer 3: CLI flag overrides
    if cli_model is not None:
        cfg.model = cli_model
    if cli_email is not None:
        cfg.email = cli_email
    if cli_db_path is not None:
        cfg.db_path = cli_db_path

    return cfg
