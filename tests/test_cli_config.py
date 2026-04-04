"""Tests for cli.config — configuration loading."""

import os
import tempfile
from pathlib import Path

from cli.settings import CLIConfig, load_config


def test_defaults():
    """Default config has sensible values."""
    cfg = CLIConfig()
    assert cfg.model == "ollama:qwen3.5-9b-biasbuster"
    assert cfg.provider == "ollama"
    assert cfg.model_name == "qwen3.5-9b-biasbuster"
    assert cfg.temperature == 0.1


def test_provider_extraction():
    """Provider and model_name are extracted from model string."""
    cfg = CLIConfig(model="anthropic:claude-sonnet-4-6")
    assert cfg.provider == "anthropic"
    assert cfg.model_name == "claude-sonnet-4-6"

    cfg2 = CLIConfig(model="deepseek:deepseek-reasoner")
    assert cfg2.provider == "deepseek"
    assert cfg2.model_name == "deepseek-reasoner"

    # Bare model name → defaults to ollama
    cfg3 = CLIConfig(model="qwen3.5-9b-biasbuster")
    assert cfg3.provider == "ollama"
    assert cfg3.model_name == "qwen3.5-9b-biasbuster"


def test_load_config_from_toml():
    """Config values are read from TOML file."""
    toml_content = """\
[model]
default = "anthropic:claude-sonnet-4-6"

[api_keys]
anthropic = "test-key-123"

[general]
email = "test@example.com"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()

        cfg = load_config(config_path=f.name)
        assert cfg.model == "anthropic:claude-sonnet-4-6"
        assert cfg.anthropic_api_key == "test-key-123"
        assert cfg.email == "test@example.com"

    os.unlink(f.name)


def test_cli_overrides_config_file():
    """CLI flags take precedence over config file."""
    toml_content = """\
[model]
default = "anthropic:claude-sonnet-4-6"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()

        cfg = load_config(
            config_path=f.name,
            cli_model="ollama:custom-model",
        )
        assert cfg.model == "ollama:custom-model"

    os.unlink(f.name)


def test_env_overrides_config_file(monkeypatch):
    """Environment variables override config file values."""
    toml_content = """\
[api_keys]
anthropic = "file-key"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        cfg = load_config(config_path=f.name)
        assert cfg.anthropic_api_key == "env-key"

    os.unlink(f.name)


def test_missing_config_file_uses_defaults():
    """Missing config file results in default values."""
    cfg = load_config(config_path="/nonexistent/path.toml")
    assert cfg.model == "ollama:qwen3.5-9b-biasbuster"
    assert cfg.email == ""
