"""
Agent configuration.

Centralises all tunables for the verification agent: Ollama connection,
generation parameters, retry settings, and API credentials.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the BiasBuster verification agent."""

    # Ollama connection
    ollama_endpoint: str = "http://localhost:11434"
    model_id: str = "qwen3.5-9b-biasbuster"

    # Generation parameters
    temperature: float = 0.1
    num_ctx: int = 4096
    max_tokens: int = 4000

    # Network resilience
    request_timeout_seconds: float = 600.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 5.0

    # Tool execution
    tool_timeout_seconds: float = 30.0
    max_concurrent_tools: int = 3

    # API credentials (passed through to collectors/enrichers)
    ncbi_api_key: str = ""
    crossref_mailto: str = ""
