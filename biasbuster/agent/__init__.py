"""
BiasBuster Verification Agent.

An agent loop that:
1. Calls the fine-tuned model for initial bias assessment
2. Synthesizes verification steps programmatically from assessment flags
3. Executes those steps concurrently using existing API clients
4. Feeds verification results back to the model for a refined assessment
"""

from biasbuster.agent.agent_config import AgentConfig
from biasbuster.agent.runner import AgentResult, format_tool_results_for_model, run_agent
from biasbuster.agent.tools import ToolResult

__all__ = [
    "AgentConfig",
    "AgentResult",
    "ToolResult",
    "format_tool_results_for_model",
    "run_agent",
]
