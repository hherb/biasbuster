"""Verification pipeline wrapper for the BiasBuster CLI.

Thin wrapper around the existing agent/ verification infrastructure.
Takes an initial LLM assessment, runs programmatic verification tools
(ClinicalTrials.gov, CMS Open Payments, ORCID, etc.), and produces
a refined assessment incorporating verification findings.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from bmlib.llm import LLMMessage

from biasbuster.agent.agent_config import AgentConfig
from biasbuster.agent.runner import format_tool_results_for_model
from biasbuster.agent.tool_router import AbstractContext, ToolCall, route_verification_steps
from biasbuster.agent.tools import ToolResult, execute_tool_call
from biasbuster.agent.verification_planner import plan_verification
from biasbuster.annotators import parse_llm_json
from biasbuster.cli.analysis import create_llm_client
from biasbuster.cli.settings import CLIConfig

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TOOLS = 3

# Refinement system prompt — same as agent/model_client.py
_REFINEMENT_SYSTEM_PROMPT = """\
You are a biomedical research integrity analyst. You previously assessed a
clinical trial abstract and recommended specific verification steps.

The verification steps have now been executed automatically, and the results
are provided below. Incorporate these findings into a REFINED bias assessment.

Where verification results confirm or contradict your initial findings,
update your severity ratings and confidence accordingly. If a verification
tool failed or returned no data, note this limitation but do not change your
rating solely because of missing verification data.

Output the REFINED assessment as a JSON object using the same schema as the
initial assessment. Respond ONLY with the JSON object."""


@dataclass
class VerificationResult:
    """Results from the verification pipeline."""

    verification_steps: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    refined_assessment: dict[str, Any] | None = None
    error: str | None = None


def run_verification(
    initial_assessment: dict[str, Any],
    content_meta: dict[str, Any],
    config: CLIConfig,
) -> VerificationResult:
    """Run the verification pipeline on an initial assessment.

    Args:
        initial_assessment: Parsed JSON from the initial LLM assessment.
        content_meta: Dict with pmid, doi, title, abstract, authors keys.
        config: CLI configuration.

    Returns:
        VerificationResult with tool results and refined assessment.
    """
    return asyncio.run(_run_verification_async(
        initial_assessment, content_meta, config
    ))


async def _run_verification_async(
    initial_assessment: dict[str, Any],
    content_meta: dict[str, Any],
    config: CLIConfig,
) -> VerificationResult:
    """Async implementation of the verification pipeline."""
    result = VerificationResult()

    # Step 1: Plan verification steps from assessment flags
    import json
    assessment_json = json.dumps(initial_assessment)
    steps, _parsed = plan_verification(assessment_json, parse_llm_json)

    if not steps:
        logger.info("No verification steps generated")
        return result

    result.verification_steps = steps
    logger.info("Planned %d verification steps", len(steps))

    # Step 2: Route steps to tool calls
    context = AbstractContext(
        pmid=content_meta.get("pmid", ""),
        doi=content_meta.get("doi", ""),
        title=content_meta.get("title", ""),
        abstract=content_meta.get("abstract", ""),
        authors=content_meta.get("authors", []),
        nct_id=content_meta.get("nct_id", ""),
    )
    tool_calls = route_verification_steps(steps, context)

    # Filter out unsupported tools
    tool_calls = [tc for tc in tool_calls if tc.tool_name != "unsupported"]
    if not tool_calls:
        logger.info("No supported tool calls generated")
        return result

    logger.info("Executing %d verification tools", len(tool_calls))

    # Step 3: Execute tools concurrently
    agent_config = _build_agent_config(config)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

    async def _run_tool(call: ToolCall) -> ToolResult:
        async with semaphore:
            try:
                return await execute_tool_call(call, agent_config)
            except Exception as exc:
                logger.warning("Tool %s failed: %s", call.tool_name, exc)
                return ToolResult(
                    tool_name=call.tool_name,
                    success=False,
                    error=str(exc),
                )

    tool_results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

    result.tool_results = [
        {
            "tool": tr.tool_name,
            "success": tr.success,
            "summary": tr.summary,
            "detail": tr.detail,
            "error": tr.error,
        }
        for tr in tool_results
    ]

    # Step 4: Refine assessment with tool results
    verification_text = format_tool_results_for_model(list(tool_results))

    llm = create_llm_client(config)

    user_msg = (
        f"## Initial Assessment\n\n"
        f"```json\n{json.dumps(initial_assessment, indent=2)}\n```\n\n"
        f"## Verification Results\n\n"
        f"{verification_text}"
    )

    messages = [
        LLMMessage(role="system", content=_REFINEMENT_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]

    logger.info("Sending verification results to %s for refined assessment", config.model)
    response = llm.chat(
        messages,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        json_mode=True,
    )

    refined = parse_llm_json(response.content, pmid=content_meta.get("pmid", ""))
    if refined is not None:
        result.refined_assessment = refined
    else:
        logger.warning("Failed to parse refined assessment")
        result.error = "Failed to parse refined assessment from LLM"

    return result


def _build_agent_config(config: CLIConfig) -> AgentConfig:
    """Build an AgentConfig from CLI config for tool execution."""
    return AgentConfig(
        ollama_endpoint=config.ollama_endpoint,
        model_id=config.model_name,
        ncbi_api_key=config.ncbi_api_key,
        crossref_mailto=config.crossref_mailto or config.email,
    )
