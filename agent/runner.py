"""
Verification agent runner.

Orchestrates the single-pass agent loop:
1. Call the fine-tuned model for initial bias assessment
2. Synthesize verification steps programmatically from assessment flags
3. Execute verification tools concurrently
4. Call the model again with verification results for a refined assessment
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from agent.agent_config import AgentConfig
from agent.model_client import get_initial_assessment, get_refined_assessment
from agent.tool_router import (
    AbstractContext,
    ToolCall,
    route_verification_steps,
)
from agent.tools import ToolResult, execute_tool_call, get_tool_display_name
from agent.verification_planner import plan_verification

logger = logging.getLogger(__name__)

# Type alias for stage callbacks: (stage_name, optional_data)
StageCallback = Callable[[str, Any], None]


@dataclass
class AgentResult:
    """Complete result from the verification agent."""

    pmid: str = ""
    title: str = ""
    abstract: str = ""
    initial_assessment: str = ""
    verification_steps: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    refined_assessment: str = ""
    total_time_seconds: float = 0.0
    error: Optional[str] = None


def format_tool_results_for_model(tool_results: list[ToolResult]) -> str:
    """Format tool results as structured markdown for the refinement prompt."""
    sections: list[str] = []
    for result in tool_results:
        display = get_tool_display_name(result.tool_name)
        status = "SUCCESS" if result.success else "FAILED"
        header = f"### {display} [{status}]"

        if result.error:
            body = f"Error: {result.error}"
        elif result.detail:
            body = result.detail
        else:
            body = result.summary or "No data returned."

        sections.append(f"{header}\n\n{body}")

    return "\n\n---\n\n".join(sections)


async def _fetch_abstract(
    config: AgentConfig, pmid: str,
) -> Optional[dict]:
    """Fetch an abstract from PubMed by PMID."""
    from collectors.retraction_watch import RetractionWatchCollector

    async with RetractionWatchCollector(
        mailto=config.crossref_mailto or "biasbuster@example.com",
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        return await collector.fetch_pubmed_abstract(pmid)


async def run_agent(
    config: AgentConfig,
    pmid: str = "",
    title: str = "",
    abstract: str = "",
    metadata: Optional[dict] = None,
    on_stage: Optional[StageCallback] = None,
) -> AgentResult:
    """Run the single-pass verification agent.

    Args:
        config: Agent configuration.
        pmid: PubMed ID (if provided without abstract, will be fetched).
        title: Paper title.
        abstract: Abstract text.
        metadata: Optional metadata dict (authors, grants, journal, etc.).
        on_stage: Optional callback invoked at each stage with
            ``(stage_name, stage_data)``. Stages: ``fetching_abstract``,
            ``initial_assessment``, ``planning_verification``,
            ``executing_tools``, ``refining``, ``complete``, ``error``.

    Returns:
        AgentResult with all intermediate and final outputs.
    """
    start = time.monotonic()

    def _notify(stage: str, data: Any = None) -> None:
        if on_stage is not None:
            on_stage(stage, data)

    result = AgentResult(pmid=pmid, title=title, abstract=abstract)

    try:
        # --- Stage 1: Fetch abstract if needed ---
        if not abstract and pmid:
            _notify("fetching_abstract", {"pmid": pmid})
            article = await _fetch_abstract(config, pmid)
            if article is None:
                result.error = f"Could not fetch abstract for PMID {pmid}"
                _notify("error", result.error)
                return result
            result.abstract = article.get("abstract", "")
            result.title = article.get("title", result.title)
            abstract = result.abstract
            title = result.title
            logger.info("Fetched abstract for PMID %s: %s", pmid, title[:80])

        if not abstract:
            result.error = "No abstract provided and no PMID to fetch from."
            _notify("error", result.error)
            return result

        # --- Stage 2: Initial assessment ---
        _notify("initial_assessment", None)
        async with httpx.AsyncClient(
            timeout=config.request_timeout_seconds,
        ) as client:
            initial = await get_initial_assessment(
                client, config,
                title=title, abstract=abstract,
                pmid=pmid, metadata=metadata,
            )
        result.initial_assessment = initial
        _notify("initial_assessment_done", {"length": len(initial)})

        # --- Stage 3: Synthesize verification steps from assessment ---
        _notify("planning_verification", None)
        steps, parsed_annotation = plan_verification(initial)
        result.verification_steps = steps
        logger.info("Synthesized %d verification steps", len(steps))

        # Route steps to tool calls
        authors = []
        if metadata:
            from annotators import _ensure_parsed
            authors = _ensure_parsed(metadata.get("authors", []))

        context = AbstractContext(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
        )
        tool_calls = route_verification_steps(steps, context)
        result.tool_calls = tool_calls
        _notify("planning_verification_done", {
            "steps": len(steps),
            "tool_calls": len(tool_calls),
        })

        # --- Stage 4: Execute tools concurrently ---
        _notify("executing_tools", {
            "count": len(tool_calls),
            "tools": [tc.tool_name for tc in tool_calls],
        })

        semaphore = asyncio.Semaphore(config.max_concurrent_tools)

        async def _run_tool(call: ToolCall) -> ToolResult:
            async with semaphore:
                _notify("tool_started", {"tool": call.tool_name})
                tr = await execute_tool_call(call, config)
                _notify("tool_done", {
                    "tool": call.tool_name,
                    "success": tr.success,
                    "summary": tr.summary,
                })
                return tr

        tool_results = await asyncio.gather(
            *[_run_tool(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        # Convert exceptions to ToolResult errors
        resolved_results: list[ToolResult] = []
        for i, tr in enumerate(tool_results):
            if isinstance(tr, Exception):
                resolved_results.append(ToolResult(
                    tool_name=tool_calls[i].tool_name,
                    success=False,
                    error=str(tr),
                    summary=f"Tool execution failed: {tr}",
                ))
            else:
                resolved_results.append(tr)

        result.tool_results = resolved_results

        # --- Stage 5: Refined assessment ---
        _notify("refining", None)
        verification_text = format_tool_results_for_model(resolved_results)

        async with httpx.AsyncClient(
            timeout=config.request_timeout_seconds,
        ) as client:
            refined = await get_refined_assessment(
                client, config,
                initial_assessment=initial,
                verification_results_text=verification_text,
            )
        result.refined_assessment = refined
        _notify("refining_done", {"length": len(refined)})

    except Exception as exc:
        logger.exception("Agent run failed")
        result.error = str(exc)
        _notify("error", str(exc))

    result.total_time_seconds = time.monotonic() - start
    _notify("complete", result)
    return result
