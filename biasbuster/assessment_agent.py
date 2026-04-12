"""v4 Assessment Agent — tool-calling LLM loop for bias assessment.

Takes a v3 extraction, hands it to an LLM agent with access to the
``run_mechanical_assessment`` Python aggregator plus verification
tools, and produces a final assessment with explicit overrides for
any mechanical rule the LLM contextually overrode.

See ``docs/two_step_approach/V4_AGENT_DESIGN.md`` for the design,
``biasbuster.prompts_v4`` for the system prompt, and
``biasbuster.assessment`` for the Python rules. Verification
tool wrappers are reused from ``biasbuster.agent.tools``.

Phase 2 scope: Anthropic Claude only. Phase 3 adds local models
via bmlib's feature/tool-calling branch.

Companion modules (split to keep this file under the 500-line
guideline): ``assessment_agent_tools`` for TOOL_DEFINITIONS,
``assessment_agent_enforcement`` for the post-hoc enforcement pass.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

from biasbuster.annotators import parse_llm_json
from biasbuster.assessment import assess_extraction
from biasbuster.assessment_agent_enforcement import enforce_hard_rules
from biasbuster.assessment_agent_tools import TOOL_DEFINITIONS
from biasbuster.prompts_v4 import (
    ASSESSMENT_AGENT_SYSTEM_PROMPT,
    build_agent_user_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AgentLoopResult:
    """Captures the full state of one assessment agent run.

    Attributes:
        assessment: The final parsed assessment dict (v3 schema shape
            plus `_overrides` and `_mechanical_provenance` keys).
            None if the agent did not produce valid JSON.
        iterations: Number of LLM round-trips used.
        tool_calls_made: List of (tool_name, args) tuples for every
            tool the agent invoked. Useful for cost analysis.
        hit_iteration_cap: True if the agent loop reached max_iterations
            without a clean terminal answer (and a forced final-answer
            call was issued).
        mechanical_assessment: The draft assessment returned by the very
            first run_mechanical_assessment call. Preserved separately
            so the enforcement layer can diff against the final.
        enforcement_notes: Any audit notes produced by
            `_enforce_hard_rules` when the LLM attempted to downgrade
            a non-overridable severity.
    """
    assessment: Optional[dict[str, Any]] = None
    iterations: int = 0
    tool_calls_made: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    hit_iteration_cap: bool = False
    mechanical_assessment: Optional[dict[str, Any]] = None
    enforcement_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AssessmentAgent
# ---------------------------------------------------------------------------

class AssessmentAgent:
    """LLM agent that performs v4 bias assessment via tool calls.

    Use via ``async with AssessmentAgent(...) as agent: await agent.assess(...)``.

    Phase 2 implementation: Anthropic Claude only. The
    ``_call_llm_with_tools_anthropic`` method talks directly to
    ``anthropic.AsyncAnthropic`` because the existing ``LLMAnnotator``
    already uses it — no need to route through bmlib for the
    reference backend. Phase 3 will add a bmlib-based code path
    for local Ollama models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_iterations: int = 5,
        max_tokens_per_turn: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.temperature = temperature
        self._client: Optional[anthropic.AsyncAnthropic] = None

    async def __aenter__(self) -> "AssessmentAgent":
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def assess(
        self,
        pmid: str,
        title: str,
        extraction: dict[str, Any],
        paper_metadata: Optional[dict[str, Any]] = None,
    ) -> AgentLoopResult:
        """Run the assessment agent loop on a single paper.

        Args:
            pmid: Paper identifier.
            title: Paper title.
            extraction: The v3-format extraction dict (Stage 1 output)
                that the agent will reason over.
            paper_metadata: Optional additional context (authors, DOI,
                retraction status, etc.) — used by some verification
                tool wrappers. Not passed to the LLM directly.

        Returns:
            AgentLoopResult with the final assessment and full provenance.
        """
        if self._client is None:
            raise RuntimeError("AssessmentAgent must be used as an async context manager")

        result = AgentLoopResult()
        extraction_json = json.dumps(extraction, indent=2, default=str)
        user_message = build_agent_user_message(pmid, title, extraction_json)

        # Running conversation. Anthropic's `messages` list is
        # everything except the system prompt. Content can be text or
        # content-block lists depending on whether the turn carried
        # tool_use / tool_result.
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]

        for iteration in range(1, self.max_iterations + 1):
            result.iterations = iteration
            logger.info(f"[agent {pmid}] iteration {iteration}/{self.max_iterations}")

            # Force a final-answer turn on the last iteration so we
            # don't spin indefinitely if the model keeps calling tools.
            force_final = iteration == self.max_iterations
            response = await self._anthropic_turn(
                conversation, force_final=force_final
            )

            # Collect text content and tool use blocks from this turn.
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_uses.append({
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": dict(getattr(block, "input", {}) or {}),
                    })

            if not tool_uses:
                # Terminal turn — extract final JSON and exit the loop
                final_text = "".join(text_parts)
                logger.info(
                    f"[agent {pmid}] terminal turn in {iteration} iterations, "
                    f"{len(result.tool_calls_made)} tool calls made"
                )
                parsed = parse_llm_json(final_text, pmid=pmid)
                if parsed is None:
                    logger.error(
                        f"[agent {pmid}] final turn did not produce valid "
                        f"JSON; raw length={len(final_text)}"
                    )
                    return result
                result.assessment = parsed
                # Enforce non-overridable rules before returning
                self._enforce_hard_rules(result)
                return result

            # Append the assistant turn to the conversation as-is so
            # the model can correlate the upcoming tool results.
            conversation.append({
                "role": "assistant",
                "content": [
                    block.model_dump() if hasattr(block, "model_dump")
                    else _block_to_dict(block)
                    for block in response.content
                ],
            })

            # Dispatch every tool call in this turn and collect results.
            tool_result_blocks: list[dict[str, Any]] = []
            for tu in tool_uses:
                tool_name = tu["name"]
                tool_args = tu["input"]
                result.tool_calls_made.append((tool_name, dict(tool_args)))

                logger.info(
                    f"[agent {pmid}] tool call: {tool_name}({tool_args})"
                )
                tool_output = await self._dispatch_tool(
                    tool_name, tool_args, pmid, title, extraction, paper_metadata, result,
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(tool_output, default=str),
                })

            # Feed all tool results back in a single user turn —
            # Anthropic's preferred shape for parallel tool calls
            conversation.append({
                "role": "user",
                "content": tool_result_blocks,
            })

            if force_final:
                # We issued the loop with tool_choice={"type": "none"}
                # but the model still emitted a tool call (should be
                # impossible, but guard anyway). Treat as failure.
                result.hit_iteration_cap = True
                logger.error(
                    f"[agent {pmid}] hit iteration cap while still calling "
                    f"tools — assessment did not terminate cleanly"
                )
                return result

        result.hit_iteration_cap = True
        return result

    # ------------------------------------------------------------------
    # Anthropic tool-calling transport
    # ------------------------------------------------------------------

    async def _anthropic_turn(
        self,
        conversation: list[dict[str, Any]],
        force_final: bool,
    ) -> Any:
        """Send one agent turn to Anthropic and return the raw response.

        Args:
            conversation: The running message list (Anthropic format,
                no system message — that goes in the top-level `system`
                parameter).
            force_final: If True, set tool_choice={"type": "none"} so
                the model cannot call any more tools and must produce
                a final answer.

        Returns:
            The raw anthropic.types.Message response object.
        """
        assert self._client is not None

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens_per_turn,
            "temperature": self.temperature,
            "system": ASSESSMENT_AGENT_SYSTEM_PROMPT,
            "messages": conversation,
            "tools": TOOL_DEFINITIONS,
        }
        if force_final:
            request_kwargs["tool_choice"] = {"type": "none"}

        return await self._client.messages.create(**request_kwargs)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        pmid: str,
        title: str,
        extraction: dict[str, Any],
        paper_metadata: Optional[dict[str, Any]],
        loop_result: AgentLoopResult,
    ) -> dict[str, Any]:
        """Route a tool call to its handler and return a JSON-serialisable result.

        Every handler returns a plain dict that gets json.dumps'd and sent
        back to the model as the content of a tool_result block.
        """
        try:
            if tool_name == "run_mechanical_assessment":
                return self._handle_mechanical_assessment(extraction, loop_result)

            # Verification tools — lazy import to avoid paying the
            # agent/tools.py import cost on every test run
            if tool_name in (
                "check_clinicaltrials",
                "check_open_payments",
                "check_orcid",
                "check_europmc_funding",
                "check_retraction_status",
                "run_effect_size_audit",
            ):
                return await self._handle_verification_tool(
                    tool_name, tool_args, pmid, title, extraction, paper_metadata,
                )

            return {
                "error": f"Unknown tool: {tool_name}",
                "success": False,
            }
        except Exception as exc:
            logger.exception(
                f"[agent {pmid}] tool {tool_name} raised {exc!r}"
            )
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "success": False,
            }

    def _handle_mechanical_assessment(
        self,
        extraction: dict[str, Any],
        loop_result: AgentLoopResult,
    ) -> dict[str, Any]:
        """Run the Python aggregator and cache the result for enforcement."""
        assessment = assess_extraction(extraction)
        # Deep-copy so enforcement sees the untouched draft even if
        # the LLM's conversation view somehow mutates the dict.
        if loop_result.mechanical_assessment is None:
            loop_result.mechanical_assessment = copy.deepcopy(assessment)
        return assessment

    async def _handle_verification_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        pmid: str,
        title: str,
        extraction: dict[str, Any],
        paper_metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Dispatch to one of the existing biasbuster/agent/tools.py wrappers.

        These wrappers need an AgentConfig for network credentials and
        rate limits. We synthesise a minimal AgentConfig from the paper
        metadata when the caller didn't provide one.
        """
        from biasbuster.agent.agent_config import AgentConfig
        from biasbuster.agent.tools import (
            check_clinicaltrials,
            check_europmc_funding,
            check_open_payments,
            check_orcid,
            check_retraction_status,
            run_effect_size_audit,
        )

        # Build a minimal AgentConfig — the verification tools only
        # use ncbi_api_key, crossref_mailto, and timeouts. API keys
        # come from the Config() defaults if not passed explicitly.
        try:
            from config import Config
            cfg = AgentConfig(
                ncbi_api_key=Config().ncbi_api_key or "",
                crossref_mailto=Config().crossref_mailto or "",
            )
        except Exception:
            cfg = AgentConfig()

        # Pull paper metadata fields that the tools need
        abstract = (paper_metadata or {}).get("abstract") or ""
        doi = (paper_metadata or {}).get("doi") or ""
        authors_meta = (paper_metadata or {}).get("authors") or []
        # Some tools want a list[str] of names, others want the full list of dicts
        author_names = _author_names(authors_meta)

        if tool_name == "check_clinicaltrials":
            nct_id = tool_args.get("nct_id", "")
            search_title = tool_args.get("title", "") or title
            return _tool_result_to_dict(
                await check_clinicaltrials(
                    nct_id=nct_id, abstract=abstract, title=search_title,
                    pmid=pmid, config=cfg,
                )
            )

        if tool_name == "check_open_payments":
            wanted = tool_args.get("authors") or author_names[:3]
            return _tool_result_to_dict(
                await check_open_payments(authors=wanted, config=cfg),
            )

        if tool_name == "check_orcid":
            wanted = tool_args.get("authors") or author_names[:3]
            return _tool_result_to_dict(
                await check_orcid(authors=wanted, config=cfg),
            )

        if tool_name == "check_europmc_funding":
            return _tool_result_to_dict(
                await check_europmc_funding(pmid=pmid, doi=doi, config=cfg),
            )

        if tool_name == "check_retraction_status":
            return _tool_result_to_dict(
                await check_retraction_status(pmid=pmid, doi=doi, config=cfg),
            )

        if tool_name == "run_effect_size_audit":
            return _tool_result_to_dict(
                await run_effect_size_audit(pmid=pmid, title=title, abstract=abstract),
            )

        return {"error": f"Unhandled verification tool: {tool_name}"}

    # ------------------------------------------------------------------
    # Post-hoc hard-rule enforcement
    # ------------------------------------------------------------------

    def _enforce_hard_rules(self, result: AgentLoopResult) -> None:
        """Delegate to the enforcement module; collect notes on the result."""
        if result.assessment is None:
            return
        _assessment, notes = enforce_hard_rules(
            result.assessment, result.mechanical_assessment,
        )
        result.enforcement_notes.extend(notes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert a non-pydantic Anthropic content block to a plain dict.

    Handles the case where the anthropic SDK returns plain objects
    rather than pydantic models (older SDK versions, or MagicMocks
    in test code).
    """
    out: dict[str, Any] = {}
    for attr in ("type", "id", "name", "input", "text", "tool_use_id", "content"):
        if hasattr(block, attr):
            val = getattr(block, attr)
            if val is not None:
                out[attr] = val
    return out


def _author_names(authors: Any) -> list[str]:
    """Pull plain name strings from a heterogeneous authors list.

    Handles both list[str] and list[dict{"name": ...}] formats that
    various collectors produce.
    """
    if not isinstance(authors, list):
        return []
    out: list[str] = []
    for a in authors:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            name = a.get("name") or a.get("full_name") or ""
            if name:
                out.append(str(name))
    return out


def _tool_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a biasbuster.agent.tools.ToolResult into a plain dict.

    The ToolResult dataclass is what the existing verification tool
    wrappers return. We flatten it into a shape the LLM can easily
    read — name, success, summary, detail, and raw_data.
    """
    return {
        "tool_name": getattr(result, "tool_name", "unknown"),
        "success": bool(getattr(result, "success", False)),
        "summary": getattr(result, "summary", "") or "",
        "detail": getattr(result, "detail", "") or "",
        "raw_data": getattr(result, "raw_data", {}) or {},
        "error": getattr(result, "error", None),
    }
