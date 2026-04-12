"""v4 Assessment Agent — tool-calling LLM loop for bias assessment.

Takes a v3 extraction, hands it to an LLM agent with access to the
``run_mechanical_assessment`` Python aggregator plus verification
tools, and produces a final assessment with explicit overrides for
any mechanical rule the LLM contextually overrode.

See ``docs/two_step_approach/V4_AGENT_DESIGN.md`` for the design,
``biasbuster.prompts_v4`` for the system prompt, and
``biasbuster.assessment`` for the Python rules. Verification
tool wrappers are reused from ``biasbuster.agent.tools``.

Supports two provider backends:
  - ``provider="anthropic"`` (Phase 2): direct ``anthropic.AsyncAnthropic``
    SDK. Requires an Anthropic API key.
  - ``provider="bmlib"`` (Phase 3): bmlib's ``LLMClient.chat(tools=...)``
    which routes to any bmlib-supported provider (Ollama for local models,
    Anthropic, DeepSeek, etc.). Requires a bmlib model string.

Companion modules: ``assessment_agent_tools`` for TOOL_DEFINITIONS,
``assessment_agent_enforcement`` for the post-hoc enforcement pass.
"""
from __future__ import annotations

import asyncio
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
# Provider-agnostic turn result
# ---------------------------------------------------------------------------

@dataclass
class _TurnResult:
    """Normalised result from one agent turn, regardless of provider.

    Both the Anthropic and bmlib code paths return this shape so the
    main ``assess()`` loop is provider-agnostic.
    """
    text: str = ""
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    # Each tool_use dict has: {"id": str, "name": str, "input": dict}
    # This is the union shape that both Anthropic and bmlib produce.

    # Raw content blocks from the provider (Anthropic only — used to
    # reconstruct the assistant turn when re-sending the conversation).
    # For bmlib this stays empty; the LLMMessage reconstruction happens
    # inside the bmlib path instead.
    raw_blocks: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AssessmentAgent
# ---------------------------------------------------------------------------

class AssessmentAgent:
    """LLM agent that performs v4 bias assessment via tool calls.

    Use via ``async with AssessmentAgent(...) as agent: await agent.assess(...)``.

    Two provider backends:

    - ``provider="anthropic"`` (default): direct ``anthropic.AsyncAnthropic``
      SDK. Pass ``api_key=``.
    - ``provider="bmlib"``: bmlib's ``LLMClient.chat(tools=...)``. Pass
      ``model=`` in bmlib format (e.g. ``"ollama:gemma4:26b-a4b-it-q8_0"``).
      The LLMClient is synchronous; calls are wrapped in
      ``asyncio.to_thread`` so the agent loop remains async.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-5-20250929",
        max_iterations: int = 5,
        max_tokens_per_turn: int = 4096,
        temperature: float = 0.0,
        provider: str = "anthropic",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.temperature = temperature
        self.provider = provider
        self._anthropic_client: Optional[anthropic.AsyncAnthropic] = None
        self._bmlib_client: Any = None  # bmlib.llm.LLMClient (lazy import)

    async def __aenter__(self) -> "AssessmentAgent":
        if self.provider == "anthropic":
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "bmlib":
            from bmlib.llm import LLMClient
            self._bmlib_client = LLMClient(
                anthropic_api_key=self.api_key or None,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")
        return self

    async def __aexit__(self, *args) -> None:
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None
        self._bmlib_client = None

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

        The loop is provider-agnostic: each iteration calls
        ``_do_turn()`` which dispatches to the Anthropic or bmlib
        backend and returns a normalised ``_TurnResult``. The loop
        logic itself — terminal detection, tool dispatch, conversation
        management — is shared.

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
        if self.provider == "anthropic" and self._anthropic_client is None:
            raise RuntimeError("AssessmentAgent must be used as an async context manager")
        if self.provider == "bmlib" and self._bmlib_client is None:
            raise RuntimeError("AssessmentAgent must be used as an async context manager")

        result = AgentLoopResult()
        extraction_json = json.dumps(extraction, indent=2, default=str)
        user_message = build_agent_user_message(pmid, title, extraction_json)

        # Running conversation — provider-specific format.
        # Anthropic: list of dicts with role + content (blocks or text).
        # bmlib: list of bmlib.llm.data_types.LLMMessage objects.
        conversation: list[Any] = self._init_conversation(user_message)

        for iteration in range(1, self.max_iterations + 1):
            result.iterations = iteration
            logger.info(f"[agent {pmid}] iteration {iteration}/{self.max_iterations}")

            force_final = iteration == self.max_iterations
            turn = await self._do_turn(conversation, force_final=force_final)

            if not turn.tool_uses:
                # Terminal turn — extract final JSON and exit the loop
                logger.info(
                    f"[agent {pmid}] terminal turn in {iteration} iterations, "
                    f"{len(result.tool_calls_made)} tool calls made"
                )
                parsed = parse_llm_json(turn.text, pmid=pmid)
                if parsed is None:
                    logger.error(
                        f"[agent {pmid}] final turn did not produce valid "
                        f"JSON; raw length={len(turn.text)}"
                    )
                    return result
                result.assessment = parsed
                self._enforce_hard_rules(result)
                return result

            # Append the assistant turn and dispatch tool calls.
            self._append_assistant_turn(conversation, turn)

            tool_results: list[tuple[str, str]] = []  # [(tool_use_id, json_output)]
            for tu in turn.tool_uses:
                tool_name = tu["name"]
                tool_args = tu["input"]
                result.tool_calls_made.append((tool_name, dict(tool_args)))
                logger.info(f"[agent {pmid}] tool call: {tool_name}({tool_args})")
                tool_output = await self._dispatch_tool(
                    tool_name, tool_args, pmid, title, extraction, paper_metadata, result,
                )
                tool_results.append((tu["id"], json.dumps(tool_output, default=str)))

            self._append_tool_results(conversation, tool_results)

            if force_final:
                result.hit_iteration_cap = True
                logger.error(
                    f"[agent {pmid}] hit iteration cap while still calling "
                    f"tools — assessment did not terminate cleanly"
                )
                return result

        result.hit_iteration_cap = True
        return result

    # ------------------------------------------------------------------
    # Provider-agnostic conversation helpers
    # ------------------------------------------------------------------

    def _init_conversation(self, user_message: str) -> list[Any]:
        """Create the initial conversation with the first user message."""
        if self.provider == "bmlib":
            from bmlib.llm import LLMMessage
            return [LLMMessage(role="user", content=user_message)]
        # Anthropic format
        return [{"role": "user", "content": user_message}]

    def _append_assistant_turn(self, conversation: list[Any], turn: _TurnResult) -> None:
        """Append the assistant's tool-calling turn to the conversation."""
        if self.provider == "bmlib":
            from bmlib.llm import LLMMessage, LLMToolCall
            tool_calls = [
                LLMToolCall(id=tu["id"], name=tu["name"], arguments=tu["input"])
                for tu in turn.tool_uses
            ]
            conversation.append(
                LLMMessage(role="assistant", content=turn.text, tool_calls=tool_calls)
            )
        else:
            # Anthropic — re-send the raw content blocks so the model
            # can correlate tool results
            conversation.append({
                "role": "assistant",
                "content": [
                    block.model_dump() if hasattr(block, "model_dump")
                    else _block_to_dict(block)
                    for block in turn.raw_blocks
                ],
            })

    def _append_tool_results(
        self, conversation: list[Any], results: list[tuple[str, str]],
    ) -> None:
        """Append tool results to the conversation."""
        if self.provider == "bmlib":
            from bmlib.llm import LLMMessage
            for tool_use_id, output_json in results:
                conversation.append(
                    LLMMessage(role="tool", content=output_json, tool_call_id=tool_use_id)
                )
        else:
            # Anthropic — all tool results in one user message
            conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output_json,
                    }
                    for tool_use_id, output_json in results
                ],
            })

    async def _do_turn(
        self, conversation: list[Any], force_final: bool,
    ) -> _TurnResult:
        """Send one agent turn and return a normalised result."""
        if self.provider == "bmlib":
            return await self._bmlib_turn(conversation, force_final)
        return await self._anthropic_turn(conversation, force_final)

    # ------------------------------------------------------------------
    # Anthropic tool-calling transport
    # ------------------------------------------------------------------

    async def _anthropic_turn(
        self,
        conversation: list[Any],
        force_final: bool,
    ) -> _TurnResult:
        """Send one agent turn to Anthropic and return a normalised result."""
        assert self._anthropic_client is not None

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

        response = await self._anthropic_client.messages.create(**request_kwargs)

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

        return _TurnResult(
            text="".join(text_parts),
            tool_uses=tool_uses,
            raw_blocks=list(response.content),
        )

    async def _bmlib_turn(
        self,
        conversation: list[Any],
        force_final: bool,
    ) -> _TurnResult:
        """Send one agent turn via bmlib LLMClient and return a normalised result.

        bmlib's ``LLMClient.chat()`` is synchronous. We wrap it in
        ``asyncio.to_thread`` so the agent loop stays async (consistent
        with how ``BmlibAnnotator`` works).
        """
        from bmlib.llm import LLMMessage, LLMToolDefinition

        assert self._bmlib_client is not None

        # Convert our tool definitions to bmlib LLMToolDefinition objects
        bmlib_tools = [
            LLMToolDefinition(
                name=td["name"],
                description=td["description"],
                parameters=td.get("input_schema", {}),
            )
            for td in TOOL_DEFINITIONS
        ]

        # Prepend system message to the conversation for bmlib
        # (bmlib forwards it to the provider which handles it natively)
        messages = [
            LLMMessage(role="system", content=ASSESSMENT_AGENT_SYSTEM_PROMPT),
            *conversation,
        ]

        tool_choice = "none" if force_final else "auto"

        def _sync_call():
            return self._bmlib_client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_turn,
                tools=bmlib_tools,
                tool_choice=tool_choice,
            )

        response = await asyncio.to_thread(_sync_call)

        # Normalise bmlib's LLMResponse into _TurnResult
        tool_uses: list[dict[str, Any]] = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_uses.append({
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })

        return _TurnResult(
            text=response.content or "",
            tool_uses=tool_uses,
        )

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
