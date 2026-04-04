"""LLM bias assessment for the BiasBuster CLI.

Supports two analysis modes:
- Single-pass: abstract or short text → one LLM call → structured assessment
- Map-reduce: full-text chunks → per-section analysis → synthesis → assessment
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bmlib.llm import LLMClient, LLMMessage

from annotators import build_user_message, parse_llm_json
from cli.chunking import TextChunk, chunk_jats_article, chunk_plain_text
from cli.settings import CLIConfig
from cli.content import AcquiredContent
from prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    SEVERITY_SCALE,
    DOMAIN_CRITERIA,
    CALIBRATION_NOTE,
)

logger = logging.getLogger(__name__)

# Section-level analysis prompt (map phase)
_SECTION_ANALYSIS_PROMPT = f"""\
You are a biomedical research integrity analyst. You are reviewing ONE SECTION
of a clinical trial paper. Identify bias signals relevant to this section.

{SEVERITY_SCALE}

{DOMAIN_CRITERIA}

{CALIBRATION_NOTE}

For this section, report ONLY the bias signals you observe. Output a JSON object:
{{
  "section": "<section name>",
  "signals": [
    {{
      "domain": "statistical_reporting|spin|outcome_reporting|conflict_of_interest|methodology",
      "observation": "What you observed",
      "severity_estimate": "none|low|moderate|high|critical",
      "evidence_quote": "Relevant quote from the text"
    }}
  ]
}}

If no bias signals are found in this section, return {{"section": "<name>", "signals": []}}.
Respond ONLY with the JSON object."""

# Synthesis prompt (reduce phase)
_SYNTHESIS_PROMPT = f"""\
You are a biomedical research integrity analyst. You have reviewed a full-text
clinical trial paper section by section. Below are the per-section bias signals
identified. Synthesize them into a single unified 5-domain assessment.

{SEVERITY_SCALE}

{DOMAIN_CRITERIA}

{CALIBRATION_NOTE}

Now produce a SINGLE assessment JSON covering all 5 domains, using this schema:

{{
  "statistical_reporting": {{
    "relative_only": boolean,
    "absolute_reported": boolean,
    "nnt_reported": boolean,
    "baseline_risk_reported": boolean,
    "selective_p_values": boolean,
    "subgroup_emphasis": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  }},
  "spin": {{
    "spin_level": "none|low|moderate|high",
    "conclusion_matches_results": boolean,
    "causal_language_from_observational": boolean,
    "focus_on_secondary_when_primary_ns": boolean,
    "inappropriate_extrapolation": boolean,
    "title_spin": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  }},
  "outcome_reporting": {{
    "primary_outcome_type": "patient_centred|surrogate|composite|unclear",
    "surrogate_without_validation": boolean,
    "composite_not_disaggregated": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  }},
  "conflict_of_interest": {{
    "funding_type": "industry|public|mixed|not_reported|unclear",
    "funding_disclosed_in_abstract": boolean,
    "industry_author_affiliations": boolean,
    "coi_disclosed": boolean,
    "severity": "none|low|moderate|high|critical"
  }},
  "methodology": {{
    "inappropriate_comparator": boolean,
    "enrichment_design": boolean,
    "per_protocol_only": boolean,
    "premature_stopping": boolean,
    "short_follow_up": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  }},
  "overall_severity": "none|low|moderate|high|critical",
  "overall_bias_probability": float (0.0 to 1.0),
  "reasoning": "Step-by-step reasoning",
  "confidence": "low|medium|high",
  "recommended_verification_steps": ["..."]
}}

Respond ONLY with the JSON object."""


def create_llm_client(config: CLIConfig) -> LLMClient:
    """Create a bmlib LLMClient configured from CLI settings.

    Args:
        config: CLI configuration with provider credentials and endpoints.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(
        default_provider=config.provider,
        ollama_host=config.ollama_endpoint,
        anthropic_api_key=config.anthropic_api_key,
        api_key=config.deepseek_api_key or config.anthropic_api_key,
        base_url=config.deepseek_base if config.provider == "deepseek" else None,
    )


def analyse(content: AcquiredContent, config: CLIConfig) -> dict[str, Any]:
    """Run bias assessment on acquired content.

    Chooses single-pass or map-reduce based on content type.

    Args:
        content: Acquired content (abstract, JATS full text, or plain full text).
        config: CLI configuration.

    Returns:
        Parsed assessment dict with 5-domain bias analysis.
    """
    llm = create_llm_client(config)

    if content.has_fulltext:
        return _analyse_fulltext(llm, content, config)
    else:
        return _analyse_abstract(llm, content, config)


def _analyse_abstract(
    llm: LLMClient,
    content: AcquiredContent,
    config: CLIConfig,
) -> dict[str, Any]:
    """Single-pass analysis of an abstract."""
    metadata = _build_metadata_dict(content)
    user_msg = build_user_message(
        content.pmid, content.title, content.abstract, metadata
    )

    messages = [
        LLMMessage(role="system", content=ANNOTATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]

    logger.info("Sending abstract to %s for single-pass analysis", config.model)
    response = llm.chat(
        messages,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        json_mode=True,
    )

    assessment = parse_llm_json(response.content, pmid=content.pmid)
    if assessment is None:
        raise ValueError(
            f"Failed to parse LLM response as valid bias assessment. "
            f"Raw response: {response.content[:500]}"
        )

    return assessment


def _analyse_fulltext(
    llm: LLMClient,
    content: AcquiredContent,
    config: CLIConfig,
) -> dict[str, Any]:
    """Map-reduce analysis of full-text content.

    Map phase: analyse each section/chunk for bias signals.
    Reduce phase: synthesize all signals into a unified assessment.
    """
    # Chunk the content
    if content.jats_article is not None:
        chunks = chunk_jats_article(content.jats_article)
        logger.info(
            "Split JATS article into %d section chunks", len(chunks)
        )
    else:
        chunks = chunk_plain_text(content.plain_fulltext)
        logger.info(
            "Split plain text into %d token-window chunks", len(chunks)
        )

    if not chunks:
        logger.warning("No chunks produced, falling back to abstract analysis")
        return _analyse_abstract(llm, content, config)

    # Map phase: per-section analysis
    section_results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        logger.info(
            "Analysing chunk %d/%d: %s (%d tokens est.)",
            i + 1, len(chunks), chunk.section, chunk.token_estimate,
        )

        user_msg = (
            f"Paper: {content.title}\n"
            f"Section: {chunk.section}\n\n"
            f"{chunk.text}"
        )

        messages = [
            LLMMessage(role="system", content=_SECTION_ANALYSIS_PROMPT),
            LLMMessage(role="user", content=user_msg),
        ]

        response = llm.chat(
            messages,
            model=config.model,
            temperature=config.temperature,
            max_tokens=2000,
            json_mode=True,
        )

        parsed = parse_llm_json(response.content)
        if parsed is not None:
            section_results.append(parsed)
        else:
            logger.warning("Failed to parse section analysis for chunk %d", i + 1)

    if not section_results:
        logger.warning("No section results, falling back to abstract analysis")
        return _analyse_abstract(llm, content, config)

    # Reduce phase: synthesize into unified assessment
    return _synthesize(llm, content, section_results, config)


def _synthesize(
    llm: LLMClient,
    content: AcquiredContent,
    section_results: list[dict[str, Any]],
    config: CLIConfig,
) -> dict[str, Any]:
    """Synthesize per-section results into a unified 5-domain assessment."""
    # Format section results for the synthesis prompt
    sections_text = json.dumps(section_results, indent=2)

    user_msg = (
        f"Paper: {content.title}\n"
        f"PMID: {content.pmid}\n"
        f"DOI: {content.doi}\n\n"
        f"Per-section bias signals:\n{sections_text}"
    )

    messages = [
        LLMMessage(role="system", content=_SYNTHESIS_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]

    logger.info("Synthesizing %d section results into unified assessment", len(section_results))
    response = llm.chat(
        messages,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        json_mode=True,
    )

    assessment = parse_llm_json(response.content, pmid=content.pmid)
    if assessment is None:
        raise ValueError(
            f"Failed to parse synthesis response. "
            f"Raw response: {response.content[:500]}"
        )

    return assessment


def _build_metadata_dict(content: AcquiredContent) -> dict[str, Any] | None:
    """Build metadata dict from AcquiredContent for build_user_message()."""
    meta: dict[str, Any] = {}
    if content.authors:
        meta["authors"] = content.authors
    if content.journal:
        meta["journal"] = content.journal
    return meta if meta else None
