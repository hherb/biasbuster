"""
LLM Pre-Labelling Annotator

Uses Claude API to generate structured bias assessments for abstracts.
These are PRE-LABELS that must be human-validated before use in training.

Strategy:
1. Send abstract + metadata to Claude with detailed bias assessment prompt
2. Parse structured response into BiasAssessment objects
3. Flag low-confidence assessments for priority human review
4. Export as JSONL for review in a simple web UI or spreadsheet
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic

from . import (
    build_user_message,
    generate_review_csv,
    parse_llm_json,
    save_annotations,
    strip_markdown_fences,
)

logger = logging.getLogger(__name__)

# The system prompt encodes the full bias taxonomy and verification knowledge
ANNOTATION_SYSTEM_PROMPT = """You are an expert biomedical research integrity analyst helping to build
a training dataset for an AI bias detection system. Your task is to assess clinical trial
abstracts for potential bias across multiple dimensions.

For each abstract, provide a structured assessment in JSON format with the following fields:

{
  "statistical_reporting": {
    "relative_only": boolean,       // Only relative measures (RRR/OR/HR) without absolute (ARR/NNT)
    "absolute_reported": boolean,    // ARR, absolute risk, or NNT present
    "nnt_reported": boolean,
    "baseline_risk_reported": boolean,
    "selective_p_values": boolean,   // Only favourable p-values highlighted
    "subgroup_emphasis": boolean,    // Subgroup results emphasised over primary
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "spin": {
    "spin_level": "none|low|moderate|high",  // Boutron classification
    "conclusion_matches_results": boolean,
    "causal_language_from_observational": boolean,
    "focus_on_secondary_when_primary_ns": boolean,
    "inappropriate_extrapolation": boolean,
    "title_spin": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "outcome_reporting": {
    "primary_outcome_type": "patient_centred|surrogate|composite|unclear",
    "surrogate_without_validation": boolean,
    "composite_not_disaggregated": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "conflict_of_interest": {
    "funding_type": "industry|public|mixed|not_reported|unclear",
    "funding_disclosed_in_abstract": boolean,
    "industry_author_affiliations": boolean,
    "coi_disclosed": boolean,
    "severity": "none|low|moderate|high|critical"
  },
  "methodology": {
    "inappropriate_comparator": boolean,
    "enrichment_design": boolean,
    "per_protocol_only": boolean,
    "premature_stopping": boolean,
    "short_follow_up": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "overall_severity": "none|low|moderate|high|critical",
  "overall_bias_probability": float,  // 0.0 to 1.0
  "reasoning": "Step-by-step reasoning explaining the assessment",
  "recommended_verification_steps": [
    "Specific actionable verification steps, citing databases and URLs"
  ],
  "confidence": "low|medium|high"  // Your confidence in this assessment
}

KEY PRINCIPLES:
1. RELATIVE vs ABSOLUTE: The sole use of relative measures (RRR, OR, HR) without
   absolute measures (ARR, NNT, baseline risk) is a STRONG indicator of potential bias.
   A "50% reduction" means very different things at 2% vs 20% baseline risk.

2. VERIFICATION SOURCES: Always suggest specific verification steps:
   - CMS Open Payments (openpaymentsdata.cms.gov) for US physician payments
   - ClinicalTrials.gov for registered outcomes, sponsors, protocol amendments
   - ORCID for author affiliation histories
   - Europe PMC for funder metadata
   - Medicines Australia for AU physician payments
   - EFPIA databases for European physician payments
   - Retraction Watch / Crossref for post-publication notices

3. SPIN CLASSIFICATION (Boutron):
   - HIGH: No uncertainty in conclusions, no recommendation for further trials,
     no acknowledgment of non-significant primary outcomes, or recommends clinical use
   - MODERATE: Some uncertainty OR recommendation for further trials,
     but no acknowledgment of non-significant primary
   - LOW: Uncertainty AND further trials recommended, OR acknowledges NS primary
   - NONE: Conclusions accurately reflect results

4. Be calibrated. Not every industry-funded study is biased. Not every study
   reporting only relative measures is intentionally misleading. Assess the totality.

Respond ONLY with the JSON object. No preamble, no markdown fences."""


class LLMAnnotator:
    """Pre-label abstracts using Claude API via the official Anthropic SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> None:
        """Initialise the annotator.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Claude model identifier.
            max_tokens: Maximum tokens per annotation response.
            max_retries: Number of retries for transient API failures.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client: Optional[anthropic.AsyncAnthropic] = None

    async def __aenter__(self) -> "LLMAnnotator":
        """Create the Anthropic async client with built-in retry."""
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Close the Anthropic client."""
        if self.client:
            await self.client.close()

    async def annotate_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Send a single abstract to Claude for bias assessment.
        Returns parsed JSON assessment or None on failure.
        """
        if not self.api_key:
            logger.error("No Anthropic API key configured")
            return None

        user_message = build_user_message(pmid, title, abstract, metadata)

        last_error = None
        for attempt in range(self.max_retries):
            text = ""
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=ANNOTATION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                text = "".join(
                    block.text for block in message.content
                    if block.type == "text"
                )

                if not text.strip():
                    logger.warning(
                        f"PMID {pmid}: empty text response "
                        f"(stop_reason={message.stop_reason}, "
                        f"content_types={[b.type for b in message.content]})"
                    )

                assessment = parse_llm_json(text, pmid=pmid)
                if assessment is not None:
                    assessment["pmid"] = pmid
                    assessment["title"] = title
                    assessment["_annotation_model"] = self.model
                    return assessment

                # parse_llm_json returned None — retry with the API
                last_error = "JSON parse failure after repair attempt"
                logger.warning(
                    f"PMID {pmid}: JSON parse failed "
                    f"(attempt {attempt + 1}/{self.max_retries}), retrying"
                )
                await asyncio.sleep(2 ** attempt)
                continue

            except anthropic.APIError as e:
                last_error = str(e)
                logger.warning(
                    f"API error for PMID {pmid} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error(f"Annotation failed for PMID {pmid}: {e}")
                return None

        logger.error(
            f"All {self.max_retries} attempts failed for PMID {pmid}: {last_error}"
        )
        return None

    async def annotate_batch(
        self,
        items: list[dict],  # Each: {pmid, title, abstract, metadata?}
        concurrency: int = 3,
        delay: float = 1.0,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """Annotate a batch of abstracts with rate limiting.

        Supports incremental save: if output_path is provided, results are
        flushed to disk periodically and already-annotated PMIDs are skipped
        on resume.

        Args:
            items: List of dicts with pmid, title, abstract, metadata keys.
            concurrency: Max concurrent API requests.
            delay: Seconds between requests.
            output_path: Optional path for incremental JSONL saves.

        Returns:
            List of successful annotations.
        """
        # Resume support: skip already-annotated PMIDs
        results: list[dict] = []
        already_done: set[str] = set()
        if output_path and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        ann = json.loads(line)
                        results.append(ann)
                        already_done.add(ann.get("pmid", ""))
                    except json.JSONDecodeError:
                        continue
            if already_done:
                logger.info(
                    f"Resuming: {len(already_done)} already annotated in {output_path}"
                )

        remaining = [it for it in items if it["pmid"] not in already_done]
        if not remaining:
            logger.info("All items already annotated, nothing to do")
            return results

        semaphore = asyncio.Semaphore(concurrency)
        flush_every = 10

        async def process_one(item):
            async with semaphore:
                result = await self.annotate_abstract(
                    pmid=item["pmid"],
                    title=item["title"],
                    abstract=item["abstract"],
                    metadata=item.get("metadata"),
                )
                await asyncio.sleep(delay)
                return result

        # Process in chunks for incremental save
        for chunk_start in range(0, len(remaining), flush_every):
            chunk = remaining[chunk_start : chunk_start + flush_every]
            batch_results = await asyncio.gather(
                *(process_one(item) for item in chunk)
            )
            new_results = [r for r in batch_results if r is not None]
            results.extend(new_results)

            # Incremental save
            if output_path and new_results:
                with open(output_path, "a") as f:
                    for ann in new_results:
                        f.write(json.dumps(ann) + "\n")
                logger.info(
                    f"Checkpoint: {len(results)}/{len(items)} annotations saved"
                )

        logger.info(
            f"Annotated {len(results)}/{len(items)} abstracts successfully"
        )
        return results

    # Delegate to shared implementations
    save_annotations = staticmethod(save_annotations)
    generate_review_csv = staticmethod(generate_review_csv)
