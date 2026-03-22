"""
LLM Pre-Labelling Annotator

Uses Claude API to generate structured bias assessments for abstracts.
These are PRE-LABELS that must be human-validated before use in training.

Strategy:
1. Send abstract + metadata to Claude with detailed bias assessment prompt
2. Parse structured response into BiasAssessment objects
3. Flag low-confidence assessments for priority human review
4. Store in SQLite database for review
"""

import asyncio
import logging
import os
from typing import Optional

import anthropic

from . import (
    build_user_message,
    parse_llm_json,
)

logger = logging.getLogger(__name__)

# The system prompt encodes the full bias taxonomy and verification knowledge
# Canonical prompt imported from the single source of truth.
# See docs/MISTAKES_ROUND_1_AND_FIXES.md for why prompt unification matters.
from prompts import ANNOTATION_SYSTEM_PROMPT  # noqa: E402


class LLMAnnotator:
    """Pre-label abstracts using Claude API via the official Anthropic SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client: Optional[anthropic.AsyncAnthropic] = None

    async def __aenter__(self) -> "LLMAnnotator":
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )
        return self

    async def __aexit__(self, *args) -> None:
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

                # Detect refusals — no point retrying, model will always refuse
                if message.stop_reason == "refusal":
                    logger.warning(
                        f"PMID {pmid}: model refused to process this abstract "
                        f"(content likely triggered safety filters). Skipping."
                    )
                    return None

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
        items: list[dict],
        concurrency: int = 3,
        delay: float = 1.0,
        already_done: Optional[set[str]] = None,
        on_result: Optional[callable] = None,
    ) -> list[dict]:
        """Annotate a batch of abstracts with rate limiting.

        Skips PMIDs already in already_done (for resume support).
        Each successful annotation is passed to on_result immediately
        for incremental persistence (e.g. saving to database).

        Args:
            items: List of dicts with pmid, title, abstract, metadata keys.
            concurrency: Max concurrent API requests.
            delay: Seconds between requests.
            already_done: Set of PMIDs to skip (already annotated).
            on_result: Optional callback(annotation_dict) called immediately
                       on each successful annotation for incremental save.

        Returns:
            List of successful annotations.
        """
        if already_done is None:
            already_done = set()

        # Deduplicate by PMID
        seen_pmids: set[str] = set(already_done)
        remaining = []
        for it in items:
            pmid = it["pmid"]
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                remaining.append(it)
        if not remaining:
            logger.info("All items already annotated, nothing to do")
            return []

        semaphore = asyncio.Semaphore(concurrency)
        successful: list[dict] = []
        flush_every = 10

        async def process_one(item):
            async with semaphore:
                result = await self.annotate_abstract(
                    pmid=item["pmid"],
                    title=item["title"],
                    abstract=item["abstract"],
                    metadata=item.get("metadata"),
                )
                # Save incrementally as each result arrives
                if result is not None:
                    successful.append(result)
                    if on_result:
                        on_result(result)
                    if len(successful) % flush_every == 0:
                        logger.info(
                            f"Checkpoint: {len(successful)}/{len(remaining)} "
                            f"annotations completed"
                        )
                await asyncio.sleep(delay)
                return result

        await asyncio.gather(
            *(process_one(item) for item in remaining)
        )

        logger.info(
            f"Annotated {len(successful)}/{len(remaining)} abstracts successfully"
        )
        return successful
