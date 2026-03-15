"""
OpenAI-Compatible LLM Annotator

Uses any OpenAI-compatible API (DeepSeek, OpenAI, local vLLM/SGLang, etc.)
for structured bias assessments. Shares the same prompt and output format as
the Anthropic annotator so results are directly comparable.

DeepSeek API: https://api-docs.deepseek.com/
"""

import asyncio
import logging
import os
from typing import Optional

import httpx

from . import (
    build_user_message,
    parse_llm_json,
)
from .llm_prelabel import ANNOTATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAICompatAnnotator:
    """Pre-label abstracts using any OpenAI-compatible API (DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.temperature = temperature
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OpenAICompatAnnotator":
        self.client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, *args) -> None:
        if self.client:
            await self.client.aclose()

    async def annotate_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Send a single abstract for bias assessment. Returns parsed JSON or None."""
        if not self.api_key:
            logger.error("No API key configured for OpenAI-compatible endpoint")
            return None

        user_message = build_user_message(pmid, title, abstract, metadata)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.post(
                    f"{self.api_base}/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", 2 ** attempt))
                    logger.warning(
                        f"Rate limited (PMID {pmid}), retrying in {retry_after}s"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(
                        f"HTTP {resp.status_code} for PMID {pmid}: {resp.text[:200]}"
                    )
                    last_error = f"HTTP {resp.status_code}"
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                assessment = parse_llm_json(text, pmid=pmid)
                if assessment is not None:
                    assessment["pmid"] = pmid
                    assessment["title"] = title
                    assessment["_annotation_model"] = self.model
                    return assessment

                last_error = "JSON parse failure after repair attempt"
                logger.warning(
                    f"PMID {pmid}: JSON parse failed "
                    f"(attempt {attempt + 1}/{self.max_retries}), retrying"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except httpx.TimeoutException:
                logger.warning(
                    f"Timeout for PMID {pmid} (attempt {attempt + 1}/{self.max_retries})"
                )
                last_error = "timeout"
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
    ) -> list[dict]:
        """Annotate a batch of abstracts with rate limiting.

        Skips PMIDs already in already_done (for resume support).
        Returns list of successful annotations.

        Args:
            items: List of dicts with pmid, title, abstract, metadata keys.
            concurrency: Max concurrent API requests.
            delay: Seconds between requests.
            already_done: Set of PMIDs to skip (already annotated).

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

        # Process all items
        results = await asyncio.gather(
            *(process_one(item) for item in remaining)
        )
        successful = [r for r in results if r is not None]

        logger.info(
            f"Annotated {len(successful)}/{len(remaining)} abstracts successfully "
            f"(model: {self.model})"
        )
        return successful
