"""
OpenAI-Compatible LLM Annotator

Uses any OpenAI-compatible API (DeepSeek, OpenAI, local vLLM/SGLang, etc.)
for structured bias assessments. Shares the same prompt and output format as
the Anthropic annotator so results are directly comparable.

DeepSeek API: https://api-docs.deepseek.com/
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from . import (
    build_user_message,
    generate_review_csv,
    parse_llm_json,
    save_annotations,
    strip_markdown_fences,
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

        text = ""
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

                # parse_llm_json returned None — retry
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

        # Deduplicate by PMID (enriched data may contain duplicates)
        seen_pmids: set[str] = set(already_done)
        remaining = []
        for it in items:
            pmid = it["pmid"]
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                remaining.append(it)
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
            f"Annotated {len(results)}/{len(items)} abstracts successfully "
            f"(model: {self.model})"
        )
        return results

    # Delegate to shared implementations
    save_annotations = staticmethod(save_annotations)
    generate_review_csv = staticmethod(generate_review_csv)
