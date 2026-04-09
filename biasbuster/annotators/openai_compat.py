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
from typing import Optional

import httpx

from . import BaseAnnotator

logger = logging.getLogger(__name__)


class OpenAICompatAnnotator(BaseAnnotator):
    """Pre-label abstracts using any OpenAI-compatible API (DeepSeek, etc.).

    Inherits single-call, two-call, and batch annotation from BaseAnnotator.
    Only the transport layer (_call_llm) and async lifecycle are specific
    to the OpenAI-compatible HTTP interface.
    """

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

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        pmid: str = "",
    ) -> Optional[str]:
        """Send a single system+user message pair and return raw text.

        Handles retries, rate limiting, and transient errors.
        Returns the raw text response or None on failure.
        """
        if not self.api_key:
            logger.error("No API key configured for OpenAI-compatible endpoint")
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
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

                if not text.strip():
                    last_error = "empty response"
                    await asyncio.sleep(2 ** attempt)
                    continue

                return text

            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e!r}"
                logger.warning(
                    f"Transient error for PMID {pmid} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                last_error = f"{type(e).__name__}: {e!r}"
                logger.warning(
                    f"Response parse error for PMID {pmid} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error(
                    f"LLM call failed for PMID {pmid} "
                    f"({type(e).__name__}): {e!r}"
                )
                return None

        logger.error(
            f"All {self.max_retries} attempts failed for PMID {pmid}: {last_error}"
        )
        return None
