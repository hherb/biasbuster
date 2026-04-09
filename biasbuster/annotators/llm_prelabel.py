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
import httpx

from . import BaseAnnotator

logger = logging.getLogger(__name__)


class LLMAnnotator(BaseAnnotator):
    """Pre-label abstracts using Claude API via the official Anthropic SDK.

    Inherits single-call, two-call, and batch annotation from BaseAnnotator.
    Only the transport layer (_call_llm) and async lifecycle are specific
    to the Anthropic SDK.
    """

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

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        pmid: str = "",
    ) -> Optional[str]:
        """Send a single system+user message pair to Claude and return raw text.

        Handles retries, refusal detection, and transient errors.
        Returns the raw text response or None on failure.
        """
        if not self.api_key:
            logger.error("No Anthropic API key configured")
            return None

        last_error = None
        for attempt in range(self.max_retries):
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                text = "".join(
                    block.text for block in message.content
                    if block.type == "text"
                )

                # Detect refusals — no point retrying, model will always refuse
                if message.stop_reason == "refusal":
                    logger.warning(
                        f"PMID {pmid}: model refused to process "
                        f"(content likely triggered safety filters). Skipping."
                    )
                    return None

                if not text.strip():
                    logger.warning(
                        f"PMID {pmid}: empty text response "
                        f"(stop_reason={message.stop_reason}, "
                        f"content_types={[b.type for b in message.content]})"
                    )
                    last_error = "empty response"
                    await asyncio.sleep(2 ** attempt)
                    continue

                return text

            except anthropic.APIError as e:
                last_error = f"APIError: {e!r}"
                logger.warning(
                    f"API error for PMID {pmid} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e!r}"
                logger.warning(
                    f"Transient error for PMID {pmid} "
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
