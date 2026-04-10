"""
Bmlib-backed annotator.

Adapts ``bmlib.LLMClient`` (synchronous, provider-agnostic) to the async
``BaseAnnotator._call_llm`` interface so the CLI and the training pipeline
can share the same prompts, merge logic, and two-call flow control.

This is the backend used by ``biasbuster/cli/analysis.py``. The annotator
pipeline (``pipeline.py``, ``annotate_single_paper.py``) uses the native
async backends (``LLMAnnotator``, ``OpenAICompatAnnotator``) instead,
because those have per-provider rate-limit handling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from bmlib.llm import LLMClient, LLMMessage

from . import BaseAnnotator

logger = logging.getLogger(__name__)


class BmlibAnnotator(BaseAnnotator):
    """Annotator backed by a bmlib LLMClient.

    Inherits single-call, two-call, full-text two-call, and batch annotation
    from BaseAnnotator. Only the transport layer (_call_llm) is specific
    to the bmlib client wrapper.

    The bmlib client is synchronous; _call_llm runs it via
    ``asyncio.to_thread`` so it can be awaited alongside the async-native
    annotators without blocking the event loop.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: int = 3,
        extra_chat_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # extra kwargs forwarded to provider (e.g. {"think": False} for Ollama)
        self._extra_chat_kwargs = extra_chat_kwargs or {}

    async def __aenter__(self) -> "BmlibAnnotator":
        # bmlib client lifecycle is managed externally — the CLI creates it
        # once per invocation and re-uses it across stages.
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        pmid: str = "",
    ) -> Optional[str]:
        """Send a system+user message pair via the bmlib client.

        Runs synchronously under ``asyncio.to_thread`` so concurrent section
        extraction calls can proceed in parallel.
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_message),
        ]

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat,
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_mode=True,
                    **self._extra_chat_kwargs,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e!r}"
                logger.warning(
                    f"PMID {pmid}: bmlib chat error "
                    f"(attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
                await asyncio.sleep(2 ** attempt)
                continue

            text = (response.content or "").strip()
            if not text:
                last_error = "empty response"
                logger.warning(
                    f"PMID {pmid}: empty response "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(2 ** attempt)
                continue

            return text

        logger.error(
            f"PMID {pmid}: all {self.max_retries} bmlib attempts failed: {last_error}"
        )
        return None
