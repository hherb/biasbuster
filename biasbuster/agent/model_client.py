"""
Ollama model client for the verification agent.

Provides async functions to call the fine-tuned BiasBuster model via Ollama's
native /api/chat endpoint. Extracted from the evaluation harness pattern with
retry logic and exponential backoff.
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

from biasbuster.agent.agent_config import AgentConfig
from biasbuster.annotators import build_user_message
from export import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


REFINEMENT_SYSTEM_PROMPT = """\
You are a biomedical research integrity analyst. You previously assessed a \
clinical trial abstract and recommended specific verification steps.

The verification steps have now been executed automatically, and the results \
are provided below. Incorporate these findings into a REFINED bias assessment.

Where verification results confirm or contradict your initial findings, \
update your severity ratings and confidence accordingly. If a verification \
step failed or returned no data, note this as a limitation.

Provide your refined assessment in the same format as your initial assessment, \
with updated severity ratings, evidence, and an overall summary that reflects \
the verification results."""


async def call_ollama(
    client: httpx.AsyncClient,
    config: AgentConfig,
    messages: list[dict[str, str]],
) -> str:
    """Call Ollama's native /api/chat endpoint with retry logic.

    Args:
        client: Shared httpx async client.
        config: Agent configuration.
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        The model's response content as a string.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    url = f"{config.ollama_endpoint.rstrip('/')}/api/chat"
    payload = {
        "model": config.model_id,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "num_predict": config.max_tokens,
        },
    }

    last_error: Optional[str] = None

    for attempt in range(config.max_retries):
        try:
            start = time.monotonic()
            resp = await client.post(url, json=payload)

            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < config.max_retries - 1:
                    wait = 2 ** attempt * config.retry_base_delay_seconds
                    logger.warning(
                        "%s, retrying in %.1fs (attempt %d/%d)",
                        last_error, wait, attempt + 1, config.max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Ollama call failed after {config.max_retries} retries: {last_error}")

            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            content = data.get("message", {}).get("content", "")
            elapsed = time.monotonic() - start

            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            logger.info(
                "Ollama response: %d input tokens, %d output tokens, %.1fs",
                input_tokens, output_tokens, elapsed,
            )
            return content

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_error = str(exc)
            if attempt < config.max_retries - 1:
                wait = 2 ** attempt * config.retry_base_delay_seconds
                logger.warning(
                    "Connection error: %s, retrying in %.1fs (attempt %d/%d)",
                    last_error, wait, attempt + 1, config.max_retries,
                )
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama call failed after {config.max_retries} retries: {last_error}"
                ) from exc

    raise RuntimeError(f"Ollama call failed: {last_error}")


async def get_initial_assessment(
    client: httpx.AsyncClient,
    config: AgentConfig,
    title: str,
    abstract: str,
    pmid: str = "",
    metadata: Optional[dict] = None,
) -> str:
    """Get the initial bias assessment from the fine-tuned model.

    Uses the canonical training system prompt and the shared
    ``build_user_message`` helper for prompt consistency.

    Returns:
        Raw model output (markdown or JSON depending on the fine-tune).
    """
    user_message = build_user_message(pmid=pmid, title=title, abstract=abstract, metadata=metadata)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return await call_ollama(client, config, messages)


async def get_refined_assessment(
    client: httpx.AsyncClient,
    config: AgentConfig,
    initial_assessment: str,
    verification_results_text: str,
) -> str:
    """Get a refined assessment incorporating verification results.

    Sends the initial assessment and tool results back to the model
    for a single refinement pass.

    Returns:
        The refined assessment text.
    """
    user_message = (
        "## Your Initial Assessment\n\n"
        f"{initial_assessment}\n\n"
        "## Verification Results\n\n"
        f"{verification_results_text}\n\n"
        "Please provide your REFINED bias assessment incorporating "
        "the verification results above."
    )
    messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return await call_ollama(client, config, messages)
