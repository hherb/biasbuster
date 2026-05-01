"""Anthropic Batch API runner for Phase 5.8 — Sonnet 4.6 evaluation.

Wraps `client.messages.batches` with prompt caching and our project-
specific request shaping. The decomposed harness from prompt_v1.md is
implemented as TWO sequential batches:

  Stage 1 — domain batch:    1,500 calls × 2 protocols = 3,000 calls
                             (5 domains × 100 RCTs × 3 passes × 2 protocols)
  Stage 2 — synthesis batch: 600 calls (after stage 1 results are written
                             to benchmark_judgment so the synthesis user
                             message can include each pass's domain
                             judgements)

Each call's `custom_id` is its source label + RCT id + domain so we can
join results back to benchmark_rct unambiguously:

  domain calls:     "{source_label}/{rct_id}/{domain}"
                    e.g. "sonnet_4_6_fulltext_pass1/RCT001/d2"
  synthesis calls:  "{source_label}/{rct_id}/overall"

Per the locked prompt spec (`prompt_v1.md` §5):
  - model: claude-sonnet-4-6
  - max_tokens: 2000
  - temperature, top_p: model defaults (do not set explicitly)
  - thinking: not enabled (the pre-reg specifies model defaults; Sonnet 4.6
    default is no thinking, so we pass nothing rather than {type:"adaptive"})
  - seed: not set (between-pass variation is what we are measuring)

Prompt caching (per `shared/prompt-caching.md`):
  - Cache breakpoint on the user message after the materials block.
  - On Sonnet 4.6 the minimum cacheable prefix is 2048 tokens, so caching
    helps the FULLTEXT protocol (typical request ~12K input tokens) and
    has no effect on ABSTRACT-only (~1.3K input — silently won't cache).
  - System prompts vary across the 5 domains, so cache reuse is across
    PASSES of the same RCT × protocol × domain (3 passes → 1 write + 2
    reads on identical-prefix calls), not across domains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DEFAULT_MODEL_ID = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2000

# Per pre-reg §6.1 retry policy; we cap parse-failure retries at 2 inside
# the orchestrator (Batch API responses arrive once — retries here are
# orchestrator-driven re-submissions, not in-batch retries).


@dataclass
class BatchSubmission:
    """Returned by submit_batch — what the orchestrator persists to state."""
    batch_id: str
    submitted_at: str
    n_requests: int


@dataclass
class BatchStatus:
    """Polling snapshot returned by poll_batch."""
    batch_id: str
    processing_status: str  # 'in_progress' | 'canceling' | 'ended'
    succeeded: int
    errored: int
    processing: int
    canceled: int
    expired: int


@dataclass
class BatchResult:
    """One per `custom_id` from fetch_results.

    `text_response` is the raw assistant text content (what the JSON parser
    will be applied to). `usage_*` capture token accounting including
    cache hits/writes for the audit trail (mirrors evaluation_run columns
    cache_read_tokens / cache_write_tokens).
    """
    custom_id: str
    succeeded: bool
    text_response: str
    error_type: str | None
    error_message: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    stop_reason: str | None


class AnthropicBatchRunner:
    """Stateless façade over the Anthropic Batch API.

    Construct once per orchestrator run. The client picks up
    ANTHROPIC_API_KEY from the environment by default.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.client = anthropic.Anthropic()
        self.model_id = model_id

    # --- Request building ---------------------------------------------

    def build_request(self, custom_id: str, system_prompt: str,
                      user_message: str,
                      max_tokens: int = DEFAULT_MAX_TOKENS) -> Request:
        """Build one Batch API request with prompt caching enabled.

        The cache_control breakpoint on the user message caches
        ``tools + system + user`` up to that point; for our 5 same-domain
        calls across 3 passes of the same RCT, that prefix is byte-
        identical, so passes 2 and 3 hit the cache. Pass 1 pays the 1.25×
        write premium.

        `cache_control` is also placed on the system prompt so that, even
        though the system varies across domains, repeated identical
        system prompts (e.g. domain D1 across many RCTs) get a cache hit
        when the same domain is processed back-to-back inside the batch.
        On Sonnet 4.6 the per-domain system prompt is below the 2048-
        token minimum, so this breakpoint is mostly a no-op — it is
        cheap to include and will activate automatically on any model
        with a smaller minimum.
        """
        return Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model_id,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": user_message,
                        "cache_control": {"type": "ephemeral"},
                    }],
                }],
            ),
        )

    # --- Submission & polling -----------------------------------------

    def submit_batch(self, requests: list[Request]) -> BatchSubmission:
        """POST all requests at once. Anthropic enforces 100,000 max."""
        if not requests:
            raise ValueError("no requests to submit")
        if len(requests) > 100_000:
            raise ValueError(
                f"batch too large: {len(requests)} > 100,000 (Anthropic limit). "
                "Split into multiple batches."
            )
        batch = self.client.messages.batches.create(requests=requests)
        return BatchSubmission(
            batch_id=batch.id,
            submitted_at=batch.created_at.isoformat() if batch.created_at else "",
            n_requests=len(requests),
        )

    def poll_batch(self, batch_id: str) -> BatchStatus:
        batch = self.client.messages.batches.retrieve(batch_id)
        rc = batch.request_counts
        return BatchStatus(
            batch_id=batch.id,
            processing_status=batch.processing_status,
            succeeded=rc.succeeded,
            errored=rc.errored,
            processing=rc.processing,
            canceled=rc.canceled,
            expired=rc.expired,
        )

    def wait_for_completion(self, batch_id: str,
                            poll_interval_s: float = 60.0,
                            max_wait_s: float = 24 * 3600,
                            on_status: callable | None = None
                            ) -> BatchStatus:
        """Block until processing_status == 'ended' or max_wait_s elapses.

        Anthropic's contract is "most batches complete within 1 hour;
        maximum 24 hours". The orchestrator typically uses this only for
        the smoke test path; for production runs the user invokes the
        `status` subcommand interactively to avoid keeping a terminal
        open for hours. `on_status` lets the caller print progress.
        """
        deadline = time.monotonic() + max_wait_s
        while True:
            status = self.poll_batch(batch_id)
            if on_status:
                on_status(status)
            if status.processing_status == "ended":
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"batch {batch_id} did not complete within "
                    f"{max_wait_s/3600:.1f}h (last status: {status.processing_status})"
                )
            time.sleep(poll_interval_s)

    # --- Result fetching ----------------------------------------------

    def fetch_results(self, batch_id: str) -> Iterator[BatchResult]:
        """Stream parsed results. Caller pipes into the DB."""
        for result in self.client.messages.batches.results(batch_id):
            yield self._parse_result(result)

    @staticmethod
    def _parse_result(raw_result) -> BatchResult:
        """Extract our flat shape from Anthropic's nested result object."""
        custom_id = raw_result.custom_id
        outcome = raw_result.result

        if outcome.type == "succeeded":
            msg = outcome.message
            text = ""
            for block in msg.content:
                if block.type == "text":
                    text += block.text
            usage = msg.usage
            return BatchResult(
                custom_id=custom_id,
                succeeded=True,
                text_response=text,
                error_type=None,
                error_message=None,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    usage, "cache_creation_input_tokens", None
                ),
                cache_read_input_tokens=getattr(
                    usage, "cache_read_input_tokens", None
                ),
                stop_reason=msg.stop_reason,
            )

        # Failure paths: errored / canceled / expired all share a similar shape.
        # Anthropic returns the inner error type+message on the .error attribute
        # for "errored" and a status string for "canceled"/"expired".
        err_type = outcome.type
        err_message = ""
        if hasattr(outcome, "error") and outcome.error is not None:
            err_type = outcome.error.type
            err_message = outcome.error.message
        return BatchResult(
            custom_id=custom_id,
            succeeded=False,
            text_response="",
            error_type=err_type,
            error_message=err_message,
            input_tokens=None,
            output_tokens=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            stop_reason=None,
        )

    # --- Cancellation (rarely needed; smoke-test helper) --------------

    def cancel_batch(self, batch_id: str) -> str:
        cancelled = self.client.messages.batches.cancel(batch_id)
        return cancelled.processing_status
