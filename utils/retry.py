"""
Retry utilities with exponential backoff for network operations.

Provides both a standalone async helper and a decorator for methods
that make HTTP requests or API calls.
"""

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retryable HTTP status codes
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted. Last error: {last_error}"
        )


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_status_codes: frozenset[int] = RETRYABLE_STATUS_CODES,
    operation_name: str = "operation",
) -> T:
    """
    Execute an async callable with exponential backoff on transient failures.

    Args:
        coro_factory: A zero-argument callable that returns an awaitable.
            Called fresh on each attempt (do NOT pass an already-awaited coroutine).
        max_retries: Maximum number of retry attempts (total attempts = max_retries + 1).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        retryable_status_codes: HTTP status codes considered transient/retryable.
        operation_name: Human-readable name for log messages.

    Returns:
        The result of the awaitable on success.

    Raises:
        RetryExhaustedError: If all attempts fail with retryable errors.
        Exception: If a non-retryable error occurs, it is raised immediately.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retryable_status_codes:
                raise
            last_error = e
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
        except Exception as e:
            # Check for anthropic-style rate limit / server errors
            error_type = type(e).__name__
            if "RateLimitError" in error_type or "InternalServerError" in error_type:
                last_error = e
            elif "APIConnectionError" in error_type:
                last_error = e
            else:
                raise

        if attempt < max_retries:
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                f"{operation_name}: attempt {attempt + 1}/{max_retries + 1} failed "
                f"({last_error}), retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RetryExhaustedError(max_retries + 1, last_error)


async def fetch_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> httpx.Response:
    """
    Make an HTTP request with automatic retry and exponential backoff.

    This is the primary helper for all httpx-based network calls.
    Non-retryable errors (4xx except 429) are raised immediately.

    Args:
        client: The httpx.AsyncClient instance.
        method: HTTP method ("GET", "POST", etc.).
        url: Request URL.
        max_retries: Maximum retry attempts.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.
        **kwargs: Additional arguments passed to client.request().

    Returns:
        The httpx.Response on success.

    Raises:
        RetryExhaustedError: If all retries are exhausted.
        httpx.HTTPStatusError: If a non-retryable HTTP error occurs.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                # Fall through to retry logic below
            else:
                return resp
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e

        if attempt < max_retries:
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                f"{method} {url}: attempt {attempt + 1}/{max_retries + 1} failed "
                f"({last_error}), retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RetryExhaustedError(max_retries + 1, last_error)
