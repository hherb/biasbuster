"""Tests for utils.retry module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from biasbuster.utils.retry import (
    retry_with_backoff,
    fetch_with_retry,
    RetryExhaustedError,
    RETRYABLE_STATUS_CODES,
)


@pytest.fixture
def mock_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


def _make_response(status_code: int, url: str = "https://example.com") -> httpx.Response:
    """Create a mock httpx.Response with the given status code."""
    request = httpx.Request("GET", url)
    return httpx.Response(status_code=status_code, request=request)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self):
        factory = AsyncMock(return_value="ok")
        result = await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert factory.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_http_error(self):
        resp = _make_response(429)
        error = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        factory = AsyncMock(side_effect=[error, error, "success"])
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert factory.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_on_non_retryable_http_error(self):
        resp = _make_response(404)
        error = httpx.HTTPStatusError("not found", request=resp.request, response=resp)
        factory = AsyncMock(side_effect=error)
        with pytest.raises(httpx.HTTPStatusError):
            await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert factory.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        error = httpx.ConnectError("connection refused")
        factory = AsyncMock(side_effect=[error, "recovered"])
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        error = httpx.TimeoutException("timed out")
        factory = AsyncMock(side_effect=[error, "ok"])
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exhausted_raises_error(self):
        resp = _make_response(500)
        error = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        factory = AsyncMock(side_effect=error)
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RetryExhaustedError) as exc_info:
                await retry_with_backoff(factory, max_retries=2, base_delay=0.01)
        assert exc_info.value.attempts == 3  # max_retries + 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        resp = _make_response(429)
        error = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        factory = AsyncMock(side_effect=[error, error, error, "ok"])
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await retry_with_backoff(factory, max_retries=3, base_delay=1.0, max_delay=60.0)
        # Delays should be 1.0, 2.0, 4.0
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        error = httpx.ConnectError("refused")
        factory = AsyncMock(side_effect=[error, error, error, "ok"])
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await retry_with_backoff(factory, max_retries=3, base_delay=10.0, max_delay=15.0)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert all(d <= 15.0 for d in delays)

    @pytest.mark.asyncio
    async def test_raises_unknown_exception_immediately(self):
        factory = AsyncMock(side_effect=ValueError("bad value"))
        with pytest.raises(ValueError):
            await retry_with_backoff(factory, max_retries=3, base_delay=0.01)
        assert factory.call_count == 1


class TestFetchWithRetry:
    """Tests for fetch_with_retry function."""

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_client):
        mock_client.request.return_value = _make_response(200)
        result = await fetch_with_retry(mock_client, "GET", "https://example.com", base_delay=0.01)
        assert result.status_code == 200
        mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_429(self, mock_client):
        mock_client.request.side_effect = [
            _make_response(429),
            _make_response(200),
        ]
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_with_retry(
                mock_client, "GET", "https://example.com", max_retries=3, base_delay=0.01
            )
        assert result.status_code == 200
        assert mock_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500(self, mock_client):
        mock_client.request.side_effect = [
            _make_response(500),
            _make_response(200),
        ]
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_with_retry(
                mock_client, "GET", "https://example.com", max_retries=3, base_delay=0.01
            )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_non_retryable_status(self, mock_client):
        """Non-retryable status codes (e.g., 404) are returned directly."""
        mock_client.request.return_value = _make_response(404)
        result = await fetch_with_retry(
            mock_client, "GET", "https://example.com", base_delay=0.01
        )
        assert result.status_code == 404
        assert mock_client.request.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_on_persistent_500(self, mock_client):
        mock_client.request.return_value = _make_response(500)
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RetryExhaustedError):
                await fetch_with_retry(
                    mock_client, "GET", "https://example.com",
                    max_retries=2, base_delay=0.01,
                )
        assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, mock_client):
        mock_client.request.side_effect = [
            httpx.ConnectError("refused"),
            _make_response(200),
        ]
        with patch("biasbuster.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_with_retry(
                mock_client, "GET", "https://example.com", max_retries=3, base_delay=0.01
            )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_passes_kwargs(self, mock_client):
        mock_client.request.return_value = _make_response(200)
        await fetch_with_retry(
            mock_client, "POST", "https://example.com",
            base_delay=0.01, json={"key": "value"}, headers={"X-Test": "1"},
        )
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs.get("json") == {"key": "value"}
        assert call_kwargs.kwargs.get("headers") == {"X-Test": "1"}


class TestRetryableStatusCodes:
    """Test the RETRYABLE_STATUS_CODES constant."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 529])
    def test_expected_codes(self, code):
        assert code in RETRYABLE_STATUS_CODES

    @pytest.mark.parametrize("code", [200, 201, 400, 401, 403, 404])
    def test_non_retryable_codes(self, code):
        assert code not in RETRYABLE_STATUS_CODES
