"""Ollama runner for Phase 5 local-model evaluation.

Wraps the Ollama HTTP API (default localhost:11434) for the three local
candidates: gpt-oss:20b, gemma4:26b-a4b-it-q8_0, qwen3.6:35b-a3b-q8_0.

The Ollama /api/chat endpoint accepts an OpenAI-style messages list with
a system prompt and user message, and returns the assistant's reply.
We use it in non-streaming mode for simplicity — token counts arrive in
a single response payload.

Per prompt_v1.md, each pass per RCT × protocol issues 6 calls:
  5 per-domain calls (d1..d5) using `prompts.py::build_system_prompt`
  1 synthesis call using the synthesize prompt

Per pre-reg §8: parse failures are retried up to 2 times. After the
third failure, the result is recorded as ``parse_failure`` with raw
output preserved for audit. If the parse-failure rate exceeds 20%
across a pass, the orchestrator should halt and revise prompts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
HTTP_TIMEOUT = httpx.Timeout(10.0, read=600.0)  # generous read for slow local models

# Map from our domain code (used in DB) to the prompts.py stage name.
DOMAIN_TO_STAGE = {
    "d1": "domain_randomization",
    "d2": "domain_deviations_from_interventions",
    "d3": "domain_missing_outcome_data",
    "d4": "domain_outcome_measurement",
    "d5": "domain_selection_of_reported_result",
    "overall": "synthesize",
}

VALID_JUDGMENTS = {"low", "some_concerns", "high"}


@dataclass
class OllamaCallResult:
    """Outcome of one Ollama /api/chat call."""
    judgment: str | None       # canonical 'low' | 'some_concerns' | 'high', None on parse failure
    rationale: str | None
    raw_response: str
    parse_status: str          # 'ok' | 'retry_succeeded' | 'parse_failure' | 'api_error'
    parse_attempts: int
    duration_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


class OllamaRunner:
    """Stateless runner for Ollama-hosted models.

    Construct once, call `domain_call` or `synthesis_call` per (rct, domain).
    The runner is stateless across calls — Ollama's /api/chat is given a
    full message history each time and we never share context between
    calls (matches the pre-reg's pass-independence guarantee).
    """

    def __init__(self, model_id: str, host: str = OLLAMA_DEFAULT_HOST):
        self.model_id = model_id
        self.host = host
        self._client = httpx.Client(timeout=HTTP_TIMEOUT)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaRunner:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _chat(self, system_prompt: str, user_message: str) -> tuple[str, dict[str, int | None], float, str | None]:
        """Single Ollama /api/chat call. Returns (text, token_counts, duration, error)."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            # We accept Ollama's defaults for temperature/top_p; pre-reg §5
            # commits to "model defaults" and we don't override per model.
        }
        t0 = time.monotonic()
        try:
            r = self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            return "", {"input": None, "output": None}, time.monotonic() - t0, repr(exc)
        duration = time.monotonic() - t0
        if r.status_code != 200:
            return "", {"input": None, "output": None}, duration, f"HTTP {r.status_code}: {r.text[:300]}"
        try:
            data = r.json()
        except ValueError as exc:
            return "", {"input": None, "output": None}, duration, f"non-JSON response: {exc}"
        text = (data.get("message", {}) or {}).get("content", "") or ""
        # Ollama reports counts as eval_count (output) and prompt_eval_count (input).
        input_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")
        return text, {"input": input_tokens, "output": output_tokens}, duration, None

    def domain_call(self,
                    domain: str,
                    system_prompt: str,
                    user_message: str,
                    max_attempts: int = 3) -> OllamaCallResult:
        """Call the model for one RoB 2 domain, with retry on parse failure."""
        return self._call_with_retry(
            domain=domain,
            system_prompt=system_prompt,
            user_message=user_message,
            max_attempts=max_attempts,
            output_kind="domain",
        )

    def synthesis_call(self,
                       system_prompt: str,
                       user_message: str,
                       max_attempts: int = 3) -> OllamaCallResult:
        return self._call_with_retry(
            domain="overall",
            system_prompt=system_prompt,
            user_message=user_message,
            max_attempts=max_attempts,
            output_kind="synthesis",
        )

    def _call_with_retry(self, *, domain: str, system_prompt: str,
                         user_message: str, max_attempts: int,
                         output_kind: str) -> OllamaCallResult:
        last_text = ""
        last_error: str | None = None
        last_tokens: dict[str, int | None] = {"input": None, "output": None}
        last_duration = 0.0
        for attempt in range(1, max_attempts + 1):
            text, tokens, duration, error = self._chat(system_prompt, user_message)
            last_text = text
            last_tokens = tokens
            last_duration = duration
            last_error = error
            if error:
                # Network/HTTP failures don't benefit from retry-with-same-prompt
                # in the normal case, but we retry once anyway in case of transient.
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
                return OllamaCallResult(
                    judgment=None, rationale=None,
                    raw_response=text, parse_status="api_error",
                    parse_attempts=attempt, duration_seconds=duration,
                    input_tokens=tokens["input"], output_tokens=tokens["output"],
                    error=error,
                )
            judgment, rationale = parse_response(text, output_kind)
            if judgment is not None:
                status = "ok" if attempt == 1 else "retry_succeeded"
                return OllamaCallResult(
                    judgment=judgment, rationale=rationale,
                    raw_response=text, parse_status=status,
                    parse_attempts=attempt, duration_seconds=duration,
                    input_tokens=tokens["input"], output_tokens=tokens["output"],
                    error=None,
                )
            # Parse failure — retry with same prompt (pre-reg §6).
            if attempt < max_attempts:
                time.sleep(0.5)
        return OllamaCallResult(
            judgment=None, rationale=None,
            raw_response=last_text, parse_status="parse_failure",
            parse_attempts=max_attempts, duration_seconds=last_duration,
            input_tokens=last_tokens["input"], output_tokens=last_tokens["output"],
            error=last_error,
        )


# --- Response parsing (used by both Ollama and Anthropic runners) ------

# Some models wrap JSON in markdown code fences; strip those before parsing.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?", re.MULTILINE)
_CLOSING_FENCE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block in text.

    Some models produce a small preamble despite being told to emit pure
    JSON; we tolerate that by scanning for the first ``{`` and then
    bracket-matching to the corresponding closing brace. Strings and
    escapes are honoured so braces inside string literals don't fool the
    scanner.
    """
    text = _FENCE_RE.sub("", text, count=1)
    text = _CLOSING_FENCE_RE.sub("", text, count=1)
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return ""


_JUDGMENT_NORM_MAP = {
    "low": "low",
    "low risk": "low",
    "low risk of bias": "low",
    "some_concerns": "some_concerns",
    "some concerns": "some_concerns",
    "high": "high",
    "high risk": "high",
    "high risk of bias": "high",
}


def _normalize_judgment(raw: Any) -> str | None:
    """Map a raw judgment string to canonical form, or None if unrecognised."""
    if not isinstance(raw, str):
        return None
    canonical = _JUDGMENT_NORM_MAP.get(raw.strip().lower())
    if canonical is None:
        return None
    return canonical


def parse_response(text: str, output_kind: str) -> tuple[str | None, str | None]:
    """Extract (judgment, rationale) from the model's raw response text.

    Returns (None, None) on parse failure so the caller can retry.
    `output_kind` is "domain" or "synthesis" — they expect different JSON
    shapes per prompt_v1.md §3.
    """
    json_str = _extract_json_object(text)
    if not json_str:
        return None, None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None

    if output_kind == "domain":
        judgment = _normalize_judgment(data.get("judgement"))
        rationale = data.get("justification")
        if judgment is None:
            return None, None
        return judgment, rationale if isinstance(rationale, str) else None

    if output_kind == "synthesis":
        judgment = _normalize_judgment(data.get("overall_judgement"))
        rationale = data.get("overall_rationale")
        if judgment is None:
            return None, None
        return judgment, rationale if isinstance(rationale, str) else None

    raise ValueError(f"unknown output_kind: {output_kind!r}")
