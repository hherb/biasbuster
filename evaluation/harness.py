"""
Evaluation Harness

Runs the same test set through multiple models and collects raw outputs.
Supports any OpenAI-compatible API endpoint, so works with:
- SGLang (recommended for DGX Spark)
- vLLM
- Ollama
- llama.cpp server
- Remote APIs (OpenAI, Anthropic, etc.)

Usage:
    # Zero-shot evaluation (before fine-tuning)
    python -m evaluation.harness \
        --test-set dataset/test/test.jsonl \
        --models qwen3.5-27b,olmo-3.1-32b \
        --endpoints http://localhost:8000,http://localhost:8001 \
        --mode zero-shot \
        --output eval_results/

    # Post-fine-tuning evaluation
    python -m evaluation.harness \
        --test-set dataset/test/test.jsonl \
        --models qwen3.5-27b-lora,olmo-3.1-32b-lora \
        --endpoints http://localhost:8000,http://localhost:8001 \
        --mode fine-tuned \
        --output eval_results/
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "TestExample",
    "ModelOutput",
    "EvalConfig",
    "EvalHarness",
    "load_test_set",
    "build_user_message",
    "load_model_outputs",
]


# System prompt for zero-shot evaluation (no domain-specific training)
ZERO_SHOT_SYSTEM_PROMPT = """You are a biomedical research integrity analyst. Assess the following
clinical trial abstract for potential bias. Provide a structured assessment covering:

1. Statistical Reporting: Are only relative measures (HR, OR, RRR) reported without
   absolute measures (ARR, NNT, baseline risk)?
2. Spin: Do conclusions match the results? Is there emphasis on secondary outcomes
   when the primary was non-significant?
3. Outcome Reporting: Are outcomes patient-centred or surrogate?
4. Conflict of Interest: Is funding disclosed? Industry involvement?
5. Methodology: Appropriate comparator? Blinding? ITT analysis?

For each dimension, rate severity as: none, low, moderate, high, or critical.
Provide an overall bias probability (0.0 to 1.0).
Recommend specific verification steps citing databases like CMS Open Payments,
ClinicalTrials.gov, ORCID, Europe PMC, or Retraction Watch.

Respond in JSON format."""

# System prompt for fine-tuned models (matches training format)
FINE_TUNED_SYSTEM_PROMPT = """You are a biomedical research integrity analyst. Given a clinical trial abstract,
assess it for potential bias across five domains:

1. STATISTICAL REPORTING: Does the abstract report only relative measures (RRR, OR, HR)
   without absolute measures (ARR, NNT, baseline risk)? Sole reliance on relative measures
   inflates perceived benefit and is a strong indicator of potential bias.

2. SPIN: Do the conclusions match the actual results? Look for claims of benefit when
   the primary outcome was not statistically significant, inappropriate causal language,
   and emphasis on secondary analyses.

3. OUTCOME REPORTING: Are outcomes patient-centred or surrogate? Is there evidence of
   outcome switching from the trial registry? Are composite endpoints disaggregated?

4. CONFLICT OF INTEREST: Is funding disclosed? Are authors affiliated with the sponsor?
   Suggest verification steps using CMS Open Payments, ClinicalTrials.gov, ORCID,
   Europe PMC, and country-specific databases (Medicines Australia, EFPIA).

5. METHODOLOGICAL RED FLAGS: Inappropriate comparator? Enrichment design?
   Per-protocol only? Premature stopping?

Provide your reasoning step by step, then a structured assessment with recommended
verification steps citing specific databases and URLs."""


@dataclass
class TestExample:
    """A single test example with ground truth."""
    pmid: str = ""
    title: str = ""
    abstract: str = ""
    ground_truth: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    source: str = ""  # e.g. "retracted", "high_suspicion", "low_suspicion"


@dataclass
class ModelOutput:
    """Raw output from a model for a single test example."""
    pmid: str = ""
    model_id: str = ""
    raw_output: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""
    # Model endpoints (OpenAI-compatible)
    models: dict = field(default_factory=dict)  # {model_id: endpoint_url}

    # Generation parameters (identical across models for fairness)
    temperature: float = 0.1       # Low temperature for more deterministic output
    max_tokens: int = 4000
    top_p: float = 0.95

    # Evaluation mode
    mode: str = "zero-shot"        # zero-shot or fine-tuned

    # Rate limiting
    requests_per_second: float = 2.0
    max_concurrent: int = 3

    # Network resilience
    request_timeout_seconds: float = 600.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 5.0

    # Ollama-specific
    num_ctx: Optional[int] = None  # Context window size (reduces KV cache; huge speedup on Ollama)

    # Output
    output_dir: str = "eval_results"


def load_test_set(path: Path) -> list[TestExample]:
    """
    Load test examples from JSONL.
    Each line should have: pmid, title, abstract, and ground_truth annotations.
    """
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(TestExample(
                pmid=data.get("pmid", ""),
                title=data.get("title", ""),
                abstract=data.get("abstract", data.get("abstract_text", "")),
                ground_truth=data,  # The full annotation IS the ground truth
                source=data.get("source", ""),
            ))
    logger.info(f"Loaded {len(examples)} test examples from {path}")
    return examples


def build_user_message(example: TestExample) -> str:
    """Build the user message for a test example."""
    return (
        f"Assess the following clinical trial abstract for potential bias:\n\n"
        f"Title: {example.title}\n"
        f"PMID: {example.pmid}\n\n"
        f"Abstract:\n{example.abstract}"
    )


class EvalHarness:
    """
    Runs test examples through multiple models via OpenAI-compatible APIs.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def query_model(
        self,
        model_id: str,
        endpoint: str,
        example: TestExample,
    ) -> ModelOutput:
        """
        Send a single example to a model endpoint and collect the response.
        Uses OpenAI-compatible /v1/chat/completions format.
        Retries on transient errors (5xx, timeouts, connection errors) with
        exponential backoff.
        """
        system_prompt = (
            FINE_TUNED_SYSTEM_PROMPT if self.config.mode == "fine-tuned"
            else ZERO_SHOT_SYSTEM_PROMPT
        )
        user_message = build_user_message(example)

        use_ollama_native = self.config.num_ctx is not None

        if use_ollama_native:
            # Ollama native /api/chat — reliably supports num_ctx via options
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_ctx": self.config.num_ctx,
                    "num_predict": self.config.max_tokens,
                },
            }
            url = f"{endpoint.rstrip('/')}/api/chat"
        else:
            # OpenAI-compatible endpoint (works with SGLang, vLLM, Ollama, etc.)
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            url = f"{endpoint.rstrip('/')}/v1/chat/completions"

        start_time = time.monotonic()
        last_error = None
        max_retries = self.config.max_retries
        base_delay = self.config.retry_base_delay_seconds

        for attempt in range(max_retries):
            try:
                resp = await self.client.post(url, json=payload)

                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt * base_delay
                        logger.warning(
                            f"PMID {example.pmid}: {last_error}, "
                            f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    return ModelOutput(
                        pmid=example.pmid,
                        model_id=model_id,
                        error=last_error,
                        latency_seconds=time.monotonic() - start_time,
                    )

                if resp.status_code != 200:
                    return ModelOutput(
                        pmid=example.pmid,
                        model_id=model_id,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        latency_seconds=time.monotonic() - start_time,
                    )

                data = resp.json()

                if use_ollama_native:
                    # Ollama native response format
                    raw_output = data.get("message", {}).get("content", "")
                    usage = {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    }
                else:
                    # OpenAI-compatible response format
                    choice = data.get("choices", [{}])[0]
                    raw_output = choice.get("message", {}).get("content", "")
                    usage = data.get("usage", {})

                return ModelOutput(
                    pmid=example.pmid,
                    model_id=model_id,
                    raw_output=raw_output,
                    latency_seconds=time.monotonic() - start_time,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * base_delay
                    logger.warning(
                        f"PMID {example.pmid}: {last_error}, "
                        f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue

            except Exception as e:
                return ModelOutput(
                    pmid=example.pmid,
                    model_id=model_id,
                    error=str(e),
                    latency_seconds=time.monotonic() - start_time,
                )

        return ModelOutput(
            pmid=example.pmid,
            model_id=model_id,
            error=f"Failed after {max_retries} retries: {last_error}",
            latency_seconds=time.monotonic() - start_time,
        )

    async def evaluate_model(
        self,
        model_id: str,
        endpoint: str,
        examples: list[TestExample],
        on_result: Callable[[ModelOutput], None] | None = None,
    ) -> list[ModelOutput]:
        """Run all test examples through a single model with progress bar.

        Args:
            on_result: Optional callback invoked with each ModelOutput
                       immediately after it completes (for incremental persistence).
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        delay = 1.0 / self.config.requests_per_second
        errors = 0

        pbar = tqdm(
            total=len(examples),
            desc=f"{model_id}",
            unit="ex",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
        )
        pbar.set_postfix(errors=0)

        async def process_one(example: TestExample) -> ModelOutput:
            nonlocal errors
            async with semaphore:
                result = await self.query_model(model_id, endpoint, example)
                await asyncio.sleep(delay)
                if result.error:
                    errors += 1
                if on_result:
                    on_result(result)
                pbar.set_postfix(
                    errors=errors,
                    last_latency=f"{result.latency_seconds:.1f}s",
                )
                pbar.update(1)
                return result

        tasks = [process_one(ex) for ex in examples]
        results = await asyncio.gather(*tasks)
        outputs = list(results)
        pbar.close()

        logger.info(
            f"Model {model_id}: {len(outputs)} examples, "
            f"{errors} errors, "
            f"mean latency {sum(o.latency_seconds for o in outputs)/len(outputs):.1f}s"
        )

        return outputs

    async def run_all(
        self, examples: list[TestExample]
    ) -> dict[str, list[ModelOutput]]:
        """Run all examples through all configured models."""
        all_outputs = {}

        for model_id, endpoint in self.config.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_id}")
            logger.info(f"Endpoint:   {endpoint}")
            logger.info(f"Examples:   {len(examples)}")
            logger.info(f"Mode:       {self.config.mode}")
            logger.info(f"{'='*60}")

            outputs = await self.evaluate_model(model_id, endpoint, examples)
            all_outputs[model_id] = outputs

        return all_outputs

    def save_outputs(
        self, all_outputs: dict[str, list[ModelOutput]], output_dir: Path
    ):
        """Save raw model outputs for later analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_id, outputs in all_outputs.items():
            safe_name = model_id.replace("/", "_").replace(" ", "_")
            path = output_dir / f"{safe_name}_outputs.jsonl"
            with open(path, "w") as f:
                for o in outputs:
                    f.write(json.dumps({
                        "pmid": o.pmid,
                        "model_id": o.model_id,
                        "raw_output": o.raw_output,
                        "latency_seconds": o.latency_seconds,
                        "input_tokens": o.input_tokens,
                        "output_tokens": o.output_tokens,
                        "error": o.error,
                    }) + "\n")
            logger.info(f"Saved {len(outputs)} outputs to {path}")


def load_model_outputs(path: Path) -> list[ModelOutput]:
    """Reload saved model outputs for re-analysis."""
    outputs = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            outputs.append(ModelOutput(**data))
    return outputs
