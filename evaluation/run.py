#!/usr/bin/env python3
"""
Evaluation Runner

End-to-end CLI for running the bias detection model comparison.
Ties together: harness (inference) -> scorer (parsing) -> metrics -> comparison report.

Usage:
    # Run both models simultaneously (both servers must be up)
    python -m evaluation.run \
        --test-set dataset/export/alpaca/test.jsonl \
        --model-a qwen3.5-27b --endpoint-a http://localhost:8000 \
        --model-b olmo-3.1-32b --endpoint-b http://localhost:8001 \
        --mode zero-shot \
        --output eval_results/

    # Sequential mode: run one model at a time (single GPU)
    # Evaluates model A, then waits for model B server to appear
    python -m evaluation.run \
        --test-set dataset/export/alpaca/test.jsonl \
        --model-a qwen3.5-27b --endpoint-a http://localhost:8000 \
        --model-b olmo-3.1-32b --endpoint-b http://localhost:8000 \
        --mode zero-shot --sequential \
        --output eval_results/

    # Re-analyse from saved outputs (no inference needed)
    python -m evaluation.run \
        --reanalyse eval_results/ \
        --test-set dataset/export/alpaca/test.jsonl \
        --output eval_results/

    # Run a single model (useful for debugging)
    python -m evaluation.run \
        --test-set dataset/export/alpaca/test.jsonl \
        --model-a qwen3.5-27b --endpoint-a http://localhost:8000 \
        --mode zero-shot \
        --output eval_results/

Serving models on DGX Spark (run these first):
    # Terminal 1: Qwen3.5-27B
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen3.5-27B \\
        --port 8000 --tp-size 1 \\
        --mem-fraction-static 0.45 \\
        --context-length 4096 \\
        --reasoning-parser qwen3

    # Terminal 2: OLMo 3.1 32B Instruct
    python -m sglang.launch_server \\
        --model-path allenai/Olmo-3.1-32B-Instruct \\
        --port 8001 --tp-size 1 \\
        --mem-fraction-static 0.45 \\
        --context-length 4096

    Note: With 128GB unified memory on the Spark, you can serve one model
    at a time with full bf16, or both simultaneously with quantisation.
    Sequential evaluation (one model at a time) is recommended for
    reproducibility and to avoid memory pressure.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bias Detection Model Evaluation & Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Test set
    parser.add_argument(
        "--test-set", type=Path, required=True,
        help="Path to test set JSONL (annotated examples with ground truth)",
    )

    # Model A
    parser.add_argument("--model-a", type=str, help="Model A identifier")
    parser.add_argument("--endpoint-a", type=str, help="Model A OpenAI-compatible endpoint")

    # Model B
    parser.add_argument("--model-b", type=str, help="Model B identifier")
    parser.add_argument("--endpoint-b", type=str, help="Model B OpenAI-compatible endpoint")

    # Mode
    parser.add_argument(
        "--mode", choices=["zero-shot", "fine-tuned"], default="zero-shot",
        help="Evaluation mode (affects system prompt)",
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-concurrent", type=int, default=2)

    # Sequential mode (one model at a time, pause between)
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run models one at a time, pausing between for server swap "
             "(useful when GPU memory only fits one model)",
    )

    # Re-analysis from saved outputs
    parser.add_argument(
        "--reanalyse", type=Path, default=None,
        help="Re-analyse from saved outputs directory (skip inference)",
    )

    # Output
    parser.add_argument(
        "--output", type=Path, default=Path("eval_results"),
        help="Output directory for results",
    )

    return parser.parse_args()


def _outputs_to_dicts(all_outputs: dict) -> dict[str, list]:
    """Convert ModelOutput objects to dicts for downstream processing."""
    raw_outputs = {}
    for model_id, outputs in all_outputs.items():
        raw_outputs[model_id] = [
            {
                "pmid": o.pmid,
                "model_id": o.model_id,
                "raw_output": o.raw_output,
                "latency_seconds": o.latency_seconds,
                "input_tokens": o.input_tokens,
                "output_tokens": o.output_tokens,
                "error": o.error,
            }
            for o in outputs
        ]
    return raw_outputs


ENDPOINT_WAIT_TIMEOUT_SECONDS = 300
ENDPOINT_POLL_INTERVAL_SECONDS = 5
ENDPOINT_HEALTH_CHECK_TIMEOUT_SECONDS = 5


def _wait_for_endpoint(
    endpoint: str,
    model_id: str,
    timeout: int = ENDPOINT_WAIT_TIMEOUT_SECONDS,
):
    """Wait for an endpoint to become ready, polling periodically."""
    import time
    import httpx

    url = f"{endpoint.rstrip('/')}/v1/models"
    logger.info(f"Waiting for {model_id} at {endpoint} to become ready...")
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = httpx.get(url, timeout=ENDPOINT_HEALTH_CHECK_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                logger.info(f"{model_id} is ready at {endpoint}")
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(ENDPOINT_POLL_INTERVAL_SECONDS)
    logger.error(f"Timed out waiting for {model_id} at {endpoint} after {timeout}s")
    return False


async def _run_single_model(
    model_id: str,
    endpoint: str,
    examples: list,
    args,
) -> dict[str, list]:
    """Run inference for a single model and save outputs."""
    from .harness import EvalConfig, EvalHarness

    config = EvalConfig(
        models={model_id: endpoint},
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        mode=args.mode,
        output_dir=str(args.output),
    )

    async with EvalHarness(config) as harness:
        all_outputs = await harness.run_all(examples)
        harness.save_outputs(all_outputs, args.output)

    return _outputs_to_dicts(all_outputs)


async def run_inference(args) -> dict[str, list]:
    """Run inference through the harness."""
    from .harness import EvalConfig, EvalHarness, load_test_set

    # Build model list
    models = {}
    if args.model_a and args.endpoint_a:
        models[args.model_a] = args.endpoint_a
    if args.model_b and args.endpoint_b:
        models[args.model_b] = args.endpoint_b

    if not models:
        logger.error("No models configured. Provide --model-a/--endpoint-a and/or --model-b/--endpoint-b")
        sys.exit(1)

    # Load test set
    examples = load_test_set(args.test_set)
    logger.info(f"Loaded {len(examples)} test examples")

    # Sequential mode: run one model, pause, run the next
    if args.sequential and len(models) > 1:
        raw_outputs = {}
        model_items = list(models.items())

        for i, (model_id, endpoint) in enumerate(model_items):
            # Wait for endpoint to be reachable before starting
            if not _wait_for_endpoint(endpoint, model_id):
                logger.error(
                    f"Endpoint for {model_id} not reachable. "
                    f"Start the server at {endpoint} and re-run, or use "
                    f"--reanalyse to score already-saved outputs."
                )
                sys.exit(1)

            outputs = await _run_single_model(
                model_id, endpoint, examples, args,
            )
            raw_outputs.update(outputs)

            # Pause between models (not after the last one)
            if i < len(model_items) - 1:
                next_model, next_endpoint = model_items[i + 1]
                logger.info(f"\n{'='*60}")
                logger.info(f"Finished {model_id}. Next: {next_model}")
                logger.info(f"Stop the current server and start {next_model}")
                logger.info(f"  Expected endpoint: {next_endpoint}")
                logger.info(f"{'='*60}")
                logger.info(
                    "Waiting for server to appear (will poll every 5s, "
                    "timeout 5min)..."
                )

        return raw_outputs

    # Default: run all models concurrently (expects all endpoints up)
    config = EvalConfig(
        models=models,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        mode=args.mode,
        output_dir=str(args.output),
    )

    async with EvalHarness(config) as harness:
        all_outputs = await harness.run_all(examples)
        harness.save_outputs(all_outputs, args.output)

    return _outputs_to_dicts(all_outputs)


def load_saved_outputs(output_dir: Path) -> dict[str, list]:
    """Load previously saved model outputs for re-analysis."""
    raw_outputs = {}
    for path in output_dir.glob("*_outputs.jsonl"):
        model_id = path.stem.replace("_outputs", "")
        outputs = []
        with open(path) as f:
            for line in f:
                outputs.append(json.loads(line))
        raw_outputs[model_id] = outputs
        logger.info(f"Loaded {len(outputs)} saved outputs for {model_id}")
    return raw_outputs


def _assessment_to_annotation(parsed) -> dict:
    """Convert a ParsedAssessment into the annotation dict format used by the DB."""

    def _dim_dict(dim_score) -> dict:
        d = {"severity": dim_score.predicted_severity}
        d.update(dim_score.predicted_flags)
        return d

    return {
        "statistical_reporting": _dim_dict(parsed.statistical_reporting),
        "spin": _dim_dict(parsed.spin),
        "outcome_reporting": _dim_dict(parsed.outcome_reporting),
        "conflict_of_interest": _dim_dict(parsed.conflict_of_interest),
        "methodology": _dim_dict(parsed.methodology),
        "overall_severity": parsed.overall_severity,
        "overall_bias_probability": parsed.overall_bias_probability,
        "confidence": "eval_unrated",
        "recommended_verification_steps": parsed.verification_steps_mentioned,
        "reasoning": parsed.thinking_text or parsed.raw_output,
    }


def _store_assessments_in_db(
    all_assessments: dict[str, list],
    db_path: Optional[Path] = None,
):
    """Persist parsed eval assessments into the SQLite annotations table."""
    # Import from project root — evaluation/ is a subpackage
    import importlib
    Database = importlib.import_module("database").Database

    if db_path is None:
        try:
            Config = importlib.import_module("config").Config
            db_path = Path(Config().db_path)
        except Exception:
            db_path = Path("dataset/biasbuster.db")

    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}, skipping DB storage")
        return

    db = Database(str(db_path))

    # Ensure papers exist for all PMIDs (FK constraint)
    existing_pmids = {r["pmid"] for r in db.conn.execute(
        "SELECT pmid FROM papers"
    ).fetchall()}

    stored = 0
    skipped = 0
    fk_missing = 0
    for model_id, assessments in all_assessments.items():
        for parsed in assessments:
            if not parsed.pmid or not parsed.parse_success:
                skipped += 1
                continue
            if parsed.pmid not in existing_pmids:
                fk_missing += 1
                continue
            annotation = _assessment_to_annotation(parsed)
            db.upsert_annotation(parsed.pmid, model_id, annotation)
            stored += 1

    if fk_missing:
        logger.warning(
            f"{fk_missing} PMIDs not in papers table — run the collection "
            f"pipeline first or use --reanalyse with the matching database"
        )
    logger.info(f"Stored {stored} annotations in DB, skipped {skipped}")


def analyse_outputs(
    raw_outputs: dict[str, list],
    test_set_path: Path,
    mode: str,
    output_dir: Path,
):
    """Score outputs, compute metrics, generate comparison report."""
    from .harness import load_test_set
    from .scorer import parse_model_output, attach_ground_truth
    from .metrics import evaluate_model
    from .comparison import generate_comparison, save_report

    # Load test set for ground truth
    examples = load_test_set(test_set_path)
    gt_by_pmid = {ex.pmid: ex.ground_truth for ex in examples}

    # Parse and score each model's outputs
    all_assessments = {}
    all_evaluations = {}

    for model_id, outputs in raw_outputs.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Scoring: {model_id}")
        logger.info(f"{'='*40}")

        assessments = []
        for output in outputs:
            if output.get("error"):
                continue

            # Parse the raw model output
            parsed = parse_model_output(
                raw_output=output.get("raw_output", ""),
                pmid=output.get("pmid", ""),
                model_id=model_id,
            )

            # Attach ground truth labels
            gt = gt_by_pmid.get(output.get("pmid", ""), {})
            if gt:
                parsed = attach_ground_truth(parsed, gt)

            assessments.append(parsed)

        logger.info(f"Parsed {len(assessments)} assessments for {model_id}")
        logger.info(f"Parse failures: {sum(1 for a in assessments if not a.parse_success)}")

        # Compute metrics
        evaluation = evaluate_model(assessments, model_id=model_id)
        all_assessments[model_id] = assessments
        all_evaluations[model_id] = evaluation

        # Log headline metrics
        logger.info(f"Overall F1: {evaluation.overall_binary.f1:.3f}")
        logger.info(f"Overall κ:  {evaluation.overall_ordinal.weighted_kappa():.3f}")
        logger.info(f"Calibration Error: {evaluation.calibration_error:.3f}")
        logger.info(f"Verification Score: {evaluation.mean_verification_score:.3f}")

        for dim in ["statistical_reporting", "spin", "outcome_reporting",
                     "conflict_of_interest", "methodology"]:
            dim_eval = getattr(evaluation, dim)
            logger.info(f"  {dim}: F1={dim_eval.binary.f1:.3f}, κ={dim_eval.ordinal.weighted_kappa():.3f}")

    # Store assessments in SQLite (primary storage)
    _store_assessments_in_db(all_assessments)

    # Save individual model evaluations (secondary export)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, evaluation in all_evaluations.items():
        safe_name = model_id.replace("/", "_").replace(" ", "_")
        eval_path = output_dir / f"{safe_name}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation to {eval_path}")

    # Generate comparison report if we have 2+ models
    if len(all_evaluations) >= 2:
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPARISON REPORT")
        logger.info("=" * 60)

        report = generate_comparison(
            evaluations=all_evaluations,
            assessments=all_assessments,
            raw_outputs=raw_outputs,
            mode=mode,
        )
        save_report(report, output_dir, evaluations=all_evaluations)

        # Print the summary to stdout
        print("\n" + report.summary)
    else:
        logger.info("Single model evaluation complete (need 2+ for comparison)")


def main():
    args = parse_args()

    if args.reanalyse:
        # Re-analysis mode: load saved outputs
        logger.info(f"Re-analysing from saved outputs in {args.reanalyse}")
        raw_outputs = load_saved_outputs(args.reanalyse)
    else:
        # Inference mode: run models
        raw_outputs = asyncio.run(run_inference(args))

    # Analyse and generate report
    analyse_outputs(
        raw_outputs=raw_outputs,
        test_set_path=args.test_set,
        mode=args.mode,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
