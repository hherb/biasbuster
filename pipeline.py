"""
Pipeline Orchestrator

Coordinates the full dataset building workflow:
1. Collect candidate abstracts from multiple sources
2. Enrich with heuristic analysis (effect size audit, author COI)
3. Pre-label with one or more LLMs (Claude, DeepSeek, etc.)
4. Export for human review
5. After human validation, export for fine-tuning
6. Compare model annotations against human ground truth

Usage:
    python pipeline.py --stage collect    # Stage 1: Collect raw abstracts
    python pipeline.py --stage enrich     # Stage 2: Heuristic enrichment
    python pipeline.py --stage annotate   # Stage 3: LLM pre-labelling (Claude)
    python pipeline.py --stage annotate --models anthropic,deepseek  # Both
    python pipeline.py --stage export     # Stage 4: Export for training
    python pipeline.py --stage compare    # Stage 5: Compare models vs human
    python pipeline.py --stage all        # Run stages 1-4
"""

import argparse
import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional, Union

import httpx

from config import Config
from utils.retry import fetch_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def stage_collect(config: Config) -> None:
    """Collect candidate abstracts from multiple sources.

    Sources:
    - Retraction Watch (via Crossref) -> known-biased positive examples
    - PubMed RCTs with effect sizes -> candidates for heuristic screening
    - Cochrane RoB assessments -> expert-validated bias labels
    """
    from collectors.retraction_watch import RetractionWatchCollector
    from collectors.cochrane_rob import CochraneRoBCollector

    output_dir = Path(config.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1a. Retracted papers (high-confidence positive examples)
    logger.info("=== Collecting retracted papers ===")
    retracted_path = output_dir / "retracted_papers.jsonl"
    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        await collector.collect_retracted_with_abstracts(
            max_papers=config.retraction_watch_max,
            output_path=retracted_path,
            flush_every=config.retraction_flush_every,
        )

    # 1b. PubMed RCTs for heuristic screening
    logger.info("=== Collecting RCT abstracts for screening ===")
    await collect_rcts_from_pubmed(config, output_dir / "rct_abstracts.jsonl")

    # 1c. Cochrane Risk of Bias assessments (expert ground truth)
    logger.info("=== Collecting Cochrane RoB assessments ===")
    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        assessments = await collector.collect_rob_dataset(
            domains=config.focus_domains[:5],
            max_reviews=config.cochrane_max_reviews,
            max_studies=config.cochrane_rob_max,
        )
        collector.save_results(assessments, str(output_dir / "cochrane_rob.jsonl"))

    logger.info(f"Collection complete. Raw data in {output_dir}")


async def collect_rcts_from_pubmed(config: Config, output_path: Path) -> None:
    """Search PubMed for recent RCTs in focus domains.

    Results are flushed to disk after each domain completes.
    Supports resuming: skips PMIDs already present in the output file.

    Args:
        config: Application configuration.
        output_path: Path to write the JSONL output.
    """
    from collectors.pubmed_xml import parse_pubmed_xml_batch

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint: find PMIDs already collected
    already_collected: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("pmid"):
                        already_collected.add(record["pmid"])
                except json.JSONDecodeError:
                    continue
        if already_collected:
            logger.info(
                f"Resuming: found {len(already_collected)} RCTs "
                f"already in {output_path}"
            )

    total_saved = len(already_collected)

    async with httpx.AsyncClient(timeout=config.http_timeout) as client:
        for domain in config.focus_domains:
            end_date = date.today().strftime("%Y/%m/%d")
            query = (
                f'"{domain}"[MeSH Terms] AND '
                f'"randomized controlled trial"[Publication Type] AND '
                f'"{config.pubmed_rct_start_date}"[Date - Publication] : '
                f'"{end_date}"[Date - Publication]'
            )

            params = {
                "db": "pubmed",
                "term": query,
                "retmax": min(500, config.spin_screening_max // len(config.focus_domains)),
                "retmode": "json",
                "sort": "date",
            }
            if config.ncbi_api_key:
                params["api_key"] = config.ncbi_api_key

            domain_articles: list[dict] = []
            try:
                resp = await fetch_with_retry(
                    client, "GET",
                    f"{config.pubmed_base}/esearch.fcgi",
                    params=params,
                    max_retries=config.max_retries,
                    base_delay=config.retry_base_delay,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    logger.info(f"Found {len(pmids)} RCTs for '{domain}'")

                    # Fetch abstracts in batches
                    batch_size = config.pubmed_fetch_batch
                    for i in range(0, len(pmids), batch_size):
                        batch = pmids[i : i + batch_size]
                        fetch_params = {
                            "db": "pubmed",
                            "id": ",".join(batch),
                            "rettype": "abstract",
                            "retmode": "xml",
                        }
                        if config.ncbi_api_key:
                            fetch_params["api_key"] = config.ncbi_api_key

                        fetch_resp = await fetch_with_retry(
                            client, "GET",
                            f"{config.pubmed_base}/efetch.fcgi",
                            params=fetch_params,
                            max_retries=config.max_retries,
                            base_delay=config.retry_base_delay,
                        )
                        if fetch_resp.status_code == 200:
                            articles = parse_pubmed_xml_batch(fetch_resp.text)
                            # Skip already-collected PMIDs (resume support)
                            new_articles = [
                                a for a in articles.values()
                                if a.get("pmid") not in already_collected
                            ]
                            domain_articles.extend(new_articles)

                        await asyncio.sleep(config.ncbi_rate_delay)

            except Exception as e:
                logger.warning(f"PubMed search failed for '{domain}': {e}")

            # Flush this domain's results to disk
            if domain_articles:
                with open(output_path, "a") as f:
                    for article in domain_articles:
                        f.write(json.dumps(article) + "\n")
                        already_collected.add(article.get("pmid", ""))
                total_saved += len(domain_articles)
                logger.info(
                    f"Flushed {len(domain_articles)} new articles for '{domain}' "
                    f"({total_saved} total so far)"
                )

            await asyncio.sleep(config.ncbi_rate_delay_slow)

    logger.info(f"Saved {total_saved} RCT abstracts to {output_path}")


async def stage_enrich(config: Config) -> None:
    """Enrich collected abstracts with heuristic analysis.

    Runs effect size audit, outcome switching detection, and copies
    retracted papers and Cochrane RoB data to the enriched directory.
    """
    from enrichers.effect_size_auditor import audit_abstract, ReportingPattern

    raw_dir = Path(config.raw_dir)
    output_dir = Path(config.enriched_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2a. Effect size audit on RCT abstracts
    logger.info("=== Running effect size audit ===")
    rct_path = raw_dir / "rct_abstracts.jsonl"
    if rct_path.exists():
        audited: list[dict] = []
        with open(rct_path) as f:
            for line in f:
                article = json.loads(line)
                audit = audit_abstract(
                    pmid=article.get("pmid") or "",
                    title=article.get("title") or "",
                    abstract=article.get("abstract") or "",
                )
                article["effect_size_audit"] = {
                    "pattern": audit.pattern.value,
                    "reporting_bias_score": audit.reporting_bias_score,
                    "relative_only": audit.pattern == ReportingPattern.RELATIVE_ONLY,
                    "flags": audit.flags,
                    "relative_measures": audit.relative_measures_found[:5],
                    "absolute_measures": audit.absolute_measures_found[:5],
                }
                audited.append(article)

        # Separate into high-suspicion and low-suspicion
        high_suspicion = [
            a for a in audited
            if a["effect_size_audit"]["reporting_bias_score"] >= config.high_suspicion_threshold
        ]
        low_suspicion = [
            a for a in audited
            if a["effect_size_audit"]["reporting_bias_score"] < config.low_suspicion_threshold
        ]

        with open(output_dir / "high_suspicion.jsonl", "w") as f:
            for a in high_suspicion:
                f.write(json.dumps(a) + "\n")

        with open(output_dir / "low_suspicion.jsonl", "w") as f:
            for a in low_suspicion:
                f.write(json.dumps(a) + "\n")

        logger.info(
            f"Effect size audit: {len(high_suspicion)} high-suspicion, "
            f"{len(low_suspicion)} low-suspicion out of {len(audited)} total"
        )

    # 2b. Outcome switching check via ClinicalTrials.gov
    logger.info("=== Checking for outcome switching ===")
    from collectors.clinicaltrials_gov import ClinicalTrialsGovCollector
    high_suspicion_path = output_dir / "high_suspicion.jsonl"
    if high_suspicion_path.exists():
        items: list[dict] = []
        with open(high_suspicion_path) as f:
            for line in f:
                items.append(json.loads(line))

        async with ClinicalTrialsGovCollector() as ctgov:
            for item in items[:config.outcome_switching_check_limit]:
                abstract = item.get("abstract", "")
                nct_id = await ctgov.extract_nct_from_abstract(abstract)
                if nct_id:
                    report = await ctgov.detect_outcome_switching(
                        nct_id=nct_id,
                        published_abstract=abstract,
                        published_title=item.get("title", ""),
                        pmid=item.get("pmid", ""),
                    )
                    item["outcome_switching"] = {
                        "nct_id": report.nct_id,
                        "primary_switched": report.primary_outcome_switched,
                        "outcomes_omitted": report.outcomes_omitted,
                        "sponsor": report.sponsor,
                        "sponsor_type": report.sponsor_type,
                        "evidence": report.evidence,
                    }
                await asyncio.sleep(config.ctgov_rate_delay)

        # Re-write with enriched data
        with open(high_suspicion_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Outcome switching check complete for {len(items)} items")

    # 2c. Retracted papers are pre-classified as high-suspicion
    retracted_path = raw_dir / "retracted_papers.jsonl"
    if retracted_path.exists():
        import shutil
        shutil.copy(retracted_path, output_dir / "retracted_papers.jsonl")
        logger.info("Retracted papers copied to enriched directory")

    # 2d. Cochrane RoB assessments copied to enriched
    cochrane_path = raw_dir / "cochrane_rob.jsonl"
    if cochrane_path.exists():
        import shutil
        shutil.copy(cochrane_path, output_dir / "cochrane_rob.jsonl")
        logger.info("Cochrane RoB assessments copied to enriched directory")


async def stage_annotate(config: Config, models: Optional[list[str]] = None) -> None:
    """Pre-label abstracts with one or more LLMs for structured bias assessment.

    Each model's outputs are always saved in a separate subdirectory under
    labelled_dir (e.g. labelled/anthropic/, labelled/deepseek/) for consistent
    layout regardless of how many models are used.

    Args:
        config: Application configuration.
        models: List of annotator backends to use. Defaults to ["anthropic"].
                Supported: "anthropic", "deepseek".
    """
    if models is None:
        models = ["anthropic"]

    enriched_dir = Path(config.enriched_dir)
    source_files = [
        "high_suspicion.jsonl",
        "retracted_papers.jsonl",
        "cochrane_rob.jsonl",
        "low_suspicion.jsonl",
    ]

    # Load items once (shared across models)
    all_items: dict[str, list[dict]] = {}
    for source_file in source_files:
        source_path = enriched_dir / source_file
        if not source_path.exists():
            continue
        items: list[dict] = []
        with open(source_path) as f:
            for line in f:
                data = json.loads(line)
                items.append({
                    "pmid": data.get("pmid", ""),
                    "title": data.get("title", ""),
                    "abstract": data.get("abstract", ""),
                    "metadata": data,
                })
        source_key = source_file.replace(".jsonl", "")
        max_items = config.annotation_max_per_source.get(source_key, 200)
        all_items[source_file] = items[:max_items]

    for model_name in models:
        annotator = _create_annotator(config, model_name)
        if annotator is None:
            continue

        # Always use subdirectories for consistent layout
        output_dir = Path(config.labelled_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"ANNOTATING WITH: {model_name}")
        logger.info(f"{'='*60}")

        async with annotator:
            for source_file, items in all_items.items():
                logger.info(f"=== Annotating {source_file} ({model_name}) ===")

                output_name = source_file.replace(".jsonl", "_annotated.jsonl")
                output_path = output_dir / output_name

                # annotate_batch handles incremental save and resume
                annotations = await annotator.annotate_batch(
                    items,
                    concurrency=config.annotation_concurrency,
                    delay=config.annotation_delay,
                    output_path=output_path,
                )

                # Merge original abstract text into annotations
                abstract_map = {item["pmid"]: item["abstract"] for item in items}
                source_key = source_file.replace(".jsonl", "")
                for ann in annotations:
                    ann["abstract_text"] = abstract_map.get(ann.get("pmid", ""), "")
                    ann["source"] = source_key

                # Write final complete file (with abstract_text and source merged)
                annotator.save_annotations(annotations, output_path)

                csv_name = source_file.replace(".jsonl", "_review.csv")
                annotator.generate_review_csv(annotations, output_dir / csv_name)


def _create_annotator(
    config: Config, model_name: str
) -> Optional[Union["LLMAnnotator", "OpenAICompatAnnotator"]]:
    """Factory: create an annotator instance by backend name.

    Args:
        config: Application configuration.
        model_name: Backend identifier ("anthropic" or "deepseek").

    Returns:
        Annotator instance, or None if the model name is unknown.
    """
    if model_name == "anthropic":
        from annotators.llm_prelabel import LLMAnnotator
        return LLMAnnotator(
            api_key=config.anthropic_api_key,
            model=config.annotation_model,
            max_tokens=config.annotation_max_tokens,
        )
    elif model_name == "deepseek":
        from annotators.openai_compat import OpenAICompatAnnotator
        return OpenAICompatAnnotator(
            api_key=config.deepseek_api_key,
            api_base=config.deepseek_api_base,
            model=config.deepseek_model,
            max_tokens=config.deepseek_max_tokens,
        )
    else:
        logger.error(f"Unknown annotator model: {model_name}")
        return None


async def stage_export(config: Config) -> None:
    """Export validated annotations for fine-tuning.

    Reads human-validated annotations from labelled_dir subdirectories
    and exports in training format with configurable train/val/test splits.
    """
    from export import export_dataset

    labelled_dir = Path(config.labelled_dir)
    output_dir = Path(config.export_dir)

    # Collect all validated annotations (search subdirectories too)
    all_annotations: list[dict] = []
    for path in labelled_dir.rglob("*_annotated.jsonl"):
        with open(path) as f:
            for line in f:
                ann = json.loads(line)
                # In production, filter to human_validated == True
                # For now, include all pre-labels
                all_annotations.append(ann)

    if not all_annotations:
        logger.warning("No annotations found. Run --stage annotate first.")
        return

    logger.info(f"Exporting {len(all_annotations)} annotations")

    # Export in multiple formats
    for output_fmt in ["alpaca", "sharegpt"]:
        export_dataset(
            all_annotations,
            output_dir=output_dir / output_fmt,
            fmt=output_fmt,
            include_thinking=True,
            train_split=config.train_split,
            val_split=config.val_split,
            seed=config.export_seed,
        )


async def stage_compare(config: Config) -> None:
    """Compare annotations from multiple models against human-validated labels.

    Reads annotations from labelled_dir/{model_name}/ subdirectories and
    human labels from labelled_dir/human/, then generates a comparison report
    using the evaluation framework.

    Expected directory layout:
        labelled/
            anthropic/     <- annotations from Claude
            deepseek/      <- annotations from DeepSeek
            human/         <- human-validated ground truth (copy from review CSVs)
    """
    from evaluation.scorer import parse_model_output, attach_ground_truth
    from evaluation.metrics import evaluate_model
    from evaluation.comparison import generate_comparison, save_report

    labelled_dir = Path(config.labelled_dir)
    output_dir = Path(config.output_dir) / "annotation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover model subdirectories
    model_dirs = [
        d for d in labelled_dir.iterdir()
        if d.is_dir() and d.name != "human"
    ]
    if not model_dirs:
        logger.warning(
            "No model subdirectories found in labelled_dir. "
            "Run --stage annotate --models anthropic,deepseek first."
        )
        return

    # Load human ground truth (if available)
    human_dir = labelled_dir / "human"
    gt_by_pmid: dict[str, dict] = {}
    has_human_gt = False
    if human_dir.exists():
        for path in human_dir.glob("*_annotated.jsonl"):
            with open(path) as f:
                for line in f:
                    ann = json.loads(line)
                    pmid = ann.get("pmid", "")
                    if pmid:
                        gt_by_pmid[pmid] = ann
        if gt_by_pmid:
            has_human_gt = True
            logger.info(f"Loaded {len(gt_by_pmid)} human ground truth labels")
        else:
            logger.warning(
                f"Human labels directory exists at {human_dir} but contains "
                "no annotated JSONL files."
            )

    if not has_human_gt:
        logger.warning(
            f"No human labels found at {human_dir}. "
            "Comparison will use the first model's annotations as reference. "
            "NOTE: Metrics will reflect inter-model agreement, NOT accuracy. "
            "For proper evaluation, create labelled/human/ with validated annotations."
        )

    # Load and score each model's annotations
    all_assessments = {}
    all_evaluations = {}

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        logger.info(f"\n{'='*40}")
        logger.info(f"Scoring annotations: {model_name}")
        logger.info(f"{'='*40}")

        # Load all annotations for this model
        annotations = []
        for path in model_dir.glob("*_annotated.jsonl"):
            with open(path) as f:
                for line in f:
                    annotations.append(json.loads(line))

        if not annotations:
            logger.warning(f"No annotations found for {model_name}")
            continue

        # If no human ground truth, use the first model as reference
        # (useful for inter-rater agreement, but NOT accuracy measurement)
        if not gt_by_pmid and not all_assessments:
            for ann in annotations:
                pmid = ann.get("pmid", "")
                if pmid:
                    gt_by_pmid[pmid] = ann
            logger.info(
                f"Using {model_name} as reference (no human labels available)"
            )

        # Parse each annotation and attach ground truth
        assessments = []
        for ann in annotations:
            pmid = ann.get("pmid", "")
            # Reconstruct raw_output as JSON for the scorer
            raw_json = json.dumps({
                k: v for k, v in ann.items()
                if k not in ("pmid", "title", "abstract_text", "source", "_annotation_model")
            })
            parsed = parse_model_output(
                raw_output=raw_json,
                pmid=pmid,
                model_id=model_name,
            )
            gt = gt_by_pmid.get(pmid)
            if gt:
                parsed = attach_ground_truth(parsed, gt)
            assessments.append(parsed)

        evaluation = evaluate_model(assessments, model_id=model_name)
        all_assessments[model_name] = assessments
        all_evaluations[model_name] = evaluation

        logger.info(f"Overall F1: {evaluation.overall_binary.f1:.3f}")
        logger.info(f"Overall kappa: {evaluation.overall_ordinal.weighted_kappa():.3f}")
        logger.info(f"Calibration Error: {evaluation.calibration_error:.3f}")

    # Generate comparison report
    if len(all_evaluations) >= 2:
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING ANNOTATION COMPARISON REPORT")
        logger.info("=" * 60)

        comparison_mode = (
            "annotation-comparison"
            if has_human_gt
            else "inter-model-agreement"
        )
        report = generate_comparison(
            evaluations=all_evaluations,
            assessments=all_assessments,
            mode=comparison_mode,
        )

        # Add caveat to summary when no human ground truth
        if not has_human_gt:
            caveat = (
                "\n\n> **NOTE:** No human ground truth was available. "
                "All metrics above reflect inter-model agreement, not "
                "accuracy against a validated reference. To measure true "
                "accuracy, place human-validated annotations in "
                "labelled/human/ and re-run."
            )
            report.summary += caveat

        save_report(report, output_dir, evaluations=all_evaluations)
        print("\n" + report.summary)
    elif len(all_evaluations) == 1:
        # Save single-model evaluation
        model_name = list(all_evaluations.keys())[0]
        eval_path = output_dir / f"{model_name}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(all_evaluations[model_name].to_dict(), f, indent=2)
        logger.info(f"Single model evaluation saved to {eval_path}")
    else:
        logger.warning("No models to compare.")


def main() -> None:
    """CLI entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Bias Detection Dataset Builder")
    parser.add_argument(
        "--stage",
        choices=["collect", "enrich", "annotate", "export", "compare", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of annotator models (e.g. anthropic,deepseek). "
             "Only used with --stage annotate. Default: anthropic",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (optional)",
    )
    args = parser.parse_args()

    config = Config()

    # Parse model list for annotation stage
    annotation_models = None
    if args.models:
        annotation_models = [m.strip() for m in args.models.split(",")]

    stages = {
        "collect": stage_collect,
        "enrich": stage_enrich,
        "annotate": lambda cfg: stage_annotate(cfg, models=annotation_models),
        "export": stage_export,
        "compare": stage_compare,
    }

    if args.stage == "all":
        async def run_all() -> None:
            for name in ["collect", "enrich", "annotate", "export"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"STAGE: {name.upper()}")
                logger.info(f"{'='*60}")
                await stages[name](config)
        asyncio.run(run_all())
    else:
        asyncio.run(stages[args.stage](config))


if __name__ == "__main__":
    main()
