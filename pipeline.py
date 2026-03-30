"""
Pipeline Orchestrator

Coordinates the full dataset building workflow:
1. Collect candidate abstracts from multiple sources
2. Enrich with heuristic analysis (effect size audit, author COI)
3. Pre-label with one or more LLMs (Claude, DeepSeek, etc.)
4. Export for human review
5. After human validation, export for fine-tuning
6. Compare model annotations against human ground truth

All data is stored in a SQLite database (dataset/biasbuster.db by default).

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
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional, Union

import httpx

from config import Config
from database import Database
from utils.retry import fetch_with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logging (httpx logs every request at INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def stage_collect(config: Config, db: Database) -> None:
    """Collect candidate abstracts from multiple sources.

    Sources:
    - Retraction Watch (via Crossref) -> known-biased positive examples
    - PubMed RCTs with effect sizes -> candidates for heuristic screening
    - Cochrane RoB assessments -> expert-validated bias labels
    """
    from collectors.retraction_watch import RetractionWatchCollector
    from collectors.cochrane_rob import (
        CochraneRoBCollector, rob_assessment_to_paper_dict,
    )

    # 1a. Retracted papers (high-confidence positive examples)
    logger.info("=== Collecting retracted papers ===")
    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        papers = await collector.collect_retracted_with_abstracts(
            max_papers=config.retraction_watch_max,
        )
        count = 0
        for paper in papers:
            paper_dict = asdict(paper)
            paper_dict["source"] = "retraction_watch"
            if db.insert_paper(paper_dict):
                count += 1
        logger.info(f"Inserted {count} new retracted papers")

    # 1b. PubMed RCTs for heuristic screening
    logger.info("=== Collecting RCT abstracts for screening ===")
    await collect_rcts_from_pubmed(config, db)

    # 1c. Cochrane Risk of Bias assessments (expert ground truth)
    logger.info("=== Collecting Cochrane RoB assessments ===")
    cochrane_saved = 0

    def save_cochrane(a) -> None:
        nonlocal cochrane_saved
        paper_dict = rob_assessment_to_paper_dict(a)
        if db.upsert_cochrane_paper(paper_dict):
            cochrane_saved += 1

    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
        llm_api_key=config.deepseek_api_key,
        llm_api_base=config.deepseek_api_base,
        llm_model=config.deepseek_model,
        llm_max_tokens=config.deepseek_max_tokens,
        max_retries=config.max_retries,
    ) as collector:
        await collector.collect_rob_dataset(
            domains=config.focus_domains[:5],
            max_reviews=config.cochrane_max_reviews,
            max_studies=config.cochrane_rob_max,
            on_result=save_cochrane,
        )
    logger.info(f"Saved {cochrane_saved} Cochrane RoB papers")

    stats = db.get_stats()
    logger.info(
        f"Collection complete. Total papers in DB: {stats['total_papers']}"
    )


async def stage_collect_rob(config: Config, db: Database) -> None:
    """Collect only Cochrane Risk of Bias assessments.

    Use this to expand RoB data without re-running the full collection
    pipeline (retraction watch + PubMed RCTs).  Fetches abstracts from
    PubMed immediately so the seed step can be skipped.

    Usage:
        python pipeline.py --stage collect-rob
    """
    from collectors.cochrane_rob import (
        CochraneRoBCollector, rob_assessment_to_paper_dict,
    )

    before = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob'"
    ).fetchone()[0]
    logger.info(f"Cochrane RoB papers before: {before}")

    logger.info("=== Collecting RoB assessments (regex + LLM fallback) ===")

    # Incremental save callback — each assessment is persisted immediately
    # so progress survives interruptions.  Uses upsert so domain ratings
    # and review metadata are updated for papers already in the DB.
    saved_count = 0

    def save_assessment(a) -> None:
        nonlocal saved_count
        paper_dict = rob_assessment_to_paper_dict(a)
        if db.upsert_cochrane_paper(paper_dict):
            saved_count += 1
            logger.info(
                f"  Saved PMID {a.pmid} (RoB={a.overall_rob}) — "
                f"total saved: {saved_count}"
            )

    async with CochraneRoBCollector(
        ncbi_api_key=config.ncbi_api_key,
        llm_api_key=config.deepseek_api_key,
        llm_api_base=config.deepseek_api_base,
        llm_model=config.deepseek_model,
        llm_max_tokens=config.deepseek_max_tokens,
        max_retries=config.max_retries,
    ) as collector:
        await collector.collect_rob_dataset(
            domains=config.focus_domains[:5],
            max_reviews=config.cochrane_max_reviews,
            max_studies=config.cochrane_rob_max,
            on_result=save_assessment,
        )

    logger.info(f"Saved {saved_count} Cochrane RoB papers")

    # Fetch abstracts from PubMed for newly inserted papers
    if saved_count > 0:
        logger.info("Fetching abstracts for new Cochrane papers...")
        from seed_database import fetch_missing_abstracts
        await fetch_missing_abstracts(config, db)

    after = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob'"
    ).fetchone()[0]
    with_abs = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob' "
        "AND abstract IS NOT NULL AND length(abstract) > 100"
    ).fetchone()[0]

    # RoB distribution
    rob_dist = db.conn.execute(
        "SELECT overall_rob, COUNT(*) FROM papers "
        "WHERE source = 'cochrane_rob' GROUP BY overall_rob"
    ).fetchall()

    logger.info(f"Cochrane RoB papers: {before} → {after} (+{after - before})")
    logger.info(f"  With abstracts: {with_abs}")
    for rob, cnt in rob_dist:
        logger.info(f"  {rob}: {cnt}")


async def collect_rcts_from_pubmed(config: Config, db: Database) -> None:
    """Search PubMed for recent RCTs in focus domains.

    Uses the database for checkpoint/resume: skips PMIDs already present.

    Args:
        config: Application configuration.
        db: Database instance.
    """
    from collectors.pubmed_xml import parse_pubmed_xml_batch

    already_collected = db.get_paper_pmids(source="pubmed_rct")
    if already_collected:
        logger.info(
            f"Resuming: found {len(already_collected)} RCTs already in DB"
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

            domain_count = 0
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
                            for article in articles.values():
                                pmid = article.get("pmid")
                                if pmid and pmid not in already_collected:
                                    article["source"] = "pubmed_rct"
                                    if db.insert_paper(article):
                                        domain_count += 1
                                        already_collected.add(pmid)

                        await asyncio.sleep(config.ncbi_rate_delay)

            except Exception as e:
                logger.warning(f"PubMed search failed for '{domain}': {e}")

            total_saved += domain_count
            if domain_count:
                logger.info(
                    f"Inserted {domain_count} new articles for '{domain}' "
                    f"({total_saved} total so far)"
                )

            await asyncio.sleep(config.ncbi_rate_delay_slow)

    logger.info(f"Total RCT abstracts in DB: {total_saved}")


async def stage_enrich(config: Config, db: Database) -> None:
    """Enrich collected abstracts with heuristic analysis.

    Runs effect size audit and outcome switching detection on RCT abstracts.
    Retracted papers and Cochrane RoB data don't need enrichment.
    """
    from enrichers.effect_size_auditor import audit_abstract, ReportingPattern

    # 2a. Effect size audit on RCT abstracts
    logger.info("=== Running effect size audit ===")
    rct_papers = db.get_papers(source="pubmed_rct")

    if not rct_papers:
        logger.info("No RCT papers found in DB, skipping enrichment")
        return

    high_count = 0
    low_count = 0
    for article in rct_papers:
        pmid = article["pmid"]
        audit = audit_abstract(
            pmid=pmid,
            title=article.get("title") or "",
            abstract=article.get("abstract") or "",
        )

        score = audit.reporting_bias_score
        effect_size_audit = {
            "pattern": audit.pattern.value,
            "reporting_bias_score": score,
            "relative_only": audit.pattern == ReportingPattern.RELATIVE_ONLY,
            "flags": audit.flags,
            "relative_measures": audit.relative_measures_found[:5],
            "absolute_measures": audit.absolute_measures_found[:5],
        }

        if score >= config.high_suspicion_threshold:
            suspicion_level = "high"
            high_count += 1
        elif score < config.low_suspicion_threshold:
            suspicion_level = "low"
            low_count += 1
        else:
            suspicion_level = "medium"

        # Defer commits for batch performance
        db.upsert_enrichment(pmid, {
            "suspicion_level": suspicion_level,
            "reporting_bias_score": score,
            "effect_size_audit": effect_size_audit,
        }, commit=False)

    db.commit()

    logger.info(
        f"Effect size audit: {high_count} high-suspicion, "
        f"{low_count} low-suspicion out of {len(rct_papers)} total"
    )

    # 2b. Outcome switching check via ClinicalTrials.gov
    logger.info("=== Checking for outcome switching ===")
    from collectors.clinicaltrials_gov import ClinicalTrialsGovCollector

    high_suspicion = db.get_enriched_papers(suspicion_level="high")
    if high_suspicion:
        async with ClinicalTrialsGovCollector() as ctgov:
            checked = 0
            for item in high_suspicion[:config.outcome_switching_check_limit]:
                abstract = item.get("abstract", "")
                nct_id = await ctgov.extract_nct_from_abstract(abstract)
                if nct_id:
                    report = await ctgov.detect_outcome_switching(
                        nct_id=nct_id,
                        published_abstract=abstract,
                        published_title=item.get("title", ""),
                        pmid=item.get("pmid", ""),
                    )
                    # Update enrichment with outcome switching data
                    enrichment = {
                        "suspicion_level": "high",
                        "reporting_bias_score": item.get("reporting_bias_score"),
                        "effect_size_audit": item.get("effect_size_audit"),
                        "outcome_switching": {
                            "nct_id": report.nct_id,
                            "primary_switched": report.primary_outcome_switched,
                            "outcomes_omitted": report.outcomes_omitted,
                            "sponsor": report.sponsor,
                            "sponsor_type": report.sponsor_type,
                            "evidence": report.evidence,
                        },
                    }
                    db.upsert_enrichment(item["pmid"], enrichment)
                    checked += 1
                await asyncio.sleep(config.ctgov_rate_delay)

        logger.info(f"Outcome switching check complete for {checked} items")


async def stage_annotate(
    config: Config, db: Database, models: Optional[list[str]] = None
) -> None:
    """Pre-label abstracts with one or more LLMs for structured bias assessment.

    Each model's annotations are stored in the annotations table with
    the model_name as part of the composite primary key.

    Args:
        config: Application configuration.
        db: Database instance.
        models: List of annotator backends to use. Defaults to ["anthropic"].
                Supported: "anthropic", "deepseek".
    """
    if models is None:
        models = ["anthropic"]

    # Source keys and their DB query parameters
    source_configs = [
        ("high_suspicion", "high_suspicion"),
        ("retracted_papers", "retracted_papers"),
        ("cochrane_rob", "cochrane_rob"),
        ("low_suspicion", "low_suspicion"),
    ]

    from annotators import is_retraction_notice

    # Load items once (shared across models)
    all_items: dict[str, list[dict]] = {}
    total_filtered = 0
    for source_key, db_source in source_configs:
        max_items = config.annotation_max_per_source.get(source_key, 200)
        papers = db.get_papers_by_source_for_annotation(
            db_source, limit=max_items
        )
        if papers:
            items = []
            for p in papers:
                title = p.get("title", "")
                abstract = p.get("abstract", "")
                # Skip bare retraction/withdrawal notices — no content to assess
                if is_retraction_notice(title, abstract, p):
                    total_filtered += 1
                    continue
                items.append({
                    "pmid": p.get("pmid", ""),
                    "title": title,
                    "abstract": abstract,
                    "metadata": p,
                })
            if items:
                all_items[source_key] = items

    if total_filtered:
        logger.info(
            f"Filtered {total_filtered} retraction/withdrawal notices "
            f"(no assessable content)"
        )

    for model_name in models:
        annotator = create_annotator(config, model_name)
        if annotator is None:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"ANNOTATING WITH: {model_name}")
        logger.info(f"{'='*60}")

        # Get already-annotated PMIDs for resume
        already_done = db.get_annotated_pmids(model_name)
        if already_done:
            logger.info(
                f"Resuming: {len(already_done)} already annotated "
                f"by {model_name}"
            )

        async with annotator:
            for source_key, items in all_items.items():
                logger.info(
                    f"=== Annotating {source_key} ({model_name}) ==="
                )

                abstract_map = {
                    item["pmid"]: item["abstract"] for item in items
                }
                new_count = 0

                # Callback saves each annotation to DB as it completes
                def save_annotation(ann, _source=source_key):
                    nonlocal new_count
                    pmid = ann.get("pmid", "")
                    if not pmid:
                        return
                    ann["abstract_text"] = abstract_map.get(pmid, "")
                    ann["source"] = _source
                    if db.insert_annotation(pmid, model_name, ann):
                        new_count += 1
                        already_done.add(pmid)

                await annotator.annotate_batch(
                    items,
                    concurrency=config.annotation_concurrency,
                    delay=config.annotation_delay,
                    already_done=already_done,
                    on_result=save_annotation,
                )

                logger.info(
                    f"Stored {new_count} new annotations for "
                    f"{source_key}/{model_name}"
                )


def create_annotator(
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


async def stage_export(
    config: Config, db: Database, export_dir: str | None = None,
) -> None:
    """Export validated annotations for fine-tuning.

    Reads annotations from the database and exports in training format
    with configurable train/val/test splits.

    Args:
        export_dir: Override export directory. If None, uses config.export_dir.
    """
    from export import export_dataset

    output_dir = Path(export_dir) if export_dir else Path(config.export_dir)

    all_annotations = db.get_all_annotations_for_export()

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


async def stage_compare(config: Config, db: Database) -> None:
    """Compare annotations from multiple models against human-validated labels.

    Reads annotations from the database and generates a comparison report.
    """
    from evaluation.scorer import parse_model_output, attach_ground_truth
    from evaluation.metrics import evaluate_model
    from evaluation.comparison import generate_comparison, save_report

    output_dir = Path(config.output_dir) / "annotation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover models
    model_names = [m for m in db.get_model_names() if m != "human"]
    if not model_names:
        logger.warning(
            "No model annotations found in DB. "
            "Run --stage annotate --models anthropic,deepseek first."
        )
        return

    # Load human ground truth (if available)
    human_annotations = db.get_annotations(model_name="human")
    gt_by_pmid: dict[str, dict] = {}
    has_human_gt = False
    if human_annotations:
        for ann in human_annotations:
            pmid = ann.get("pmid", "")
            if pmid:
                gt_by_pmid[pmid] = ann
        has_human_gt = True
        logger.info(f"Loaded {len(gt_by_pmid)} human ground truth labels")

    if not has_human_gt:
        logger.warning(
            "No human labels found in DB. "
            "Comparison will use the first model's annotations as reference. "
            "NOTE: Metrics will reflect inter-model agreement, NOT accuracy. "
            "For proper evaluation, add human annotations with model_name='human'."
        )

    # Load and score each model's annotations
    all_assessments = {}
    all_evaluations = {}

    for model_name in sorted(model_names):
        logger.info(f"\n{'='*40}")
        logger.info(f"Scoring annotations: {model_name}")
        logger.info(f"{'='*40}")

        annotations = db.get_annotations(model_name=model_name)

        if not annotations:
            logger.warning(f"No annotations found for {model_name}")
            continue

        # If no human ground truth, use the first model as reference
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
            raw_json = json.dumps({
                k: v for k, v in ann.items()
                if k not in (
                    "pmid", "title", "abstract_text", "source",
                    "_annotation_model", "model_name", "annotated_at",
                )
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

        if not has_human_gt:
            caveat = (
                "\n\n> **NOTE:** No human ground truth was available. "
                "All metrics above reflect inter-model agreement, not "
                "accuracy against a validated reference. To measure true "
                "accuracy, add human-validated annotations with "
                "model_name='human' and re-run."
            )
            report.summary += caveat

        save_report(report, output_dir, evaluations=all_evaluations)
        print("\n" + report.summary)
    elif len(all_evaluations) == 1:
        model_name = list(all_evaluations.keys())[0]
        eval_path = output_dir / f"{model_name}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(all_evaluations[model_name].to_dict(), f, indent=2)
        logger.info(f"Single model evaluation saved to {eval_path}")
    else:
        logger.warning("No models to compare.")


def _reset_undetectable_annotations(db: "Database", dry_run: bool = False) -> None:
    """Delete annotations for papers with abstract-undetectable retraction reasons.

    These papers (fabrication, fraud, manipulation, etc.) need to be
    re-annotated without retraction context so the LLM rates the abstract
    on its own merits.
    """
    from enrichers.retraction_classifier import classify_retraction

    retracted = db.get_retracted_paper_pmids_with_reasons()
    if not retracted:
        logger.info("No retracted papers found in database.")
        return

    undetectable_pmids = []
    for paper in retracted:
        reasons_raw = paper["retraction_reasons"]
        if isinstance(reasons_raw, str):
            try:
                reasons = json.loads(reasons_raw)
            except (json.JSONDecodeError, TypeError):
                reasons = []
        else:
            reasons = reasons_raw or []

        _floor, _category, detectable = classify_retraction(
            reasons, title=paper.get("title") or "",
        )
        if not detectable:
            undetectable_pmids.append(paper["pmid"])

    if not undetectable_pmids:
        logger.info("No abstract-undetectable retracted papers found.")
        return

    # Count existing annotations that would be deleted
    placeholders = ",".join("?" * len(undetectable_pmids))
    count = db.conn.execute(
        f"SELECT COUNT(*) FROM annotations WHERE pmid IN ({placeholders})",
        undetectable_pmids,
    ).fetchone()[0]

    if count == 0:
        print(f"\n{len(undetectable_pmids)} abstract-undetectable papers found, "
              "but none have annotations yet. Nothing to delete.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would delete {count} annotations for "
              f"{len(undetectable_pmids)} abstract-undetectable retracted papers.")
        print("Run without --dry-run to actually delete.")
        return

    print(f"\nAbout to delete {count} annotations for "
          f"{len(undetectable_pmids)} abstract-undetectable retracted papers.")
    response = input("Proceed? [y/N] ").strip().lower()
    if response not in ("y", "yes"):
        print("Aborted.")
        return

    deleted = db.delete_annotations_for_pmids(undetectable_pmids)
    logger.info(
        "Deleted %d annotations for %d abstract-undetectable retracted papers. "
        "Run --stage annotate to re-annotate them without retraction context.",
        deleted, len(undetectable_pmids),
    )
    print(f"Deleted {deleted} annotations.")
    print("Run the following to re-annotate:")
    print("  uv run python pipeline.py --stage annotate --models deepseek")


def main() -> None:
    """CLI entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Bias Detection Dataset Builder")
    parser.add_argument(
        "--stage",
        choices=["collect", "collect-rob", "seed", "enrich", "annotate", "export", "compare", "all"],
        default="all",
        help="Pipeline stage to run. collect-rob = Cochrane RoB only; "
             "seed = enrich RW reasons + fetch abstracts + clean notices",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of annotator models (e.g. anthropic,deepseek). "
             "Only used with --stage annotate. Default: anthropic",
    )
    parser.add_argument(
        "--reset-undetectable-annotations", action="store_true",
        help="Delete annotations for papers with abstract-undetectable "
             "retraction reasons (fabrication, fraud, etc.) so they can be "
             "re-annotated without retraction context. Run --stage annotate "
             "after this to re-annotate them.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes. "
             "Currently used with --reset-undetectable-annotations.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--export-dir", type=str, default=None,
        help="Override export directory (default: config.export_dir). "
             "Example: --export-dir dataset_V2/export",
    )
    args = parser.parse_args()

    config = Config()

    # Parse model list for annotation stage
    annotation_models = None
    if args.models:
        annotation_models = [m.strip() for m in args.models.split(",")]

    # Initialize database
    db = Database(config.db_path)
    db.initialize()

    async def stage_seed(cfg: Config, database: Database) -> None:
        """Seed step: enrich RW reasons, fetch missing abstracts, clean notices."""
        from seed_database import run_steps, ALL_STEPS, print_summary
        await run_steps(ALL_STEPS, cfg, database)
        print_summary(database)

    stages = {
        "collect": lambda cfg: stage_collect(cfg, db),
        "collect-rob": lambda cfg: stage_collect_rob(cfg, db),
        "seed": lambda cfg: stage_seed(cfg, db),
        "enrich": lambda cfg: stage_enrich(cfg, db),
        "annotate": lambda cfg: stage_annotate(cfg, db, models=annotation_models),
        "export": lambda cfg: stage_export(cfg, db, export_dir=args.export_dir),
        "compare": lambda cfg: stage_compare(cfg, db),
    }

    # Handle --reset-undetectable-annotations before running stages
    if args.reset_undetectable_annotations:
        _reset_undetectable_annotations(db, dry_run=args.dry_run)
        db.close()
        return

    try:
        if args.stage == "all":
            async def run_all() -> None:
                for name in ["collect", "seed", "enrich", "annotate", "export"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"STAGE: {name.upper()}")
                    logger.info(f"{'='*60}")
                    await stages[name](config)
            asyncio.run(run_all())
        else:
            asyncio.run(stages[args.stage](config))
    finally:
        db.close()


if __name__ == "__main__":
    main()
