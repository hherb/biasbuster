"""
Pipeline Orchestrator

Coordinates the full dataset building workflow:
1. Collect candidate abstracts from multiple sources
2. Enrich with heuristic analysis (effect size audit, author COI)
3. Pre-label with Claude API
4. Export for human review
5. After human validation, export for fine-tuning

Usage:
    python pipeline.py --stage collect    # Stage 1: Collect raw abstracts
    python pipeline.py --stage enrich     # Stage 2: Heuristic enrichment
    python pipeline.py --stage annotate   # Stage 3: LLM pre-labelling
    python pipeline.py --stage export     # Stage 4: Export for review/training
    python pipeline.py --stage all        # Run all stages
"""

import argparse
import asyncio
import json
import logging
from datetime import date
from pathlib import Path

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def stage_collect(config: Config):
    """
    Stage 1: Collect candidate abstracts from multiple sources.

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
            max_reviews=50,
            max_studies=config.cochrane_rob_max,
        )
        collector.save_results(assessments, str(output_dir / "cochrane_rob.jsonl"))

    logger.info(f"Collection complete. Raw data in {output_dir}")


async def collect_rcts_from_pubmed(config: Config, output_path: Path):
    """
    Search PubMed for recent RCTs in focus domains.
    These will be screened by the effect size auditor.
    Results are flushed to disk after each domain completes.
    Supports resuming: skips PMIDs already present in the output file.
    """
    import httpx
    from collectors.retraction_watch import parse_pubmed_xml_batch

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

    async with httpx.AsyncClient(timeout=30.0) as client:
        for domain in config.focus_domains:
            end_date = date.today().strftime("%Y/%m/%d")
            query = (
                f'"{domain}"[MeSH Terms] AND '
                f'"randomized controlled trial"[Publication Type] AND '
                f'"2020/01/01"[Date - Publication] : "{end_date}"[Date - Publication]'
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

            domain_articles = []
            try:
                resp = await client.get(
                    f"{config.pubmed_base}/esearch.fcgi", params=params
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pmids = data.get("esearchresult", {}).get("idlist", [])
                    logger.info(f"Found {len(pmids)} RCTs for '{domain}'")

                    # Fetch abstracts in batches
                    for i in range(0, len(pmids), 200):
                        batch = pmids[i : i + 200]
                        fetch_params = {
                            "db": "pubmed",
                            "id": ",".join(batch),
                            "rettype": "abstract",
                            "retmode": "xml",
                        }
                        if config.ncbi_api_key:
                            fetch_params["api_key"] = config.ncbi_api_key

                        fetch_resp = await client.get(
                            f"{config.pubmed_base}/efetch.fcgi",
                            params=fetch_params,
                        )
                        if fetch_resp.status_code == 200:
                            articles = parse_pubmed_xml_batch(fetch_resp.text)
                            # Skip already-collected PMIDs (resume support)
                            new_articles = [
                                a for a in articles.values()
                                if a.get("pmid") not in already_collected
                            ]
                            domain_articles.extend(new_articles)

                        await asyncio.sleep(0.35)

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

            await asyncio.sleep(0.5)

    logger.info(f"Saved {total_saved} RCT abstracts to {output_path}")


async def stage_enrich(config: Config):
    """
    Stage 2: Enrich collected abstracts with heuristic analysis.

    - Effect size audit (relative vs absolute reporting)
    - Author COI cross-referencing
    """
    from enrichers.effect_size_auditor import audit_abstract, ReportingPattern

    raw_dir = Path(config.raw_dir)
    output_dir = Path(config.enriched_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2a. Effect size audit on RCT abstracts
    logger.info("=== Running effect size audit ===")
    rct_path = raw_dir / "rct_abstracts.jsonl"
    if rct_path.exists():
        audited = []
        with open(rct_path) as f:
            for line in f:
                article = json.loads(line)
                audit = audit_abstract(
                    pmid=article.get("pmid", ""),
                    title=article.get("title", ""),
                    abstract=article.get("abstract", ""),
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
        high_suspicion = [a for a in audited if a["effect_size_audit"]["reporting_bias_score"] >= 0.3]
        low_suspicion = [a for a in audited if a["effect_size_audit"]["reporting_bias_score"] < 0.1]

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
        items = []
        with open(high_suspicion_path) as f:
            for line in f:
                items.append(json.loads(line))

        async with ClinicalTrialsGovCollector() as ctgov:
            for item in items[:100]:  # Limit to avoid excessive API calls
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
                await asyncio.sleep(0.5)

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


async def stage_annotate(config: Config):
    """
    Stage 3: Pre-label with Claude API.

    Sends high-suspicion and retracted abstracts for structured assessment.
    Low-suspicion abstracts are also annotated (as negative examples).
    """
    from annotators.llm_prelabel import LLMAnnotator

    enriched_dir = Path(config.enriched_dir)
    output_dir = Path(config.labelled_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    async with LLMAnnotator(
        api_key=config.anthropic_api_key,
        model=config.annotation_model,
    ) as annotator:
        for source_file in ["high_suspicion.jsonl", "retracted_papers.jsonl", "cochrane_rob.jsonl", "low_suspicion.jsonl"]:
            source_path = enriched_dir / source_file
            if not source_path.exists():
                continue

            logger.info(f"=== Annotating {source_file} ===")
            items = []
            with open(source_path) as f:
                for line in f:
                    data = json.loads(line)
                    items.append({
                        "pmid": data.get("pmid", ""),
                        "title": data.get("title", ""),
                        "abstract": data.get("abstract", ""),
                        "metadata": data,
                    })

            # Limit batch size for cost control
            max_per_source = {
                "high_suspicion.jsonl": 500,
                "retracted_papers.jsonl": 200,
                "cochrane_rob.jsonl": 300,
                "low_suspicion.jsonl": 300,  # Negative examples
            }
            items = items[: max_per_source.get(source_file, 200)]

            annotations = await annotator.annotate_batch(items)

            # Merge original abstract text into annotations
            abstract_map = {item["pmid"]: item["abstract"] for item in items}
            for ann in annotations:
                ann["abstract_text"] = abstract_map.get(ann.get("pmid", ""), "")
                ann["source"] = source_file.replace(".jsonl", "")

            output_name = source_file.replace(".jsonl", "_annotated.jsonl")
            annotator.save_annotations(annotations, output_dir / output_name)

            # Also generate CSV for human review
            csv_name = source_file.replace(".jsonl", "_review.csv")
            annotator.generate_review_csv(annotations, output_dir / csv_name)


async def stage_export(config: Config):
    """
    Stage 4: Export validated annotations for fine-tuning.

    Reads human-validated annotations and exports in training format.
    """
    from export import export_dataset

    labelled_dir = Path(config.labelled_dir)
    output_dir = Path(config.export_dir)

    # Collect all validated annotations
    all_annotations = []
    for path in labelled_dir.glob("*_annotated.jsonl"):
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
        )


def main():
    parser = argparse.ArgumentParser(description="Bias Detection Dataset Builder")
    parser.add_argument(
        "--stage",
        choices=["collect", "enrich", "annotate", "export", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (optional)",
    )
    args = parser.parse_args()

    config = Config()

    stages = {
        "collect": stage_collect,
        "enrich": stage_enrich,
        "annotate": stage_annotate,
        "export": stage_export,
    }

    if args.stage == "all":
        async def run_all():
            for name, stage_fn in stages.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"STAGE: {name.upper()}")
                logger.info(f"{'='*60}")
                await stage_fn(config)
        asyncio.run(run_all())
    else:
        asyncio.run(stages[args.stage](config))


if __name__ == "__main__":
    main()
