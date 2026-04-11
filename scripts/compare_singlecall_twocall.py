#!/usr/bin/env python3
"""Compare v1 single-call vs v3 two-call annotation on the same paper.

The purpose of the two-call pipeline is to improve *small-model*
performance by splitting extraction from judgment. This script runs both
modes on the same paper with the same local model so you can eyeball
whether the two-call approach closes the gap on a known failure case.

The typical workflow is:

    uv run python scripts/compare_singlecall_twocall.py \
        --pmid 41750436 --model "ollama:gpt-oss:120b"

    # or multiple PMIDs at once:
    uv run python scripts/compare_singlecall_twocall.py \
        --pmids 41750436,12345678 --model "ollama:gpt-oss:120b"

    # or by DOI:
    uv run python scripts/compare_singlecall_twocall.py \
        --doi 10.3390/antibiotics15020138 --model "ollama:gpt-oss:120b"

The script:
1. Resolves/fetches/enriches each paper using the same path as
   ``annotate_single_paper.py`` (so the prompts see identical context).
2. Runs ``annotate_abstract`` (v1 single-call) with the old prompt.
3. Runs ``annotate_abstract_two_call`` (v3 two-call) with the new prompts.
4. Saves both annotations to the DB under tagged model names:
   ``<safe_model>_singlecall`` and ``<safe_model>_twocall``.
5. Prints a side-by-side diff of the key fields.
6. Optionally writes the full annotations to a JSON file with ``--output``.

Both annotations go through BmlibAnnotator and therefore use identical
transport, temperature, retry logic, and JSON parsing — the only
difference is the prompt architecture.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from bmlib.llm import LLMClient

from biasbuster.annotators.bmlib_backend import BmlibAnnotator
from biasbuster.cli.settings import load_config
from biasbuster.database import Database

# These utilities are re-used from the existing single-paper CLI so
# paper acquisition, enrichment, and DB storage behave identically.
from annotate_single_paper import (
    resolve_pmid,
    fetch_paper,
    enrich_paper,
)
from config import Config as PipelineConfig
from biasbuster.annotators import is_retraction_notice

logger = logging.getLogger(__name__)


def _safe_model_tag(model: str) -> str:
    """Turn a bmlib model string into a DB-safe tag.

    Example: "ollama:gpt-oss:120b" -> "ollama_gpt-oss_120b"
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


async def _ensure_paper_in_db(
    identifier: str,
    is_doi: bool,
    db: Database,
    pipeline_config: PipelineConfig,
    source_tag: str,
) -> Optional[dict]:
    """Resolve → fetch → enrich → return the paper dict from the DB.

    Mirrors the acquisition logic in ``annotate_single_paper.main`` but
    returns the paper dict instead of running a single annotation. Returns
    None on any failure.
    """
    if is_doi:
        logger.info(f"Resolving DOI {identifier}...")
        pmid = await resolve_pmid(identifier, pipeline_config)
        if pmid is None:
            return None
        logger.info(f"Resolved {identifier} -> PMID {pmid}")
    else:
        pmid = identifier

    paper = db.get_paper(pmid)
    if paper is None:
        logger.info(f"Paper {pmid} not in database, fetching from PubMed...")
        paper = await fetch_paper(pmid, pipeline_config)
        if paper is None:
            return None
        paper["source"] = source_tag
        db.insert_paper(paper)

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if not abstract or not abstract.strip():
        logger.error(f"PMID {pmid} has no abstract — cannot annotate")
        return None
    if is_retraction_notice(title, abstract, paper):
        logger.error(f"PMID {pmid} is a retraction notice — skipping")
        return None

    enrich_paper(pmid, title, abstract, db, pipeline_config)
    # reload to pick up the enrichment metadata
    return db.get_paper(pmid)


def _severity_row(label: str, ann: Optional[dict]) -> str:
    """Format one row of the comparison table for a single annotation."""
    if ann is None:
        return f"  {label:<22} FAILED"

    def dom(k: str) -> str:
        v = ann.get(k, {})
        if not isinstance(v, dict):
            return "?"
        return str(v.get("severity", "?")).upper()

    return (
        f"  {label:<22} "
        f"overall={ann.get('overall_severity', '?').upper():<8} "
        f"prob={ann.get('overall_bias_probability', 0.0):.2f}  "
        f"stat={dom('statistical_reporting'):<8} "
        f"spin={dom('spin'):<8} "
        f"out={dom('outcome_reporting'):<8} "
        f"coi={dom('conflict_of_interest'):<8} "
        f"meth={dom('methodology'):<8}"
    )


def _key_flag_diff(ann: Optional[dict]) -> list[str]:
    """Extract the methodology flags that distinguish good from bad analysis."""
    if ann is None:
        return ["  (none — annotation failed)"]
    meth = ann.get("methodology", {}) if isinstance(ann.get("methodology"), dict) else {}
    coi = ann.get("conflict_of_interest", {}) if isinstance(ann.get("conflict_of_interest"), dict) else {}
    stat = ann.get("statistical_reporting", {}) if isinstance(ann.get("statistical_reporting"), dict) else {}
    flags_of_interest = [
        ("methodology.high_attrition", meth.get("high_attrition")),
        ("methodology.differential_attrition", meth.get("differential_attrition")),
        ("methodology.per_protocol_only", meth.get("per_protocol_only")),
        ("methodology.inadequate_sample_size", meth.get("inadequate_sample_size")),
        ("methodology.no_multiplicity_correction", meth.get("no_multiplicity_correction")),
        ("methodology.analytical_flexibility", meth.get("analytical_flexibility")),
        ("coi.sponsor_controls_analysis", coi.get("sponsor_controls_analysis")),
        ("coi.sponsor_controls_manuscript", coi.get("sponsor_controls_manuscript")),
        ("stat.selective_p_values", stat.get("selective_p_values")),
        ("stat.subgroup_emphasis", stat.get("subgroup_emphasis")),
    ]
    return [f"  {name}: {value!r}" for name, value in flags_of_interest]


def _print_comparison(pmid: str, title: str, single: Optional[dict], two: Optional[dict]) -> None:
    print()
    print("=" * 88)
    print(f"PMID {pmid}: {title[:70]}")
    print("=" * 88)
    print("Severity summary:")
    print(_severity_row("single-call (v1)", single))
    print(_severity_row("two-call  (v3)", two))
    print()
    print("Key methodology / COI flags (these are what the local model")
    print("tended to miss in single-call mode per docs/two_step_approach):")
    print("  single-call (v1):")
    for line in _key_flag_diff(single):
        print("  " + line)
    print("  two-call (v3):")
    for line in _key_flag_diff(two):
        print("  " + line)

    # Reasoning snippet (first 400 chars) to see whether the two-call
    # version produces more grounded reasoning
    print()
    print("Reasoning (truncated to 400 chars):")
    if single:
        print(f"  v1: {str(single.get('reasoning', ''))[:400]}")
    if two:
        print(f"  v3: {str(two.get('reasoning', ''))[:400]}")
    print()


async def _run_pair(
    annotator: BmlibAnnotator,
    db: Database,
    paper: dict,
    model_tag: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """Run both single-call and two-call on the same paper, save to DB."""
    pmid = paper.get("pmid", "")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Single-call v1 (uses ANNOTATION_SYSTEM_PROMPT from biasbuster.prompts)
    logger.info(f"[{pmid}] running single-call v1...")
    single = await annotator.annotate_abstract(
        pmid=pmid, title=title, abstract=abstract, metadata=paper,
    )
    if single is not None:
        single["abstract_text"] = abstract
        single["source"] = paper.get("source", "comparison_experiment")
        single["_annotation_mode"] = "single_call_v1"
        single_tag = f"{model_tag}_singlecall"
        # Delete any stale result from a prior run so --force semantics apply
        db.delete_annotation(pmid, single_tag)
        db.insert_annotation(pmid, single_tag, single)
        logger.info(
            f"[{pmid}] v1 single-call saved as {single_tag}: "
            f"overall={single.get('overall_severity')}, "
            f"prob={single.get('overall_bias_probability')}"
        )
    else:
        logger.error(f"[{pmid}] single-call v1 FAILED")

    # Two-call v3 (uses EXTRACTION_SYSTEM_PROMPT then ASSESSMENT_SYSTEM_PROMPT)
    logger.info(f"[{pmid}] running two-call v3...")
    two = await annotator.annotate_abstract_two_call(
        pmid=pmid, title=title, abstract=abstract, metadata=paper,
    )
    if two is not None:
        two["abstract_text"] = abstract
        two["source"] = paper.get("source", "comparison_experiment")
        two_tag = f"{model_tag}_twocall"
        db.delete_annotation(pmid, two_tag)
        db.insert_annotation(pmid, two_tag, two)
        logger.info(
            f"[{pmid}] v3 two-call saved as {two_tag}: "
            f"overall={two.get('overall_severity')}, "
            f"prob={two.get('overall_bias_probability')}"
        )
    else:
        logger.error(f"[{pmid}] two-call v3 FAILED")

    return single, two


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve the list of (identifier, is_doi) tuples to process
    targets: list[tuple[str, bool]] = []
    if args.pmid:
        targets.append((args.pmid, False))
    if args.pmids:
        for p in args.pmids.split(","):
            p = p.strip()
            if p:
                targets.append((p, False))
    if args.doi:
        targets.append((args.doi, True))
    if not targets:
        logger.error("Provide at least one --pmid, --pmids, or --doi")
        return 1

    # Pipeline config for paper acquisition (PubMed fetch, enrichment)
    pipeline_config = PipelineConfig()
    db = Database(pipeline_config.db_path)
    db.initialize()

    # CLI config for the LLM client (Ollama endpoint, temperature, etc.)
    cli_config = load_config(cli_model=args.model)

    # bmlib LLMClient shared across all runs
    llm = LLMClient(
        default_provider=cli_config.provider,
        ollama_host=cli_config.ollama_endpoint,
        anthropic_api_key=cli_config.anthropic_api_key,
        api_key=cli_config.deepseek_api_key or cli_config.anthropic_api_key,
        base_url=cli_config.deepseek_base if cli_config.provider == "deepseek" else None,
    )
    extra_kwargs: dict = {}
    if cli_config.provider == "ollama":
        # Extended thinking wastes token budget on extraction; disable it
        # per the rationale in cli/analysis.py.
        extra_kwargs["think"] = False
    annotator = BmlibAnnotator(
        client=llm,
        model=cli_config.model,
        temperature=cli_config.temperature,
        max_tokens=cli_config.max_tokens,
        extra_chat_kwargs=extra_kwargs,
    )

    model_tag = _safe_model_tag(cli_config.model)
    logger.info(f"Using model: {cli_config.model}")
    logger.info(f"Saving results under tags: {model_tag}_singlecall, {model_tag}_twocall")

    all_results: list[dict] = []

    try:
        async with annotator:
            for identifier, is_doi in targets:
                paper = await _ensure_paper_in_db(
                    identifier, is_doi, db, pipeline_config, source_tag=args.source,
                )
                if paper is None:
                    logger.error(f"Skipping {identifier}: acquisition failed")
                    continue

                pmid = paper.get("pmid", "")
                title = paper.get("title", "")
                single, two = await _run_pair(annotator, db, paper, model_tag)
                _print_comparison(pmid, title, single, two)
                all_results.append({
                    "pmid": pmid,
                    "title": title,
                    "single_call_v1": single,
                    "two_call_v3": two,
                })
    finally:
        db.close()

    if args.output and all_results:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        logger.info(f"Wrote full comparison to {out_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare v1 single-call vs v3 two-call annotation on the same paper",
    )
    parser.add_argument(
        "--pmid", type=str, help="Single PMID to compare"
    )
    parser.add_argument(
        "--pmids", type=str, help="Comma-separated list of PMIDs"
    )
    parser.add_argument(
        "--doi", type=str, help="DOI to resolve and compare"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help='Model in "provider:model" form, e.g. "ollama:gpt-oss:120b"',
    )
    parser.add_argument(
        "--source", type=str, default="comparison_experiment",
        help="Source tag for any newly imported papers",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write full annotation JSON for all papers to this file",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
