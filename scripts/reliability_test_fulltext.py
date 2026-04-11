#!/usr/bin/env python3
"""Run the full-text two-call pipeline N times on one paper and tag each
result distinctly so we can measure run-to-run determinism.

Why this exists
---------------
The full-text two-call path (``annotate_full_text_two_call``) runs
section-level extraction on each JATS chunk and merges the partial
extractions. Earlier runs on gpt-oss:120b produced non-deterministic
failures where one chunk's correct extraction silently became ``null``
in the merged output, even though the same chunk extracted correctly
when run in isolation. Commit ``66bca3c`` (sequential section
extraction + per-chunk partial persistence) was intended to fix that,
but we never empirically verified the fix.

This script runs the pipeline N times (default 3) back-to-back on a
single paper and saves each run under a distinct DB tag
(``<model>_fulltext_twocall_rel1``, ``_rel2``, ...). You can then:

1. Inspect whether key extracted fields are stable across runs
2. Score each run against a reference annotation
3. Catch any run that regresses even though the rest succeed

Typical usage::

    uv run python scripts/reliability_test_fulltext.py \\
        --pmid 41750436 \\
        --model "ollama:gpt-oss:120b" \\
        --runs 3

Each run's log is prefixed with ``[run N/M]`` so you can follow progress
across long sequential extractions. Results remain in the DB under the
run-numbered tags until a future invocation deletes them (the script
deletes its own target tags at the start of each run so repeated
invocations overwrite instead of appending).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from typing import Optional

from bmlib.llm import LLMClient

from biasbuster.annotators.bmlib_backend import BmlibAnnotator
from biasbuster.cli.chunking import chunk_jats_article, chunk_plain_text
from biasbuster.cli.content import acquire_content
from biasbuster.cli.settings import load_config
from biasbuster.database import Database
from biasbuster.annotators import is_retraction_notice

from annotate_single_paper import resolve_pmid, fetch_paper, enrich_paper
from config import Config as PipelineConfig

logger = logging.getLogger(__name__)


def _safe_model_tag(model: str) -> str:
    """Turn ``ollama:gpt-oss:120b`` into ``ollama_gpt-oss_120b``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


async def _ensure_paper_in_db(
    identifier: str,
    is_doi: bool,
    db: Database,
    pipeline_config: PipelineConfig,
) -> Optional[dict]:
    """Resolve the identifier and make sure the paper is in the DB.

    Mirrors the acquisition path in ``annotate_single_paper.main`` so the
    paper metadata (and enrichment) are available to the annotator even
    when the caller only supplies a PMID with no abstract in hand.
    """
    if is_doi:
        pmid = await resolve_pmid(identifier, pipeline_config)
        if pmid is None:
            return None
    else:
        pmid = identifier

    paper = db.get_paper(pmid)
    if paper is None:
        paper = await fetch_paper(pmid, pipeline_config)
        if paper is None:
            return None
        paper["source"] = "reliability_test"
        db.insert_paper(paper)

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    if not abstract.strip():
        logger.error(f"PMID {pmid} has no abstract — cannot annotate")
        return None
    if is_retraction_notice(title, abstract, paper):
        logger.error(f"PMID {pmid} is a retraction notice — skipping")
        return None

    enrich_paper(pmid, title, abstract, db, pipeline_config)
    return db.get_paper(pmid)


def _acquire_sections(identifier: str, cli_config) -> Optional[list[tuple[str, str]]]:
    """Fetch full text + chunk into (section, text) tuples.

    Returns None if the paper isn't available in full text. JATS preferred
    (semantic chunks), plain-text window fallback when only PDF is
    available.
    """
    try:
        content = acquire_content(identifier, cli_config)
    except Exception as exc:
        logger.error(f"Full-text acquisition failed for {identifier}: {exc!r}")
        return None
    if not content.has_fulltext:
        logger.error(
            f"Full text unavailable for {identifier} "
            f"(content_type={content.content_type})"
        )
        return None
    if content.jats_article is not None:
        chunks = chunk_jats_article(content.jats_article, jats_xml=content.jats_xml)
        logger.info(f"Acquired JATS: {len(chunks)} section chunks")
    else:
        chunks = chunk_plain_text(content.plain_fulltext)
        logger.info(f"Acquired plain text: {len(chunks)} window chunks")
    return [(c.section, c.text) for c in chunks]


async def run_once(
    run_number: int,
    total_runs: int,
    annotator: BmlibAnnotator,
    db: Database,
    paper: dict,
    sections: list[tuple[str, str]],
    safe_model: str,
    run_prefix: str,
) -> Optional[dict]:
    """Execute one full-text two-call run and save it under its tagged slot."""
    pmid = paper.get("pmid", "")
    title = paper.get("title", "")
    tag = f"{safe_model}_{run_prefix}{run_number}"
    logger.info(f"[run {run_number}/{total_runs}] starting: tag={tag}")

    # Start each run from a clean slot so re-invocations don't stack
    db.delete_annotation(pmid, tag)

    t0 = time.time()
    result = await annotator.annotate_full_text_two_call(
        pmid=pmid,
        title=title,
        sections=sections,
        metadata=paper,
    )
    elapsed = time.time() - t0

    if result is None:
        logger.error(f"[run {run_number}/{total_runs}] FAILED after {elapsed:.0f}s")
        return None

    result["abstract_text"] = paper.get("abstract", "")
    result["source"] = paper.get("source", "reliability_test")
    db.insert_annotation(pmid, tag, result)

    # One-line summary of the key extracted facts for at-a-glance inspection
    ext = result.get("extraction", {}) if isinstance(result.get("extraction"), dict) else {}
    sample = ext.get("sample", {}) if isinstance(ext.get("sample"), dict) else {}
    n_rand = sample.get("n_randomised")
    n_anal = sample.get("n_analysed")
    attr_stated = sample.get("attrition_stated")
    attrition_ok = (n_rand is not None) and (n_anal is not None) and attr_stated

    logger.info(
        f"[run {run_number}/{total_runs}] done in {elapsed:.0f}s: "
        f"overall={result.get('overall_severity')} "
        f"prob={result.get('overall_bias_probability')} "
        f"n_rand={n_rand} n_anal={n_anal} attrition_stated={attr_stated} "
        f"{'✓' if attrition_ok else '✗'}"
    )
    logger.info(f"[run {run_number}/{total_runs}] saved as {tag}")
    return result


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    identifier: str
    is_doi: bool
    if args.pmid:
        identifier, is_doi = args.pmid, False
    elif args.doi:
        identifier, is_doi = args.doi, True
    else:
        logger.error("Provide either --pmid or --doi")
        return 1

    pipeline_config = PipelineConfig()
    db = Database(pipeline_config.db_path)
    db.initialize()

    cli_config = load_config(cli_model=args.model)
    safe_model = _safe_model_tag(cli_config.model)

    paper = await _ensure_paper_in_db(identifier, is_doi, db, pipeline_config)
    if paper is None:
        logger.error("Paper acquisition failed")
        db.close()
        return 1

    sections = _acquire_sections(identifier, cli_config)
    if sections is None:
        logger.error("Full-text acquisition failed — nothing to run")
        db.close()
        return 1

    llm = LLMClient(
        default_provider=cli_config.provider,
        ollama_host=cli_config.ollama_endpoint,
        anthropic_api_key=cli_config.anthropic_api_key,
        api_key=cli_config.deepseek_api_key or cli_config.anthropic_api_key,
        base_url=cli_config.deepseek_base if cli_config.provider == "deepseek" else None,
    )
    extra_kwargs: dict = {}
    if cli_config.provider == "ollama":
        extra_kwargs["think"] = False
    annotator = BmlibAnnotator(
        client=llm,
        model=cli_config.model,
        temperature=cli_config.temperature,
        max_tokens=cli_config.max_tokens,
        extra_chat_kwargs=extra_kwargs,
    )

    logger.info(
        f"Reliability test: model={cli_config.model}, "
        f"runs={args.runs}, paper={paper.get('pmid')}, "
        f"sections={len(sections)}"
    )

    results = []
    async with annotator:
        for i in range(1, args.runs + 1):
            res = await run_once(
                run_number=i, total_runs=args.runs,
                annotator=annotator, db=db, paper=paper, sections=sections,
                safe_model=safe_model, run_prefix=args.run_prefix,
            )
            results.append(res)

    successes = sum(1 for r in results if r is not None)
    logger.info(
        f"Reliability test complete: {successes}/{args.runs} runs succeeded"
    )
    db.close()
    return 0 if successes == args.runs else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full-text two-call N times on one paper to "
                    "measure run-to-run determinism",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pmid", type=str, help="PubMed ID of the paper")
    group.add_argument("--doi", type=str, help="DOI of the paper")
    parser.add_argument(
        "--model", type=str, required=True,
        help='Model in "provider:model" form, e.g. "ollama:gpt-oss:120b"',
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of repeat runs (default: 3)",
    )
    parser.add_argument(
        "--run-prefix", type=str, default="fulltext_twocall_rel",
        help="DB tag suffix template; each run gets <prefix><N>",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
