#!/usr/bin/env python3
"""Import and annotate a single paper by PMID or DOI.

If the paper already exists in the database, skips straight to annotation.
If not, fetches it from PubMed, stores it, runs enrichment, then annotates.

Three annotation modes:
- abstract two-call (default): Stage 1 extracts facts from the abstract,
  Stage 2 assesses bias.
- abstract single-call (--single-call): legacy v1 prompt, one LLM call.
- full-text two-call (--full-text): fetch JATS via Europe PMC, chunk by
  section, run section-level extraction sequentially, merge, then assess.
  Stored under a distinct DB tag (``<model>_fulltext``) so the abstract
  annotation isn't overwritten.

Usage:
    uv run python annotate_single_paper.py --pmid 41271640
    uv run python annotate_single_paper.py --pmid 41271640 --model anthropic
    uv run python annotate_single_paper.py --pmid 41271640 --force
    uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001
    uv run python annotate_single_paper.py --doi 10.1016/j.example.2024.01.001 --source cochrane_rob
    uv run python annotate_single_paper.py --pmid 41750436 --model anthropic --full-text
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from biasbuster.annotators import is_retraction_notice
from biasbuster.cli.chunking import chunk_jats_article, chunk_plain_text
from biasbuster.cli.content import acquire_content
from biasbuster.cli.settings import load_config
from config import Config
from biasbuster.database import Database
from biasbuster.pipeline import create_annotator

logger = logging.getLogger(__name__)


async def resolve_pmid(
    doi: str, config: Config
) -> Optional[str]:
    """Convert a DOI to a PMID via the NCBI ID Converter API.

    Args:
        doi: The DOI to resolve.
        config: Application configuration (provides mailto, API key).

    Returns:
        PMID string, or None if the DOI could not be resolved.
    """
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector

    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        mapping = await collector.doi_to_pmid([doi])
    pmid = mapping.get(doi)
    if not pmid:
        logger.error(f"Could not resolve DOI {doi} to a PMID")
    return pmid


async def fetch_paper(
    pmid: str, config: Config
) -> Optional[dict]:
    """Fetch a paper from PubMed by PMID.

    Args:
        pmid: The PubMed ID to fetch.
        config: Application configuration (provides mailto, API key).

    Returns:
        Paper dict with title, abstract, authors, etc., or None on failure.
    """
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector

    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        paper = await collector.fetch_pubmed_abstract(pmid)
    if not paper:
        logger.error(f"Could not fetch PMID {pmid} from PubMed")
    return paper


def enrich_paper(
    pmid: str, title: str, abstract: str, db: Database, config: Config
) -> None:
    """Run heuristic enrichment (effect size audit) and store the result.

    Args:
        pmid: Paper PMID.
        title: Paper title.
        abstract: Paper abstract text.
        db: Database instance.
        config: Application configuration (provides suspicion thresholds).
    """
    from biasbuster.enrichers.effect_size_auditor import ReportingPattern, audit_abstract

    audit = audit_abstract(pmid=pmid, title=title, abstract=abstract)
    score = audit.reporting_bias_score

    if score >= config.high_suspicion_threshold:
        suspicion_level = "high"
    elif score < config.low_suspicion_threshold:
        suspicion_level = "low"
    else:
        suspicion_level = "medium"

    db.upsert_enrichment(pmid, {
        "suspicion_level": suspicion_level,
        "reporting_bias_score": score,
        "effect_size_audit": {
            "pattern": audit.pattern.value,
            "reporting_bias_score": score,
            "relative_only": audit.pattern == ReportingPattern.RELATIVE_ONLY,
            "flags": audit.flags,
            "relative_measures": audit.relative_measures_found[:5],
            "absolute_measures": audit.absolute_measures_found[:5],
        },
    })
    logger.info(
        f"Enrichment: suspicion={suspicion_level}, "
        f"reporting_bias_score={score:.3f}, flags={audit.flags}"
    )


def _fetch_sections_for_full_text(
    identifier: str,
) -> Optional[list[tuple[str, str]]]:
    """Acquire full text for *identifier* and return chunked (section, text) tuples.

    Uses the same path as the BiasBuster CLI: tries Europe PMC JATS first,
    then PMC PDF fallback. Returns None if no full text is available
    (caller should treat this as a hard failure when --full-text was
    requested explicitly).

    Args:
        identifier: PMID, DOI, or any value accepted by
                    ``biasbuster.cli.content.acquire_content``.

    Returns:
        List of (section_name, section_text) tuples ready to feed
        ``annotate_full_text_two_call``, or None on failure.
    """
    cli_config = load_config()
    try:
        content = acquire_content(identifier, cli_config)
    except Exception as exc:
        logger.error(f"Full-text acquisition failed for {identifier}: {exc!r}")
        return None

    if not content.has_fulltext:
        logger.error(
            f"Full text unavailable for {identifier} "
            f"(content_type={content.content_type}). "
            f"--full-text requires JATS or PDF, abstract-only is not enough."
        )
        return None

    if content.jats_article is not None:
        chunks = chunk_jats_article(
            content.jats_article, jats_xml=content.jats_xml,
        )
        logger.info(
            f"Acquired JATS full text: {len(chunks)} section chunks"
        )
    else:
        chunks = chunk_plain_text(content.plain_fulltext)
        logger.info(
            f"Acquired plain-text full text: {len(chunks)} window chunks"
        )

    return [(c.section, c.text) for c in chunks]


async def annotate_paper(
    pmid: str,
    paper: dict,
    db: Database,
    config: Config,
    model_name: str,
    force: bool = False,
    two_call: bool = True,
    full_text: bool = False,
    agentic: bool = False,
    decomposed: bool = False,
    identifier: Optional[str] = None,
) -> Optional[dict]:
    """Annotate a single paper and store the result.

    Five modes:
    - Abstract single-call (``two_call=False, full_text=False``): legacy v1.
    - Abstract two-call (``two_call=True, full_text=False``): default v3.
    - Full-text two-call (``full_text=True``): map-reduce v3 over JATS chunks.
    - Full-text agentic (``agentic=True``): v4 extraction + tool-calling
      assessment agent.
    - Full-text decomposed (``decomposed=True``): v5A extraction + Python
      mechanical assessment + per-domain LLM override calls. Result is
      stored under ``<model>_fulltext_decomposed``.

    The full-text result is saved under a distinct DB tag
    (``<model>_fulltext``) so it does not overwrite the abstract
    annotation, which lives under the bare ``<model>`` tag.
    The agentic result is saved under ``<model>_fulltext_agentic``.
    The decomposed result is saved under ``<model>_fulltext_decomposed``.

    Args:
        pmid: Paper PMID.
        paper: Full paper dict from the database (used for metadata even
               in full-text mode, since enrichment lives there).
        db: Database instance.
        config: Application configuration.
        model_name: Annotator backend ("anthropic" or "deepseek").
        force: If True, delete any existing annotation under this tag
               and re-annotate.
        two_call: If True (default), use v3 two-call pipeline.
        full_text: If True, fetch JATS and use the map-reduce path.
                   Requires ``two_call=True``.
        identifier: PMID or DOI used to fetch full text (only consulted
                    when ``full_text=True``). Defaults to *pmid* if not
                    explicitly provided.

    Returns:
        The annotation dict if successful, None otherwise.
    """
    if full_text and not two_call:
        logger.error(
            "full_text=True requires two_call=True "
            "(no v1 single-call full-text path is supported)"
        )
        return None

    annotator = create_annotator(config, model_name)
    if annotator is None:
        return None

    # Pipeline uses the backend name (e.g. "deepseek") as the DB key,
    # not the specific model string (e.g. "deepseek-reasoner"). Full-text
    # annotations get their own tag so they don't overwrite the abstract
    # version under the bare backend name. Agentic mode gets its own tag
    # so it doesn't overwrite the v3 full-text results either.
    if decomposed:
        db_model_name = f"{model_name}_fulltext_decomposed"
    elif agentic:
        db_model_name = f"{model_name}_fulltext_agentic"
    elif full_text:
        db_model_name = f"{model_name}_fulltext"
    else:
        db_model_name = model_name

    if db.has_annotation(pmid, db_model_name):
        if force:
            db.delete_annotation(pmid, db_model_name)
            logger.info(
                f"Deleted existing annotation for PMID {pmid}/{db_model_name}"
            )
        else:
            logger.info(
                f"PMID {pmid} already annotated by {db_model_name}, skipping. "
                f"Use --force to re-annotate."
            )
            existing = db.get_annotations(model_name=db_model_name, pmid=pmid)
            if existing:
                return existing[0]
            return None

    if agentic or decomposed or full_text:
        # Fetch + chunk *before* opening the annotator context so a
        # full-text acquisition failure doesn't waste an API connection.
        sections = _fetch_sections_for_full_text(identifier or pmid)
        if not sections:
            return None

        if agentic or decomposed:
            # Determine the assessor provider from the model name:
            # "anthropic" or any "anthropic:*" → provider="anthropic"
            # anything else (ollama, deepseek, etc.) → provider="bmlib"
            if model_name.startswith("anthropic"):
                agent_provider = "anthropic"
                agent_model = annotator.model
            else:
                agent_provider = "bmlib"
                agent_model = model_name

        if agentic:
            logger.info(
                f"Using v4 agentic assessment mode "
                f"(agent_provider={agent_provider}, agent_model={agent_model}, "
                f"{len(sections)} sections)"
            )
            async with annotator:
                result = await annotator.annotate_full_text_agentic(
                    pmid=pmid,
                    title=paper.get("title", ""),
                    sections=sections,
                    metadata=paper,
                    agent_provider=agent_provider,
                    agent_model=agent_model,
                )
        elif decomposed:
            logger.info(
                f"Using v5A decomposed assessment mode "
                f"(agent_provider={agent_provider}, agent_model={agent_model}, "
                f"{len(sections)} sections)"
            )
            async with annotator:
                result = await annotator.annotate_full_text_decomposed(
                    pmid=pmid,
                    title=paper.get("title", ""),
                    sections=sections,
                    metadata=paper,
                    agent_provider=agent_provider,
                    agent_model=agent_model,
                )
        else:
            logger.info(
                f"Using full-text two-call (v3) annotation mode "
                f"({len(sections)} sections)"
            )
            async with annotator:
                result = await annotator.annotate_full_text_two_call(
                    pmid=pmid,
                    title=paper.get("title", ""),
                    sections=sections,
                    metadata=paper,
                )
    else:
        annotate_fn = (
            annotator.annotate_abstract_two_call if two_call
            else annotator.annotate_abstract
        )
        mode_label = "two-call v3" if two_call else "single-call v1"
        logger.info(f"Using {mode_label} annotation mode (abstract)")
        async with annotator:
            result = await annotate_fn(
                pmid=pmid,
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                metadata=paper,
            )

    if result is None:
        logger.error(f"Annotation failed for PMID {pmid}")
        return None

    # Store — mirror the pipeline's save_annotation logic
    result["abstract_text"] = paper.get("abstract", "")
    result["source"] = paper.get("source", "manual_import")
    db.insert_annotation(pmid, db_model_name, result)
    logger.info(
        f"Annotation saved as {db_model_name}: "
        f"severity={result.get('overall_severity')}, "
        f"bias_prob={result.get('overall_bias_probability')}, "
        f"confidence={result.get('confidence')}"
    )
    return result


async def main() -> int:
    """Entry point: resolve identifier, fetch/enrich/annotate the paper.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Import and annotate a single paper by PMID or DOI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pmid", type=str, help="PubMed ID of the paper")
    group.add_argument("--doi", type=str, help="DOI of the paper")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek",
        help="Annotator backend. Built-in: 'anthropic', 'deepseek'. "
             "For local models via Ollama, pass a bmlib model string "
             "(e.g. 'ollama:gemma4:26b-a4b-it-q8_0'). Default: deepseek.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="manual_import",
        help="Source label for newly imported papers (default: manual_import)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if an annotation already exists",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single-call",
        action="store_true",
        help="Use single-call annotation (v1) instead of two-call (v3). "
             "Default is two-call: Stage 1 extracts facts, Stage 2 assesses bias. "
             "Mutually exclusive with --full-text.",
    )
    mode_group.add_argument(
        "--full-text",
        action="store_true",
        help="Annotate the paper's full text via the v3 map-reduce pipeline "
             "(per-section extraction → merge → assessment). Requires JATS or "
             "PDF availability via Europe PMC; errors out if only the abstract "
             "is reachable. Result is stored under the DB tag '<model>_fulltext' "
             "so it does not overwrite the abstract annotation.",
    )
    mode_group.add_argument(
        "--agentic",
        action="store_true",
        help="Use the v4 agentic assessment pipeline: extraction (identical "
             "to --full-text) followed by a tool-calling assessment agent that "
             "runs the mechanical rules in Python and lets the LLM apply "
             "contextual overrides. Requires Anthropic API key (Phase 2: "
             "Claude only). Result is stored under '<model>_fulltext_agentic'. "
             "See docs/two_step_approach/V4_AGENT_DESIGN.md for design details.",
    )
    mode_group.add_argument(
        "--decomposed",
        action="store_true",
        help="Use the v5A decomposed assessment pipeline: extraction "
             "(identical to --full-text) → Python mechanical assessment → "
             "one focused LLM call per elevated overridable domain → "
             "deterministic synthesis. Designed to make small local "
             "models (gemma4-26B, gpt-oss-20B) reliably produce "
             "contextual overrides by giving each call a single narrow "
             "task. Result is stored under '<model>_fulltext_decomposed'. "
             "See docs/three_step_approach/V5A_DECOMPOSED.md for design.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write full annotation JSON to this file. "
             "Prints to stdout if set to '-'. Default: no file output.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config()
    db = Database(config.db_path)
    db.initialize()

    try:
        # --- Resolve PMID ---
        pmid: Optional[str] = args.pmid
        if args.doi:
            logger.info(f"Resolving DOI {args.doi} to PMID...")
            pmid = await resolve_pmid(args.doi, config)
            if pmid is None:
                return 1
            logger.info(f"Resolved DOI {args.doi} -> PMID {pmid}")

        assert pmid is not None  # guaranteed by argparse

        # --- Fetch / verify paper in DB ---
        paper = db.get_paper(pmid)
        if paper:
            logger.info(
                f"Paper already in database: "
                f"source={paper.get('source')}, "
                f"title={paper.get('title', '')[:80]}"
            )
        else:
            logger.info(f"Paper not in database, fetching from PubMed...")
            paper = await fetch_paper(pmid, config)
            if paper is None:
                return 1
            paper["source"] = args.source
            db.insert_paper(paper)
            logger.info(
                f"Imported: title={paper.get('title', '')[:80]}"
            )

        # --- Validate abstract ---
        # Full-text modes (--full-text / --agentic / --decomposed) get
        # their content from cached JATS, so a missing DB abstract is
        # not a blocker for them. Abstract-only modes still require a
        # usable abstract.
        is_full_text_mode = args.full_text or args.decomposed or args.agentic
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        abstract_present = bool(abstract and abstract.strip())

        if not abstract_present and not is_full_text_mode:
            logger.error(
                f"PMID {pmid} has no abstract — cannot annotate"
            )
            return 1
        if not abstract_present and is_full_text_mode:
            logger.info(
                f"PMID {pmid} has no DB abstract; proceeding with cached full text"
            )
        if abstract_present and is_retraction_notice(title, abstract, paper):
            logger.error(
                f"PMID {pmid} is a bare retraction/withdrawal notice — "
                f"no assessable content"
            )
            return 1

        # --- Enrich ---
        enrich_paper(pmid, title, abstract, db, config)

        # Reload paper to include enrichment data in metadata
        paper = db.get_paper(pmid)
        assert paper is not None

        # --- Annotate ---
        # Pass the original identifier (PMID or DOI) so full-text mode
        # can use whichever the user gave us — acquire_content's cache
        # is keyed by id type.
        original_identifier = args.doi if args.doi else pmid
        result = await annotate_paper(
            pmid, paper, db, config, args.model,
            force=args.force,
            two_call=not args.single_call,
            full_text=args.full_text or args.agentic or args.decomposed,
            agentic=args.agentic,
            decomposed=args.decomposed,
            identifier=original_identifier,
        )
        if result is None:
            return 1

        # --- Output ---
        if args.output:
            import json
            formatted = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output == "-":
                print(formatted)
            else:
                with open(args.output, "w") as f:
                    f.write(formatted)
                logger.info(f"Annotation written to {args.output}")

        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
