#!/usr/bin/env python3
"""Compare v1 single-call vs v3 two-call annotation on abstracts and full text.

Runs up to four annotation modes on the same paper with the same local
model via BmlibAnnotator, saves each to the DB under a distinct tag, and
prints a side-by-side comparison against Claude's annotation (if present)
or each mode's pairwise agreement.

Modes:
- a1 = abstract + single-call (v1 prompt)
- a2 = abstract + two-call   (v3 prompts, extraction → assessment)
- f1 = full text + single-call (naive baseline — whole paper stuffed into
       the v1 prompt; may truncate on small context windows)
- f2 = full text + two-call   (v3 map-reduce: per-section extraction →
       merge → assessment — the full paper-level architecture)

Without --full-text, only a1 and a2 run. With --full-text, the script
fetches the paper via the CLI's acquire_content() path (Europe PMC JATS
preferred, PDF fallback), chunks it via the CLI's chunking utilities,
and runs all four modes. Modes fall back gracefully: if full text cannot
be retrieved, only a1 and a2 run and this is reported.

DB tagging scheme:
    <safe_model>_abstract_singlecall   # a1
    <safe_model>_abstract_twocall      # a2
    <safe_model>_fulltext_singlecall   # f1  (only written if --full-text
                                       #       succeeds and f1 completes)
    <safe_model>_fulltext_twocall      # f2  (same)

On first run with a new version of this script, legacy tags from earlier
experiments (<safe_model>_singlecall, <safe_model>_twocall — without the
_abstract_ qualifier) are automatically renamed to the new scheme so the
DB stays consistent.

Typical workflow:
    uv run python scripts/compare_singlecall_twocall.py \\
        --pmid 41750436 --model "ollama:gpt-oss:120b" --full-text \\
        -o comparison_seed_health_fulltext.json
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
from biasbuster.cli.chunking import chunk_jats_article, chunk_plain_text
from biasbuster.cli.content import AcquiredContent, acquire_content
from biasbuster.cli.settings import load_config
from biasbuster.database import Database
from biasbuster.annotators import is_retraction_notice

# Re-used from the existing single-paper CLI so paper acquisition,
# enrichment, and DB storage match the training pipeline.
from annotate_single_paper import (
    resolve_pmid,
    fetch_paper,
    enrich_paper,
)
from config import Config as PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB tag helpers
# ---------------------------------------------------------------------------

def _safe_model_tag(model: str) -> str:
    """Turn a bmlib model string into a DB-safe tag.

    Example: ``ollama:gpt-oss:120b`` -> ``ollama_gpt-oss_120b``
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def _tag(safe_model: str, kind: str) -> str:
    """Build a full DB model_name tag from the safe model name and kind."""
    return f"{safe_model}_{kind}"


# Kind labels, used both as DB tag suffixes and as short column headers.
MODE_A1 = "abstract_singlecall"
MODE_A2 = "abstract_twocall"
MODE_F1 = "fulltext_singlecall"
MODE_F2 = "fulltext_twocall"

MODE_LABELS = {
    MODE_A1: "a1-sgl",
    MODE_A2: "a2-two",
    MODE_F1: "f1-sgl",
    MODE_F2: "f2-two",
}


def _migrate_legacy_tags(db: Database, safe_model: str) -> None:
    """One-time migration: rename pre-multi-mode tags to the new scheme.

    Earlier versions of this script saved results under
    ``<safe_model>_singlecall`` and ``<safe_model>_twocall`` (no mode
    qualifier — at the time only abstract mode existed). Rename those rows
    to ``<safe_model>_abstract_singlecall`` and ``<safe_model>_abstract_twocall``
    so every tag in the DB tells you which mode produced it.

    Skips rows whose target slot is already occupied (avoids overwriting
    a result from a fresh run).
    """
    conn = db.conn
    legacy_to_new = {
        f"{safe_model}_singlecall": _tag(safe_model, MODE_A1),
        f"{safe_model}_twocall": _tag(safe_model, MODE_A2),
    }
    for legacy, new in legacy_to_new.items():
        # Find all (pmid) rows under the legacy tag
        rows = conn.execute(
            "SELECT pmid FROM annotations WHERE model_name = ?", (legacy,),
        ).fetchall()
        for r in rows:
            pmid = r["pmid"]
            collision = conn.execute(
                "SELECT 1 FROM annotations WHERE pmid = ? AND model_name = ?",
                (pmid, new),
            ).fetchone()
            if collision:
                logger.info(
                    "migrate: skipping %s → %s for pmid=%s (target already exists)",
                    legacy, new, pmid,
                )
                continue
            conn.execute(
                "UPDATE annotations SET model_name = ? WHERE pmid = ? AND model_name = ?",
                (new, pmid, legacy),
            )
            conn.execute(
                "UPDATE human_reviews SET model_name = ? WHERE pmid = ? AND model_name = ?",
                (new, pmid, legacy),
            )
            logger.info("migrate: %s → %s (pmid=%s)", legacy, new, pmid)
    conn.commit()


# ---------------------------------------------------------------------------
# Paper acquisition
# ---------------------------------------------------------------------------

async def _ensure_paper_in_db(
    identifier: str,
    is_doi: bool,
    db: Database,
    pipeline_config: PipelineConfig,
    source_tag: str,
) -> Optional[dict]:
    """Resolve → fetch (abstract) → enrich → return the paper dict.

    Identical to the acquisition path in ``annotate_single_paper.main``.
    This populates the abstract in the DB and runs the effect-size audit
    so the abstract modes get the same metadata as production runs.
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
    return db.get_paper(pmid)


def _acquire_fulltext(identifier: str, cli_config) -> Optional[AcquiredContent]:
    """Fetch full-text via the CLI's content acquisition path.

    Returns an AcquiredContent with JATS structure when available, plain
    text when only a PDF is fetchable, or None when only the abstract is
    reachable (caller falls back to abstract-only modes).

    Wraps ``biasbuster.cli.content.acquire_content`` so we use the same
    cache and fetch logic as the production CLI — no duplication.
    """
    try:
        content = acquire_content(identifier, cli_config)
    except Exception as exc:
        logger.warning(f"Full-text acquisition failed for {identifier}: {exc!r}")
        return None

    if not content.has_fulltext:
        logger.warning(
            f"Full-text unavailable for {identifier} "
            f"(acquired content_type={content.content_type}); "
            f"will skip full-text modes for this paper"
        )
        return None

    logger.info(
        f"Acquired full text for {identifier}: "
        f"content_type={content.content_type}, "
        f"jats={content.jats_article is not None}, "
        f"plain_text_chars={len(content.plain_fulltext)}"
    )
    return content


def _build_sections_from_content(
    content: AcquiredContent,
) -> list[tuple[str, str]]:
    """Turn AcquiredContent into a list of (section_name, text) tuples.

    Uses ``chunk_jats_article`` when the paper was fetched with structure;
    falls back to ``chunk_plain_text`` for PDFs. Both chunkers cap chunk
    size at DEFAULT_MAX_TOKENS so the per-section LLM call fits in the
    model's context window.
    """
    if content.jats_article is not None:
        chunks = chunk_jats_article(
            content.jats_article, jats_xml=content.jats_xml,
        )
        logger.info(
            f"Split JATS article into {len(chunks)} section chunks"
        )
    else:
        chunks = chunk_plain_text(content.plain_fulltext)
        logger.info(
            f"Split plain text into {len(chunks)} token-window chunks"
        )
    return [(c.section, c.text) for c in chunks]


def _sections_to_single_blob(sections: list[tuple[str, str]]) -> str:
    """Concatenate section texts into one blob for the f1 (naive) mode.

    Used only for the full-text single-call baseline. Formatting mirrors
    what ``build_user_message`` would produce if you fed the whole paper
    as a single "abstract". Kept simple on purpose: the point of f1 is to
    show what happens when you throw a long document at the v1 prompt
    with no architectural help. If the model truncates or degrades, that
    *is* the signal.
    """
    return "\n\n".join(
        f"[Section: {name}]\n{text}" for name, text in sections
    )


# ---------------------------------------------------------------------------
# Running the four modes
# ---------------------------------------------------------------------------

async def _run_abstract_singlecall(
    annotator: BmlibAnnotator, paper: dict,
) -> Optional[dict]:
    """a1 — single-call v1 on abstract only."""
    logger.info(f"[{paper.get('pmid')}] running a1 (abstract + single-call)...")
    result = await annotator.annotate_abstract(
        pmid=paper.get("pmid", ""),
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        metadata=paper,
    )
    if result is not None:
        result["_annotation_mode"] = "single_call_v1"
    return result


async def _run_abstract_twocall(
    annotator: BmlibAnnotator, paper: dict,
) -> Optional[dict]:
    """a2 — two-call v3 on abstract only."""
    logger.info(f"[{paper.get('pmid')}] running a2 (abstract + two-call)...")
    return await annotator.annotate_abstract_two_call(
        pmid=paper.get("pmid", ""),
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        metadata=paper,
    )


async def _run_fulltext_singlecall(
    annotator: BmlibAnnotator,
    paper: dict,
    sections: list[tuple[str, str]],
) -> Optional[dict]:
    """f1 — naive full-text single-call.

    Concatenates all sections into a single blob and feeds it to
    ``annotate_abstract`` (which uses the v1 prompt). This is the honest
    baseline for "what happens if you throw a long paper at v1 with no
    architectural help". Expected failure modes: context overflow,
    truncation, schema drift.
    """
    logger.info(f"[{paper.get('pmid')}] running f1 (full text + single-call)...")
    blob = _sections_to_single_blob(sections)
    # Stuff the full paper into the "abstract" slot. The v1 prompt talks
    # about abstracts, so this is intentionally a worst-case framing.
    return await annotator.annotate_abstract(
        pmid=paper.get("pmid", ""),
        title=paper.get("title", ""),
        abstract=blob,
        metadata=paper,
    )


async def _run_fulltext_twocall(
    annotator: BmlibAnnotator,
    paper: dict,
    sections: list[tuple[str, str]],
) -> Optional[dict]:
    """f2 — full map-reduce two-call (extract per section, merge, assess)."""
    logger.info(
        f"[{paper.get('pmid')}] running f2 (full text + two-call, "
        f"{len(sections)} sections)..."
    )
    return await annotator.annotate_full_text_two_call(
        pmid=paper.get("pmid", ""),
        title=paper.get("title", ""),
        sections=sections,
        metadata=paper,
    )


def _save(db: Database, paper: dict, tag: str, result: Optional[dict]) -> None:
    """Save an annotation result under the given DB tag (overwrites stale)."""
    if result is None:
        return
    pmid = paper.get("pmid", "")
    result["abstract_text"] = paper.get("abstract", "")
    result["source"] = paper.get("source", "comparison_experiment")
    db.delete_annotation(pmid, tag)
    db.insert_annotation(pmid, tag, result)
    logger.info(
        f"[{pmid}] saved {tag}: "
        f"overall={result.get('overall_severity')}, "
        f"prob={result.get('overall_bias_probability')}"
    )


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

# Fields we care about for the side-by-side table.
COMPARISON_FLAGS = {
    "statistical_reporting": [
        "relative_only", "absolute_reported", "nnt_reported",
        "baseline_risk_reported", "selective_p_values",
        "subgroup_emphasis", "inflated_effect_sizes",
    ],
    "spin": [
        "spin_level", "conclusion_matches_results",
        "causal_language_from_observational",
        "focus_on_secondary_when_primary_ns",
        "inappropriate_extrapolation", "title_spin",
    ],
    "outcome_reporting": [
        "primary_outcome_type", "surrogate_without_validation",
        "composite_not_disaggregated", "registered_outcome_not_reported",
    ],
    "conflict_of_interest": [
        "funding_type", "funding_disclosed_in_abstract",
        "industry_author_affiliations", "coi_disclosed",
        "sponsor_controls_analysis", "sponsor_controls_manuscript",
    ],
    "methodology": [
        "inappropriate_comparator", "enrichment_design",
        "per_protocol_only", "premature_stopping", "short_follow_up",
        "high_attrition", "differential_attrition",
        "inadequate_sample_size", "no_multiplicity_correction",
        "analytical_flexibility",
    ],
}


def _dom(ann: Optional[dict], name: str) -> dict:
    if ann is None:
        return {}
    v = ann.get(name, {})
    return v if isinstance(v, dict) else {}


def _fmt(v) -> str:
    if v is True:
        return "T"
    if v is False:
        return "F"
    if v is None:
        return "·"
    return str(v)[:8]


def _agreement(target: Optional[dict], ref: Optional[dict]) -> tuple[int, int]:
    """Count matching fields between target and reference.

    Fields where the reference is None are ignored (Claude didn't have an
    opinion, so there's nothing to agree or disagree with).
    """
    if target is None or ref is None:
        return 0, 0
    total = 0
    match = 0
    for domain, flags in COMPARISON_FLAGS.items():
        c_sev = _dom(ref, domain).get("severity")
        if c_sev is not None:
            total += 1
            if _dom(target, domain).get("severity") == c_sev:
                match += 1
        for flag in flags:
            c = _dom(ref, domain).get(flag)
            if c is not None:
                total += 1
                if _dom(target, domain).get(flag) == c:
                    match += 1
    return match, total


def _print_comparison(
    pmid: str,
    title: str,
    results: dict[str, Optional[dict]],
    claude_ann: Optional[dict],
) -> None:
    """Print the field-by-field table for this paper.

    Columns: CLAUDE (if present) plus every mode in ``results`` that has
    a non-None entry. Rows: per-domain severity first, then individual
    flags (only shown where Claude has an opinion or there's
    cross-mode disagreement).
    """
    active_modes = [m for m in (MODE_A1, MODE_A2, MODE_F1, MODE_F2)
                    if results.get(m) is not None]

    print()
    print("=" * 100)
    print(f"PMID {pmid}: {title[:80]}")
    print("=" * 100)

    # Build the header
    cols = []
    if claude_ann is not None:
        cols.append(("CLAUDE", claude_ann))
    for m in active_modes:
        cols.append((MODE_LABELS[m], results[m]))

    header = f"{'FIELD':<48}"
    for label, _ in cols:
        header += f"  {label:>10}"
    print(header)
    print("-" * len(header))

    # Per-domain severities
    for domain in COMPARISON_FLAGS:
        ref_sev = _dom(claude_ann, domain).get("severity") if claude_ann else None
        row = f"{'[' + domain[:30] + '.severity]':<48}"
        for label, ann in cols:
            v = _dom(ann, domain).get("severity")
            if claude_ann is None or label == "CLAUDE":
                row += f"  {_fmt(v):>10}"
            else:
                marker = "✓" if v == ref_sev else "✗"
                row += f"  {_fmt(v):>8}{marker}"
        print(row)

    # Overall severity
    ref_overall = claude_ann.get("overall_severity") if claude_ann else None
    row = f"{'[overall_severity]':<48}"
    for label, ann in cols:
        v = ann.get("overall_severity") if ann else None
        if claude_ann is None or label == "CLAUDE":
            row += f"  {_fmt(v):>10}"
        else:
            marker = "✓" if v == ref_overall else "✗"
            row += f"  {_fmt(v):>8}{marker}"
    print(row)
    print()

    # Individual flags — only where Claude has an opinion or modes disagree
    for domain, flags in COMPARISON_FLAGS.items():
        for flag in flags:
            ref_val = _dom(claude_ann, domain).get(flag) if claude_ann else None
            all_vals = [_dom(ann, domain).get(flag) for _, ann in cols]
            if ref_val is None and len(set(map(str, all_vals))) <= 1:
                continue
            path = f"{domain}.{flag}"[:48]
            row = f"{path:<48}"
            for label, ann in cols:
                v = _dom(ann, domain).get(flag)
                if claude_ann is None or label == "CLAUDE":
                    row += f"  {_fmt(v):>10}"
                else:
                    marker = "✓" if v == ref_val else "✗"
                    row += f"  {_fmt(v):>8}{marker}"
            print(row)

    # Agreement summary
    if claude_ann is not None:
        print()
        print("Agreement with Claude (on fields Claude populates):")
        for mode in active_modes:
            m, t = _agreement(results[mode], claude_ann)
            if t == 0:
                pct = 0.0
            else:
                pct = 100 * m / t
            bar = "█" * int(20 * m / max(t, 1))
            print(f"  {MODE_LABELS[mode]:<10} ({mode:<20}): {m:>2}/{t} = {pct:>3.0f}%  {bar}")
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Collect the list of (identifier, is_doi) tuples
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

    pipeline_config = PipelineConfig()
    db = Database(pipeline_config.db_path)
    db.initialize()

    cli_config = load_config(cli_model=args.model)
    safe_model = _safe_model_tag(cli_config.model)

    # One-time migration of legacy tags to the new scheme
    _migrate_legacy_tags(db, safe_model)

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

    logger.info(f"Using model: {cli_config.model}")
    modes_run = [MODE_A1, MODE_A2] + ([MODE_F1, MODE_F2] if args.full_text else [])
    logger.info(
        f"Modes to run for each paper: {', '.join(MODE_LABELS[m] for m in modes_run)}"
    )
    logger.info(
        f"DB tag prefix: {safe_model}_(abstract|fulltext)_(singlecall|twocall)"
    )

    all_results: list[dict] = []

    try:
        async with annotator:
            for identifier, is_doi in targets:
                paper = await _ensure_paper_in_db(
                    identifier, is_doi, db, pipeline_config,
                    source_tag=args.source,
                )
                if paper is None:
                    logger.error(f"Skipping {identifier}: acquisition failed")
                    continue

                pmid = paper.get("pmid", "")
                title = paper.get("title", "")

                results: dict[str, Optional[dict]] = {}

                # Abstract modes always run
                results[MODE_A1] = await _run_abstract_singlecall(annotator, paper)
                _save(db, paper, _tag(safe_model, MODE_A1), results[MODE_A1])
                results[MODE_A2] = await _run_abstract_twocall(annotator, paper)
                _save(db, paper, _tag(safe_model, MODE_A2), results[MODE_A2])

                # Full-text modes only when requested and available
                if args.full_text:
                    content = _acquire_fulltext(identifier, cli_config)
                    if content is None:
                        logger.warning(
                            f"[{pmid}] full-text modes skipped — "
                            f"acquisition returned abstract-only"
                        )
                    else:
                        sections = _build_sections_from_content(content)
                        if not sections:
                            logger.warning(
                                f"[{pmid}] no sections produced from full text — "
                                f"skipping full-text modes"
                            )
                        else:
                            results[MODE_F1] = await _run_fulltext_singlecall(
                                annotator, paper, sections,
                            )
                            _save(db, paper, _tag(safe_model, MODE_F1), results[MODE_F1])
                            results[MODE_F2] = await _run_fulltext_twocall(
                                annotator, paper, sections,
                            )
                            _save(db, paper, _tag(safe_model, MODE_F2), results[MODE_F2])

                # Look up Claude's annotation (if present) for comparison
                claude_row = db.conn.execute(
                    "SELECT annotation FROM annotations "
                    "WHERE pmid = ? AND model_name = 'anthropic'",
                    (pmid,),
                ).fetchone()
                claude_ann = None
                if claude_row and claude_row["annotation"]:
                    try:
                        claude_ann = json.loads(claude_row["annotation"])
                    except json.JSONDecodeError:
                        logger.warning(
                            f"[{pmid}] Claude annotation JSON invalid, "
                            f"skipping comparison column"
                        )

                _print_comparison(pmid, title, results, claude_ann)

                all_results.append({
                    "pmid": pmid,
                    "title": title,
                    "results": {mode: results.get(mode) for mode in modes_run},
                    "claude": claude_ann,
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
        description="Compare abstract/full-text × single-call/two-call annotations",
    )
    parser.add_argument("--pmid", type=str, help="Single PMID to compare")
    parser.add_argument("--pmids", type=str, help="Comma-separated list of PMIDs")
    parser.add_argument("--doi", type=str, help="DOI to resolve and compare")
    parser.add_argument(
        "--model", type=str, required=True,
        help='Model in "provider:model" form, e.g. "ollama:gpt-oss:120b"',
    )
    parser.add_argument(
        "--full-text", action="store_true",
        help="Also run full-text modes (f1, f2). Requires the paper to be "
             "fetchable via Europe PMC JATS or PMC PDF. Falls back to "
             "abstract-only modes when full text is unavailable.",
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
