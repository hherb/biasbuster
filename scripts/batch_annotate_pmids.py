"""Batch-annotate a list of PMIDs with any registered methodology.

Reads a newline-delimited PMID file and runs the full single-paper
annotation flow (fetch-if-needed, enrich, annotate) on each one,
continuing on per-paper failure so one broken JATS doesn't abort a
whole batch run.

Resume is automatic: ``annotate_paper`` checks
``db.has_annotation(pmid, model_name, methodology=...)`` and skips
already-done papers unless ``--force`` is set. So re-running after a
partial batch picks up exactly where it left off, without duplicating
LLM calls.

Usage::

    # 15-paper seed corpus from the legacy DB → fresh DB already done;
    # now run Cochrane RoB 2 on each paper:
    uv run python scripts/batch_annotate_pmids.py \\
        --pmids-file dataset/seed_15.txt \\
        --model anthropic \\
        --methodology cochrane_rob2 \\
        --decomposed

    # Follow up with the faithfulness report:
    uv run python -m biasbuster.evaluation.rob2_faithfulness \\
        --model anthropic

The PMID file format matches the migration script's ``--only-pmids``:
one PMID per line, ``#`` comments ignored.

This script is a thin loop over ``annotate_paper``; every correctness
invariant (applicability guard, full-text requirement check, DB
schema enforcement, methodology-scoped dedup) comes from there.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_pmids(path: Path) -> list[str]:
    """Read PMIDs from a newline-delimited file. Same format as the migration script.

    Blank lines and ``#``-comment lines are ignored. Order is preserved
    so batch runs are reproducible (the progress log ties back to file
    line numbers that way).
    """
    if not path.exists():
        raise SystemExit(f"PMID file not found: {path}")
    pmids: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line in seen:
            logger.warning("Duplicate PMID %s in file; keeping first occurrence", line)
            continue
        seen.add(line)
        pmids.append(line)
    if not pmids:
        raise SystemExit(f"PMID file {path} contained no PMIDs")
    return pmids


async def _run_one(
    pmid: str,
    db,
    config,
    args: argparse.Namespace,
) -> Optional[dict]:
    """Fetch/enrich/annotate one PMID, mirroring annotate_single_paper.main().

    Returns the annotation dict on success, None on any failure. Never
    raises — a broken paper in the middle of a 15-paper batch must not
    abort the remaining 14.
    """
    # Deferred import so this script can be --help'd without a config.py.
    from annotate_single_paper import (
        annotate_paper, enrich_paper, fetch_paper, resolve_pmid,
    )
    from biasbuster.annotators import is_retraction_notice

    # --- Fetch / verify paper in DB ---
    paper = db.get_paper(pmid)
    if paper is None:
        logger.info("PMID %s not in DB; fetching from PubMed", pmid)
        paper = await fetch_paper(pmid, config, source=args.source)
        if paper is None:
            logger.error("PMID %s: fetch failed, skipping", pmid)
            return None

    abstract = paper.get("abstract") or ""
    title = paper.get("title") or ""
    is_full_text_mode = args.full_text or args.agentic or args.decomposed
    if not abstract and not is_full_text_mode:
        logger.error("PMID %s: no abstract and no full-text mode; skipping", pmid)
        return None
    if abstract and is_retraction_notice(title, abstract, paper):
        logger.error("PMID %s: bare retraction notice; skipping", pmid)
        return None

    # --- Enrich (idempotent) ---
    enrich_paper(pmid, title, abstract, db, config)
    paper = db.get_paper(pmid)
    if paper is None:  # defensive; should not happen after enrich
        logger.error("PMID %s: paper vanished after enrich; skipping", pmid)
        return None

    # --- Annotate ---
    result = await annotate_paper(
        pmid, paper, db, config, args.model,
        force=args.force,
        two_call=not args.single_call,
        full_text=is_full_text_mode,
        agentic=args.agentic,
        decomposed=args.decomposed,
        identifier=pmid,  # batch mode always uses PMID as the identifier
        methodology=args.methodology,
    )
    if result is None:
        logger.error("PMID %s: annotation failed", pmid)
        return None
    logger.info(
        "PMID %s: annotated. severity=%s bias_prob=%s",
        pmid,
        result.get("overall_severity"),
        result.get("overall_bias_probability"),
    )
    return result


async def _run_batch(pmids: list[str], args: argparse.Namespace) -> dict[str, int]:
    """Loop over PMIDs, returning per-outcome counts."""
    # Deferred imports so --help doesn't require config.py.
    from config import Config
    from biasbuster.database import Database

    config = Config()
    db = Database(config.db_path)
    db.initialize()
    stats = {"total": len(pmids), "annotated": 0, "skipped": 0, "failed": 0}
    try:
        for i, pmid in enumerate(pmids, 1):
            logger.info("[%d/%d] PMID %s", i, stats["total"], pmid)
            try:
                result = await _run_one(pmid, db, config, args)
            except Exception as exc:  # noqa: BLE001 — batch must not crash on one
                logger.exception("PMID %s: unhandled exception: %s", pmid, exc)
                stats["failed"] += 1
                continue
            if result is None:
                stats["failed"] += 1
            elif result.get("_skipped_existing"):
                # annotate_paper does not set this today, but if it ever
                # starts signalling resume skips explicitly we're ready.
                stats["skipped"] += 1
            else:
                stats["annotated"] += 1
    finally:
        db.close()
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--pmids-file", type=Path, required=True,
        help="Newline-delimited PMID list (one PMID per line; # comments ignored).",
    )
    p.add_argument(
        "--model", type=str, default="anthropic",
        help="Annotator backend slug (e.g. 'anthropic', 'deepseek'). "
             "Default: anthropic.",
    )
    p.add_argument(
        "--methodology", type=str, default="biasbuster",
        help="Risk-of-bias methodology slug. Default: biasbuster.",
    )
    p.add_argument(
        "--source", type=str, default="manual_import",
        help="Source tag for newly fetched papers not already in the DB. "
             "Default: manual_import.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-annotate papers that already have an annotation under "
             "(pmid, model, methodology). Without this flag, existing "
             "annotations are preserved and the batch resumes cleanly.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-call", action="store_true",
        help="Abstract-only v1 single-call mode (biasbuster-only).",
    )
    mode.add_argument(
        "--full-text", action="store_true",
        help="Full-text v3 two-call (map-reduce).",
    )
    mode.add_argument(
        "--agentic", action="store_true",
        help="Full-text v4 agentic assessor.",
    )
    mode.add_argument(
        "--decomposed", action="store_true",
        help="Full-text v5A/RoB-2 decomposed assessor. Required for "
             "--methodology=cochrane_rob2.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    # Pre-flight: validate the methodology + flag combination before
    # touching config / DB / network. Mirrors annotate_single_paper's
    # validator so batch errors look identical to single-paper errors.
    if args.methodology != "biasbuster":
        forbidden = [
            name for name, flag in (
                ("--single-call", args.single_call),
                ("--full-text", args.full_text),
                ("--agentic", args.agentic),
                ("--decomposed", args.decomposed),
            ) if flag
        ]
        # cochrane_rob2 requires --decomposed; if no orchestration flag
        # was given the pre-flight in annotate_paper will refuse because
        # the methodology's requires_full_text=True. Give a better error here.
        if not args.decomposed and args.methodology == "cochrane_rob2":
            raise SystemExit(
                "--methodology=cochrane_rob2 requires --decomposed "
                "(RoB 2's declared orchestration is decomposed_full_text)."
            )
        if forbidden and not args.decomposed:
            # Only --decomposed is allowed alongside cochrane_rob2; the
            # other orchestration flags are incompatible.
            raise SystemExit(
                f"{', '.join(forbidden)} is incompatible with "
                f"--methodology={args.methodology}. Non-biasbuster "
                "methodologies follow their declared orchestration."
            )

    pmids = load_pmids(args.pmids_file)
    logger.info(
        "Running %s/%s on %d PMID(s) from %s",
        args.model, args.methodology, len(pmids), args.pmids_file,
    )
    stats = asyncio.run(_run_batch(pmids, args))
    logger.info(
        "Batch complete. total=%d annotated=%d skipped=%d failed=%d",
        stats["total"], stats["annotated"], stats["skipped"], stats["failed"],
    )
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
