"""Cochrane RoB 2 faithfulness — thin shim over the generic harness.

All logic now lives in :mod:`biasbuster.evaluation.methodology_faithfulness`.
This module preserves the original module-level API (``collect_paired_papers``,
``build_report``, ``render_markdown``, ``JudgementSeries``,
``FaithfulnessReport``, ``_extract_prediction_view``) and the
``python -m biasbuster.evaluation.rob2_faithfulness --model X`` CLI so
callers that pinned to the old entry point keep working. New code
should prefer the generic harness::

    uv run python -m biasbuster.evaluation.methodology_faithfulness \\
        --methodology cochrane_rob2 --model anthropic
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from biasbuster.database import Database
from biasbuster.evaluation.methodology_faithfulness import (
    FaithfulnessReport,
    FaithfulnessSpec,
    JudgementSeries,
    PairedPaper,
    build_report as _generic_build_report,
    collect_paired_papers as _generic_collect_paired,
    get_spec,
    render_markdown as _generic_render_markdown,
)
from biasbuster.methodologies.cochrane_rob2 import METHODOLOGY_VERSION

logger = logging.getLogger(__name__)

_METHODOLOGY = "cochrane_rob2"


def _spec() -> FaithfulnessSpec:
    """Lookup helper so get_spec is called once per entry point."""
    return get_spec(_METHODOLOGY)


# ---- Back-compat API --------------------------------------------------

def collect_paired_papers(
    db: Database, model_name: str,
) -> list[PairedPaper]:
    """RoB 2-scoped wrapper over
    :func:`methodology_faithfulness.collect_paired_papers`."""
    return _generic_collect_paired(db, _spec(), model_name)


def build_report(
    paired: list[PairedPaper],
    model_name: str,
    n_model_annotations: int,
) -> FaithfulnessReport:
    """RoB 2-scoped wrapper over the generic ``build_report``."""
    return _generic_build_report(
        paired, _spec(), model_name, n_model_annotations,
    )


def render_markdown(report: FaithfulnessReport) -> str:
    """RoB 2-scoped wrapper over the generic ``render_markdown``."""
    return _generic_render_markdown(report, _spec())


def _extract_prediction_view(ann: dict) -> Optional[dict]:
    """Back-compat alias for the spec's prediction-view loader.

    Existing tests import this directly; the implementation moved into
    ``biasbuster.methodologies.cochrane_rob2._load_prediction_view``.
    """
    return _spec().load_prediction_view(ann)


# ---- CLI --------------------------------------------------------------

def _write_outputs(
    report: FaithfulnessReport, output_dir: Path,
) -> tuple[Path, Path]:
    """Write the Markdown + JSON files and return their paths.

    Filename stem stays ``rob2_faithfulness_<model>_<timestamp>`` for
    compatibility with report consumers that glob on that pattern.
    """
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"rob2_faithfulness_{report.model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8",
    )
    return md_path, json_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--model", required=True,
        help="Annotator backend slug (e.g. 'anthropic', 'deepseek') to score.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("dataset/annotation_comparison"),
        help="Directory to write the Markdown + JSON reports.",
    )
    p.add_argument(
        "--db", type=Path, default=None,
        help="Override the DB path (defaults to config.db_path).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    db_path: Path
    if args.db is not None:
        db_path = args.db
    else:
        from config import Config
        db_path = Path(Config().db_path)

    db = Database(db_path)
    db.initialize()
    try:
        annotations = db.get_annotations(
            model_name=args.model, methodology=_METHODOLOGY,
        )
        paired = collect_paired_papers(db, args.model)
    finally:
        db.close()

    report = build_report(
        paired=paired,
        model_name=args.model,
        n_model_annotations=len(annotations),
    )
    md_path, json_path = _write_outputs(report, args.output)
    logger.info("Markdown report: %s", md_path)
    logger.info("JSON sidecar: %s", json_path)
    if report.n_paired > 0:
        logger.info(
            "overall_exact_match=%.3f, weighted_kappa=%.3f (n=%d paired)",
            report.overall.exact_match(),
            report.overall.weighted_kappa(),
            report.n_paired,
        )
    return 0


# The METHODOLOGY_VERSION re-export preserves the old import site
# (``from ... rob2_faithfulness import METHODOLOGY_VERSION`` is not used,
# but the shim should stay self-contained for sanity).
__all__ = [
    "FaithfulnessReport",
    "JudgementSeries",
    "METHODOLOGY_VERSION",
    "PairedPaper",
    "build_report",
    "collect_paired_papers",
    "main",
    "render_markdown",
]


if __name__ == "__main__":
    sys.exit(main())
