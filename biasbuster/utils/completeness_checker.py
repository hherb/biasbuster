"""
Dataset Completeness Checker

Checks whether all enriched abstracts have been annotated by each model.
Reports missing items per source and per model using the SQLite database.

Usage:
    uv run python -m utils.completeness_checker
    uv run python -m utils.completeness_checker --models anthropic,deepseek
"""

import logging

from biasbuster.database import Database

logger = logging.getLogger(__name__)

# Source keys used in annotations and their corresponding DB queries
SOURCE_KEYS = [
    "high_suspicion",
    "retracted_papers",
    "cochrane_rob",
    "low_suspicion",
]


def check_completeness(
    db: Database,
    models: list[str] | None = None,
    annotation_limits: dict[str, int] | None = None,
) -> dict[str, dict[str, dict]]:
    """Check annotation completeness for each source per model.

    Args:
        db: Database instance.
        models: Model names to check (auto-discovers if None).
        annotation_limits: Per-source max items from config.

    Returns:
        Nested dict: {source_name: {model_name: {
            "target": int,
            "total_labelled": int,
            "missing_count": int,
            "completion_pct": float,
        }}}
    """
    if models is None:
        models = db.get_model_names()
        models = [m for m in models if m != "human"]
    if not models:
        logger.warning("No model annotations found in database")
        return {}

    results: dict[str, dict[str, dict]] = {}

    for source_key in SOURCE_KEYS:
        # Count papers available for this source
        papers = db.get_papers_by_source_for_annotation(source_key)
        total_available = len(papers)
        available_pmids = {p["pmid"] for p in papers}

        limit = (
            annotation_limits.get(source_key)
            if annotation_limits
            else None
        )
        target = min(total_available, limit) if limit else total_available

        results[source_key] = {}

        for model_name in models:
            annotated_pmids = db.get_annotated_pmids(model_name)
            labelled_in_source = annotated_pmids & available_pmids
            labelled_count = len(labelled_in_source)
            effective_missing = max(0, target - labelled_count)

            results[source_key][model_name] = {
                "total_available": total_available,
                "target": target,
                "total_labelled": labelled_count,
                "missing_count": effective_missing,
                "completion_pct": (
                    labelled_count / target * 100 if target > 0 else 100.0
                ),
            }

    return results


def print_report(
    results: dict,
    show_limits: bool = True,
) -> str:
    """Format completeness results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET COMPLETENESS REPORT")
    lines.append("=" * 70)

    if not results:
        lines.append("No data found.")
        text = "\n".join(lines)
        print(text)
        return text

    all_models = set()
    for source_data in results.values():
        all_models.update(source_data.keys())
    all_models = sorted(all_models)

    # Summary table
    lines.append("")
    header = f"{'Source':<25}"
    for model in all_models:
        header += f" | {model:>20}"
    lines.append(header)
    lines.append("-" * len(header))

    total_target_all = 0
    total_labelled_all = {m: 0 for m in all_models}

    for source_name, source_data in sorted(results.items()):
        row = f"{source_name:<25}"
        first_model_info = next(iter(source_data.values()), {})
        target = first_model_info.get("target", 0)
        total_available = first_model_info.get("total_available", 0)

        for model in all_models:
            info = source_data.get(model, {})
            labelled = info.get("total_labelled", 0)
            pct = info.get("completion_pct", 0.0)
            row += f" | {labelled:>5}/{target:<5} ({pct:5.1f}%)"
            total_labelled_all[model] += labelled

        total_target_all += target

        if show_limits and target != total_available:
            row += f"  [of {total_available} available]"

        lines.append(row)

    # Totals row
    lines.append("-" * len(header))
    row = f"{'TOTAL':<25}"
    for model in all_models:
        labelled = total_labelled_all[model]
        pct = (
            labelled / total_target_all * 100
            if total_target_all > 0
            else 0.0
        )
        row += f" | {labelled:>5}/{total_target_all:<5} ({pct:5.1f}%)"
    lines.append(row)

    # Overall status
    lines.append("")
    all_complete = all(
        info.get("completion_pct", 0) >= 100.0
        for source_data in results.values()
        for info in source_data.values()
    )
    if all_complete:
        lines.append("Status: ALL TARGETS MET")
    else:
        lines.append("Remaining work:")
        for source_name, source_data in sorted(results.items()):
            for model, info in sorted(source_data.items()):
                missing = info.get("missing_count", 0)
                if missing > 0:
                    lines.append(
                        f"  {source_name} / {model}: "
                        f"{missing} items to annotate"
                    )

    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Check dataset labelling completeness"
    )
    parser.add_argument(
        "--db-path", default=None, help="Path to SQLite database"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: auto-discover)",
    )
    parser.add_argument(
        "--no-limits",
        action="store_true",
        help="Ignore config caps, show progress against full available set",
    )
    args = parser.parse_args()

    from config import Config

    cfg = Config()
    db_path = args.db_path or cfg.db_path
    models = args.models.split(",") if args.models else None
    limits = None if args.no_limits else cfg.annotation_max_per_source

    db = Database(db_path)
    db.initialize()
    try:
        results = check_completeness(db, models, limits)
        print_report(results)
    finally:
        db.close()
