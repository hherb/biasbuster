"""
Dataset Completeness Checker

Checks whether all enriched abstracts have been labelled by each model.
Reports missing items per source file and per model.

Usage:
    uv run python -m utils.completeness_checker
    uv run python -m utils.completeness_checker --show-missing
    uv run python -m utils.completeness_checker --models anthropic,deepseek
"""

import logging
from pathlib import Path

from utils import load_pmids_from_jsonl

logger = logging.getLogger(__name__)


def check_completeness(
    enriched_dir: str | Path,
    labelled_dir: str | Path,
    models: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Check labelling completeness for each source file per model.

    Returns:
        Nested dict: {source_name: {model_name: {
            "total_enriched": int,
            "total_labelled": int,
            "missing_count": int,
            "missing_pmids": list[str],
            "extra_pmids": list[str],
            "completion_pct": float,
        }}}
    """
    enriched_dir = Path(enriched_dir)
    labelled_dir = Path(labelled_dir)

    # Discover enriched source files
    enriched_files = sorted(enriched_dir.glob("*.jsonl"))
    if not enriched_files:
        logger.warning(f"No JSONL files found in {enriched_dir}")
        return {}

    # Discover models (subdirectories of labelled_dir, excluding 'human')
    if models is None:
        model_dirs = sorted(
            d for d in labelled_dir.iterdir()
            if d.is_dir() and d.name != "human"
        )
        models = [d.name for d in model_dirs]
    if not models:
        logger.warning(f"No model subdirectories found in {labelled_dir}")
        return {}

    results: dict[str, dict[str, dict]] = {}

    for enriched_path in enriched_files:
        source_name = enriched_path.stem  # e.g. "high_suspicion"
        enriched_pmids = load_pmids_from_jsonl(enriched_path)
        results[source_name] = {}

        for model_name in models:
            annotated_path = (
                labelled_dir / model_name / f"{source_name}_annotated.jsonl"
            )

            if annotated_path.exists():
                labelled_pmids = load_pmids_from_jsonl(annotated_path)
            else:
                labelled_pmids = set()

            missing = enriched_pmids - labelled_pmids
            extra = labelled_pmids - enriched_pmids
            total = len(enriched_pmids)

            results[source_name][model_name] = {
                "total_enriched": total,
                "total_labelled": len(labelled_pmids),
                "missing_count": len(missing),
                "missing_pmids": sorted(missing),
                "extra_pmids": sorted(extra),
                "completion_pct": (
                    (total - len(missing)) / total * 100 if total > 0 else 0.0
                ),
            }

    return results


def print_report(results: dict, show_missing: bool = False) -> str:
    """Format completeness results as a readable report.

    Returns the report text (also prints it).
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET COMPLETENESS REPORT")
    lines.append("=" * 70)

    if not results:
        lines.append("No data found.")
        text = "\n".join(lines)
        print(text)
        return text

    # Collect all models
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

    total_enriched_all = 0
    total_labelled_all = {m: 0 for m in all_models}
    total_missing_all = {m: 0 for m in all_models}

    for source_name, source_data in sorted(results.items()):
        row = f"{source_name:<25}"
        for model in all_models:
            info = source_data.get(model, {})
            enriched = info.get("total_enriched", 0)
            labelled = info.get("total_labelled", 0)
            pct = info.get("completion_pct", 0.0)
            row += f" | {labelled:>5}/{enriched:<5} ({pct:5.1f}%)"

            total_labelled_all[model] += labelled
            total_missing_all[model] += info.get("missing_count", 0)
        total_enriched_all += next(iter(source_data.values()), {}).get(
            "total_enriched", 0
        )
        lines.append(row)

    # Totals row
    lines.append("-" * len(header))
    row = f"{'TOTAL':<25}"
    for model in all_models:
        labelled = total_labelled_all[model]
        pct = (
            labelled / total_enriched_all * 100
            if total_enriched_all > 0
            else 0.0
        )
        row += f" | {labelled:>5}/{total_enriched_all:<5} ({pct:5.1f}%)"
    lines.append(row)

    # Missing details
    if show_missing:
        lines.append("")
        lines.append("MISSING PMIDs:")
        lines.append("-" * 40)
        for source_name, source_data in sorted(results.items()):
            for model, info in sorted(source_data.items()):
                missing = info.get("missing_pmids", [])
                if missing:
                    lines.append(
                        f"\n  {source_name} / {model} "
                        f"({len(missing)} missing):"
                    )
                    for pmid in missing[:50]:
                        lines.append(f"    - {pmid}")
                    if len(missing) > 50:
                        lines.append(f"    ... and {len(missing) - 50} more")

    # Extra PMIDs (labelled but not in enriched — diagnostic)
    has_extra = any(
        info.get("extra_pmids")
        for source_data in results.values()
        for info in source_data.values()
    )
    if has_extra:
        lines.append("")
        lines.append("NOTE: Some labelled PMIDs not found in enriched data:")
        for source_name, source_data in sorted(results.items()):
            for model, info in sorted(source_data.items()):
                extra = info.get("extra_pmids", [])
                if extra:
                    lines.append(
                        f"  {source_name} / {model}: "
                        f"{len(extra)} extra PMIDs"
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
        "--enriched-dir", default=None, help="Path to enriched directory"
    )
    parser.add_argument(
        "--labelled-dir", default=None, help="Path to labelled directory"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: auto-discover)",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print individual missing PMIDs",
    )
    args = parser.parse_args()

    from config import Config

    cfg = Config()
    enriched = args.enriched_dir or cfg.enriched_dir
    labelled = args.labelled_dir or cfg.labelled_dir
    models = args.models.split(",") if args.models else None

    results = check_completeness(enriched, labelled, models)
    print_report(results, show_missing=args.show_missing)
