"""
Inter-Model Agreement Analyzer

Compares annotations from two models on shared PMIDs to measure
congruence/divergence without requiring human ground truth.

Uses the SQLite database instead of reading JSONL files directly.

Usage:
    uv run python -m utils.agreement_analyzer
    uv run python -m utils.agreement_analyzer --model-a anthropic --model-b deepseek
    uv run python -m utils.agreement_analyzer --output report.md
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from database import Database

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
]

DIMENSION_LABELS = {
    "statistical_reporting": "Statistical Reporting",
    "spin": "Spin (Boutron)",
    "outcome_reporting": "Outcome Reporting",
    "conflict_of_interest": "Conflict of Interest",
    "methodology": "Methodology",
}

SEVERITY_LABELS = ["none", "low", "moderate", "high", "critical"]
SEVERITY_MAP = {s: i for i, s in enumerate(SEVERITY_LABELS)}

# Flags to compare across models
FLAG_SPECS = [
    ("statistical_reporting", "relative_only", "bool"),
    ("spin", "spin_level", "str"),
    ("conflict_of_interest", "funding_type", "str"),
]


@dataclass
class DimensionAgreement:
    dimension: str = ""
    exact_agreement: float = 0.0
    within_one: float = 0.0
    cohens_kappa: float = 0.0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    n: int = 0


@dataclass
class FlagAgreement:
    flag_name: str = ""
    agreement_rate: float = 0.0
    n: int = 0
    model_a_rate: float = 0.0  # positive/non-default rate
    model_b_rate: float = 0.0


@dataclass
class AgreementReport:
    model_a: str = ""
    model_b: str = ""
    n_shared: int = 0
    n_model_a_only: int = 0
    n_model_b_only: int = 0

    dimensions: list[DimensionAgreement] = field(default_factory=list)

    overall_severity_exact: float = 0.0
    overall_severity_kappa: float = 0.0
    probability_mae: float = 0.0
    probability_correlation: float = 0.0

    flags: list[FlagAgreement] = field(default_factory=list)

    most_divergent: list[dict] = field(default_factory=list)


def _weighted_kappa(
    ratings_a: list[int], ratings_b: list[int], n_cats: int = 5
) -> float:
    """Cohen's kappa with linear weights for ordinal data."""
    n = len(ratings_a)
    if n == 0:
        return 0.0

    observed = [[0] * n_cats for _ in range(n_cats)]
    for a, b in zip(ratings_a, ratings_b):
        if 0 <= a < n_cats and 0 <= b < n_cats:
            observed[a][b] += 1

    row_totals = [sum(observed[i]) for i in range(n_cats)]
    col_totals = [
        sum(observed[i][j] for i in range(n_cats)) for j in range(n_cats)
    ]

    weights = [
        [abs(i - j) / (n_cats - 1) for j in range(n_cats)]
        for i in range(n_cats)
    ]

    observed_weighted = (
        sum(
            weights[i][j] * observed[i][j]
            for i in range(n_cats)
            for j in range(n_cats)
        )
        / n
        if n > 0
        else 0
    )

    expected_weighted = (
        sum(
            weights[i][j] * row_totals[i] * col_totals[j]
            for i in range(n_cats)
            for j in range(n_cats)
        )
        / (n * n)
        if n > 0
        else 0
    )

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    return 1.0 - (observed_weighted / expected_weighted)


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def _load_annotations_by_pmid(db: Database, model_name: str) -> dict[str, dict]:
    """Load all annotations for a model, indexed by PMID."""
    annotations = db.get_annotations(model_name=model_name)
    return {ann["pmid"]: ann for ann in annotations if ann.get("pmid")}


def analyze_agreement(
    db: Database,
    model_a: str = "anthropic",
    model_b: str = "deepseek",
) -> AgreementReport:
    """Compare two models' annotations on shared PMIDs."""
    records_a = _load_annotations_by_pmid(db, model_a)
    records_b = _load_annotations_by_pmid(db, model_b)

    shared_pmids = sorted(set(records_a) & set(records_b))
    report = AgreementReport(
        model_a=model_a,
        model_b=model_b,
        n_shared=len(shared_pmids),
        n_model_a_only=len(set(records_a) - set(records_b)),
        n_model_b_only=len(set(records_b) - set(records_a)),
    )

    if not shared_pmids:
        logger.warning("No shared PMIDs between models.")
        return report

    # --- Per-dimension severity agreement ---
    for dim in DIMENSIONS:
        ratings_a_int = []
        ratings_b_int = []
        for pmid in shared_pmids:
            sev_a = records_a[pmid].get(dim, {}).get("severity", "")
            sev_b = records_b[pmid].get(dim, {}).get("severity", "")
            if isinstance(sev_a, dict):
                sev_a = sev_a.get("severity", "")
            if isinstance(sev_b, dict):
                sev_b = sev_b.get("severity", "")
            idx_a = SEVERITY_MAP.get(str(sev_a).lower(), -1)
            idx_b = SEVERITY_MAP.get(str(sev_b).lower(), -1)
            if idx_a >= 0 and idx_b >= 0:
                ratings_a_int.append(idx_a)
                ratings_b_int.append(idx_b)

        n = len(ratings_a_int)
        if n == 0:
            report.dimensions.append(DimensionAgreement(dimension=dim))
            continue

        exact = sum(a == b for a, b in zip(ratings_a_int, ratings_b_int))
        within = sum(
            abs(a - b) <= 1 for a, b in zip(ratings_a_int, ratings_b_int)
        )
        kappa = _weighted_kappa(ratings_a_int, ratings_b_int)

        matrix = {
            gt: {pred: 0 for pred in SEVERITY_LABELS}
            for gt in SEVERITY_LABELS
        }
        for a, b in zip(ratings_a_int, ratings_b_int):
            matrix[SEVERITY_LABELS[a]][SEVERITY_LABELS[b]] += 1

        report.dimensions.append(
            DimensionAgreement(
                dimension=dim,
                exact_agreement=exact / n,
                within_one=within / n,
                cohens_kappa=kappa,
                confusion_matrix=matrix,
                n=n,
            )
        )

    # --- Overall severity agreement ---
    overall_a_int = []
    overall_b_int = []
    for pmid in shared_pmids:
        sev_a = records_a[pmid].get("overall_severity", "")
        sev_b = records_b[pmid].get("overall_severity", "")
        idx_a = SEVERITY_MAP.get(str(sev_a).lower(), -1)
        idx_b = SEVERITY_MAP.get(str(sev_b).lower(), -1)
        if idx_a >= 0 and idx_b >= 0:
            overall_a_int.append(idx_a)
            overall_b_int.append(idx_b)

    if overall_a_int:
        n = len(overall_a_int)
        report.overall_severity_exact = (
            sum(a == b for a, b in zip(overall_a_int, overall_b_int)) / n
        )
        report.overall_severity_kappa = _weighted_kappa(
            overall_a_int, overall_b_int
        )

    # --- Overall bias probability agreement ---
    probs_a = []
    probs_b = []
    for pmid in shared_pmids:
        pa = records_a[pmid].get("overall_bias_probability")
        pb = records_b[pmid].get("overall_bias_probability")
        if pa is not None and pb is not None:
            try:
                probs_a.append(float(pa))
                probs_b.append(float(pb))
            except (ValueError, TypeError):
                continue

    if probs_a:
        report.probability_mae = sum(
            abs(a - b) for a, b in zip(probs_a, probs_b)
        ) / len(probs_a)
        report.probability_correlation = _pearson_r(probs_a, probs_b)

    # --- Flag-level agreement ---
    for dim, flag_name, flag_type in FLAG_SPECS:
        agree = 0
        n = 0
        a_positive = 0
        b_positive = 0
        for pmid in shared_pmids:
            dim_a = records_a[pmid].get(dim, {})
            dim_b = records_b[pmid].get(dim, {})
            if isinstance(dim_a, str):
                try:
                    dim_a = json.loads(dim_a)
                except (json.JSONDecodeError, TypeError):
                    dim_a = {}
            if isinstance(dim_b, str):
                try:
                    dim_b = json.loads(dim_b)
                except (json.JSONDecodeError, TypeError):
                    dim_b = {}
            val_a = dim_a.get(flag_name)
            val_b = dim_b.get(flag_name)
            if val_a is None or val_b is None:
                continue
            n += 1
            if flag_type == "bool":
                va = bool(val_a)
                vb = bool(val_b)
                if va == vb:
                    agree += 1
                if va:
                    a_positive += 1
                if vb:
                    b_positive += 1
            else:
                sa = str(val_a).lower()
                sb = str(val_b).lower()
                if sa == sb:
                    agree += 1
                if sa not in ("none", "not_reported", ""):
                    a_positive += 1
                if sb not in ("none", "not_reported", ""):
                    b_positive += 1

        report.flags.append(
            FlagAgreement(
                flag_name=f"{dim}.{flag_name}",
                agreement_rate=agree / n if n > 0 else 0.0,
                n=n,
                model_a_rate=a_positive / n if n > 0 else 0.0,
                model_b_rate=b_positive / n if n > 0 else 0.0,
            )
        )

    # --- Most divergent cases ---
    divergences = []
    for pmid in shared_pmids:
        pa = records_a[pmid].get("overall_bias_probability")
        pb = records_b[pmid].get("overall_bias_probability")
        try:
            diff = abs(float(pa) - float(pb))
        except (TypeError, ValueError):
            diff = 0.0
        divergences.append(
            {
                "pmid": pmid,
                "title": str(records_a[pmid].get("title", ""))[:100],
                f"{model_a}_severity": records_a[pmid].get(
                    "overall_severity", ""
                ),
                f"{model_b}_severity": records_b[pmid].get(
                    "overall_severity", ""
                ),
                f"{model_a}_probability": pa,
                f"{model_b}_probability": pb,
                "probability_diff": round(diff, 3),
            }
        )
    divergences.sort(key=lambda x: x["probability_diff"], reverse=True)
    report.most_divergent = divergences[:50]

    return report


def format_report(report: AgreementReport, top_divergent: int = 10) -> str:
    """Format agreement report as Markdown-style text."""
    lines = []
    lines.append(
        f"# Inter-Model Agreement: {report.model_a} vs {report.model_b}"
    )
    lines.append(f"Shared PMIDs: {report.n_shared}")
    lines.append(
        f"{report.model_a}-only: {report.n_model_a_only} | "
        f"{report.model_b}-only: {report.n_model_b_only}"
    )
    lines.append("")

    if report.n_shared == 0:
        lines.append("No shared PMIDs to compare.")
        return "\n".join(lines)

    # Overall
    lines.append("## Overall Agreement")
    lines.append(
        f"- Severity exact match: {report.overall_severity_exact:.1%}"
    )
    lines.append(
        f"- Severity weighted kappa: {report.overall_severity_kappa:.3f}"
    )
    lines.append(f"- Probability MAE: {report.probability_mae:.3f}")
    lines.append(
        f"- Probability Pearson r: {report.probability_correlation:.3f}"
    )
    lines.append("")

    # Per-dimension table
    lines.append("## Per-Dimension Severity Agreement")
    lines.append(
        f"| {'Dimension':<25} | {'Exact':>7} | {'Within 1':>9} | "
        f"{'Kappa':>7} | {'N':>5} |"
    )
    lines.append(f"|{'-'*27}|{'-'*9}|{'-'*11}|{'-'*9}|{'-'*7}|")
    for da in report.dimensions:
        label = DIMENSION_LABELS.get(da.dimension, da.dimension)
        lines.append(
            f"| {label:<25} | {da.exact_agreement:>6.1%} | "
            f"{da.within_one:>8.1%} | {da.cohens_kappa:>6.3f} | "
            f"{da.n:>5} |"
        )
    lines.append("")

    lines.append("Kappa interpretation: "
                 "<0 poor | 0-.20 slight | .21-.40 fair | "
                 ".41-.60 moderate | .61-.80 substantial | .81-1 almost perfect")
    lines.append("")

    # Flag agreement
    lines.append("## Flag-Level Agreement")
    lines.append(
        f"| {'Flag':<40} | {'Agreement':>10} | "
        f"{report.model_a + ' rate':>15} | {report.model_b + ' rate':>15} | "
        f"{'N':>5} |"
    )
    lines.append(
        f"|{'-'*42}|{'-'*12}|{'-'*17}|{'-'*17}|{'-'*7}|"
    )
    for fa in report.flags:
        lines.append(
            f"| {fa.flag_name:<40} | {fa.agreement_rate:>9.1%} | "
            f"{fa.model_a_rate:>14.1%} | {fa.model_b_rate:>14.1%} | "
            f"{fa.n:>5} |"
        )
    lines.append("")

    # Most divergent
    lines.append(f"## Top {top_divergent} Most Divergent Cases")
    lines.append(
        f"| {'PMID':<12} | {report.model_a + ' sev':>12} | "
        f"{report.model_b + ' sev':>12} | {'Prob diff':>10} | Title |"
    )
    lines.append(
        f"|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*12}|-------|"
    )
    for case in report.most_divergent[:top_divergent]:
        lines.append(
            f"| {case['pmid']:<12} | "
            f"{case.get(f'{report.model_a}_severity', ''):>12} | "
            f"{case.get(f'{report.model_b}_severity', ''):>12} | "
            f"{case['probability_diff']:>9.3f} | "
            f"{case.get('title', '')[:50]} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compare model annotations for agreement"
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--model-a", default="anthropic")
    parser.add_argument("--model-b", default="deepseek")
    parser.add_argument("--top-divergent", type=int, default=10)
    parser.add_argument(
        "--output", type=str, default=None, help="Save report to file"
    )
    args = parser.parse_args()

    from config import Config

    cfg = Config()
    db_path = args.db_path or cfg.db_path

    db = Database(db_path)
    db.initialize()
    try:
        report = analyze_agreement(db, args.model_a, args.model_b)
        text = format_report(report, top_divergent=args.top_divergent)
        print(text)
        if args.output:
            Path(args.output).write_text(text)
            print(f"\nReport saved to {args.output}")
    finally:
        db.close()
