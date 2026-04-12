"""
Model Comparison Report Generator

Takes ModelEvaluation results from multiple models and generates:
1. Per-dimension head-to-head comparison tables
2. Statistical significance tests (McNemar's for binary, Wilcoxon for ordinal)
3. Radar chart data for visualisation
4. Verification quality comparison
5. Efficiency metrics (latency, token usage)
6. A combined Markdown report

This is the module that answers: "Which model is better at detecting bias,
and specifically WHERE does each one excel?"
"""

import json
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .metrics import ModelEvaluation, BinaryMetrics, OrdinalMetrics
from .scorer import ParsedAssessment, severity_to_int

logger = logging.getLogger(__name__)

__all__ = [
    "PairwiseTest",
    "EfficiencyMetrics",
    "ComparisonReport",
    "generate_comparison",
    "save_report",
    "mcnemar_test",
    "wilcoxon_approx",
]


DIMENSIONS = [
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
]

DIMENSION_LABELS = {
    "statistical_reporting": "Statistical Reporting (RRR vs ARR)",
    "spin": "Spin (Boutron)",
    "outcome_reporting": "Outcome Reporting",
    "conflict_of_interest": "Conflict of Interest",
    "methodology": "Methodological Red Flags",
}


@dataclass
class PairwiseTest:
    """Result of a pairwise statistical comparison."""
    metric_name: str = ""
    model_a: str = ""
    model_b: str = ""
    value_a: float = 0.0
    value_b: float = 0.0
    difference: float = 0.0       # b - a (positive = b is better)
    p_value: float = 1.0
    significant: bool = False     # at alpha=0.05
    effect_size: float = 0.0
    test_used: str = ""
    winner: str = ""              # model_id of winner, or "tie"


@dataclass
class EfficiencyMetrics:
    """Efficiency comparison between models."""
    model_id: str = ""
    mean_latency_s: float = 0.0
    median_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    mean_output_tokens: float = 0.0
    tokens_per_second: float = 0.0
    error_rate: float = 0.0


@dataclass
class ComparisonReport:
    """Complete head-to-head comparison report."""
    timestamp: str = ""
    mode: str = ""                # zero-shot or fine-tuned
    n_examples: int = 0

    models: list[str] = field(default_factory=list)
    evaluations: dict = field(default_factory=dict)  # model_id -> ModelEvaluation.to_dict()

    # Dimension-level comparisons
    dimension_winners: dict = field(default_factory=dict)   # dimension -> model_id
    pairwise_tests: list[dict] = field(default_factory=list)

    # Key flag comparisons (relative_only is the star)
    flag_comparison: dict = field(default_factory=dict)

    # Verification quality
    verification_comparison: dict = field(default_factory=dict)

    # Efficiency
    efficiency: dict = field(default_factory=dict)

    # Radar chart data (for visualisation)
    radar_data: dict = field(default_factory=dict)

    # Executive summary
    summary: str = ""


def mcnemar_test(paired_predictions: list[tuple[bool, bool, bool]]) -> tuple[float, float]:
    """
    McNemar's test for paired binary predictions.

    Input: list of (model_a_correct, model_b_correct, ground_truth) tuples.
    Returns: (chi2, p_value)

    Uses exact binomial test when cell counts < 25 (per standard practice).
    """
    # Count discordant pairs
    b = sum(1 for a_ok, b_ok, _ in paired_predictions if not a_ok and b_ok)  # b right, a wrong
    c = sum(1 for a_ok, b_ok, _ in paired_predictions if a_ok and not b_ok)  # a right, b wrong

    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0

    if n_discordant < 25:
        # Exact binomial test
        from math import comb
        k = min(b, c)
        p_value = 2 * sum(
            comb(n_discordant, i) * (0.5 ** n_discordant)
            for i in range(k + 1)
        )
        return float(n_discordant), min(p_value, 1.0)
    else:
        # Chi-squared approximation with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Approximate p-value from chi2 with 1 df
        # Using simple approximation since we don't have scipy
        p_value = _chi2_sf(chi2, 1)
        return chi2, p_value


def _chi2_sf(x: float, df: int = 1) -> float:
    """
    Approximate survival function for chi-squared distribution.
    Simple approximation without scipy dependency.
    """
    if x <= 0:
        return 1.0
    # Wilson-Hilferty approximation
    z = ((x / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    # Standard normal CDF approximation
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon_approx(differences: list[float]) -> tuple[float, float]:
    """
    Approximate Wilcoxon signed-rank test for ordinal severity differences.
    Returns (test_statistic, approximate_p_value).

    Without scipy, we use the normal approximation for n > 20.
    """
    # Remove zeros
    diffs = [d for d in differences if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences
    abs_diffs = [(abs(d), i, d > 0) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of positive ranks
    w_plus = sum(ranks[k] for k in range(n) if abs_diffs[k][2])
    w_minus = sum(ranks[k] for k in range(n) if not abs_diffs[k][2])
    w = min(w_plus, w_minus)

    # Normal approximation (for n > 20)
    if n > 20:
        mean_w = n * (n + 1) / 4
        var_w = n * (n + 1) * (2 * n + 1) / 24
        z = (w - mean_w) / math.sqrt(var_w)
        p_value = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
        return w, p_value
    else:
        # For small n, return statistic without p-value
        return w, -1.0  # -1 signals exact tables needed


def compute_pairwise_binary(
    model_a_id: str,
    model_b_id: str,
    assessments_a: list[ParsedAssessment],
    assessments_b: list[ParsedAssessment],
    dimension: str,
) -> PairwiseTest:
    """
    McNemar's test comparing two models on binary detection for one dimension.
    """
    result = PairwiseTest(
        metric_name=f"{dimension}_binary",
        model_a=model_a_id,
        model_b=model_b_id,
    )

    paired = []
    for a, b in zip(assessments_a, assessments_b):
        dim_a = getattr(a, dimension)
        dim_b = getattr(b, dimension)
        truth = dim_a.ground_truth_binary  # Same ground truth for both

        a_correct = (dim_a.predicted_binary == truth)
        b_correct = (dim_b.predicted_binary == truth)
        paired.append((a_correct, b_correct, truth))

    a_acc = sum(1 for p in paired if p[0]) / len(paired) if paired else 0
    b_acc = sum(1 for p in paired if p[1]) / len(paired) if paired else 0
    result.value_a = a_acc
    result.value_b = b_acc
    result.difference = b_acc - a_acc

    chi2, p_value = mcnemar_test(paired)
    result.p_value = p_value
    result.significant = p_value < 0.05
    result.test_used = "McNemar"

    if result.significant:
        result.winner = model_b_id if result.difference > 0 else model_a_id
    else:
        result.winner = "tie"

    return result


def compute_pairwise_ordinal(
    model_a_id: str,
    model_b_id: str,
    assessments_a: list[ParsedAssessment],
    assessments_b: list[ParsedAssessment],
    dimension: str,
) -> PairwiseTest:
    """
    Wilcoxon signed-rank test comparing severity rating accuracy between models.
    """
    result = PairwiseTest(
        metric_name=f"{dimension}_ordinal",
        model_a=model_a_id,
        model_b=model_b_id,
    )

    # Compute absolute error for each model on each example
    errors_a = []
    errors_b = []
    for a, b in zip(assessments_a, assessments_b):
        dim_a = getattr(a, dimension)
        dim_b = getattr(b, dimension)

        pred_a = severity_to_int(dim_a.predicted_severity)
        pred_b = severity_to_int(dim_b.predicted_severity)
        truth = severity_to_int(dim_a.ground_truth_severity)

        if pred_a >= 0 and pred_b >= 0 and truth >= 0:
            errors_a.append(abs(pred_a - truth))
            errors_b.append(abs(pred_b - truth))

    if not errors_a:
        return result

    result.value_a = sum(errors_a) / len(errors_a)  # MAE for model A
    result.value_b = sum(errors_b) / len(errors_b)  # MAE for model B (lower is better)
    result.difference = result.value_a - result.value_b  # Positive = B is better

    # Wilcoxon on paired error differences
    diffs = [ea - eb for ea, eb in zip(errors_a, errors_b)]
    w, p_value = wilcoxon_approx(diffs)
    result.p_value = p_value if p_value >= 0 else -1.0
    result.significant = 0 < p_value < 0.05
    result.test_used = "Wilcoxon signed-rank"

    if result.significant:
        result.winner = model_b_id if result.difference > 0 else model_a_id
    else:
        result.winner = "tie"

    return result


def compute_efficiency(
    model_id: str, outputs: list[dict]
) -> EfficiencyMetrics:
    """Compute efficiency metrics from raw model outputs."""
    latencies = [o.get("latency_seconds", 0) for o in outputs if not o.get("error")]
    tokens = [o.get("output_tokens", 0) for o in outputs if not o.get("error")]
    errors = sum(1 for o in outputs if o.get("error"))

    if not latencies:
        return EfficiencyMetrics(model_id=model_id)

    latencies.sort()
    n = len(latencies)

    return EfficiencyMetrics(
        model_id=model_id,
        mean_latency_s=sum(latencies) / n,
        median_latency_s=latencies[n // 2],
        p95_latency_s=latencies[int(n * 0.95)] if n >= 20 else latencies[-1],
        mean_output_tokens=sum(tokens) / n if tokens else 0,
        tokens_per_second=(
            sum(tokens) / sum(latencies) if sum(latencies) > 0 else 0
        ),
        error_rate=errors / (n + errors) if (n + errors) > 0 else 0,
    )


def generate_comparison(
    evaluations: dict[str, ModelEvaluation],
    assessments: dict[str, list[ParsedAssessment]],
    raw_outputs: dict[str, list[dict]] = None,
    mode: str = "zero-shot",
) -> ComparisonReport:
    """Generate a comparison report for N models (N >= 2).

    Computes pairwise statistical tests for every pair of models and
    builds an N-column summary table so all models are visible.
    """
    model_ids = list(evaluations.keys())
    report = ComparisonReport(
        timestamp=datetime.now().isoformat(),
        mode=mode,
        n_examples=min(e.n_examples for e in evaluations.values()) if evaluations else 0,
        models=model_ids,
        evaluations={k: v.to_dict() for k, v in evaluations.items()},
    )

    if len(model_ids) < 2:
        report.summary = "Need at least 2 models for comparison."
        return report

    # ---- Pairwise tests for ALL model pairs ----
    from itertools import combinations
    all_tests: list[PairwiseTest] = []
    for mid_a, mid_b in combinations(model_ids, 2):
        for dim in DIMENSIONS:
            if assessments.get(mid_a) and assessments.get(mid_b):
                all_tests.append(compute_pairwise_binary(
                    mid_a, mid_b, assessments[mid_a], assessments[mid_b], dim,
                ))
                all_tests.append(compute_pairwise_ordinal(
                    mid_a, mid_b, assessments[mid_a], assessments[mid_b], dim,
                ))

    report.pairwise_tests = [
        {
            "metric": t.metric_name,
            "model_a": t.model_a,
            "model_b": t.model_b,
            "value_a": round(t.value_a, 4),
            "value_b": round(t.value_b, 4),
            "difference": round(t.difference, 4),
            "p_value": round(t.p_value, 4) if t.p_value >= 0 else None,
            "p_value_note": "exact tables needed (n <= 20)" if t.p_value < 0 else None,
            "significant": t.significant,
            "winner": t.winner,
            "test": t.test_used,
        }
        for t in all_tests
    ]

    # Dimension winners: model with most pairwise wins per dimension
    for dim in DIMENSIONS:
        dim_tests = [t for t in all_tests if t.metric_name == f"{dim}_binary"]
        win_counts: dict[str, int] = defaultdict(int)
        for t in dim_tests:
            if t.winner != "tie":
                win_counts[t.winner] += 1
        if win_counts:
            report.dimension_winners[dim] = max(win_counts, key=win_counts.get)
        else:
            report.dimension_winners[dim] = "tie"

    # ---- Flag-level comparison (all models) ----
    flag_comparison: dict[str, dict] = {}
    for dim in DIMENSIONS:
        # Collect all flag names from the first model that has them
        all_flag_names: set[str] = set()
        for mid in model_ids:
            dim_eval = getattr(evaluations[mid], dim)
            all_flag_names.update(f.flag_name for f in dim_eval.flags)
        for flag_name in sorted(all_flag_names):
            entry: dict = {}
            best_acc, best_mid = -1.0, "tie"
            for mid in model_ids:
                dim_eval = getattr(evaluations[mid], dim)
                flag_data = next((f for f in dim_eval.flags if f.flag_name == flag_name), None)
                if flag_data:
                    entry[mid] = flag_data.to_dict()
                    if flag_data.accuracy > best_acc:
                        best_acc = flag_data.accuracy
                        best_mid = mid
                    elif flag_data.accuracy == best_acc:
                        best_mid = "tie"
            entry["winner"] = best_mid
            flag_comparison[f"{dim}.{flag_name}"] = entry
    report.flag_comparison = flag_comparison

    # ---- Verification quality comparison (all models) ----
    report.verification_comparison = {
        mid: {
            "mean_score": evaluations[mid].mean_verification_score,
            "source_rates": evaluations[mid].verification_source_rates,
        }
        for mid in model_ids
    }

    # ---- Efficiency ----
    if raw_outputs:
        for mid in model_ids:
            if mid in raw_outputs:
                report.efficiency[mid] = compute_efficiency(
                    mid, raw_outputs[mid]
                ).__dict__

    # ---- Radar chart data ----
    for mid in model_ids:
        ev = evaluations[mid]
        report.radar_data[mid] = {
            dim: getattr(ev, dim).binary.f1
            for dim in DIMENSIONS
        }
        report.radar_data[mid]["verification"] = ev.mean_verification_score
        report.radar_data[mid]["overall_f1"] = ev.overall_binary.f1

    # ---- Executive summary ----
    report.summary = _generate_summary(report, evaluations, model_ids, all_tests)

    return report


def _generate_summary(
    report: ComparisonReport,
    evaluations: dict[str, ModelEvaluation],
    model_ids: list[str],
    tests: list[PairwiseTest],
) -> str:
    """Generate a human-readable executive summary for N models."""
    lines: list[str] = []

    # Short display names for readability in tables
    short = _short_model_names(model_ids)

    title_models = " vs ".join(short[m] for m in model_ids)
    lines.append(f"# Bias Detection Model Comparison: {title_models}")
    lines.append(f"Mode: {report.mode} | Examples: {report.n_examples}")
    lines.append(f"Date: {report.timestamp}")
    if any(short[m] != m for m in model_ids):
        lines.append("")
        lines.append("Model key:")
        for mid in model_ids:
            lines.append(f"  - **{short[mid]}** = `{mid}`")
    lines.append("")

    # --- Overall performance (N-column table) ---
    lines.append("## Overall Performance")
    hdr = "| Metric | " + " | ".join(short[m] for m in model_ids) + " |"
    sep = "|--------|" + "|".join("-" * (len(short[m]) + 2) for m in model_ids) + "|"
    lines.append(hdr)
    lines.append(sep)

    # Severity kappa is the meaningful overall metric; binary F1 is
    # trivially 1.0 when all papers have bias so we omit it.
    def _row(label: str, getter) -> str:
        vals = " | ".join(f"{getter(evaluations[m]):.3f}" for m in model_ids)
        return f"| {label} | {vals} |"

    lines.append(_row("Severity κ (overall)", lambda e: e.overall_ordinal.weighted_kappa()))
    lines.append(_row("Calibration Error", lambda e: e.calibration_error))
    pf_vals = " | ".join(str(evaluations[m].n_parse_failures) for m in model_ids)
    lines.append(f"| Parse Failures | {pf_vals} |")
    lines.append("")

    # --- Per-dimension severity kappa (the metric that matters) ---
    lines.append("## Per-Dimension Severity Agreement (κ)")
    hdr = "| Dimension | " + " | ".join(short[m] for m in model_ids) + " |"
    sep = "|-----------|" + "|".join("-" * (len(short[m]) + 2) for m in model_ids) + "|"
    lines.append(hdr)
    lines.append(sep)
    for dim in DIMENSIONS:
        label = DIMENSION_LABELS.get(dim, dim)
        vals = []
        for mid in model_ids:
            kappa = getattr(evaluations[mid], dim).ordinal.weighted_kappa()
            vals.append(f"{kappa:.3f}")
        lines.append(f"| {label} | {' | '.join(vals)} |")
    lines.append("")

    # --- Per-dimension binary F1 ---
    lines.append("## Per-Dimension Binary F1")
    hdr = "| Dimension | " + " | ".join(short[m] for m in model_ids) + " | Winner |"
    sep = "|-----------|" + "|".join("-" * (len(short[m]) + 2) for m in model_ids) + "|--------|"
    lines.append(hdr)
    lines.append(sep)
    for dim in DIMENSIONS:
        label = DIMENSION_LABELS.get(dim, dim)
        vals = []
        for mid in model_ids:
            f1 = getattr(evaluations[mid], dim).binary.f1
            vals.append(f"{f1:.3f}")
        winner = report.dimension_winners.get(dim, "tie")
        winner_display = short.get(winner, winner)
        lines.append(f"| {label} | {' | '.join(vals)} | {winner_display} |")
    lines.append("")

    # --- Key flag: relative_only ---
    lines.append("## Key Flag: Relative-Only Reporting Detection")
    rel_flag = report.flag_comparison.get("statistical_reporting.relative_only", {})
    if rel_flag:
        for mid in model_ids:
            data = rel_flag.get(mid, {})
            lines.append(
                f"- {short[mid]}: {data.get('accuracy', 0):.1%} accuracy "
                f"({data.get('correct', 0)}/{data.get('total', 0)})"
            )
        lines.append(f"- Winner: **{short.get(rel_flag.get('winner', 'tie'), 'tie')}**")
    else:
        lines.append("- Not enough data for this flag comparison")
    lines.append("")

    # --- Verification quality ---
    lines.append("## Verification Source Knowledge")
    hdr = "| Source | " + " | ".join(short[m] for m in model_ids) + " |"
    sep = "|--------|" + "|".join("-" * (len(short[m]) + 2) for m in model_ids) + "|"
    lines.append(hdr)
    lines.append(sep)
    sources = ["open_payments", "clinicaltrials_gov", "orcid", "retraction_watch", "europmc"]
    for source in sources:
        vals = []
        for mid in model_ids:
            rate = evaluations[mid].verification_source_rates.get(source, 0)
            vals.append(f"{rate:.0%}")
        lines.append(f"| {source} | {' | '.join(vals)} |")
    mean_vals = []
    for mid in model_ids:
        mean_vals.append(f"**{evaluations[mid].mean_verification_score:.3f}**")
    lines.append(f"| **Mean score** | {' | '.join(mean_vals)} |")
    lines.append("")

    # --- Reasoning quality ---
    lines.append("## Reasoning Quality")
    for mid in model_ids:
        ev = evaluations[mid]
        lines.append(
            f"- {short[mid]}: reasoning present in "
            f"{ev.thinking_present_rate:.0%} of outputs, "
            f"mean length {ev.mean_thinking_length:.0f} chars"
        )
    lines.append("")

    # --- Efficiency ---
    if report.efficiency:
        lines.append("## Efficiency")
        for mid in model_ids:
            eff = report.efficiency.get(mid, {})
            if eff:
                lines.append(
                    f"- {short[mid]}: {eff.get('mean_latency_s', 0):.1f}s mean, "
                    f"{eff.get('tokens_per_second', 0):.1f} tok/s, "
                    f"{eff.get('error_rate', 0):.1%} errors"
                )
        lines.append("")

    # --- Bottom line ---
    lines.append("## Bottom Line")
    win_counts: dict[str, int] = defaultdict(int)
    for w in report.dimension_winners.values():
        win_counts[w] += 1
    ties = win_counts.pop("tie", 0)
    sig_wins = sum(1 for t in tests if t.significant and "binary" in t.metric_name)

    parts = []
    for mid in model_ids:
        parts.append(f"{short[mid]}={win_counts.get(mid, 0)}")
    lines.append(
        f"Dimension wins: {', '.join(parts)}, ties={ties} "
        f"({sig_wins} statistically significant)"
    )

    if win_counts:
        best = max(win_counts, key=win_counts.get)
        verdict = short[best]
    else:
        verdict = "No clear winner"
    lines.append(f"Overall verdict: **{verdict}**")

    return "\n".join(lines)


def _short_model_names(model_ids: list[str]) -> dict[str, str]:
    """Create readable short names for model IDs.

    Long Ollama model IDs like ``ollama:gemma4:26b-a4b-it-q8_0_fulltext_agentic``
    are unwieldy in table headers. This extracts a short, unique suffix.
    """
    short: dict[str, str] = {}
    for mid in model_ids:
        # Strip common prefixes
        name = mid
        if name.startswith("ollama:"):
            name = name[len("ollama:"):]
        elif name.startswith("ollama_"):
            name = name[len("ollama_"):]
        # Collapse colons to slashes for readability
        name = name.replace(":", "/")
        # Truncate very long names at the model size indicator
        # e.g. "gemma4/26b-a4b-it-q8_0_fulltext_agentic" → "gemma4/26b_agentic"
        # Keep it simple: just use the name as-is but capped
        if len(name) > 30:
            # Keep first part (model family + size) and last part (harness tag)
            parts = name.replace("-", "_").split("_")
            if len(parts) > 3:
                name = "_".join(parts[:2]) + "_…_" + parts[-1]
        short[mid] = name
    # Check for collisions — if any, fall back to full IDs
    if len(set(short.values())) < len(short):
        return {mid: mid for mid in model_ids}
    return short


def save_report(report: ComparisonReport, output_dir: Path, evaluations: dict = None):
    """Save the comparison report as JSON, Markdown, and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = report.timestamp[:10]

    # JSON (machine-readable)
    json_path = output_dir / f"comparison_{report.mode}_{date_str}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": report.timestamp,
            "mode": report.mode,
            "n_examples": report.n_examples,
            "models": report.models,
            "evaluations": report.evaluations,
            "dimension_winners": report.dimension_winners,
            "pairwise_tests": report.pairwise_tests,
            "flag_comparison": report.flag_comparison,
            "verification_comparison": report.verification_comparison,
            "efficiency": report.efficiency,
            "radar_data": report.radar_data,
        }, f, indent=2)
    logger.info(f"JSON report saved to {json_path}")

    # Markdown (human-readable)
    md_path = output_dir / f"comparison_{report.mode}_{date_str}.md"
    with open(md_path, "w") as f:
        f.write(report.summary)
    logger.info(f"Markdown report saved to {md_path}")

    # CSV (spreadsheet analysis)
    if evaluations:
        csv_path = output_dir / f"comparison_{report.mode}_{date_str}.csv"
        _save_csv(report, evaluations, csv_path)
        logger.info(f"CSV report saved to {csv_path}")


def _save_csv(report: ComparisonReport, evaluations: dict, path: Path):
    """Generate a summary CSV for spreadsheet analysis."""
    import csv

    model_ids = report.models

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["dimension", "metric"] + model_ids + ["winner"]
        writer.writerow(header)

        for dim in DIMENSIONS:
            for metric in ["binary_f1", "binary_precision", "binary_recall",
                           "ordinal_kappa", "ordinal_mae"]:
                row = [dim, metric]
                values = {}
                for mid in model_ids:
                    ev = evaluations[mid]
                    dim_eval = getattr(ev, dim)
                    if metric.startswith("binary"):
                        val = getattr(dim_eval.binary, metric.replace("binary_", ""))
                    elif metric == "ordinal_kappa":
                        val = dim_eval.ordinal.weighted_kappa()
                    elif metric == "ordinal_mae":
                        val = dim_eval.ordinal.mae
                    else:
                        val = 0
                    row.append(f"{val:.4f}")
                    values[mid] = val

                if metric == "ordinal_mae":
                    winner = min(values, key=values.get)
                else:
                    winner = max(values, key=values.get)
                row.append(winner)
                writer.writerow(row)

        for metric in ["binary_f1", "ordinal_kappa", "calibration_error",
                       "mean_verification_score"]:
            row = ["overall", metric]
            values = {}
            for mid in model_ids:
                ev = evaluations[mid]
                if metric == "binary_f1":
                    val = ev.overall_binary.f1
                elif metric == "ordinal_kappa":
                    val = ev.overall_ordinal.weighted_kappa()
                elif metric == "calibration_error":
                    val = ev.calibration_error
                elif metric == "mean_verification_score":
                    val = ev.mean_verification_score
                else:
                    val = 0
                row.append(f"{val:.4f}")
                values[mid] = val

            if metric == "calibration_error":
                winner = min(values, key=values.get)
            else:
                winner = max(values, key=values.get)
            row.append(winner)
            writer.writerow(row)
