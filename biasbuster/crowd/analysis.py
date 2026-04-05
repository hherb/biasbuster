"""Congruence analysis for crowd annotations.

Computes inter-annotator agreement (human vs human) and human-vs-AI
congruence metrics, including anchoring effect analysis.

Metrics:
- Krippendorff's alpha (ordinal) per domain severity
- Fleiss' kappa per domain (nominal)
- Cohen's kappa for crowd consensus vs AI
- Confusion matrices
- McNemar's test for anchoring effect (blind vs revised)
- Per-annotator quality scores

Usage:
    uv run python -m crowd.analysis \\
        --crowd-db dataset/crowd_annotations.db \\
        --model deepseek \\
        --output crowd_analysis_report.md
"""

import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from biasbuster.crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)

# Severity ordinal mapping
SEVERITY_LEVELS = ["none", "low", "moderate", "high", "critical"]
SEVERITY_ORD = {s: i for i, s in enumerate(SEVERITY_LEVELS)}

DOMAINS = [
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
]


# ── Statistical helpers ──────────────────────────────────────────────


def _ordinal_median(values: list[str], levels: list[str] = SEVERITY_LEVELS) -> str:
    """Compute ordinal median of severity labels."""
    ord_map = {s: i for i, s in enumerate(levels)}
    nums = sorted(ord_map.get(v, 0) for v in values)
    mid = len(nums) // 2
    if len(nums) % 2 == 0:
        med_val = round((nums[mid - 1] + nums[mid]) / 2)
    else:
        med_val = nums[mid]
    med_val = max(0, min(med_val, len(levels) - 1))
    return levels[med_val]


def _weighted_ordinal_median(
    values: list[str],
    weights: list[float],
    levels: list[str] = SEVERITY_LEVELS,
) -> str:
    """Compute weighted ordinal median."""
    ord_map = {s: i for i, s in enumerate(levels)}
    pairs = sorted(
        zip([ord_map.get(v, 0) for v in values], weights),
        key=lambda x: x[0],
    )
    total_weight = sum(w for _, w in pairs)
    if total_weight == 0:
        return _ordinal_median(values, levels)

    cumulative = 0.0
    for val, w in pairs:
        cumulative += w
        if cumulative >= total_weight / 2:
            return levels[max(0, min(val, len(levels) - 1))]
    return levels[0]


def _mode(values: list[str]) -> str:
    """Return the most common value (ties broken by first occurrence)."""
    if not values:
        return ""
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _majority_vote_bool(values: list[bool]) -> bool:
    """Majority vote for boolean values."""
    return sum(values) > len(values) / 2


def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa between two lists of labels."""
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    n = len(labels_a)
    all_labels = sorted(set(labels_a) | set(labels_b))

    # Observed agreement
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    po = agree / n

    # Expected agreement by chance
    pe = 0.0
    for label in all_labels:
        count_a = sum(1 for x in labels_a if x == label)
        count_b = sum(1 for x in labels_b if x == label)
        pe += (count_a / n) * (count_b / n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _fleiss_kappa(ratings_matrix: list[list[int]]) -> float:
    """Compute Fleiss' kappa for multiple raters.

    Args:
        ratings_matrix: List of items, each item is a list of counts
            per category (how many raters assigned each category).
    """
    if not ratings_matrix:
        return 0.0

    n_items = len(ratings_matrix)
    n_categories = len(ratings_matrix[0])
    n_raters = sum(ratings_matrix[0])

    if n_raters <= 1:
        return 0.0

    # p_j: proportion of all assignments to category j
    p_j = []
    total_assignments = n_items * n_raters
    for j in range(n_categories):
        col_sum = sum(row[j] for row in ratings_matrix)
        p_j.append(col_sum / total_assignments)

    # P_i: extent of agreement for item i
    p_i = []
    for row in ratings_matrix:
        sum_sq = sum(n * n for n in row)
        p_i.append((sum_sq - n_raters) / (n_raters * (n_raters - 1)))

    p_bar = sum(p_i) / n_items
    pe = sum(p * p for p in p_j)

    if pe == 1.0:
        return 1.0
    return (p_bar - pe) / (1 - pe)


def _mcnemar_test(
    changed_to_agree: int,
    changed_to_disagree: int,
) -> dict:
    """McNemar's test for paired nominal data.

    Tests whether the rate of changing from disagree→agree after seeing AI
    is significantly different from agree→disagree.

    Returns dict with chi2, p_value (approximate), and interpretation.
    """
    b = changed_to_agree  # blind disagree → revised agree
    c = changed_to_disagree  # blind agree → revised disagree

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with 1 df
    # P(X > chi2) = erfc(sqrt(chi2 / 2)) for 1-df chi-squared
    p_value = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "changed_to_agree": b,
        "changed_to_disagree": c,
    }


def _confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    labels: list[str] = SEVERITY_LEVELS,
) -> dict:
    """Build a confusion matrix dict."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(true_labels, pred_labels):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


# ── Core analysis ────────────────────────────────────────────────────


def analyze_crowd_annotations(
    db: CrowdDatabase,
    model_name: Optional[str] = None,
) -> dict:
    """Run full congruence analysis on crowd annotations.

    Returns a dict with all computed metrics.
    """
    completed = db.get_all_completed_annotations()
    if not completed:
        return {"error": "No completed annotations found."}

    # Group by paper
    by_paper: dict[str, list[dict]] = {}
    for ann in completed:
        by_paper.setdefault(ann["pmid"], []).append(ann)

    # Only papers with 2+ annotations for agreement metrics
    multi_rated = {
        pmid: anns for pmid, anns in by_paper.items() if len(anns) >= 2
    }

    results: dict[str, Any] = {
        "total_annotations": len(completed),
        "total_papers": len(by_paper),
        "multi_rated_papers": len(multi_rated),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Inter-annotator agreement ────────────────────────────────
    if multi_rated:
        domain_results = {}
        for domain in DOMAINS + ["overall"]:
            domain_results[domain] = _compute_domain_agreement(
                multi_rated, domain
            )
        results["inter_annotator"] = domain_results

    # ── Human vs AI ──────────────────────────────────────────────
    ai_comparison = _compute_ai_comparison(db, by_paper, model_name)
    if ai_comparison:
        results["human_vs_ai"] = ai_comparison

    # ── Anchoring effect ─────────────────────────────────────────
    anchoring = _compute_anchoring_effect(db, by_paper, model_name)
    if anchoring:
        results["anchoring_effect"] = anchoring

    # ── Annotator quality scores ─────────────────────────────────
    if multi_rated:
        results["annotator_quality"] = compute_annotator_quality(
            multi_rated
        )

    return results


def _compute_domain_agreement(
    multi_rated: dict[str, list[dict]],
    domain: str,
) -> dict:
    """Compute Fleiss' kappa for a domain across multi-rated papers."""
    ratings_matrix = []
    for pmid, annotations in multi_rated.items():
        counts = [0] * len(SEVERITY_LEVELS)
        for ann in annotations:
            if domain == "overall":
                sev = ann.get("revised_annotation", {}).get(
                    "overall_severity", "none"
                )
            else:
                domain_data = ann.get("revised_annotation", {}).get(domain, {})
                sev = domain_data.get("severity", "none") if isinstance(
                    domain_data, dict
                ) else "none"
            idx = SEVERITY_ORD.get(sev, 0)
            counts[idx] += 1
        ratings_matrix.append(counts)

    kappa = _fleiss_kappa(ratings_matrix)
    return {
        "fleiss_kappa": round(kappa, 4),
        "n_papers": len(ratings_matrix),
    }


def _compute_ai_comparison(
    db: CrowdDatabase,
    by_paper: dict[str, list[dict]],
    model_name: Optional[str],
) -> dict:
    """Compute crowd consensus vs AI agreement."""
    results: dict[str, Any] = {}

    for domain in DOMAINS + ["overall"]:
        consensus_labels = []
        ai_labels = []

        for pmid, annotations in by_paper.items():
            ai_row = db.get_ai_annotation(pmid, model_name)
            if ai_row is None:
                continue
            ai_ann = ai_row["annotation"]
            if not isinstance(ai_ann, dict):
                continue

            # Get AI severity for this domain
            if domain == "overall":
                ai_sev = ai_ann.get("overall_severity", "none")
            else:
                ai_domain = ai_ann.get(domain, {})
                ai_sev = ai_domain.get("severity", "none") if isinstance(
                    ai_domain, dict
                ) else "none"

            # Compute crowd consensus (revised annotations)
            human_sevs = []
            for ann in annotations:
                revised = ann.get("revised_annotation", {})
                if not isinstance(revised, dict):
                    continue
                if domain == "overall":
                    human_sevs.append(revised.get("overall_severity", "none"))
                else:
                    d = revised.get(domain, {})
                    if isinstance(d, dict):
                        human_sevs.append(d.get("severity", "none"))

            if human_sevs:
                consensus = _ordinal_median(human_sevs)
                consensus_labels.append(consensus)
                ai_labels.append(ai_sev)

        if consensus_labels:
            kappa = _cohens_kappa(consensus_labels, ai_labels)
            exact_agree = sum(
                c == a for c, a in zip(consensus_labels, ai_labels)
            )
            results[domain] = {
                "cohens_kappa": round(kappa, 4),
                "exact_agreement": round(exact_agree / len(consensus_labels), 4),
                "n_papers": len(consensus_labels),
                "confusion_matrix": _confusion_matrix(consensus_labels, ai_labels),
            }

    return results


def _compute_anchoring_effect(
    db: CrowdDatabase,
    by_paper: dict[str, list[dict]],
    model_name: Optional[str],
) -> dict:
    """Measure how much seeing AI output changes human ratings.

    Compares blind vs revised annotations' agreement with AI.
    """
    # Track transitions
    blind_agree_revised_agree = 0
    blind_agree_revised_disagree = 0
    blind_disagree_revised_agree = 0
    blind_disagree_revised_disagree = 0

    blind_severities = []
    revised_severities = []
    ai_severities = []

    for pmid, annotations in by_paper.items():
        ai_row = db.get_ai_annotation(pmid, model_name)
        if ai_row is None:
            continue
        ai_ann = ai_row["annotation"]
        if not isinstance(ai_ann, dict):
            continue
        ai_sev = ai_ann.get("overall_severity", "none")

        for ann in annotations:
            blind = ann.get("blind_annotation", {})
            revised = ann.get("revised_annotation", {})
            if not isinstance(blind, dict) or not isinstance(revised, dict):
                continue

            blind_sev = blind.get("overall_severity", "none")
            revised_sev = revised.get("overall_severity", "none")

            blind_agrees = blind_sev == ai_sev
            revised_agrees = revised_sev == ai_sev

            if blind_agrees and revised_agrees:
                blind_agree_revised_agree += 1
            elif blind_agrees and not revised_agrees:
                blind_agree_revised_disagree += 1
            elif not blind_agrees and revised_agrees:
                blind_disagree_revised_agree += 1
            else:
                blind_disagree_revised_disagree += 1

            blind_severities.append(blind_sev)
            revised_severities.append(revised_sev)
            ai_severities.append(ai_sev)

    total = len(blind_severities)
    if total == 0:
        return {}

    blind_agreement_rate = sum(
        b == a for b, a in zip(blind_severities, ai_severities)
    ) / total
    revised_agreement_rate = sum(
        r == a for r, a in zip(revised_severities, ai_severities)
    ) / total
    change_rate = sum(
        b != r for b, r in zip(blind_severities, revised_severities)
    ) / total

    mcnemar = _mcnemar_test(
        blind_disagree_revised_agree,
        blind_agree_revised_disagree,
    )

    return {
        "n_annotations": total,
        "blind_agreement_with_ai": round(blind_agreement_rate, 4),
        "revised_agreement_with_ai": round(revised_agreement_rate, 4),
        "change_rate": round(change_rate, 4),
        "mcnemar_test": mcnemar,
        "transition_table": {
            "agree_to_agree": blind_agree_revised_agree,
            "agree_to_disagree": blind_agree_revised_disagree,
            "disagree_to_agree": blind_disagree_revised_agree,
            "disagree_to_disagree": blind_disagree_revised_disagree,
        },
    }


def compute_annotator_quality(
    multi_rated: dict[str, list[dict]],
) -> dict:
    """Score each annotator by agreement with majority on multi-rated papers."""
    # Compute majority labels per paper
    paper_consensus: dict[str, str] = {}
    for pmid, annotations in multi_rated.items():
        sevs = []
        for ann in annotations:
            revised = ann.get("revised_annotation", {})
            if isinstance(revised, dict):
                sevs.append(revised.get("overall_severity", "none"))
        if sevs:
            paper_consensus[pmid] = _ordinal_median(sevs)

    # Score each annotator
    annotator_scores: dict[int, dict] = {}
    for pmid, annotations in multi_rated.items():
        consensus = paper_consensus.get(pmid)
        if consensus is None:
            continue
        for ann in annotations:
            uid = ann["user_id"]
            if uid not in annotator_scores:
                annotator_scores[uid] = {"agree": 0, "total": 0}
            revised = ann.get("revised_annotation", {})
            if isinstance(revised, dict):
                if revised.get("overall_severity") == consensus:
                    annotator_scores[uid]["agree"] += 1
                annotator_scores[uid]["total"] += 1

    quality = {}
    for uid, scores in annotator_scores.items():
        total = scores["total"]
        quality[uid] = {
            "agreement_rate": round(scores["agree"] / total, 4) if total > 0 else 0,
            "n_papers": total,
        }

    return quality


# ── Consensus computation (for gold standard export) ─────────────────


def compute_consensus(
    db: CrowdDatabase,
    min_raters: int = 3,
    annotator_weights: Optional[dict[int, float]] = None,
) -> list[dict]:
    """Compute consensus annotations for papers with sufficient raters.

    Args:
        db: Crowd database.
        min_raters: Minimum completed annotations required per paper.
        annotator_weights: Optional {user_id: quality_score} for weighting.

    Returns:
        List of dicts with 'pmid', 'consensus_annotation', 'n_raters',
        'agreement_alpha'.
    """
    completed = db.get_all_completed_annotations()
    by_paper: dict[str, list[dict]] = {}
    for ann in completed:
        by_paper.setdefault(ann["pmid"], []).append(ann)

    results = []
    for pmid, annotations in by_paper.items():
        if len(annotations) < min_raters:
            continue

        consensus = _build_consensus_annotation(
            annotations, annotator_weights
        )
        results.append({
            "pmid": pmid,
            "consensus_annotation": consensus,
            "n_raters": len(annotations),
        })

    return results


def _build_consensus_annotation(
    annotations: list[dict],
    weights: Optional[dict[int, float]] = None,
) -> dict:
    """Build a consensus annotation from multiple crowd annotations.

    Uses ordinal median for severities, majority vote for booleans,
    mode for categoricals.
    """
    result: dict[str, Any] = {}

    # Per-domain consensus
    for domain in DOMAINS:
        domain_data = _build_domain_consensus(annotations, domain, weights)
        result[domain] = domain_data

    # Overall fields
    overall_sevs = []
    overall_weights = []
    probs = []
    confidences = []
    for ann in annotations:
        revised = ann.get("revised_annotation", {})
        if not isinstance(revised, dict):
            continue
        overall_sevs.append(revised.get("overall_severity", "none"))
        w = weights.get(ann["user_id"], 1.0) if weights else 1.0
        overall_weights.append(w)
        try:
            probs.append(float(revised.get("overall_bias_probability", 0)))
        except (TypeError, ValueError):
            pass
        conf = revised.get("confidence")
        if conf:
            confidences.append(conf)

    if weights and overall_sevs:
        result["overall_severity"] = _weighted_ordinal_median(
            overall_sevs, overall_weights
        )
    else:
        result["overall_severity"] = _ordinal_median(overall_sevs)

    result["overall_bias_probability"] = (
        round(sum(probs) / len(probs), 2) if probs else 0.0
    )
    result["confidence"] = _mode(confidences) if confidences else "medium"

    return result


def _build_domain_consensus(
    annotations: list[dict],
    domain: str,
    weights: Optional[dict[int, float]] = None,
) -> dict:
    """Build consensus for a single domain."""
    severities = []
    sev_weights = []
    bool_fields: dict[str, list[bool]] = {}
    cat_fields: dict[str, list[str]] = {}

    # Known categoricals per domain
    categoricals = {
        "spin": ["spin_level"],
        "outcome_reporting": ["primary_outcome_type"],
        "conflict_of_interest": ["funding_type"],
    }

    for ann in annotations:
        revised = ann.get("revised_annotation", {})
        if not isinstance(revised, dict):
            continue
        domain_data = revised.get(domain, {})
        if not isinstance(domain_data, dict):
            continue

        severities.append(domain_data.get("severity", "none"))
        w = weights.get(ann["user_id"], 1.0) if weights else 1.0
        sev_weights.append(w)

        for key, val in domain_data.items():
            if key in ("severity", "evidence_quotes"):
                continue
            if key in categoricals.get(domain, []):
                cat_fields.setdefault(key, []).append(str(val))
            elif isinstance(val, bool):
                bool_fields.setdefault(key, []).append(val)

    result: dict[str, Any] = {}
    if weights and severities:
        result["severity"] = _weighted_ordinal_median(severities, sev_weights)
    else:
        result["severity"] = _ordinal_median(severities)

    for key, vals in bool_fields.items():
        result[key] = _majority_vote_bool(vals)

    for key, vals in cat_fields.items():
        result[key] = _mode(vals)

    return result


# ── Report generation ────────────────────────────────────────────────


def generate_report(analysis: dict) -> str:
    """Generate a Markdown report from analysis results."""
    lines = [
        "# Crowd Annotation Congruence Report",
        "",
        f"Generated: {analysis.get('timestamp', 'N/A')}",
        "",
        f"- Total annotations: {analysis.get('total_annotations', 0)}",
        f"- Total papers: {analysis.get('total_papers', 0)}",
        f"- Multi-rated papers (2+): {analysis.get('multi_rated_papers', 0)}",
        "",
    ]

    # Inter-annotator agreement
    inter = analysis.get("inter_annotator")
    if inter:
        lines.append("## Inter-Annotator Agreement (Fleiss' Kappa)")
        lines.append("")
        lines.append("| Domain | Fleiss' Kappa | N Papers |")
        lines.append("|--------|-------------|----------|")
        for domain in DOMAINS + ["overall"]:
            d = inter.get(domain, {})
            k = d.get("fleiss_kappa", "N/A")
            n = d.get("n_papers", 0)
            label = domain.replace("_", " ").title()
            lines.append(f"| {label} | {k} | {n} |")
        lines.append("")

    # Human vs AI
    hva = analysis.get("human_vs_ai")
    if hva:
        lines.append("## Human vs AI Agreement (Cohen's Kappa)")
        lines.append("")
        lines.append("| Domain | Cohen's Kappa | Exact Agreement | N Papers |")
        lines.append("|--------|-------------|-----------------|----------|")
        for domain in DOMAINS + ["overall"]:
            d = hva.get(domain, {})
            k = d.get("cohens_kappa", "N/A")
            ea = d.get("exact_agreement", "N/A")
            n = d.get("n_papers", 0)
            label = domain.replace("_", " ").title()
            lines.append(f"| {label} | {k} | {ea} | {n} |")
        lines.append("")

    # Anchoring effect
    anchor = analysis.get("anchoring_effect")
    if anchor:
        lines.append("## Anchoring Effect")
        lines.append("")
        lines.append(
            f"- N annotations: {anchor.get('n_annotations', 0)}"
        )
        lines.append(
            f"- Blind agreement with AI: "
            f"{anchor.get('blind_agreement_with_ai', 'N/A')}"
        )
        lines.append(
            f"- Revised agreement with AI: "
            f"{anchor.get('revised_agreement_with_ai', 'N/A')}"
        )
        lines.append(
            f"- Rate of change (blind -> revised): "
            f"{anchor.get('change_rate', 'N/A')}"
        )
        lines.append("")

        mc = anchor.get("mcnemar_test", {})
        if mc:
            lines.append("### McNemar's Test (Anchoring Significance)")
            lines.append(f"- Chi-squared: {mc.get('chi2', 'N/A')}")
            lines.append(f"- p-value: {mc.get('p_value', 'N/A')}")
            lines.append(
                f"- Significant (p < 0.05): {mc.get('significant', 'N/A')}"
            )
            lines.append(
                f"- Changed to agree with AI: "
                f"{mc.get('changed_to_agree', 0)}"
            )
            lines.append(
                f"- Changed to disagree with AI: "
                f"{mc.get('changed_to_disagree', 0)}"
            )
            lines.append("")

        tt = anchor.get("transition_table", {})
        if tt:
            lines.append("### Transition Table (Blind → Revised)")
            lines.append("")
            lines.append("| | Revised Agree | Revised Disagree |")
            lines.append("|---|---|---|")
            lines.append(
                f"| Blind Agree | {tt.get('agree_to_agree', 0)} "
                f"| {tt.get('agree_to_disagree', 0)} |"
            )
            lines.append(
                f"| Blind Disagree | {tt.get('disagree_to_agree', 0)} "
                f"| {tt.get('disagree_to_disagree', 0)} |"
            )
            lines.append("")

    # Annotator quality
    quality = analysis.get("annotator_quality")
    if quality:
        lines.append("## Annotator Quality Scores")
        lines.append("")
        lines.append("| User ID | Agreement Rate | N Papers |")
        lines.append("|---------|---------------|----------|")
        for uid, scores in sorted(
            quality.items(), key=lambda x: x[1]["agreement_rate"], reverse=True
        ):
            lines.append(
                f"| {uid} | {scores['agreement_rate']} "
                f"| {scores['n_papers']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for congruence analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze crowd annotation congruence"
    )
    parser.add_argument(
        "--crowd-db", default="dataset/crowd_annotations.db",
        help="Path to crowd database",
    )
    parser.add_argument(
        "--model", default=None,
        help="AI model name for comparison (default: first available)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output Markdown report path (default: stdout)",
    )
    parser.add_argument(
        "--json", default=None,
        help="Output raw analysis as JSON",
    )

    args = parser.parse_args()

    db = CrowdDatabase(args.crowd_db)
    db.initialize()

    try:
        analysis = analyze_crowd_annotations(db, model_name=args.model)
        report = generate_report(analysis)

        if args.output:
            Path(args.output).write_text(report)
            logger.info("Report written to %s", args.output)
        else:
            print(report)

        if args.json:
            # Serialize with special handling for non-serializable keys
            json_data = json.dumps(analysis, indent=2, default=str)
            Path(args.json).write_text(json_data)
            logger.info("JSON analysis written to %s", args.json)

    finally:
        db.close()


if __name__ == "__main__":
    main()
