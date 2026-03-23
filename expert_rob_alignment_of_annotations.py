"""
Expert RoB Alignment Report

Compares LLM bias annotations against expert Risk of Bias assessments
(Cochrane RoB 2, human reviews) to measure annotation quality. Prints
a detailed Markdown report to stdout.

Usage:
    uv run python expert_rob_alignment_of_annotations.py
    uv run python expert_rob_alignment_of_annotations.py --db dataset/biasbuster.db
    uv run python expert_rob_alignment_of_annotations.py > alignment_report.md
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from database import Database
from config import Config
from evaluation.metrics import BinaryMetrics
from evaluation.scorer import SEVERITY_ORDER
from enrichers.retraction_classifier import classify_retraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Collapse our 5-level severity to 3 levels for fair Cochrane comparison
FIVE_TO_THREE: dict[str, str] = {
    "none": "low_concern",
    "low": "low_concern",
    "moderate": "moderate_concern",
    "high": "high_concern",
    "critical": "high_concern",
}

# Collapse Cochrane RoB 2 to the same 3 levels
COCHRANE_TO_THREE: dict[str, str] = {
    "low": "low_concern",
    "some_concerns": "moderate_concern",
    "high": "high_concern",
}

THREE_LEVEL_LABELS = ["low_concern", "moderate_concern", "high_concern"]
THREE_LEVEL_ORDER: dict[str, int] = {label: i for i, label in enumerate(THREE_LEVEL_LABELS)}

# Cochrane RoB 2 domain → (our domain, mapping confidence)
# Only domains with at least moderate overlap are used for comparison.
DOMAIN_MAPPING: dict[str, tuple[Optional[str], str]] = {
    "randomization_bias": ("methodology", "moderate"),
    "reporting_bias": ("outcome_reporting", "moderate"),
    "missing_outcome_bias": ("outcome_reporting", "weak"),
    "measurement_bias": ("methodology", "weak"),
    "deviation_bias": (None, "none"),  # no clean mapping
}

# Landis & Koch (1977) interpretation of Cohen's kappa
KAPPA_THRESHOLDS: list[tuple[float, float, str]] = [
    (-1.0, 0.00, "Poor (less than chance)"),
    (0.00, 0.20, "Slight agreement"),
    (0.20, 0.40, "Fair agreement"),
    (0.40, 0.60, "Moderate agreement"),
    (0.60, 0.80, "Substantial agreement"),
    (0.80, 1.01, "Almost perfect agreement"),
]


def interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch interpretation string for a kappa value."""
    for lo, hi, label in KAPPA_THRESHOLDS:
        if lo <= kappa < hi:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Scale mapping
# ---------------------------------------------------------------------------

def collapse_annotation_severity(severity: str) -> Optional[str]:
    """Map 5-level annotation severity to 3-level collapsed scale."""
    return FIVE_TO_THREE.get(severity.lower().strip() if severity else "")


def collapse_cochrane_rob(rob: str) -> Optional[str]:
    """Map Cochrane RoB 2 rating to 3-level collapsed scale."""
    return COCHRANE_TO_THREE.get(rob.lower().strip() if rob else "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class PairedObservation:
    """One annotation paired with its Cochrane expert RoB assessment."""
    pmid: str
    model_name: str
    cochrane_rob: str          # raw Cochrane overall_rob
    annotation_severity: str   # raw annotation overall_severity
    cochrane_collapsed: str    # 3-level
    annotation_collapsed: str  # 3-level
    annotation_json: dict = field(default_factory=dict)
    # Cochrane domain-level (may be empty strings)
    cochrane_domains: dict = field(default_factory=dict)


def load_paired_data(db: Database) -> list[PairedObservation]:
    """Load annotations paired with Cochrane RoB expert assessments.

    Returns only pairs where both Cochrane RoB and annotation severity
    can be collapsed to the 3-level scale.
    """
    rows = db.conn.execute("""
        SELECT a.pmid, a.model_name, a.overall_severity, a.annotation,
               p.overall_rob,
               p.randomization_bias, p.deviation_bias,
               p.missing_outcome_bias, p.measurement_bias, p.reporting_bias
        FROM annotations a
        JOIN papers p ON a.pmid = p.pmid
        WHERE p.source = 'cochrane_rob'
          AND p.overall_rob IS NOT NULL
          AND p.excluded = 0
    """).fetchall()

    pairs = []
    for row in rows:
        (pmid, model, ann_sev, ann_json_str, overall_rob,
         rand_bias, dev_bias, miss_bias, meas_bias, rep_bias) = row

        cochrane_c = collapse_cochrane_rob(overall_rob)
        annotation_c = collapse_annotation_severity(ann_sev)

        if cochrane_c is None or annotation_c is None:
            logger.warning(
                f"PMID {pmid}: could not map scales "
                f"(cochrane={overall_rob!r}, annotation={ann_sev!r}), skipping"
            )
            continue

        try:
            ann_dict = json.loads(ann_json_str) if isinstance(ann_json_str, str) else ann_json_str
        except (json.JSONDecodeError, TypeError):
            ann_dict = {}

        pairs.append(PairedObservation(
            pmid=pmid,
            model_name=model,
            cochrane_rob=overall_rob,
            annotation_severity=ann_sev,
            cochrane_collapsed=cochrane_c,
            annotation_collapsed=annotation_c,
            annotation_json=ann_dict,
            cochrane_domains={
                "randomization_bias": rand_bias or "",
                "deviation_bias": dev_bias or "",
                "missing_outcome_bias": miss_bias or "",
                "measurement_bias": meas_bias or "",
                "reporting_bias": rep_bias or "",
            },
        ))

    return pairs


@dataclass
class CoverageStats:
    """Coverage of expert ground truth across annotations."""
    total_cochrane_papers: int = 0
    annotated_cochrane_papers: int = 0
    total_annotations: int = 0
    annotations_with_expert: int = 0
    annotations_without_expert: int = 0
    cochrane_by_rob: dict = field(default_factory=dict)  # rob_level → (annotated, total)
    total_human_reviews: int = 0


def load_coverage_stats(db: Database) -> CoverageStats:
    """Compute coverage statistics for expert ground truth."""
    stats = CoverageStats()

    # Total Cochrane papers
    stats.total_cochrane_papers = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'cochrane_rob' AND excluded = 0"
    ).fetchone()[0]

    # Annotated Cochrane papers
    stats.annotated_cochrane_papers = db.conn.execute("""
        SELECT COUNT(DISTINCT a.pmid) FROM annotations a
        JOIN papers p ON a.pmid = p.pmid
        WHERE p.source = 'cochrane_rob' AND p.excluded = 0
    """).fetchone()[0]

    # Total annotations
    stats.total_annotations = db.conn.execute(
        "SELECT COUNT(*) FROM annotations"
    ).fetchone()[0]

    # Annotations with/without Cochrane expert data
    stats.annotations_with_expert = db.conn.execute("""
        SELECT COUNT(*) FROM annotations a
        JOIN papers p ON a.pmid = p.pmid
        WHERE p.source = 'cochrane_rob' AND p.overall_rob IS NOT NULL AND p.excluded = 0
    """).fetchone()[0]

    stats.annotations_without_expert = (
        stats.total_annotations - stats.annotations_with_expert
    )

    # Breakdown by Cochrane RoB level
    rows = db.conn.execute("""
        SELECT p.overall_rob,
               COUNT(*) as total,
               SUM(CASE WHEN a.pmid IS NOT NULL THEN 1 ELSE 0 END) as annotated
        FROM papers p
        LEFT JOIN (SELECT DISTINCT pmid FROM annotations) a ON p.pmid = a.pmid
        WHERE p.source = 'cochrane_rob' AND p.overall_rob IS NOT NULL AND p.excluded = 0
        GROUP BY p.overall_rob
    """).fetchall()
    stats.cochrane_by_rob = {row[0]: (row[2], row[1]) for row in rows}

    # Human reviews
    stats.total_human_reviews = db.conn.execute(
        "SELECT COUNT(*) FROM human_reviews WHERE validated = 1"
    ).fetchone()[0]

    return stats


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    pairs: list[PairedObservation],
) -> dict[str, dict[str, int]]:
    """Build a 3x3 confusion matrix (expert rows, annotation columns)."""
    matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in THREE_LEVEL_LABELS}
        for gt in THREE_LEVEL_LABELS
    }
    for p in pairs:
        matrix[p.cochrane_collapsed][p.annotation_collapsed] += 1
    return matrix


def compute_weighted_kappa(pairs: list[PairedObservation]) -> float:
    """Linearly-weighted Cohen's kappa for the 3-level ordinal scale.

    Uses the standard formula:
        kappa_w = 1 - (sum w_ij * O_ij) / (sum w_ij * E_ij)
    where w_ij = |i - j| / (k - 1), O = observed, E = expected.
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    k = len(THREE_LEVEL_LABELS)
    # Count observed frequencies
    observed = [[0] * k for _ in range(k)]
    for p in pairs:
        i = THREE_LEVEL_ORDER[p.cochrane_collapsed]
        j = THREE_LEVEL_ORDER[p.annotation_collapsed]
        observed[i][j] += 1

    # Marginals
    row_sums = [sum(observed[i]) for i in range(k)]
    col_sums = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    # Weight matrix (linear)
    max_dist = k - 1
    if max_dist == 0:
        return 1.0

    weights = [[abs(i - j) / max_dist for j in range(k)] for i in range(k)]

    # Weighted observed and expected
    w_observed = sum(
        weights[i][j] * observed[i][j]
        for i in range(k) for j in range(k)
    )
    w_expected = sum(
        weights[i][j] * row_sums[i] * col_sums[j] / n
        for i in range(k) for j in range(k)
    )

    if w_expected == 0:
        return 1.0

    return 1.0 - (w_observed / w_expected)


@dataclass
class DirectionalBias:
    """Directional bias analysis: does the LLM over- or under-flag?"""
    mean_difference: float = 0.0  # positive = LLM rates higher than expert
    over_flags: int = 0           # LLM severity > expert
    under_flags: int = 0          # LLM severity < expert
    exact: int = 0                # exact match on 3-level scale
    n: int = 0


def compute_directional_bias(pairs: list[PairedObservation]) -> DirectionalBias:
    """Measure systematic over- or under-flagging relative to expert."""
    if not pairs:
        return DirectionalBias()

    diffs = []
    over = under = exact = 0
    for p in pairs:
        expert_ord = THREE_LEVEL_ORDER[p.cochrane_collapsed]
        ann_ord = THREE_LEVEL_ORDER[p.annotation_collapsed]
        diff = ann_ord - expert_ord
        diffs.append(diff)
        if diff > 0:
            over += 1
        elif diff < 0:
            under += 1
        else:
            exact += 1

    return DirectionalBias(
        mean_difference=sum(diffs) / len(diffs),
        over_flags=over,
        under_flags=under,
        exact=exact,
        n=len(pairs),
    )


def compute_binary_detection(pairs: list[PairedObservation]) -> BinaryMetrics:
    """Collapse to binary (low_concern=negative, else=positive) and compute metrics.

    This answers: when Cochrane says there's a concern, does the LLM detect it?
    """
    metrics = BinaryMetrics()
    for p in pairs:
        expert_pos = p.cochrane_collapsed != "low_concern"
        ann_pos = p.annotation_collapsed != "low_concern"
        if expert_pos and ann_pos:
            metrics.tp += 1
        elif expert_pos and not ann_pos:
            metrics.fn += 1
        elif not expert_pos and ann_pos:
            metrics.fp += 1
        else:
            metrics.tn += 1
    return metrics


# ---------------------------------------------------------------------------
# Domain-level analysis
# ---------------------------------------------------------------------------

def compute_domain_alignment(
    pairs: list[PairedObservation],
) -> dict[str, dict]:
    """Compute alignment for Cochrane RoB domains mapped to our domains.

    Returns dict keyed by Cochrane domain name. Each value has:
        our_domain, mapping_confidence, n, kappa, exact_match
    Returns empty dict if no domain-level data is available.
    """
    results: dict[str, dict] = {}

    for cochrane_domain, (our_domain, confidence) in DOMAIN_MAPPING.items():
        if our_domain is None:
            continue

        # Collect pairs where Cochrane domain has actual data
        domain_pairs = []
        for p in pairs:
            cochrane_val = p.cochrane_domains.get(cochrane_domain, "").strip()
            if not cochrane_val:
                continue

            cochrane_c = collapse_cochrane_rob(cochrane_val)
            if cochrane_c is None:
                continue

            # Get our domain severity from annotation JSON
            domain_data = p.annotation_json.get(our_domain, {})
            if not isinstance(domain_data, dict):
                continue
            our_severity = domain_data.get("severity", "")
            annotation_c = collapse_annotation_severity(our_severity)
            if annotation_c is None:
                continue

            domain_pairs.append(PairedObservation(
                pmid=p.pmid,
                model_name=p.model_name,
                cochrane_rob=cochrane_val,
                annotation_severity=our_severity,
                cochrane_collapsed=cochrane_c,
                annotation_collapsed=annotation_c,
            ))

        if not domain_pairs:
            continue

        kappa = compute_weighted_kappa(domain_pairs)
        exact = sum(
            1 for dp in domain_pairs
            if dp.cochrane_collapsed == dp.annotation_collapsed
        ) / len(domain_pairs)

        results[cochrane_domain] = {
            "our_domain": our_domain,
            "mapping_confidence": confidence,
            "n": len(domain_pairs),
            "kappa": kappa,
            "exact_match": exact,
            "confusion": compute_confusion_matrix(domain_pairs),
        }

    return results


# ---------------------------------------------------------------------------
# Retraction severity floor compliance
# ---------------------------------------------------------------------------

@dataclass
class RetractionFloorResult:
    """Results of checking retraction severity floor compliance."""
    total_retracted_annotated: int = 0
    total_retracted_papers: int = 0
    meets_floor: int = 0
    below_floor: int = 0
    no_floor: int = 0  # non-bias retractions (floor=None)
    by_category: dict = field(default_factory=dict)  # category → {meets, below, n, floor}
    violations: list = field(default_factory=list)    # PMIDs that violate the floor


def load_retraction_floor_compliance(db: Database) -> RetractionFloorResult:
    """Check if annotated retracted papers meet their severity floor."""
    result = RetractionFloorResult()

    result.total_retracted_papers = db.conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source = 'retraction_watch' AND excluded = 0"
    ).fetchone()[0]

    rows = db.conn.execute("""
        SELECT a.pmid, a.overall_severity, p.retraction_reasons, p.title
        FROM annotations a
        JOIN papers p ON a.pmid = p.pmid
        WHERE p.source = 'retraction_watch' AND p.excluded = 0
    """).fetchall()

    result.total_retracted_annotated = len(rows)

    for pmid, ann_sev, reasons_json, title in rows:
        try:
            reasons = json.loads(reasons_json) if reasons_json else []
        except (json.JSONDecodeError, TypeError):
            reasons = []

        floor, category = classify_retraction(reasons, title=title or "")

        ann_ord = SEVERITY_ORDER.get(ann_sev.lower().strip() if ann_sev else "", -1)

        if floor is None:
            result.no_floor += 1
            # Still track the category distribution
            cat_entry = result.by_category.setdefault(
                category, {"meets": 0, "below": 0, "no_floor": 0, "n": 0, "floor": "none"}
            )
            cat_entry["no_floor"] += 1
            cat_entry["n"] += 1
            continue

        floor_ord = SEVERITY_ORDER.get(floor, -1)

        cat_entry = result.by_category.setdefault(
            category, {"meets": 0, "below": 0, "no_floor": 0, "n": 0, "floor": floor}
        )
        cat_entry["n"] += 1

        if ann_ord >= floor_ord:
            result.meets_floor += 1
            cat_entry["meets"] += 1
        else:
            result.below_floor += 1
            cat_entry["below"] += 1
            result.violations.append({
                "pmid": pmid,
                "annotation_severity": ann_sev,
                "floor": floor,
                "category": category,
            })

    return result


# ---------------------------------------------------------------------------
# Heuristic suspicion alignment
# ---------------------------------------------------------------------------

@dataclass
class SuspicionAlignment:
    """Cross-tabulation of heuristic suspicion vs annotation severity."""
    by_suspicion: dict = field(default_factory=dict)  # suspicion_level → {severity → count}
    total: int = 0
    high_suspicion_moderate_plus: int = 0
    high_suspicion_total: int = 0
    low_suspicion_low_or_none: int = 0
    low_suspicion_total: int = 0


def load_suspicion_alignment(db: Database) -> SuspicionAlignment:
    """Compare heuristic suspicion levels against annotation severity."""
    result = SuspicionAlignment()

    rows = db.conn.execute("""
        SELECT e.suspicion_level, a.overall_severity, COUNT(*) as cnt
        FROM enrichments e
        JOIN annotations a ON e.pmid = a.pmid
        WHERE e.suspicion_level IS NOT NULL AND e.suspicion_level != ''
        GROUP BY e.suspicion_level, a.overall_severity
    """).fetchall()

    for susp, sev, cnt in rows:
        result.by_suspicion.setdefault(susp, defaultdict(int))
        result.by_suspicion[susp][sev] = cnt
        result.total += cnt

        sev_ord = SEVERITY_ORDER.get(sev.lower().strip() if sev else "", 0)
        if susp == "high":
            result.high_suspicion_total += cnt
            if sev_ord >= SEVERITY_ORDER["moderate"]:
                result.high_suspicion_moderate_plus += cnt
        elif susp == "low":
            result.low_suspicion_total += cnt
            if sev_ord <= SEVERITY_ORDER["low"]:
                result.low_suspicion_low_or_none += cnt

    return result


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceCalibration:
    """Confidence vs severity cross-tabulation."""
    by_confidence: dict = field(default_factory=dict)  # confidence → {severity → count}
    total: int = 0
    null_confidence: int = 0


def load_confidence_calibration(db: Database) -> ConfidenceCalibration:
    """Cross-tabulate annotation confidence vs severity."""
    result = ConfidenceCalibration()

    rows = db.conn.execute("""
        SELECT confidence, overall_severity, COUNT(*) as cnt
        FROM annotations
        GROUP BY confidence, overall_severity
    """).fetchall()

    for conf, sev, cnt in rows:
        result.total += cnt
        if conf is None or conf.strip() == "":
            result.null_confidence += cnt
            continue
        result.by_confidence.setdefault(conf, defaultdict(int))
        result.by_confidence[conf][sev] = cnt

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_pct(num: int, denom: int) -> str:
    """Format as 'XX.X%'."""
    if denom == 0:
        return "N/A"
    return f"{100.0 * num / denom:.1f}%"


def _fmt_confusion_table(matrix: dict[str, dict[str, int]]) -> str:
    """Format a 3x3 confusion matrix as a Markdown table."""
    # Short labels for readability
    short = {"low_concern": "Low", "moderate_concern": "Moderate", "high_concern": "High"}
    header = "| Expert \\ LLM | " + " | ".join(short[l] for l in THREE_LEVEL_LABELS) + " | Row Total |"
    sep = "|" + "---|" * (len(THREE_LEVEL_LABELS) + 2)

    rows = []
    for gt_label in THREE_LEVEL_LABELS:
        cells = [str(matrix[gt_label][pred]) for pred in THREE_LEVEL_LABELS]
        row_total = sum(matrix[gt_label].values())
        rows.append(f"| **{short[gt_label]}** | " + " | ".join(cells) + f" | {row_total} |")

    # Column totals
    col_totals = [sum(matrix[gt][pred] for gt in THREE_LEVEL_LABELS) for pred in THREE_LEVEL_LABELS]
    grand_total = sum(col_totals)
    rows.append(f"| **Col Total** | " + " | ".join(str(c) for c in col_totals) + f" | {grand_total} |")

    return "\n".join([header, sep] + rows)


def format_report(
    coverage: CoverageStats,
    pairs: list[PairedObservation],
    model_names: list[str],
    db_path: str,
    retraction: Optional[RetractionFloorResult] = None,
    suspicion: Optional[SuspicionAlignment] = None,
    confidence: Optional[ConfidenceCalibration] = None,
) -> str:
    """Build the full Markdown alignment report."""
    lines: list[str] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# Expert RoB Alignment Report")
    lines.append(f"Generated: {now} | Database: `{db_path}`\n")

    # ---- Section 1: Coverage ----
    lines.append("## 1. Coverage\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total annotations | {coverage.total_annotations} |")
    lines.append(
        f"| Annotations with Cochrane expert RoB | "
        f"{coverage.annotations_with_expert} "
        f"({_fmt_pct(coverage.annotations_with_expert, coverage.total_annotations)}) |"
    )
    lines.append(
        f"| Annotations without expert ground truth | "
        f"{coverage.annotations_without_expert} "
        f"({_fmt_pct(coverage.annotations_without_expert, coverage.total_annotations)}) |"
    )
    lines.append(
        f"| Cochrane papers total | {coverage.total_cochrane_papers} |"
    )
    lines.append(
        f"| Cochrane papers annotated | {coverage.annotated_cochrane_papers} "
        f"({_fmt_pct(coverage.annotated_cochrane_papers, coverage.total_cochrane_papers)}) |"
    )
    lines.append(
        f"| Human-validated reviews | {coverage.total_human_reviews} |"
    )

    if coverage.cochrane_by_rob:
        lines.append("\n### Cochrane Papers by RoB Level\n")
        lines.append("| Cochrane RoB | Annotated | Total | Coverage |")
        lines.append("|--------------|-----------|-------|----------|")
        for rob_level in ["low", "some_concerns", "high"]:
            if rob_level in coverage.cochrane_by_rob:
                annotated, total = coverage.cochrane_by_rob[rob_level]
                lines.append(
                    f"| {rob_level} | {annotated} | {total} | {_fmt_pct(annotated, total)} |"
                )

    if not pairs:
        lines.append("\n**No paired observations available. Cannot compute alignment metrics.**")
        return "\n".join(lines)

    # ---- Section 2: Overall Alignment ----
    lines.append("\n## 2. Overall Severity Alignment\n")
    lines.append(
        "Scale mapping: our {none, low} = Low, {moderate} = Moderate, "
        "{high, critical} = High\n"
    )

    matrix = compute_confusion_matrix(pairs)
    lines.append("### Confusion Matrix\n")
    lines.append(_fmt_confusion_table(matrix))

    kappa = compute_weighted_kappa(pairs)
    kappa_interp = interpret_kappa(kappa)
    lines.append(f"\n**Weighted Cohen's kappa: {kappa:.3f}** ({kappa_interp})\n")

    # Exact match rate on 3-level scale
    exact_rate = sum(
        1 for p in pairs if p.cochrane_collapsed == p.annotation_collapsed
    ) / len(pairs)
    lines.append(f"Exact agreement (3-level): {exact_rate:.1%} ({int(exact_rate * len(pairs))}/{len(pairs)})\n")

    # Binary detection
    binary = compute_binary_detection(pairs)
    lines.append("### Binary Detection (any concern vs low)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Precision | {binary.precision:.3f} |")
    lines.append(f"| Recall | {binary.recall:.3f} |")
    lines.append(f"| F1 | {binary.f1:.3f} |")
    lines.append(f"| Accuracy | {binary.accuracy:.3f} |")
    lines.append(
        f"| (TP={binary.tp}, FP={binary.fp}, TN={binary.tn}, FN={binary.fn}) | |"
    )

    # ---- Section 3: Directional Bias ----
    lines.append("\n## 3. Directional Bias Analysis\n")
    db_result = compute_directional_bias(pairs)
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Mean difference (LLM - Expert) | {db_result.mean_difference:+.3f} |"
    )
    lines.append(
        f"| LLM over-flags (rates higher) | "
        f"{db_result.over_flags} ({_fmt_pct(db_result.over_flags, db_result.n)}) |"
    )
    lines.append(
        f"| LLM under-flags (rates lower) | "
        f"{db_result.under_flags} ({_fmt_pct(db_result.under_flags, db_result.n)}) |"
    )
    lines.append(
        f"| Exact match | "
        f"{db_result.exact} ({_fmt_pct(db_result.exact, db_result.n)}) |"
    )

    if db_result.mean_difference > 0.1:
        lines.append(
            "\nThe LLM tends to **over-flag** relative to Cochrane experts. "
            "This is expected: abstract-level analysis can detect reporting "
            "concerns (spin, selective statistics) that full-text RoB assessment "
            "may weight differently."
        )
    elif db_result.mean_difference < -0.1:
        lines.append(
            "\nThe LLM tends to **under-flag** relative to Cochrane experts. "
            "This may reflect information asymmetry: methodology concerns "
            "(allocation concealment, blinding details) are often only visible "
            "in full text."
        )
    else:
        lines.append(
            "\nNo strong directional bias detected. The LLM's severity "
            "ratings are approximately symmetric around Cochrane expert judgments."
        )

    # ---- Section 4: Per-Model Breakdown ----
    if len(model_names) > 1:
        lines.append("\n## 4. Per-Model Breakdown\n")
        lines.append("| Model | N | Kappa | Exact Match | F1 (binary) | Mean Diff |")
        lines.append("|-------|---|-------|-------------|-------------|-----------|")
        for model in sorted(model_names):
            model_pairs = [p for p in pairs if p.model_name == model]
            if not model_pairs:
                continue
            m_kappa = compute_weighted_kappa(model_pairs)
            m_exact = sum(
                1 for p in model_pairs
                if p.cochrane_collapsed == p.annotation_collapsed
            ) / len(model_pairs)
            m_binary = compute_binary_detection(model_pairs)
            m_dir = compute_directional_bias(model_pairs)
            lines.append(
                f"| {model} | {len(model_pairs)} | {m_kappa:.3f} | "
                f"{m_exact:.1%} | {m_binary.f1:.3f} | {m_dir.mean_difference:+.3f} |"
            )
    else:
        lines.append(f"\n## 4. Per-Model Breakdown\n")
        lines.append(f"Single model: **{model_names[0] if model_names else 'unknown'}** (N={len(pairs)})")

    # ---- Section 5: Domain-Level Alignment ----
    lines.append("\n## 5. Domain-Level Alignment\n")
    domain_results = compute_domain_alignment(pairs)
    if domain_results:
        lines.append(
            "**Note:** Domain mapping between Cochrane RoB 2 and our taxonomy is "
            "approximate. Interpret with caution.\n"
        )
        lines.append("| Cochrane Domain | Our Domain | Confidence | N | Kappa | Exact Match |")
        lines.append("|-----------------|------------|------------|---|-------|-------------|")
        for coch_domain, result in domain_results.items():
            lines.append(
                f"| {coch_domain} | {result['our_domain']} | "
                f"{result['mapping_confidence']} | {result['n']} | "
                f"{result['kappa']:.3f} | {result['exact_match']:.1%} |"
            )
    else:
        lines.append(
            "Per-domain Cochrane RoB 2 ratings (randomization, deviation, "
            "missing outcome, measurement, reporting) are not yet populated "
            "in the database. The Cochrane collector currently extracts only "
            "the overall RoB judgment. To enable this section, update the "
            "LLM extraction prompt in `collectors/cochrane_rob.py` to also "
            "extract per-domain ratings, then re-run collection.\n\n"
            "*Note: this refers to Cochrane RoB 2 assessment domains, not "
            "the medical domain column (cardiovascular, metabolic, etc.) "
            "which IS populated.*"
        )

    # ---- Section 6: Retraction Severity Floor Compliance ----
    lines.append("\n## 6. Retraction Severity Floor Compliance\n")
    if retraction and retraction.total_retracted_annotated > 0:
        has_floor = retraction.meets_floor + retraction.below_floor
        lines.append(
            f"Retracted papers with annotations: {retraction.total_retracted_annotated} "
            f"/ {retraction.total_retracted_papers} "
            f"({_fmt_pct(retraction.total_retracted_annotated, retraction.total_retracted_papers)})\n"
        )
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(
            f"| Meets severity floor | {retraction.meets_floor} "
            f"({_fmt_pct(retraction.meets_floor, has_floor)}) |"
        )
        lines.append(
            f"| Below severity floor | {retraction.below_floor} "
            f"({_fmt_pct(retraction.below_floor, has_floor)}) |"
        )
        lines.append(
            f"| No floor (non-bias retraction) | {retraction.no_floor} |"
        )

        if retraction.by_category:
            lines.append("\n### By Retraction Category\n")
            lines.append("| Category | Floor | Meets | Below | No Floor | Total |")
            lines.append("|----------|-------|-------|-------|----------|-------|")
            for cat, data in sorted(
                retraction.by_category.items(),
                key=lambda x: x[1]["n"],
                reverse=True,
            ):
                lines.append(
                    f"| {cat} | {data['floor']} | {data['meets']} | "
                    f"{data['below']} | {data['no_floor']} | {data['n']} |"
                )

        if retraction.below_floor > 0:
            lines.append(
                f"\n**{retraction.below_floor} annotation(s) rate below "
                f"the retraction severity floor.** These papers were retracted "
                f"for bias-relevant reasons but the LLM assessed them as lower "
                f"severity than expected. This may indicate the abstract alone "
                f"doesn't reveal the issues that led to retraction."
            )
            # Show first few violations
            for v in retraction.violations[:5]:
                lines.append(
                    f"- PMID {v['pmid']}: annotated={v['annotation_severity']}, "
                    f"floor={v['floor']} ({v['category']})"
                )
            if len(retraction.violations) > 5:
                lines.append(
                    f"- ... and {len(retraction.violations) - 5} more"
                )
        else:
            lines.append(
                "\nAll annotated retracted papers meet their severity floor."
            )
    else:
        lines.append("No annotated retracted papers found.")

    # ---- Section 7: Heuristic Suspicion Alignment ----
    lines.append("\n## 7. Heuristic Suspicion Alignment\n")
    if suspicion and suspicion.total > 0:
        lines.append(
            "Papers flagged as high-suspicion by heuristic enrichment "
            "(effect size auditing, outcome switching) should tend toward "
            "higher annotation severity.\n"
        )
        for susp_level in ["high", "low"]:
            if susp_level in suspicion.by_suspicion:
                dist = suspicion.by_suspicion[susp_level]
                total = sum(dist.values())
                lines.append(f"### {susp_level.title()}-suspicion papers (N={total})\n")
                lines.append("| Severity | Count | % |")
                lines.append("|----------|-------|---|")
                for sev in ["none", "low", "moderate", "high", "critical"]:
                    cnt = dist.get(sev, 0)
                    lines.append(f"| {sev} | {cnt} | {_fmt_pct(cnt, total)} |")

        if suspicion.high_suspicion_total > 0:
            rate = suspicion.high_suspicion_moderate_plus / suspicion.high_suspicion_total
            lines.append(
                f"\nHigh-suspicion papers rated moderate+: "
                f"{suspicion.high_suspicion_moderate_plus}/{suspicion.high_suspicion_total} "
                f"({rate:.1%})"
            )
            if rate >= 0.7:
                lines.append(
                    "— Good: heuristic flags are confirmed by LLM assessment."
                )
            elif rate >= 0.5:
                lines.append(
                    "— Moderate alignment. Some heuristic flags may be noisy, "
                    "or the LLM may be under-flagging."
                )
            else:
                lines.append(
                    "— **Weak alignment.** The heuristic enrichment and LLM "
                    "annotation disagree frequently. Investigate whether the "
                    "heuristic is too aggressive or the LLM too lenient."
                )

        if suspicion.low_suspicion_total > 0:
            rate = suspicion.low_suspicion_low_or_none / suspicion.low_suspicion_total
            lines.append(
                f"\nLow-suspicion papers rated low/none: "
                f"{suspicion.low_suspicion_low_or_none}/{suspicion.low_suspicion_total} "
                f"({rate:.1%})"
            )
    else:
        lines.append("No enrichment data available for suspicion alignment.")

    # ---- Section 8: Confidence Calibration ----
    lines.append("\n## 8. Confidence Calibration\n")
    if confidence and confidence.total > 0:
        if confidence.null_confidence > 0:
            lines.append(
                f"*{confidence.null_confidence} annotations have no confidence "
                f"value ({_fmt_pct(confidence.null_confidence, confidence.total)} "
                f"of total).*\n"
            )
        lines.append(
            "A well-calibrated model should show high confidence on clear-cut "
            "cases (very low or very high severity) and medium/low confidence "
            "on ambiguous cases (moderate severity).\n"
        )
        for conf_level in ["high", "medium", "low"]:
            if conf_level in confidence.by_confidence:
                dist = confidence.by_confidence[conf_level]
                total = sum(dist.values())
                lines.append(f"### {conf_level.title()} confidence (N={total})\n")
                lines.append("| Severity | Count | % |")
                lines.append("|----------|-------|---|")
                for sev in ["none", "low", "moderate", "high", "critical"]:
                    cnt = dist.get(sev, 0)
                    lines.append(f"| {sev} | {cnt} | {_fmt_pct(cnt, total)} |")
                lines.append("")

        # Compute "decisiveness" for each confidence level
        # High confidence should have more extreme (none/critical) ratings
        for conf_level in ["high", "medium", "low"]:
            if conf_level in confidence.by_confidence:
                dist = confidence.by_confidence[conf_level]
                total = sum(dist.values())
                if total == 0:
                    continue
                extreme = dist.get("none", 0) + dist.get("critical", 0) + dist.get("high", 0)
                moderate = dist.get("moderate", 0)
                lines.append(
                    f"**{conf_level.title()}** confidence: "
                    f"{_fmt_pct(extreme, total)} extreme (none/high/critical), "
                    f"{_fmt_pct(moderate, total)} moderate"
                )
        lines.append("")
    else:
        lines.append("No confidence data available.")

    # ---- Section 9: Human Review Alignment ----
    lines.append("\n## 9. Human Review Alignment\n")
    if coverage.total_human_reviews > 0:
        lines.append(
            f"{coverage.total_human_reviews} validated human reviews found. "
            "Human review alignment analysis will be added in a future version."
        )
    else:
        lines.append(
            "No validated human reviews found in the database. "
            "This section will activate once human review data is available."
        )

    # ---- Section 10: Interpretation Guide ----
    lines.append("\n## 10. Interpretation Guide\n")
    lines.append("""\
### Reading the Confusion Matrix

- **Diagonal cells** (top-left to bottom-right) = agreement between expert and LLM
- **Above diagonal** = LLM rates higher severity than expert (over-flagging)
- **Below diagonal** = LLM rates lower severity than expert (under-flagging)

### Kappa Interpretation (Landis & Koch, 1977)

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0.00 | Poor (less than chance) |
| 0.00 - 0.20 | Slight agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect agreement |

### Important Caveats

1. **Different constructs**: Cochrane RoB 2 assesses internal validity of trial \
methodology from **full text**. Our annotations assess bias in how results are \
*reported* in **abstracts only**. A trial can have low methodological bias but \
high reporting bias (e.g., spin in conclusions), or vice versa.

2. **Scale compression**: Collapsing 5 levels to 3 loses information. Our "none" \
and "low" both map to Cochrane "low", but "none" (no concern) differs from "low" \
(minor concern noted).

3. **Information asymmetry**: Cochrane assessors see full text, methods sections, \
protocols, and trial registrations. LLM annotators see only the abstract. This \
systematically limits detection of methodology issues (allocation concealment, \
blinding details, per-protocol deviations).

4. **Expected asymmetry in errors**: LLMs should tend to *under*-flag methodology \
issues (hidden in full text) but may *over*-flag reporting issues (visible in \
abstract). Symmetric agreement is NOT expected, and moderate over-flagging on \
reporting dimensions may actually be correct.

5. **Sample size**: With small N, kappa estimates have wide confidence intervals. \
A kappa of 0.60 with N=50 is much less certain than with N=500.""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load data, compute metrics, print report."""
    parser = argparse.ArgumentParser(
        description="Compare LLM annotations against expert RoB assessments"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to SQLite database (default: from config.py)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    db_path = args.db or Config().db_path
    db = Database(db_path)

    try:
        # Load data
        coverage = load_coverage_stats(db)
        pairs = load_paired_data(db)
        model_names = list({p.model_name for p in pairs}) or ["unknown"]
        retraction = load_retraction_floor_compliance(db)
        suspicion = load_suspicion_alignment(db)
        confidence = load_confidence_calibration(db)

        logger.info(
            f"Loaded {len(pairs)} Cochrane pairs, "
            f"{retraction.total_retracted_annotated} retracted, "
            f"{suspicion.total} enriched"
        )

        # Generate and print report
        report = format_report(
            coverage, pairs, model_names, db_path,
            retraction=retraction,
            suspicion=suspicion,
            confidence=confidence,
        )
        print(report)

    finally:
        db.close()


if __name__ == "__main__":
    main()
