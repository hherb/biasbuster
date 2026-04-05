"""
Evaluation Metrics

Computes per-dimension and aggregate metrics for bias detection:
- Binary classification: precision, recall, F1, accuracy
- Ordinal severity: Cohen's kappa (weighted), MAE, confusion matrix
- Flag-level: per-flag accuracy for key boolean indicators
- Calibration: predicted probability vs actual bias rate
- Verification quality: how many appropriate sources the model suggests

All metrics are computed per-model to enable direct head-to-head comparison.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .scorer import (
    DimensionScore,
    ParsedAssessment,
    SEVERITY_ORDER,
    SEVERITY_LABELS,
    severity_to_int,
)

__all__ = [
    "BinaryMetrics",
    "OrdinalMetrics",
    "FlagMetrics",
    "CalibrationBin",
    "DimensionEvaluation",
    "ModelEvaluation",
    "evaluate_model",
]


@dataclass
class BinaryMetrics:
    """Standard binary classification metrics."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "n": self.n,
        }


@dataclass
class OrdinalMetrics:
    """Metrics for ordinal severity ratings."""
    predictions: list[int] = field(default_factory=list)
    ground_truths: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.predictions)

    @property
    def mae(self) -> float:
        """Mean Absolute Error on severity scale."""
        if not self.predictions:
            return 0.0
        return sum(
            abs(p - g) for p, g in zip(self.predictions, self.ground_truths)
        ) / len(self.predictions)

    @property
    def exact_match(self) -> float:
        """Proportion of exact severity matches."""
        if not self.predictions:
            return 0.0
        return sum(
            1 for p, g in zip(self.predictions, self.ground_truths) if p == g
        ) / len(self.predictions)

    @property
    def within_one(self) -> float:
        """Proportion within one severity level."""
        if not self.predictions:
            return 0.0
        return sum(
            1 for p, g in zip(self.predictions, self.ground_truths) if abs(p - g) <= 1
        ) / len(self.predictions)

    def confusion_matrix(self) -> dict:
        """5x5 confusion matrix indexed by severity label."""
        matrix = {gt: {pred: 0 for pred in SEVERITY_LABELS} for gt in SEVERITY_LABELS}
        for p, g in zip(self.predictions, self.ground_truths):
            if 0 <= p < len(SEVERITY_LABELS) and 0 <= g < len(SEVERITY_LABELS):
                matrix[SEVERITY_LABELS[g]][SEVERITY_LABELS[p]] += 1
        return matrix

    def weighted_kappa(self) -> float:
        """
        Cohen's kappa with linear weights for ordinal severity.
        Accounts for chance agreement and penalises distant disagreements more.
        """
        if not self.predictions:
            return 0.0

        n_cats = len(SEVERITY_LABELS)
        n = len(self.predictions)

        # Observed agreement matrix
        observed = [[0] * n_cats for _ in range(n_cats)]
        for p, g in zip(self.predictions, self.ground_truths):
            if 0 <= p < n_cats and 0 <= g < n_cats:
                observed[g][p] += 1

        # Marginal distributions
        row_totals = [sum(observed[i]) for i in range(n_cats)]
        col_totals = [sum(observed[i][j] for i in range(n_cats)) for j in range(n_cats)]

        # Weight matrix (linear)
        weights = [[abs(i - j) / (n_cats - 1) for j in range(n_cats)] for i in range(n_cats)]

        # Observed weighted disagreement
        observed_weighted = sum(
            weights[i][j] * observed[i][j]
            for i in range(n_cats) for j in range(n_cats)
        ) / n if n > 0 else 0

        # Expected weighted disagreement
        expected_weighted = sum(
            weights[i][j] * row_totals[i] * col_totals[j]
            for i in range(n_cats) for j in range(n_cats)
        ) / (n * n) if n > 0 else 0

        if expected_weighted == 0:
            return 1.0 if observed_weighted == 0 else 0.0

        return 1.0 - (observed_weighted / expected_weighted)

    def to_dict(self) -> dict:
        return {
            "mae": round(self.mae, 4),
            "exact_match": round(self.exact_match, 4),
            "within_one": round(self.within_one, 4),
            "weighted_kappa": round(self.weighted_kappa(), 4),
            "n": self.n,
            "confusion_matrix": self.confusion_matrix(),
        }


@dataclass
class FlagMetrics:
    """Per-flag binary accuracy for key indicators."""
    flag_name: str = ""
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "flag": self.flag_name,
            "accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "total": self.total,
        }


@dataclass
class CalibrationBin:
    """A single bin for calibration analysis."""
    bin_lower: float = 0.0
    bin_upper: float = 0.0
    predicted_mean: float = 0.0
    actual_rate: float = 0.0
    count: int = 0


@dataclass
class DimensionEvaluation:
    """Complete evaluation for one bias dimension."""
    dimension: str = ""
    binary: BinaryMetrics = field(default_factory=BinaryMetrics)
    ordinal: OrdinalMetrics = field(default_factory=OrdinalMetrics)
    flags: list[FlagMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "binary": self.binary.to_dict(),
            "ordinal": self.ordinal.to_dict(),
            "flags": [f.to_dict() for f in self.flags],
        }


@dataclass
class ModelEvaluation:
    """Complete evaluation results for one model."""
    model_id: str = ""
    n_examples: int = 0
    n_parse_failures: int = 0

    # Per-dimension evaluations
    statistical_reporting: DimensionEvaluation = field(
        default_factory=lambda: DimensionEvaluation(dimension="statistical_reporting")
    )
    spin: DimensionEvaluation = field(
        default_factory=lambda: DimensionEvaluation(dimension="spin")
    )
    outcome_reporting: DimensionEvaluation = field(
        default_factory=lambda: DimensionEvaluation(dimension="outcome_reporting")
    )
    conflict_of_interest: DimensionEvaluation = field(
        default_factory=lambda: DimensionEvaluation(dimension="conflict_of_interest")
    )
    methodology: DimensionEvaluation = field(
        default_factory=lambda: DimensionEvaluation(dimension="methodology")
    )

    # Overall
    overall_binary: BinaryMetrics = field(default_factory=BinaryMetrics)
    overall_ordinal: OrdinalMetrics = field(default_factory=OrdinalMetrics)

    # Calibration (predicted probability vs actual bias)
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    calibration_error: float = 0.0  # Expected Calibration Error

    # Verification quality
    mean_verification_score: float = 0.0
    verification_source_rates: dict = field(default_factory=dict)

    # Thinking quality (for reasoning models)
    mean_thinking_length: float = 0.0
    thinking_present_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_examples": self.n_examples,
            "n_parse_failures": self.n_parse_failures,
            "dimensions": {
                "statistical_reporting": self.statistical_reporting.to_dict(),
                "spin": self.spin.to_dict(),
                "outcome_reporting": self.outcome_reporting.to_dict(),
                "conflict_of_interest": self.conflict_of_interest.to_dict(),
                "methodology": self.methodology.to_dict(),
            },
            "overall_binary": self.overall_binary.to_dict(),
            "overall_ordinal": self.overall_ordinal.to_dict(),
            "calibration_error": round(self.calibration_error, 4),
            "mean_verification_score": round(self.mean_verification_score, 4),
            "verification_source_rates": {
                k: round(v, 4) for k, v in self.verification_source_rates.items()
            },
            "mean_thinking_length": round(self.mean_thinking_length, 1),
            "thinking_present_rate": round(self.thinking_present_rate, 4),
        }


def evaluate_model(
    assessments: list[ParsedAssessment],
    model_id: str = "",
) -> ModelEvaluation:
    """
    Compute all metrics for a list of parsed & ground-truth-attached assessments.
    """
    result = ModelEvaluation(model_id=model_id, n_examples=len(assessments))

    if not assessments:
        return result

    result.n_parse_failures = sum(1 for a in assessments if not a.parse_success)

    # ---- Per-dimension metrics ----
    dimensions = [
        ("statistical_reporting", result.statistical_reporting),
        ("spin", result.spin),
        ("outcome_reporting", result.outcome_reporting),
        ("conflict_of_interest", result.conflict_of_interest),
        ("methodology", result.methodology),
    ]

    for dim_name, dim_eval in dimensions:
        for assessment in assessments:
            dim_score: DimensionScore = getattr(assessment, dim_name)

            # Binary
            pred = dim_score.predicted_binary
            truth = dim_score.ground_truth_binary
            if pred and truth:
                dim_eval.binary.tp += 1
            elif pred and not truth:
                dim_eval.binary.fp += 1
            elif not pred and truth:
                dim_eval.binary.fn += 1
            else:
                dim_eval.binary.tn += 1

            # Ordinal
            pred_sev = severity_to_int(dim_score.predicted_severity)
            truth_sev = severity_to_int(dim_score.ground_truth_severity)
            if pred_sev >= 0 and truth_sev >= 0:
                dim_eval.ordinal.predictions.append(pred_sev)
                dim_eval.ordinal.ground_truths.append(truth_sev)

        # Flag-level metrics
        flag_counters = defaultdict(lambda: {"correct": 0, "total": 0})
        for assessment in assessments:
            dim_score = getattr(assessment, dim_name)
            for flag_name in dim_score.predicted_flags:
                if flag_name in dim_score.ground_truth_flags:
                    pred_val = dim_score.predicted_flags[flag_name]
                    truth_val = dim_score.ground_truth_flags[flag_name]
                    flag_counters[flag_name]["total"] += 1
                    if pred_val == truth_val:
                        flag_counters[flag_name]["correct"] += 1

        dim_eval.flags = [
            FlagMetrics(flag_name=k, correct=v["correct"], total=v["total"])
            for k, v in flag_counters.items()
        ]

    # ---- Overall metrics ----
    # Binary: any dimension flagged = positive
    for assessment in assessments:
        any_pred = any(
            getattr(assessment, dim).predicted_binary
            for dim, _ in dimensions
        )
        any_truth = any(
            getattr(assessment, dim).ground_truth_binary
            for dim, _ in dimensions
        )

        if any_pred and any_truth:
            result.overall_binary.tp += 1
        elif any_pred and not any_truth:
            result.overall_binary.fp += 1
        elif not any_pred and any_truth:
            result.overall_binary.fn += 1
        else:
            result.overall_binary.tn += 1

        # Overall ordinal (use max severity across dimensions)
        pred_max = max(
            severity_to_int(getattr(assessment, dim).predicted_severity)
            for dim, _ in dimensions
        )
        truth_max = max(
            severity_to_int(getattr(assessment, dim).ground_truth_severity)
            for dim, _ in dimensions
        )
        if pred_max >= 0 and truth_max >= 0:
            result.overall_ordinal.predictions.append(pred_max)
            result.overall_ordinal.ground_truths.append(truth_max)

    # ---- Calibration ----
    n_bins = 5
    prob_bins = [[] for _ in range(n_bins)]
    truth_bins = [[] for _ in range(n_bins)]

    for assessment in assessments:
        prob = assessment.overall_bias_probability
        # Ground truth: any dimension has severity > none
        is_biased = any(
            getattr(assessment, dim).ground_truth_binary
            for dim, _ in dimensions
        )
        bin_idx = max(0, min(int(prob * n_bins), n_bins - 1))
        prob_bins[bin_idx].append(prob)
        truth_bins[bin_idx].append(1.0 if is_biased else 0.0)

    calibration_bins = []
    ece = 0.0
    total_n = len(assessments)
    for i in range(n_bins):
        if prob_bins[i]:
            pred_mean = sum(prob_bins[i]) / len(prob_bins[i])
            actual_rate = sum(truth_bins[i]) / len(truth_bins[i])
            calibration_bins.append(CalibrationBin(
                bin_lower=i / n_bins,
                bin_upper=(i + 1) / n_bins,
                predicted_mean=pred_mean,
                actual_rate=actual_rate,
                count=len(prob_bins[i]),
            ))
            ece += abs(pred_mean - actual_rate) * len(prob_bins[i]) / total_n

    result.calibration_bins = calibration_bins
    result.calibration_error = ece

    # ---- Verification quality ----
    v_scores = [a.verification_score for a in assessments]
    result.mean_verification_score = sum(v_scores) / len(v_scores) if v_scores else 0

    source_counts = Counter()
    for a in assessments:
        if a.mentions_open_payments:
            source_counts["open_payments"] += 1
        if a.mentions_clinicaltrials_gov:
            source_counts["clinicaltrials_gov"] += 1
        if a.mentions_orcid:
            source_counts["orcid"] += 1
        if a.mentions_retraction_watch:
            source_counts["retraction_watch"] += 1
        if a.mentions_europmc:
            source_counts["europmc"] += 1

    n = len(assessments)
    result.verification_source_rates = {
        k: v / n for k, v in source_counts.items()
    }

    # ---- Thinking quality ----
    thinking_lengths = [
        len(a.thinking_text) for a in assessments if a.thinking_text
    ]
    result.mean_thinking_length = (
        sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
    )
    result.thinking_present_rate = (
        len(thinking_lengths) / len(assessments) if assessments else 0
    )

    return result
