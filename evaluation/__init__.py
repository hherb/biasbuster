# Evaluation harness for BMLibrarian bias detection model comparison

from .scorer import parse_model_output, attach_ground_truth, ParsedAssessment
from .metrics import evaluate_model, ModelEvaluation
from .comparison import generate_comparison, save_report, ComparisonReport

# Harness requires httpx — import lazily
try:
    from .harness import EvalHarness, EvalConfig, load_test_set
except ImportError:
    pass  # httpx not installed; scorer/metrics/comparison still work

__all__ = [
    "parse_model_output",
    "attach_ground_truth",
    "ParsedAssessment",
    "evaluate_model",
    "ModelEvaluation",
    "generate_comparison",
    "save_report",
    "ComparisonReport",
    "EvalHarness",
    "EvalConfig",
    "load_test_set",
]
