"""PROBAST+AI — prediction models with AI / ML. STUB.

Inherits PROBAST's four domains with AI-specific signalling questions
in Participants (data leakage), Predictors (image/unstructured-data
preprocessing), and Analysis (hyper-parameter tuning, fairness /
subgroup performance, transparency, deployment-shift).
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "probast+ai-2025"

METHODOLOGY = make_stub_methodology(
    name="probast_plus_ai",
    display_name="PROBAST+AI (AI/ML prediction models) — STUB",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"diagnostic_accuracy", "cohort"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "unclear", "high"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#probast-plus-ai",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
