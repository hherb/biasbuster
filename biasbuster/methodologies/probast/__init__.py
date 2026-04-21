"""PROBAST — classical prediction-model studies. STUB.

Four domains (Participants, Predictors, Outcome, Analysis). The first
three also carry a separate applicability-concern rating; domain 4
(Analysis) has no applicability dimension. Severity vocabulary is
low / high / unclear, like QUADAS-2.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "probast-2019"

METHODOLOGY = make_stub_methodology(
    name="probast",
    display_name="PROBAST (prediction-model studies) — STUB",
    version=METHODOLOGY_VERSION,
    # Applicable to studies developing, validating, or updating
    # diagnostic/prognostic prediction models. Detected study-design
    # slug for these would typically be "diagnostic_accuracy" or the
    # caller's prognostic-study equivalent; until a dedicated slug
    # exists we leave this loose (the declarative filter is "*-like"
    # in spirit but the stub will refuse any invocation anyway).
    applicable_study_designs=frozenset({"diagnostic_accuracy", "cohort"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "unclear", "high"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#probast",
        "reference": "Wolff RF et al. Ann Intern Med 2019;170:51-58.",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
