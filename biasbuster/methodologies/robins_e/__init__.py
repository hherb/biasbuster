"""ROBINS-E — non-randomized studies of exposures. STUB.

Structurally parallel to ROBINS-I but adapted for observational studies
of exposures (nutrition, occupational, environmental) rather than
deliberately-assigned interventions. Same seven domains, same
severity vocabulary.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "robins-e-2022"

METHODOLOGY = make_stub_methodology(
    name="robins_e",
    display_name="ROBINS-E (non-randomized studies of exposures) — STUB",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"cohort", "case_control"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "moderate", "serious", "critical"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#robins-e",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
