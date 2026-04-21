"""SYRCLE RoB — animal intervention studies. STUB.

Adapted from the original Cochrane RoB tool for controlled animal
experiments. Domains include randomization of animal allocation,
housing/cage randomization, blinding of caretakers and assessors, and
selective outcome reporting. Severity vocabulary is low / high / unclear
(RoB 1 convention).

Note: applies only to controlled experiments, not observational animal
studies.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "syrcle-2014"

METHODOLOGY = make_stub_methodology(
    name="syrcle",
    display_name="SYRCLE RoB (animal intervention studies) — STUB",
    version=METHODOLOGY_VERSION,
    # No dedicated study-design slug for animal studies in our detector;
    # biasbuster.methodologies.study_design would need extension for
    # proper routing. The stub's applicable_study_designs is mostly
    # documentary here since any invocation fails loud anyway.
    applicable_study_designs=frozenset({"unknown"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "unclear", "high"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#syrcle",
        "reference": "Hooijmans CR et al. BMC Med Res Methodol 2014;14:43.",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
