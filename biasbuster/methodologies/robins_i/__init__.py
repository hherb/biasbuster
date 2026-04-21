"""ROBINS-I — non-randomized studies of interventions. STUB.

Not yet implemented. Registered so ``list_methodologies()`` shows it
on the roadmap and any attempt to invoke it fails loud.

ROBINS-I has seven domains (three pre-intervention, one at-intervention,
three post-intervention) and a four-level severity vocabulary
(low / moderate / serious / critical / no information). The assessor
is expected to require a "target trial" specification as part of the
user message — see docs/ASSESSING_RISK_OF_BIAS.md#robins-i.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "robins-i-2016"

METHODOLOGY = make_stub_methodology(
    name="robins_i",
    display_name="ROBINS-I (non-randomized studies of interventions) — STUB",
    version=METHODOLOGY_VERSION,
    # Applicable to comparative observational / quasi-experimental
    # intervention studies. Case reports / series and qualitative
    # studies are out of scope.
    applicable_study_designs=frozenset({"cohort", "case_control"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "moderate", "serious", "critical"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#robins-i",
        "reference": "Sterne JAC et al. BMJ 2016;355:i4919.",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
