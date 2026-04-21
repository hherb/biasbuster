"""ROB-ME — missing evidence in meta-analyses. STUB.

Applied AFTER a meta-analysis exists to assess the likelihood of
unpublished / selectively-non-reported evidence. Complements (does not
replace) within-study tools like RoB 2 or ROBINS-I. Five signalling
questions, overall rating low / some_concerns / high / no_information.

Note: formally applicable to meta-analyses only, not the primary
studies within them.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "rob-me-2022"

METHODOLOGY = make_stub_methodology(
    name="rob_me",
    display_name="ROB-ME (missing evidence in meta-analyses) — STUB",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"meta_analysis"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "some_concerns", "high"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#rob-me",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
