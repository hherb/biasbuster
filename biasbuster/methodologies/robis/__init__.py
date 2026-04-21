"""ROBIS — risk of bias in systematic reviews. STUB.

Three-phase structure: Phase 1 (relevance, optional), Phase 2 (four
domains covering the review process — study eligibility, identification
& selection, data collection & appraisal, synthesis & findings), Phase 3
(overall review-level bias rating: low / high / unclear).

Complements AMSTAR 2 (methodological quality of reviews) but assesses
a different construct — bias in the review's conclusions rather than
compliance with checklist items.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "robis-2016"

METHODOLOGY = make_stub_methodology(
    name="robis",
    display_name="ROBIS (risk of bias in systematic reviews) — STUB",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"systematic_review", "meta_analysis"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "unclear", "high"),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#robis",
        "reference": "Whiting P et al. J Clin Epidemiol 2016;69:225-234.",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
