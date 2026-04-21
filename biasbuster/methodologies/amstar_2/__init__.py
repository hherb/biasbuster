"""AMSTAR 2 — methodological quality of systematic reviews. STUB.

Sixteen-item checklist (yes / partial_yes / no) with seven items
designated "critical". Overall rating is high / moderate / low /
critically_low, derived from decision rules over the critical-item
failures and non-critical weaknesses.

Orchestration shape is ``two_call_full_text`` (unusual among the
registered methodologies — every other active methodology uses
``decomposed_full_text``): Stage 1 extracts evidence relevant to each
of the 16 items, Stage 2 assesses the checklist and rolls up the
overall rating.

Complements ROBIS (see sibling stub): ROBIS = review-level bias;
AMSTAR 2 = review-level methodological quality.
"""

from __future__ import annotations

from biasbuster.methodologies._stub_helpers import (
    make_stub_methodology,
    register_stub_once,
)

METHODOLOGY_VERSION: str = "amstar2-2017"

METHODOLOGY = make_stub_methodology(
    name="amstar_2",
    display_name="AMSTAR 2 (systematic-review methodological quality) — STUB",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"systematic_review", "meta_analysis"}),
    requires_full_text=True,
    orchestration="two_call_full_text",
    severity_rollup_levels=(
        "critically_low", "low", "moderate", "high",
    ),
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#amstar-2",
        "reference": "Shea BJ et al. BMJ 2017;358:j4008.",
    },
)


def _register_once():
    return register_stub_once(METHODOLOGY)


_register_once()
