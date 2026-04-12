"""Algorithmic assessment aggregator for v3 extraction JSON.

This is the v4 proof-of-concept Python aggregator that consumes a v3
extraction JSON blob (as produced by Stage 1 of the two-call pipeline)
and computes bias domain severities, overall severity, and an overall
bias probability using deterministic rules — **no LLM involvement at
this stage**.

The motivating observation (calibration test, 2026-04-12): every
calibration failure in the Round 10 v3 pipeline was an arithmetic or
boolean-logic bug, not a text-reasoning failure:

- 120b/20b extracted n_primary_endpoints=0 on the lidocaine paper
  despite the paper having 1 primary + 7 secondary outcomes
- gemma4 extracted those counts correctly but failed to apply the
  "total_endpoints >= 6 AND no_multiplicity_correction → HIGH"
  threshold rule
- 20b simultaneously over-called consulting-tie COI (rTMS paper)
  and under-called methodology-high (lidocaine paper)
- Claude's own structured methodology.severity = "moderate" while
  its reasoning text said HIGH and overall=high — inconsistent

All of these disappear when the rule logic is moved out of the LLM
prompt and into Python. The LLM remains responsible for:

  1. Reading the paper (extraction stage, unchanged)
  2. Qualitative boolean judgments that require text reading
     (in a future v4 mini-prompt — not yet wired in for Option B)

Python is responsible for:

  1. Counting items in lists
  2. Applying thresholds (> 20%, >= 6 endpoints, etc.)
  3. Applying boolean trigger rules (trigger (a), (b), (c), (d))
  4. Cascading flags into domain severities
  5. Computing overall severity via the max-domain rule
  6. Computing overall bias probability from domain severities
  7. Emitting a full provenance trace so every rating is auditable

For the Option B proof-of-concept, the aggregator operates on the
*existing* v3 extraction JSON already in the database. It does not
yet require a new qualitative-judgment LLM stage — it takes the
existing extraction flags as-is and recomputes the severities from
them. This isolates the hypothesis ("does Python-computed severity
produce better calibration results than LLM-computed severity on
the same extraction inputs?") from any prompt engineering work.

Public API:
    assess_extraction(extraction: dict) -> dict
        Main entry point. Takes a v3 extraction JSON blob and
        returns an assessment dict with the same shape as the
        v3 assessment stage output, but with every severity and
        flag computed deterministically, plus a _provenance key
        listing which rules fired.

    DomainSeverity, OverallSeverity
        String enums for severity values.

    Rule, RuleOutcome
        Provenance types for the audit trail.

This module has no external dependencies. Every function is pure.
"""

from biasbuster.assessment.aggregate import assess_extraction
from biasbuster.assessment.rules import (
    DomainSeverity,
    OverallSeverity,
    Rule,
    RuleOutcome,
)

__all__ = [
    "assess_extraction",
    "DomainSeverity",
    "OverallSeverity",
    "Rule",
    "RuleOutcome",
]
