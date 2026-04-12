"""Post-hoc hard-rule enforcement for the v4 assessment agent.

Extracted from assessment_agent.py to keep the agent module under
the 500-line guideline. The enforcement layer runs AFTER the agent
loop has produced a candidate final assessment, walks the mechanical
assessment's ``domain_overridable`` map, and forces any downgraded
non-overridable severities back to their mechanical value with an
audit note.

This is where the policy from
``docs/two_step_approach/DESIGN_RATIONALE_COI.md`` is structurally
enforced: the LLM may attempt any override it likes, but this
Python pass prevents it from taking effect on domains marked
overridable=False (notably the COI domain when it rates HIGH via
one of the structural triggers a/b/c/d).
"""
from __future__ import annotations

import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


# Severity ordering — needed by the rank helper to compare severity
# strings against each other. Mirrors DomainSeverity in
# biasbuster/assessment/rules.py; duplicated here so this module can
# be imported without pulling in the full rules package.
_SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

# Domains the enforcement layer cares about — matches the top-level
# keys in the v3 assessment JSON schema
_DOMAIN_NAMES = (
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
)


def _rank(severity: Optional[str]) -> int:
    """Return the integer rank of a severity string (case-insensitive)."""
    if not severity:
        return 0
    return _SEVERITY_RANK.get(str(severity).lower(), 0)


def enforce_hard_rules(
    assessment: dict[str, Any],
    mechanical_assessment: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Enforce non-overridable domain severities on a final assessment.

    Args:
        assessment: The LLM agent's final assessment dict. Modified in
            place — see the return value for the mutations.
        mechanical_assessment: The draft assessment returned by the
            agent's first call to run_mechanical_assessment (captured
            in AgentLoopResult.mechanical_assessment). The
            ``_provenance.domain_overridable`` map on this dict drives
            enforcement.

    Returns:
        (assessment, enforcement_notes). The assessment dict is the
        same object passed in, with any forced upgrades applied and
        an ``_overrides`` audit entry appended for each intervention.
        ``_mechanical_provenance`` is attached to the assessment dict
        so downstream consumers can audit which mechanical rules
        fired.
        enforcement_notes is a human-readable list of "what happened"
        strings, useful for the run log and agent result surfacing.
    """
    enforcement_notes: list[str] = []

    if mechanical_assessment is None:
        return assessment, enforcement_notes

    prov = mechanical_assessment.get("_provenance") or {}
    domain_overridable = prov.get("domain_overridable") or {}
    domain_rationales = prov.get("domain_rationales") or {}
    mechanical_severities = prov.get("domain_severities") or {}

    # Ensure the _overrides list exists — the agent may or may not
    # have populated it depending on whether it chose to override
    # anything itself
    overrides = assessment.get("_overrides")
    if not isinstance(overrides, list):
        overrides = []
        assessment["_overrides"] = overrides

    # Attach the mechanical provenance so downstream consumers can
    # audit which rules fired on which inputs
    assessment["_mechanical_provenance"] = prov

    for domain, is_overridable in domain_overridable.items():
        if is_overridable:
            continue  # LLM is free to modify this domain

        mechanical_sev = str(mechanical_severities.get(domain, "none")).lower()
        final_block = assessment.get(domain)
        if not isinstance(final_block, dict):
            continue
        final_sev = str(final_block.get("severity", "none")).lower()

        if _rank(final_sev) < _rank(mechanical_sev):
            # LLM tried to downgrade a non-overridable severity.
            # Force it back up and audit.
            audit = {
                "domain": domain,
                "mechanical_severity": mechanical_sev,
                "attempted_severity": final_sev,
                "final_severity": mechanical_sev,
                "policy": "non_overridable",
                "rationale_at_extraction": domain_rationales.get(domain, ""),
                "reason": (
                    f"LLM attempted to downgrade {domain} from "
                    f"{mechanical_sev} to {final_sev}, but this domain's "
                    f"severity came from a non-overridable rule "
                    f"({domain_rationales.get(domain, 'non-overridable trigger')}). "
                    f"Reverted per DESIGN_RATIONALE_COI.md policy."
                ),
            }
            final_block["severity"] = mechanical_sev
            overrides.append(audit)
            enforcement_notes.append(
                f"{domain}: blocked downgrade {final_sev} → {mechanical_sev}"
            )
            logger.warning(
                f"enforcement: blocked downgrade of {domain} from "
                f"{mechanical_sev} to {final_sev}"
            )

    # Overall severity follows the max rule — if we forced a domain
    # upward, the overall may also need adjusting.
    if enforcement_notes:
        max_rank = max(
            _rank(str(assessment.get(name, {}).get("severity", "none")).lower())
            for name in _DOMAIN_NAMES
            if isinstance(assessment.get(name), dict)
        )
        max_sev = next(
            (s for s, r in _SEVERITY_RANK.items() if r == max_rank), "none",
        )
        current_overall = str(assessment.get("overall_severity", "none")).lower()
        if _rank(current_overall) < _rank(max_sev):
            assessment["overall_severity"] = max_sev
            enforcement_notes.append(
                f"overall_severity bumped {current_overall} → {max_sev} "
                f"after domain enforcement"
            )

    return assessment, enforcement_notes
