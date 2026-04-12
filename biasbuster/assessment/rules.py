"""Pure-function rule catalogue for v4 algorithmic assessment.

Each function in this module is a rule: it takes a v3 extraction dict
(or a subset of it) and returns either a boolean flag value or a
structured rule outcome with provenance. No LLM involvement, no
side effects, no I/O.

The module is deliberately flat (no classes beyond simple dataclasses)
so each rule can be read and tested in isolation. The rules are
organised by domain but have no interdependencies — each reads only
what it needs from the extraction dict.

Translated from the Round 10 assessment prompt at
`biasbuster/prompts_v3.py` ASSESSMENT_DOMAIN_CRITERIA. Every rule
here corresponds to a paragraph in that prompt; see the source of
each function for a quote of the prompt rule it implements.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Severity enums
# ---------------------------------------------------------------------------

class DomainSeverity(str, Enum):
    """Severity for a single domain (statistical, spin, outcome, coi, methodology)."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def rank(self) -> int:
        return {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}[self.value]


class OverallSeverity(str, Enum):
    """Overall severity (= max of domain severities)."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Provenance types
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A rule that was evaluated — records its identity and inputs."""
    name: str                          # e.g. "methodology.no_multiplicity_correction"
    description: str                   # short human-readable summary
    inputs: dict[str, Any]             # the specific extraction values read
    result: Any                        # the value the rule produced
    fired: bool                        # whether the rule's condition was met (if applicable)


@dataclass
class RuleOutcome:
    """Per-domain bundle of rule evaluations and the resulting severity."""
    domain: str
    severity: DomainSeverity
    flags: dict[str, Optional[bool]]   # derived boolean flags (None = unknown/insufficient data)
    rules: list[Rule] = field(default_factory=list)
    severity_rationale: str = ""       # which cascade branch produced the severity


# ---------------------------------------------------------------------------
# Safe dict accessors — extraction fields are often null or missing
# ---------------------------------------------------------------------------

def _get(d: Optional[dict], *path: str, default: Any = None) -> Any:
    """Walk a nested dict by key path. Returns default at any missing step."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _as_int(v: Any) -> Optional[int]:
    """Coerce to int or return None (never raises)."""
    if isinstance(v, bool):
        return None  # bool is int subclass — exclude
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return None


def _list(v: Any) -> list:
    """Return v if v is a list, else empty list."""
    return v if isinstance(v, list) else []


# ===========================================================================
# DOMAIN 1: STATISTICAL REPORTING
# ===========================================================================

def _parse_relative_magnitude(text: str) -> float:
    """Extract the largest absolute percent change or fold change from *text*.

    Returns 0.0 if nothing found. Used by inflated_effect_sizes. Examples:
        "13,008%"         → 13008.0
        "2.1-fold-change" → 210.0   (2.1x = 210%)
        "62% increase"    → 62.0
        "5x increase"     → 500.0
    """
    if not isinstance(text, str):
        return 0.0
    max_mag = 0.0

    # Percentages: "13,008%", "305%", "-62%"
    for m in re.finditer(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%", text):
        num = m.group(1).replace(",", "")
        try:
            v = abs(float(num))
            if v > max_mag:
                max_mag = v
        except ValueError:
            pass

    # Fold changes: "2.1-fold", "2587-fold", "5x"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[-]?\s*(?:fold|x)\b", text, re.IGNORECASE):
        try:
            v = float(m.group(1)) * 100  # 2.1x = 210%
            if v > max_mag:
                max_mag = v
        except ValueError:
            pass

    return max_mag


def statistical_reporting_flags(extraction: dict) -> tuple[dict[str, Optional[bool]], list[Rule]]:
    """Compute the 7 statistical_reporting boolean flags from the extraction.

    Derivable from extraction:
      - inflated_effect_sizes (parse effect_size_quotes for magnitudes > 500%)

    Text-judgment flags (NOT reliably computable from structured extraction
    alone — passed through from the extraction's own flag if present, else
    None = "unknown"):
      - relative_only, absolute_reported, nnt_reported, baseline_risk_reported
      - selective_p_values
      - subgroup_emphasis (partially computable from subgroups section)

    For the Option B proof-of-concept, we accept the extraction's own
    self-reported values for the pure-text-judgment flags and ONLY
    recompute the ones that are genuinely arithmetic. Future work:
    a qualitative-judgment LLM mini-stage produces these as part of
    v4's three-stage architecture.
    """
    rules: list[Rule] = []
    flags: dict[str, Optional[bool]] = {}

    outcomes = _get(extraction, "outcomes") or {}
    effect_quotes = _list(outcomes.get("effect_size_quotes"))

    # --- inflated_effect_sizes: parse each effect quote for > 500% or > 5x ---
    inflated = False
    max_found = 0.0
    for quote in effect_quotes:
        if not isinstance(quote, str):
            continue
        mag = _parse_relative_magnitude(quote)
        if mag > max_found:
            max_found = mag
        if mag > 500.0:
            inflated = True
    flags["inflated_effect_sizes"] = inflated
    rules.append(Rule(
        name="stat.inflated_effect_sizes",
        description="any effect > 500% or > 5-fold in effect_size_quotes",
        inputs={"n_quotes": len(effect_quotes), "max_magnitude_pct": round(max_found, 1)},
        result=inflated,
        fired=inflated,
    ))

    # --- subgroup_emphasis: check subgroups with prominence in title/abstract_conclusion ---
    subgroups = _list(_get(extraction, "subgroups", "subgroups"))
    subgroup_emphasis = False
    subgroup_reason: Optional[dict] = None
    for sg in subgroups:
        if not isinstance(sg, dict):
            continue
        prominence = sg.get("prominence")
        pre_spec = sg.get("pre_specified")
        mult_corr = sg.get("multiplicity_corrected")
        if prominence in ("title", "abstract_conclusion"):
            if pre_spec in ("no", "unclear") or mult_corr in ("no", "unclear"):
                subgroup_emphasis = True
                subgroup_reason = {
                    "name": sg.get("name"),
                    "prominence": prominence,
                    "pre_specified": pre_spec,
                    "multiplicity_corrected": mult_corr,
                }
                break
    flags["subgroup_emphasis"] = subgroup_emphasis
    rules.append(Rule(
        name="stat.subgroup_emphasis",
        description="any prominent subgroup with unclear pre-spec or no multiplicity correction",
        inputs={"n_subgroups": len(subgroups), "reason": subgroup_reason},
        result=subgroup_emphasis,
        fired=subgroup_emphasis,
    ))

    # --- relative_only: TRUE if every outcome uses a relative-only result_format ---
    # We can compute this deterministically from the extraction when
    # result_format is populated on most outcomes.
    primary = _list(outcomes.get("primary_outcomes_stated"))
    secondary = _list(outcomes.get("secondary_outcomes_stated"))
    all_outcomes = primary + secondary
    formats = [o.get("result_format") for o in all_outcomes if isinstance(o, dict) and o.get("result_format")]
    relative_only: Optional[bool]
    if not formats:
        relative_only = None  # insufficient data
    else:
        relative_only = all(fmt in ("relative",) for fmt in formats)
    flags["relative_only"] = relative_only
    rules.append(Rule(
        name="stat.relative_only",
        description="every populated result_format is 'relative'",
        inputs={"formats": formats},
        result=relative_only,
        fired=bool(relative_only),
    ))

    # --- absolute_reported: inverse check, TRUE if any outcome has absolute/both ---
    absolute_reported: Optional[bool]
    if not formats:
        absolute_reported = None
    else:
        absolute_reported = any(fmt in ("absolute", "both") for fmt in formats)
    flags["absolute_reported"] = absolute_reported
    rules.append(Rule(
        name="stat.absolute_reported",
        description="any populated result_format is 'absolute' or 'both'",
        inputs={"formats": formats},
        result=absolute_reported,
        fired=bool(absolute_reported),
    ))

    # --- Pass-through text-judgment flags (v4 will replace these) ---
    # For Option B, read them from the extraction's own values if present
    # (some extractors pre-compute them, some don't). None = unknown.
    flags["nnt_reported"] = None
    flags["baseline_risk_reported"] = None
    flags["selective_p_values"] = None

    rules.append(Rule(
        name="stat.passthrough_flags",
        description="text-judgment flags currently not computable without LLM",
        inputs={"note": "nnt_reported / baseline_risk_reported / selective_p_values are unknown in v4 Option B"},
        result=None,
        fired=False,
    ))

    return flags, rules


def statistical_reporting_severity(
    flags: dict[str, Optional[bool]],
) -> tuple[DomainSeverity, str, bool]:
    """Apply the Round 10 statistical_reporting severity cascade.

    Rules (from prompts_v3 ASSESSMENT_DOMAIN_CRITERIA §1):
      - HIGH: two or more TRUE flags from {relative_only, inflated_effect_sizes,
              selective_p_values, subgroup_emphasis}
      - MODERATE: exactly one TRUE from that set
      - LOW: only minor omission flags
      - NONE: nothing flagged

    Returns:
        (severity, rationale, overridable). Every branch of this cascade
        produces an overridable severity — there are no structural
        statistical-reporting triggers that we refuse to let the LLM
        context-gate in v4. The LLM may downgrade these severities
        with contextual reasoning (e.g. "this is an exploratory
        secondary analysis where subgroup emphasis is customary").
    """
    concern_flags = ["relative_only", "inflated_effect_sizes", "selective_p_values", "subgroup_emphasis"]
    n_true = sum(1 for f in concern_flags if flags.get(f) is True)

    if n_true >= 2:
        return DomainSeverity.HIGH, f"{n_true} concern flags true (>=2 → HIGH)", True
    if n_true == 1:
        which = next(f for f in concern_flags if flags.get(f) is True)
        return DomainSeverity.MODERATE, f"1 concern flag true ({which}) → MODERATE", True
    # No concerns — but if NNT is missing without absolute data, that's LOW
    if flags.get("absolute_reported") is False:
        return DomainSeverity.LOW, "no absolute measures reported → LOW", True
    return DomainSeverity.NONE, "no concern flags true", True


# ===========================================================================
# DOMAIN 2: SPIN
# ===========================================================================

CLINICAL_VERBS_TITLE_SPIN = {
    "promotes", "improves", "treats", "restores", "protects", "prevents",
    "relieves", "cures", "benefits", "enhances", "strengthens", "heals",
    "recovery", "healing",
}


def spin_flags(extraction: dict) -> tuple[dict[str, Optional[bool]], list[Rule]]:
    """Compute spin flags from title + conclusion + outcome types.

    Fully computable:
      - title_spin (requires all primary outcomes surrogate/composite + clinical verb in title)

    Text-judgment pass-through (None in v4 Option B):
      - conclusion_matches_results, causal_language_from_observational,
        inappropriate_extrapolation, focus_on_secondary_when_primary_ns
    """
    rules: list[Rule] = []
    flags: dict[str, Optional[bool]] = {}

    title = (_get(extraction, "paper_metadata", "title") or "").lower()
    primary_outcomes = _list(_get(extraction, "outcomes", "primary_outcomes_stated"))
    primary_types = [o.get("type") for o in primary_outcomes if isinstance(o, dict)]

    all_surrogate_or_composite = bool(primary_types) and all(
        t in ("surrogate", "composite") for t in primary_types if t is not None
    )
    verb_in_title = any(verb in title for verb in CLINICAL_VERBS_TITLE_SPIN)

    title_spin = all_surrogate_or_composite and verb_in_title
    flags["title_spin"] = title_spin
    rules.append(Rule(
        name="spin.title_spin",
        description="all primary outcomes surrogate/composite AND title uses a therapeutic verb",
        inputs={
            "n_primary": len(primary_outcomes),
            "all_surrogate_or_composite": all_surrogate_or_composite,
            "verb_in_title": verb_in_title,
            "title_snippet": title[:120] if title else None,
        },
        result=title_spin,
        fired=title_spin,
    ))

    # --- inappropriate_extrapolation: partially computable ---
    # (a) All primary surrogate AND conclusions use clinical language
    clinical_lang = _get(extraction, "conclusions", "clinical_language_in_conclusions")
    has_uncertainty = _get(extraction, "conclusions", "uncertainty_language_present")
    inappropriate_extrapolation: Optional[bool] = None
    if clinical_lang is True and all_surrogate_or_composite:
        inappropriate_extrapolation = True
    elif has_uncertainty is False and clinical_lang is True:
        inappropriate_extrapolation = True
    elif clinical_lang is False:
        inappropriate_extrapolation = False
    flags["inappropriate_extrapolation"] = inappropriate_extrapolation
    rules.append(Rule(
        name="spin.inappropriate_extrapolation",
        description="surrogate primaries + clinical language, OR no uncertainty + clinical language",
        inputs={
            "clinical_language_in_conclusions": clinical_lang,
            "uncertainty_language_present": has_uncertainty,
            "all_surrogate_or_composite": all_surrogate_or_composite,
        },
        result=inappropriate_extrapolation,
        fired=bool(inappropriate_extrapolation),
    ))

    # Pass-throughs
    flags["conclusion_matches_results"] = None
    flags["causal_language_from_observational"] = None
    flags["focus_on_secondary_when_primary_ns"] = None

    return flags, rules


def spin_severity(
    flags: dict[str, Optional[bool]],
) -> tuple[DomainSeverity, str, bool]:
    """Apply the spin severity cascade.

    Boutron-ish:
      - HIGH: clinical claims from surrogates + no uncertainty + no further research
      - MODERATE: some hedging but overclaiming (inappropriate_extrapolation=True + title_spin=True)
      - LOW: appropriate uncertainty with minor overclaiming (title_spin only)
      - NONE: nothing flagged

    All spin severities are overridable. The LLM can legitimately
    override title_spin false positives (e.g. a descriptive title
    that happens to contain a listed verb by coincidence).
    """
    title_spin = flags.get("title_spin") is True
    inappropriate = flags.get("inappropriate_extrapolation") is True

    if inappropriate and title_spin:
        return DomainSeverity.MODERATE, "title_spin + inappropriate_extrapolation → MODERATE", True
    if inappropriate or title_spin:
        return DomainSeverity.LOW, "single spin flag → LOW", True
    return DomainSeverity.NONE, "no spin flags true", True


# ===========================================================================
# DOMAIN 3: OUTCOME REPORTING
# ===========================================================================

def outcome_reporting_flags(extraction: dict) -> tuple[dict[str, Optional[bool]], list[Rule]]:
    """Aggregate primary_outcome_type and the 3 concern flags.

    Implements the Round 9 aggregation rule:
      - n_pc / n_total >= 0.30 AND n_pc >= 2  → 'patient_centred'
      - n_comp >= 1 AND n_surr == 0           → 'composite'
      - n_surr > 0                            → 'surrogate'
      - otherwise                             → 'unclear'
    """
    rules: list[Rule] = []
    flags: dict[str, Optional[bool]] = {}

    primary = _list(_get(extraction, "outcomes", "primary_outcomes_stated"))
    types = [o.get("type") for o in primary if isinstance(o, dict) and o.get("type")]
    n_total = len(types)
    n_pc = sum(1 for t in types if t == "patient_centred")
    n_surr = sum(1 for t in types if t == "surrogate")
    n_comp = sum(1 for t in types if t == "composite")

    primary_outcome_type: str
    if n_total == 0:
        primary_outcome_type = "unclear"
    elif (n_pc / n_total >= 0.30) and n_pc >= 2:
        primary_outcome_type = "patient_centred"
    elif n_comp >= 1 and n_surr == 0:
        primary_outcome_type = "composite"
    elif n_surr > 0:
        primary_outcome_type = "surrogate"
    else:
        primary_outcome_type = "unclear"

    flags["primary_outcome_type"] = primary_outcome_type  # type: ignore[assignment]
    rules.append(Rule(
        name="outcome.primary_outcome_type",
        description="aggregate primary outcome type from per-outcome labels",
        inputs={"n_total": n_total, "n_pc": n_pc, "n_surr": n_surr, "n_comp": n_comp},
        result=primary_outcome_type,
        fired=True,
    ))

    # --- surrogate_without_validation ---
    surrogate_without_validation = (primary_outcome_type == "surrogate")
    flags["surrogate_without_validation"] = surrogate_without_validation
    rules.append(Rule(
        name="outcome.surrogate_without_validation",
        description="primary type = surrogate (extraction lacks a 'validation cited' field)",
        inputs={"primary_outcome_type": primary_outcome_type},
        result=surrogate_without_validation,
        fired=surrogate_without_validation,
    ))

    # --- composite_not_disaggregated ---
    disaggregated = _get(extraction, "outcomes", "composite_components_disaggregated")
    composite_not_disaggregated: Optional[bool] = None
    if primary_outcome_type == "composite":
        composite_not_disaggregated = disaggregated is False
    flags["composite_not_disaggregated"] = composite_not_disaggregated
    rules.append(Rule(
        name="outcome.composite_not_disaggregated",
        description="primary is composite and components are not disaggregated",
        inputs={"primary_outcome_type": primary_outcome_type, "disaggregated": disaggregated},
        result=composite_not_disaggregated,
        fired=bool(composite_not_disaggregated),
    ))

    # --- registered_outcome_not_reported: requires verification data we don't have in Option B ---
    flags["registered_outcome_not_reported"] = None

    return flags, rules


def outcome_reporting_severity(
    flags: dict[str, Optional[bool]],
) -> tuple[DomainSeverity, str, bool]:
    """Apply the outcome_reporting severity cascade.

    - HIGH: surrogate_without_validation AND registered_outcome_not_reported
    - MODERATE: surrogate_without_validation OR composite_not_disaggregated
                OR registered_outcome_not_reported
    - LOW: patient_centred primary with secondary surrogate emphasis (not
           computable here without the subgroup/secondary analysis)
    - NONE: patient_centred primary, nothing else flagged

    All overridable. The LLM can override surrogate_without_validation
    when the paper is explicitly proof-of-mechanism or when the
    surrogate has external validation the extraction didn't capture.
    """
    swv = flags.get("surrogate_without_validation") is True
    cnd = flags.get("composite_not_disaggregated") is True
    ronr = flags.get("registered_outcome_not_reported") is True
    primary_type = flags.get("primary_outcome_type")

    if swv and ronr:
        return DomainSeverity.HIGH, "surrogate + registered outcome missing → HIGH", True
    if swv:
        return DomainSeverity.MODERATE, "surrogate primary without validation → MODERATE", True
    if cnd:
        return DomainSeverity.MODERATE, "composite not disaggregated → MODERATE", True
    if ronr:
        return DomainSeverity.MODERATE, "registered outcome not reported → MODERATE", True
    if primary_type == "patient_centred":
        return DomainSeverity.LOW, "patient-centred primary → LOW (no concerns found)", True
    return DomainSeverity.NONE, "no concern flags true", True


# ===========================================================================
# DOMAIN 4: CONFLICT OF INTEREST — the Round 10 trigger (d) domain
# ===========================================================================

SPONSOR_CONTROLS_ROLES = {"employee", "shareholder"}


def conflict_of_interest_flags(extraction: dict) -> tuple[dict[str, Optional[bool]], list[Rule]]:
    """Compute COI flags including the Round 10 trigger (d) check.

    All five flags are deterministic under the Round 10 design:
      - industry_author_affiliations
      - coi_disclosed
      - sponsor_controls_analysis (from data_analyst.is_sponsor_affiliated)
      - sponsor_controls_manuscript (from manuscript_drafter.is_sponsor_affiliated
        OR any author with role ∈ {employee, shareholder})
      - plus funding_type (passthrough)
    """
    rules: list[Rule] = []
    flags: dict[str, Optional[bool]] = {}

    conflicts = _get(extraction, "conflicts") or {}
    funding_type = conflicts.get("funding_type")
    flags["funding_type"] = funding_type  # type: ignore[assignment]
    flags["funding_disclosed_in_abstract"] = conflicts.get("funding_disclosed_in_abstract")

    authors = _list(conflicts.get("authors_with_industry_affiliation"))
    industry_author_affiliations = len(authors) > 0
    flags["industry_author_affiliations"] = industry_author_affiliations
    rules.append(Rule(
        name="coi.industry_author_affiliations",
        description="authors_with_industry_affiliation is non-empty",
        inputs={"n_authors_with_affiliation": len(authors)},
        result=industry_author_affiliations,
        fired=industry_author_affiliations,
    ))

    # --- coi_disclosed ---
    coi_statement_present = conflicts.get("coi_statement_present")
    coi_statement_text = conflicts.get("coi_statement_text") or ""
    # Substantive = present AND text has per-author disclosure patterns
    has_author_disclosure = bool(
        isinstance(coi_statement_text, str)
        and (
            re.search(r"\b[A-Z]\.[A-Z]\.", coi_statement_text)  # "B.A.N." initials
            or "employees" in coi_statement_text.lower()
            or "shareholders" in coi_statement_text.lower()
            or "consultant" in coi_statement_text.lower()
            or "advisor" in coi_statement_text.lower()
            or "declare" in coi_statement_text.lower()
        )
    )
    coi_disclosed = bool(coi_statement_present) and has_author_disclosure
    flags["coi_disclosed"] = coi_disclosed
    rules.append(Rule(
        name="coi.coi_disclosed",
        description="statement present AND contains per-author disclosures",
        inputs={
            "coi_statement_present": coi_statement_present,
            "has_author_disclosure": has_author_disclosure,
        },
        result=coi_disclosed,
        fired=coi_disclosed,
    ))

    # --- sponsor_controls_analysis ---
    data_analyst = conflicts.get("data_analyst") or {}
    sponsor_controls_analysis = data_analyst.get("is_sponsor_affiliated") is True
    flags["sponsor_controls_analysis"] = sponsor_controls_analysis
    rules.append(Rule(
        name="coi.sponsor_controls_analysis",
        description="data_analyst.is_sponsor_affiliated = True",
        inputs={"data_analyst": data_analyst},
        result=sponsor_controls_analysis,
        fired=sponsor_controls_analysis,
    ))

    # --- sponsor_controls_manuscript (two paths) ---
    manuscript_drafter = conflicts.get("manuscript_drafter") or {}
    drafter_path = manuscript_drafter.get("is_sponsor_affiliated") is True
    # Second path: any author in the sponsor-controls role set
    authorship_path = any(
        isinstance(a, dict) and (a.get("role") or "").lower() in SPONSOR_CONTROLS_ROLES
        for a in authors
    )
    sponsor_controls_manuscript = drafter_path or authorship_path
    flags["sponsor_controls_manuscript"] = sponsor_controls_manuscript
    rules.append(Rule(
        name="coi.sponsor_controls_manuscript",
        description="drafter.is_sponsor_affiliated OR any author role in {employee,shareholder}",
        inputs={
            "drafter_path": drafter_path,
            "authorship_path": authorship_path,
            "author_roles": [a.get("role") for a in authors if isinstance(a, dict)],
        },
        result=sponsor_controls_manuscript,
        fired=sponsor_controls_manuscript,
    ))

    return flags, rules


def conflict_of_interest_severity(
    flags: dict[str, Optional[bool]],
    outcome_flags: dict[str, Optional[bool]],
) -> tuple[DomainSeverity, str, bool]:
    """Apply the Round 10 COI severity cascade with all 4 HIGH triggers.

    HIGH triggers (apply mechanically — any one is sufficient):
      (a) sponsor_controls_analysis AND sponsor_controls_manuscript
      (b) industry funding AND undisclosed COI AND industry author affiliations
      (c) sponsor_controls_analysis AND primary_outcome_type = "surrogate"
      (d) funding_type ∈ {industry, mixed} AND >= 1 author with role ∈ {employee, shareholder}

    This is the canonical implementation of the
    docs/two_step_approach/DESIGN_RATIONALE_COI.md policy. Python here
    is the single source of truth — any disagreement with the prompt
    version should be resolved by updating the prompt to match this
    module.

    Overridability policy:
      ALL FOUR HIGH triggers (a/b/c/d) are NON-OVERRIDABLE. These are
      structural risk signals that the v4 assessment agent is not
      permitted to downgrade below HIGH regardless of contextual
      judgment. The post-hoc enforcement check in
      AssessmentAgent._enforce_hard_rules inspects the third element
      of this return tuple and forces the severity back up if the LLM
      attempts a downgrade.

      Why non-overridable: DESIGN_RATIONALE_COI.md documents the
      risk-of-bias-not-proof-of-bias framing and the explicit user
      policy that structural COI (sponsor-employed authors, sponsor
      controlling analysis, etc.) always warrants HIGH categorically.
      The LLM may adjust overall_bias_probability within the HIGH
      anchor band (0.65-0.85) but cannot downgrade the categorical
      rating.

      Moderate/low/none severities are overridable — they reflect
      gradations within the "non-structural COI" space where
      contextual judgment is valid.
    """
    funding_type = flags.get("funding_type")
    sca = flags.get("sponsor_controls_analysis") is True
    scm = flags.get("sponsor_controls_manuscript") is True
    iaa = flags.get("industry_author_affiliations") is True
    coi_disclosed = flags.get("coi_disclosed") is True
    primary_type = outcome_flags.get("primary_outcome_type")

    # Trigger (a) — NON-OVERRIDABLE
    if sca and scm:
        return DomainSeverity.HIGH, (
            "trigger (a) [non-overridable]: sponsor_controls_analysis "
            "AND sponsor_controls_manuscript"
        ), False

    # Trigger (d) — the Round 10 addition, most commonly firing rule
    # NB: we derive this directly from the extraction authors list rather
    # than re-checking flags["sponsor_controls_manuscript"], because (d)
    # is supposed to short-circuit on the structural signal even when
    # the drafter field is unpopulated. The scm flag already covers
    # this via the authorship_path — but we keep (d) as an explicit
    # named trigger for provenance clarity. NON-OVERRIDABLE.
    if funding_type in ("industry", "mixed") and scm:
        # Note: scm=True when an employee/shareholder author is present
        # because of how sponsor_controls_manuscript is computed
        return DomainSeverity.HIGH, (
            f"trigger (d) [non-overridable]: funding_type={funding_type} "
            f"AND sponsor-employed/shareholder author present"
        ), False

    # Trigger (c) — NON-OVERRIDABLE
    if sca and primary_type == "surrogate":
        return DomainSeverity.HIGH, (
            "trigger (c) [non-overridable]: sponsor_controls_analysis "
            "AND surrogate primary outcomes"
        ), False

    # Trigger (b) — NON-OVERRIDABLE
    if funding_type == "industry" and iaa and not coi_disclosed:
        return DomainSeverity.HIGH, (
            "trigger (b) [non-overridable]: industry funding + undisclosed COI "
            "+ industry author affiliations"
        ), False

    # --- Moderate (overridable) ---
    # Industry funding with incomplete disclosure, OR sponsor employees
    # involved in analysis with disclosed independent oversight
    if funding_type == "industry" and not coi_disclosed:
        return DomainSeverity.MODERATE, "industry funding with incomplete COI disclosure", True
    if sca and coi_disclosed:
        return DomainSeverity.MODERATE, "sponsor-affiliated analyst but COI disclosed", True

    # --- Low (overridable) ---
    # Any industry involvement without triggering HIGH gets at least LOW
    if funding_type in ("industry", "mixed") and iaa:
        return DomainSeverity.LOW, "industry funding + industry authors, fully disclosed", True
    if funding_type in ("industry", "mixed"):
        return DomainSeverity.LOW, "industry funding, no industry author affiliations", True
    if iaa:
        return DomainSeverity.LOW, "industry author affiliations without industry funding", True

    # --- None (overridable) ---
    return DomainSeverity.NONE, "no COI concerns found", True


# ===========================================================================
# DOMAIN 5: METHODOLOGY — where arithmetic matters most
# ===========================================================================

def _count_endpoints(extraction: dict) -> tuple[int, dict[str, Any]]:
    """Compute total_endpoints = len(primary) + len(secondary) from extraction.

    THIS IS THE CRITICAL FIX: the extraction's own n_primary_endpoints and
    n_secondary_endpoints fields are unreliable (calibration test showed
    120b/20b setting them to 0 on papers with 1+7 actual endpoints; even
    Claude sets them to 0 on the Seed Health paper while the list has 10
    items). We ignore those count fields entirely and count the list
    elements directly.
    """
    primary = _list(_get(extraction, "outcomes", "primary_outcomes_stated"))
    secondary = _list(_get(extraction, "outcomes", "secondary_outcomes_stated"))
    n_primary = len(primary)
    n_secondary = len(secondary)
    total = n_primary + n_secondary

    # Fallback: if both lists are empty but effect_size_quotes has items,
    # each quote represents a hypothesis test.
    if total == 0:
        quotes = _list(_get(extraction, "outcomes", "effect_size_quotes"))
        total = len(quotes)
        return total, {
            "source": "effect_size_quotes fallback",
            "n_quotes": total,
            "n_primary_list": 0,
            "n_secondary_list": 0,
        }

    return total, {
        "source": "primary+secondary outcome lists",
        "n_primary": n_primary,
        "n_secondary": n_secondary,
    }


def methodology_flags(
    extraction: dict,
    total_endpoints: int,
) -> tuple[dict[str, Optional[bool]], list[Rule]]:
    """Compute the 10 methodology boolean flags from the extraction."""
    rules: list[Rule] = []
    flags: dict[str, Optional[bool]] = {}

    sample = _get(extraction, "sample") or {}
    analysis = _get(extraction, "analysis") or {}
    meth_details = _get(extraction, "methodology_details") or {}

    # --- ATTRITION ---
    n_rand = _as_int(sample.get("n_randomised"))
    n_anal = _as_int(sample.get("n_analysed"))

    attrition_pct: Optional[float] = None
    high_attrition: Optional[bool] = None
    if n_rand is not None and n_anal is not None and n_rand > 0:
        attrition_pct = (n_rand - n_anal) / n_rand * 100
        high_attrition = attrition_pct > 20.0
    flags["high_attrition"] = high_attrition
    rules.append(Rule(
        name="meth.high_attrition",
        description="overall attrition > 20%",
        inputs={"n_randomised": n_rand, "n_analysed": n_anal, "attrition_pct": attrition_pct},
        result=high_attrition,
        fired=bool(high_attrition),
    ))

    # --- DIFFERENTIAL ATTRITION ---
    per_arm_rand = sample.get("n_per_arm_randomised")
    per_arm_anal = sample.get("n_per_arm_analysed")
    differential_attrition: Optional[bool] = None
    per_arm_rates: dict[str, float] = {}
    if isinstance(per_arm_rand, dict) and isinstance(per_arm_anal, dict):
        # Match arms by shared key names (best effort)
        shared_keys = set(per_arm_rand.keys()) & set(per_arm_anal.keys())
        # If keys don't match exactly, try pairing by position
        if len(shared_keys) < 2 and len(per_arm_rand) == len(per_arm_anal) == 2:
            shared_keys = set(per_arm_rand.keys())
            anal_values = list(per_arm_anal.values())
            per_arm_anal_paired = dict(zip(per_arm_rand.keys(), anal_values))
        else:
            per_arm_anal_paired = per_arm_anal
        for arm in shared_keys:
            r = _as_int(per_arm_rand.get(arm))
            a = _as_int(per_arm_anal_paired.get(arm))
            if r and a is not None and r > 0:
                per_arm_rates[arm] = (r - a) / r * 100
        if len(per_arm_rates) >= 2:
            spread = max(per_arm_rates.values()) - min(per_arm_rates.values())
            differential_attrition = spread > 10.0
    flags["differential_attrition"] = differential_attrition
    rules.append(Rule(
        name="meth.differential_attrition",
        description="per-arm attrition spread > 10 percentage points",
        inputs={"per_arm_rates": {k: round(v, 1) for k, v in per_arm_rates.items()}},
        result=differential_attrition,
        fired=bool(differential_attrition),
    ))

    # --- PER-PROTOCOL ONLY (3 triggers, with ITT-declared short-circuit) ---
    analysis_pop = analysis.get("analysis_population_stated")
    analysis_quote = (analysis.get("analysis_population_quote") or "").lower()
    attrition_quotes = " ".join(_list(sample.get("attrition_quotes"))).lower()
    stat_methods = (analysis.get("statistical_methods_quote") or "").lower()

    # Short-circuit: if the paper explicitly declared ITT/mITT, none of
    # the "maybe it's per-protocol" heuristics should fire. The explicit
    # field is more authoritative than prose inference.
    itt_declared = analysis_pop in ("ITT", "mITT", "itt", "mitt")

    pp_triggers: list[str] = []

    # (a) explicit per_protocol label
    if analysis_pop in ("per_protocol", "completer"):
        pp_triggers.append(f"analysis_population_stated={analysis_pop}")

    if not itt_declared:
        # (b) "only completers" language in analysis_quote or attrition_quotes
        completer_phrases = [
            "completed the trial",
            "who completed",
            "completed the study",
            "included in the statistical analysis",
            "included in the analysis",
        ]
        if any(p in analysis_quote for p in completer_phrases) or any(
            p in attrition_quotes for p in completer_phrases
        ):
            # Only fire if no ITT language is present anywhere
            itt_phrases = ["itt", "intention-to-treat", "intent-to-treat", "imputation"]
            has_itt = any(p in stat_methods or p in analysis_quote for p in itt_phrases)
            if not has_itt:
                pp_triggers.append("completer language without ITT")

        # (c) n_analysed shrunk by > 5% without ITT/imputation mentioned
        if n_rand is not None and n_anal is not None and n_rand > 0:
            shrinkage = (n_rand - n_anal) / n_rand
            if shrinkage > 0.05:
                itt_mentioned = any(
                    p in stat_methods
                    for p in ["itt", "intention-to-treat", "intent-to-treat", "imputation", "mitt"]
                )
                if not itt_mentioned:
                    pp_triggers.append(
                        f"shrinkage {shrinkage * 100:.1f}% > 5% without ITT/imputation mentioned"
                    )

    per_protocol_only = len(pp_triggers) > 0
    flags["per_protocol_only"] = per_protocol_only
    rules.append(Rule(
        name="meth.per_protocol_only",
        description="any of three triggers: explicit label, completer-language, or shrinkage>5% without ITT",
        inputs={"triggers": pp_triggers, "analysis_population_stated": analysis_pop},
        result=per_protocol_only,
        fired=per_protocol_only,
    ))

    # --- INADEQUATE SAMPLE SIZE ---
    # n_analysed < 30 per arm AND total_endpoints > 5
    # OR proof-of-mechanism / pilot / mechanistic study with > 5 endpoints
    inadequate_sample_size: Optional[bool] = None
    per_arm_min = None
    if isinstance(per_arm_anal, dict) and per_arm_anal:
        anal_ints: list[int] = []
        for v in per_arm_anal.values():
            i = _as_int(v)
            if i is not None:
                anal_ints.append(i)
        if anal_ints:
            per_arm_min = min(anal_ints)
    if per_arm_min is not None:
        inadequate_sample_size = per_arm_min < 30 and total_endpoints > 5
    else:
        # Fallback to overall n_analysed
        if n_anal is not None:
            # Assume rough split in half if per-arm unknown
            approx_per_arm = n_anal // 2
            inadequate_sample_size = approx_per_arm < 30 and total_endpoints > 5
    # Mechanistic fallback
    conclusions = _get(extraction, "conclusions") or {}
    conclusion_text = " ".join(_list(conclusions.get("conclusion_quotes"))).lower()
    mechanistic_markers = ["proof of mechanism", "proof-of-mechanism", "pilot", "mechanistic"]
    if (
        n_anal is None
        and any(m in conclusion_text for m in mechanistic_markers)
        and total_endpoints > 5
    ):
        inadequate_sample_size = True
    flags["inadequate_sample_size"] = inadequate_sample_size
    rules.append(Rule(
        name="meth.inadequate_sample_size",
        description="per-arm n_analysed < 30 AND total_endpoints > 5",
        inputs={
            "per_arm_min": per_arm_min,
            "n_analysed": n_anal,
            "total_endpoints": total_endpoints,
        },
        result=inadequate_sample_size,
        fired=bool(inadequate_sample_size),
    ))

    # --- NO MULTIPLICITY CORRECTION ---
    mult_method = analysis.get("multiplicity_correction_method")
    no_multiplicity_correction = (mult_method is None) and (total_endpoints >= 3)
    flags["no_multiplicity_correction"] = no_multiplicity_correction
    rules.append(Rule(
        name="meth.no_multiplicity_correction",
        description="multiplicity_correction_method is null AND total_endpoints >= 3",
        inputs={"mult_method": mult_method, "total_endpoints": total_endpoints},
        result=no_multiplicity_correction,
        fired=no_multiplicity_correction,
    ))

    # --- ANALYTICAL FLEXIBILITY ---
    approaches = meth_details.get("analytical_approaches_described")
    approach_quote = (meth_details.get("approach_selection_quote") or "").lower()
    post_hoc_markers = ["post hoc", "post-hoc", "post hoc selection", "chose", "selected"]
    analytical_flexibility: Optional[bool] = None
    if approaches is not None:
        n_approaches = _as_int(approaches) or 0
        if n_approaches > 1:
            analytical_flexibility = any(m in approach_quote for m in post_hoc_markers)
        else:
            analytical_flexibility = False
    flags["analytical_flexibility"] = analytical_flexibility

    # --- Pass-through: inappropriate_comparator, enrichment_design, premature_stopping ---
    enrichment_design = meth_details.get("run_in_or_enrichment")
    flags["enrichment_design"] = enrichment_design
    early_stopping = meth_details.get("early_stopping")
    flags["premature_stopping"] = early_stopping
    flags["inappropriate_comparator"] = None  # requires text judgment
    flags["short_follow_up"] = None  # requires condition-specific judgment

    return flags, rules


def methodology_severity(
    flags: dict[str, Optional[bool]],
    total_endpoints: int,
) -> tuple[DomainSeverity, str, bool]:
    """Apply the methodology severity cascade.

    HIGH triggers:
      - no_multiplicity_correction AND total_endpoints >= 6
      - per_protocol_only AND (high_attrition OR differential_attrition)
      - inadequate_sample_size AND no_multiplicity_correction
      - premature_stopping (without pre-specified rules — we don't check that)
      - two or more MODERATE concerns

    MODERATE triggers:
      - per_protocol_only (alone)
      - no_multiplicity_correction AND 3 <= total_endpoints <= 5
      - high_attrition (alone)

    ALL methodology severities are OVERRIDABLE. The Option B
    validation discovered the rTMS case (PMID 32382720) where the
    mechanical rule fires correctly but Claude's contextual
    reasoning correctly overrides it to MODERATE because the paper
    is an explicitly exploratory secondary analysis where
    multiplicity correction standards are more lenient. See
    INITIAL_FINDINGS_V3.md §3.14 and V4_AGENT_DESIGN.md §5.2.
    """
    no_mult = flags.get("no_multiplicity_correction") is True
    per_protocol = flags.get("per_protocol_only") is True
    high_attr = flags.get("high_attrition") is True
    diff_attr = flags.get("differential_attrition") is True
    inadequate = flags.get("inadequate_sample_size") is True

    # HIGH cascade
    if no_mult and total_endpoints >= 6:
        return DomainSeverity.HIGH, (
            f"no_multiplicity_correction AND total_endpoints={total_endpoints} >= 6"
        ), True
    if per_protocol and (high_attr or diff_attr):
        reasons = []
        if high_attr:
            reasons.append("high_attrition")
        if diff_attr:
            reasons.append("differential_attrition")
        return DomainSeverity.HIGH, (
            f"per_protocol_only AND ({'/'.join(reasons)})"
        ), True
    if inadequate and no_mult:
        return DomainSeverity.HIGH, "inadequate_sample_size AND no_multiplicity_correction", True

    # Count moderate-level concerns
    moderate_concerns: list[str] = []
    if per_protocol:
        moderate_concerns.append("per_protocol_only")
    if no_mult and 3 <= total_endpoints <= 5:
        moderate_concerns.append("no_multiplicity_correction (3-5 endpoints)")
    if high_attr:
        moderate_concerns.append("high_attrition")
    if inadequate and not no_mult:
        moderate_concerns.append("inadequate_sample_size")

    if len(moderate_concerns) >= 2:
        return DomainSeverity.HIGH, f"two+ moderate concerns: {', '.join(moderate_concerns)}", True
    if len(moderate_concerns) == 1:
        return DomainSeverity.MODERATE, f"one moderate concern: {moderate_concerns[0]}", True

    # LOW cascade — short follow-up etc, but those are text judgments
    return DomainSeverity.NONE, "no methodology concerns found", True
