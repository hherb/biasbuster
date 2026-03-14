"""
Effect Size Auditor

Heuristic analysis of abstracts for statistical reporting patterns.
Implements Horst's hypothesis: sole/emphasis on relative measures (RRR, OR, HR)
without absolute measures (ARR, NNT, baseline risk) is a strong predictor of bias.

This module provides RULE-BASED pre-screening to identify candidate abstracts
for the training dataset. The actual model will learn far more nuanced patterns.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class ReportingPattern(str, Enum):
    RELATIVE_ONLY = "relative_only"          # Only relative measures, no absolute
    RELATIVE_EMPHASISED = "relative_emphasised"  # Both present but relative in title/conclusions
    BALANCED = "balanced"                     # Both relative and absolute reported
    ABSOLUTE_EMPHASISED = "absolute_emphasised"  # Absolute measures prominent
    NO_EFFECT_SIZE = "no_effect_size"         # No quantitative effect size found
    INDETERMINATE = "indeterminate"


@dataclass
class EffectSizeAudit:
    """Results of heuristic effect-size reporting analysis."""
    pmid: str = ""
    pattern: ReportingPattern = ReportingPattern.INDETERMINATE

    # Relative measures found
    has_relative_risk: bool = False
    has_odds_ratio: bool = False
    has_hazard_ratio: bool = False
    has_risk_ratio: bool = False
    has_relative_risk_reduction: bool = False
    has_percent_reduction: bool = False       # "reduced X by 50%"
    relative_measures_found: list[str] = field(default_factory=list)

    # Absolute measures found
    has_absolute_risk_reduction: bool = False
    has_nnt: bool = False
    has_nnh: bool = False
    has_baseline_risk: bool = False           # Control group event rate
    has_absolute_rates: bool = False          # "X% vs Y%"
    has_risk_difference: bool = False
    absolute_measures_found: list[str] = field(default_factory=list)

    # Context
    has_confidence_intervals: bool = False
    has_p_values: bool = False
    effect_in_title: bool = False             # Effect size mentioned in title
    relative_in_conclusion: bool = False      # Relative measure in conclusion section
    absolute_in_conclusion: bool = False      # Absolute measure in conclusion section

    # Bias score (0-1, higher = more suspect)
    reporting_bias_score: float = 0.0
    flags: list[str] = field(default_factory=list)


# ---- Regex patterns for detecting statistical measures ----

# Relative measures
RELATIVE_PATTERNS = {
    "hazard_ratio": [
        r'\b[Hh]azard\s+ratio\b',
        r'\bHR\s*[=:]\s*\d',
        r'\bHR\s+\d+\.\d+',
        r'\b(?:a)?HR\b\s*[\(=:]',
    ],
    "odds_ratio": [
        r'\b[Oo]dds\s+ratio\b',
        r'\bOR\s*[=:]\s*\d',
        r'\b(?:a)?OR\b\s*[\(=:]',
    ],
    "relative_risk": [
        r'\b[Rr]elative\s+risk\b',
        r'\bRR\s*[=:]\s*\d',
        r'\bRR\b\s*[\(=:]',
    ],
    "risk_ratio": [
        r'\b[Rr]isk\s+ratio\b',
    ],
    "relative_risk_reduction": [
        r'\b[Rr]elative\s+risk\s+reduction\b',
        r'\bRRR\b',
    ],
    "percent_reduction": [
        # "reduced [outcome] by X%", "X% reduction in [outcome]"
        r'reduc(?:ed|tion|ing)\s+(?:\w+\s+){0,4}by\s+\d+(?:\.\d+)?%',
        r'\d+(?:\.\d+)?%\s+(?:relative\s+)?reduction',
        r'(?:decreased|lowered)\s+(?:\w+\s+){0,4}by\s+\d+(?:\.\d+)?%',
    ],
    "percent_lower_higher": [
        # "X% lower risk", "X% higher rate"
        r'\d+(?:\.\d+)?%\s+(?:lower|higher|greater|less)\s+(?:risk|rate|incidence|probability)',
    ],
}

# Absolute measures
ABSOLUTE_PATTERNS = {
    "absolute_risk_reduction": [
        r'\b[Aa]bsolute\s+risk\s+(?:reduction|difference)\b',
        r'\bARR\b',
        r'\bARD\b',
    ],
    "nnt": [
        r'\b[Nn]umber\s+needed\s+to\s+treat\b',
        r'\bNNT\b\s*[=:]?\s*\d',
    ],
    "nnh": [
        r'\b[Nn]umber\s+needed\s+to\s+harm\b',
        r'\bNNH\b\s*[=:]?\s*\d',
    ],
    "risk_difference": [
        r'\b[Rr]isk\s+difference\b',
        r'\bRD\b\s*[=:]\s*-?\d',
    ],
    "absolute_rates": [
        # "X.X% vs Y.Y%", "X% in the treatment group vs Y% in controls"
        r'\d+(?:\.\d+)?%\s+(?:vs\.?|versus|compared\s+(?:with|to))\s+\d+(?:\.\d+)?%',
        r'\d+(?:\.\d+)?%\s+in\s+the\s+\w+\s+group\s+(?:vs\.?|versus|and|compared)',
    ],
    "baseline_risk": [
        # "baseline risk of X%", "control group event rate X%", "placebo rate X%"
        r'\b(?:baseline|control|placebo)\s+(?:\w+\s+){0,3}(?:risk|rate|incidence)\s+(?:of\s+)?\d+(?:\.\d+)?%',
        r'\bevent\s+rate\s+(?:of\s+)?\d+(?:\.\d+)?%\s+in\s+(?:the\s+)?(?:control|placebo)',
    ],
    "events_per_group": [
        # "23/150 (15.3%) vs 45/148 (30.4%)"
        r'\d+/\d+\s*\(\d+(?:\.\d+)?%\)\s*(?:vs\.?|versus)\s*\d+/\d+\s*\(\d+(?:\.\d+)?%\)',
    ],
}

# General statistical patterns
STAT_PATTERNS = {
    "confidence_interval": [
        r'\b(?:95|99|90)%?\s*(?:CI|confidence\s+interval)\b',
        r'\(\s*\d+(?:\.\d+)?\s*[-–to]+\s*\d+(?:\.\d+)?\s*\)',
    ],
    "p_value": [
        r'\bp\s*[<=]\s*0?\.\d+',
        r'\bp\s*=\s*0?\.\d+',
        r'\bP\s*[<=]\s*0?\.\d+',
    ],
}


def _find_matches(text: str, patterns: dict[str, list[str]]) -> dict[str, list[str]]:
    """Find all regex matches grouped by category."""
    results = {}
    for category, pattern_list in patterns.items():
        matches = []
        for pattern in pattern_list:
            for m in re.finditer(pattern, text):
                matches.append(m.group())
        if matches:
            results[category] = matches
    return results


from schemas import extract_abstract_sections as _extract_abstract_sections

# ---- Scoring weights for bias score computation ----
# Each weight represents the contribution to the 0-1 reporting bias score.
SCORE_RELATIVE_ONLY = 0.4
SCORE_RELATIVE_EMPHASISED = 0.2
SCORE_PERCENT_REDUCTION_NO_CONTEXT = 0.15
SCORE_NO_NNT = 0.1
SCORE_NO_BASELINE_RISK = 0.1
SCORE_NO_CI = 0.05
SCORE_TITLE_RELATIVE = 0.1


def audit_abstract(pmid: str, title: str, abstract: str) -> EffectSizeAudit:
    """
    Perform heuristic effect-size reporting analysis on a single abstract.

    This is a rule-based pre-screener, not a substitute for the trained model.
    Its purpose is to identify candidate abstracts for the training dataset
    that are likely to exhibit relative-only reporting bias.
    """
    audit = EffectSizeAudit(pmid=pmid)
    full_text = f"{title}\n{abstract}"

    # Find relative measures
    rel_matches = _find_matches(full_text, RELATIVE_PATTERNS)
    if "hazard_ratio" in rel_matches:
        audit.has_hazard_ratio = True
    if "odds_ratio" in rel_matches:
        audit.has_odds_ratio = True
    if "relative_risk" in rel_matches:
        audit.has_relative_risk = True
    if "risk_ratio" in rel_matches:
        audit.has_risk_ratio = True
    if "relative_risk_reduction" in rel_matches:
        audit.has_relative_risk_reduction = True
    if "percent_reduction" in rel_matches or "percent_lower_higher" in rel_matches:
        audit.has_percent_reduction = True

    for cat_matches in rel_matches.values():
        audit.relative_measures_found.extend(cat_matches)

    # Find absolute measures
    abs_matches = _find_matches(full_text, ABSOLUTE_PATTERNS)
    if "absolute_risk_reduction" in abs_matches:
        audit.has_absolute_risk_reduction = True
    if "nnt" in abs_matches:
        audit.has_nnt = True
    if "nnh" in abs_matches:
        audit.has_nnh = True
    if "risk_difference" in abs_matches:
        audit.has_risk_difference = True
    if "absolute_rates" in abs_matches or "events_per_group" in abs_matches:
        audit.has_absolute_rates = True
    if "baseline_risk" in abs_matches:
        audit.has_baseline_risk = True

    for cat_matches in abs_matches.values():
        audit.absolute_measures_found.extend(cat_matches)

    # General stats
    stat_matches = _find_matches(full_text, STAT_PATTERNS)
    audit.has_confidence_intervals = "confidence_interval" in stat_matches
    audit.has_p_values = "p_value" in stat_matches

    # Check title for effect sizes
    title_rel = _find_matches(title, RELATIVE_PATTERNS)
    title_abs = _find_matches(title, ABSOLUTE_PATTERNS)
    audit.effect_in_title = bool(title_rel or title_abs)

    # Check conclusions section specifically
    sections = _extract_abstract_sections(abstract)
    conclusion_text = ""
    for key in ["CONCLUSION", "CONCLUSIONS", "INTERPRETATION"]:
        if key in sections:
            conclusion_text = sections[key]
            break

    if conclusion_text:
        conc_rel = _find_matches(conclusion_text, RELATIVE_PATTERNS)
        conc_abs = _find_matches(conclusion_text, ABSOLUTE_PATTERNS)
        audit.relative_in_conclusion = bool(conc_rel)
        audit.absolute_in_conclusion = bool(conc_abs)

    # ---- Classify pattern ----
    has_any_relative = bool(audit.relative_measures_found)
    has_any_absolute = (
        audit.has_absolute_risk_reduction
        or audit.has_nnt
        or audit.has_nnh
        or audit.has_risk_difference
        or audit.has_absolute_rates
        or audit.has_baseline_risk
    )

    if has_any_relative and not has_any_absolute:
        audit.pattern = ReportingPattern.RELATIVE_ONLY
    elif has_any_relative and has_any_absolute:
        # Check emphasis
        if audit.relative_in_conclusion and not audit.absolute_in_conclusion:
            audit.pattern = ReportingPattern.RELATIVE_EMPHASISED
        elif audit.absolute_in_conclusion and not audit.relative_in_conclusion:
            audit.pattern = ReportingPattern.ABSOLUTE_EMPHASISED
        else:
            audit.pattern = ReportingPattern.BALANCED
    elif has_any_absolute and not has_any_relative:
        audit.pattern = ReportingPattern.ABSOLUTE_EMPHASISED
    elif not has_any_relative and not has_any_absolute:
        audit.pattern = ReportingPattern.NO_EFFECT_SIZE
    else:
        audit.pattern = ReportingPattern.INDETERMINATE

    # ---- Compute bias score ----
    score = 0.0
    flags = []

    if audit.pattern == ReportingPattern.RELATIVE_ONLY:
        score += SCORE_RELATIVE_ONLY
        flags.append("RELATIVE_ONLY: No absolute measures reported alongside relative measures")

    if audit.pattern == ReportingPattern.RELATIVE_EMPHASISED:
        score += SCORE_RELATIVE_EMPHASISED
        flags.append("RELATIVE_EMPHASISED: Relative measures in conclusion without absolute context")

    if audit.has_percent_reduction and not has_any_absolute:
        score += SCORE_PERCENT_REDUCTION_NO_CONTEXT
        flags.append("PERCENT_REDUCTION_NO_CONTEXT: '% reduction' claim without baseline risk")

    if not audit.has_nnt and has_any_relative:
        score += SCORE_NO_NNT
        flags.append("NO_NNT: NNT not provided despite reporting relative benefit")

    if not audit.has_baseline_risk and has_any_relative:
        score += SCORE_NO_BASELINE_RISK
        flags.append("NO_BASELINE_RISK: Control group event rate not reported")

    if not audit.has_confidence_intervals and (has_any_relative or has_any_absolute):
        score += SCORE_NO_CI
        flags.append("NO_CI: Effect sizes reported without confidence intervals")

    if audit.effect_in_title and audit.pattern in (
        ReportingPattern.RELATIVE_ONLY, ReportingPattern.RELATIVE_EMPHASISED
    ):
        score += SCORE_TITLE_RELATIVE
        flags.append("TITLE_RELATIVE: Title promotes relative measure")

    audit.reporting_bias_score = min(score, 1.0)
    audit.flags = flags

    return audit


def batch_audit(
    abstracts: list[dict],  # Each dict has: pmid, title, abstract
    min_score: float = 0.3,
) -> list[EffectSizeAudit]:
    """
    Batch audit abstracts and return those exceeding the bias score threshold.
    Useful for pre-screening large batches to find training candidates.
    """
    results = []
    for item in abstracts:
        audit = audit_abstract(
            pmid=item.get("pmid", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", ""),
        )
        if audit.reporting_bias_score >= min_score:
            results.append(audit)
    return results


# ---- Demo / self-test ----

if __name__ == "__main__":
    # Test with a synthetic biased abstract
    test_biased = audit_abstract(
        pmid="TEST001",
        title="Wonderdrug reduces major adverse cardiac events by 47% in statin-intolerant patients",
        abstract=(
            "BACKGROUND: Statin-intolerant patients remain at elevated cardiovascular risk. "
            "We evaluated wonderdrug in this population. "
            "METHODS: Randomized, double-blind, placebo-controlled trial of 8,000 patients. "
            "Primary endpoint was time to first major adverse cardiac event (MACE). "
            "RESULTS: Wonderdrug significantly reduced MACE (HR 0.53, 95% CI 0.40-0.70, "
            "p<0.001), representing a 47% relative risk reduction. Subgroup analyses showed "
            "consistent benefit across age groups. "
            "CONCLUSIONS: Wonderdrug provides substantial cardiovascular protection in "
            "statin-intolerant patients and should be considered as first-line alternative therapy."
        ),
    )

    print(f"=== Biased example ===")
    print(f"Pattern: {test_biased.pattern.value}")
    print(f"Score: {test_biased.reporting_bias_score:.2f}")
    print(f"Flags: {test_biased.flags}")
    print(f"Relative found: {test_biased.relative_measures_found}")
    print(f"Absolute found: {test_biased.absolute_measures_found}")
    print()

    # Test with a well-reported abstract
    test_good = audit_abstract(
        pmid="TEST002",
        title="Effect of gooddrug on cardiovascular outcomes: a randomized trial",
        abstract=(
            "BACKGROUND: We assessed whether gooddrug reduces cardiovascular events. "
            "METHODS: Multicentre RCT, n=5000, median follow-up 3.2 years. "
            "RESULTS: The primary endpoint occurred in 8.2% of the gooddrug group vs "
            "10.8% of the placebo group (HR 0.75, 95% CI 0.63-0.89, p=0.001). "
            "The absolute risk reduction was 2.6% (95% CI 1.1-4.1%), corresponding to "
            "a number needed to treat of 39 over 3 years. "
            "CONCLUSIONS: Gooddrug modestly reduced cardiovascular events with an NNT of 39. "
            "The absolute benefit should be weighed against the cost and side effect profile."
        ),
    )

    print(f"=== Well-reported example ===")
    print(f"Pattern: {test_good.pattern.value}")
    print(f"Score: {test_good.reporting_bias_score:.2f}")
    print(f"Flags: {test_good.flags}")
    print(f"Relative found: {test_good.relative_measures_found}")
    print(f"Absolute found: {test_good.absolute_measures_found}")
