"""
Heuristic Spin Detector

Rule-based pre-screening for spin in clinical trial abstracts,
based on the Boutron et al. (2010) classification of spin patterns.

Spin patterns detected:
1. Focus on statistically significant results other than primary outcome
2. Focus on within-group comparisons (pre-post) rather than between-group
3. Inappropriate causal language from non-randomized designs
4. Conclusions claiming benefit despite non-significant primary outcome
5. Emphasis on "trends" or "clinically meaningful" without statistical significance
6. Selective reporting of subgroup analyses
7. Title spin (title claims benefit not supported by results)
8. Recommendation for clinical use from a single trial

References:
- Boutron I et al. JAMA 2010;303(20):2058-64
- Boutron I et al. J Clin Oncol 2014;32(36):4120-6
- Lazarus C et al. BMC Med Res Methodol 2015;15:85
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class SpinType(str, Enum):
    SECONDARY_FOCUS = "focus_on_secondary"
    WITHIN_GROUP = "within_group_comparison"
    CAUSAL_LANGUAGE = "inappropriate_causal_language"
    BENEFIT_DESPITE_NS = "benefit_claimed_despite_ns"
    TREND_EMPHASIS = "trend_or_clinical_meaning_emphasis"
    SUBGROUP_EMPHASIS = "subgroup_emphasis"
    TITLE_SPIN = "title_spin"
    CLINICAL_RECOMMENDATION = "premature_clinical_recommendation"
    P_VALUE_SPIN = "p_value_selective_reporting"
    NO_ACKNOWLEDGMENT_NS = "no_acknowledgment_of_ns_primary"


@dataclass
class SpinFlag:
    spin_type: SpinType
    description: str
    evidence: str = ""           # The text span that triggered this flag
    severity: str = "moderate"   # low, moderate, high
    section: str = ""            # title, results, conclusion


@dataclass
class SpinScreeningResult:
    pmid: str = ""
    title: str = ""
    has_structured_abstract: bool = False
    primary_outcome_significant: str = "unknown"  # yes, no, unclear
    flags: list[SpinFlag] = field(default_factory=list)
    overall_spin_score: float = 0.0  # 0-1
    boutron_level: str = "unknown"   # none, low, moderate, high


# ---- Regex-based pattern detectors ----

# Patterns suggesting primary outcome was NOT significant
NS_PRIMARY_PATTERNS = [
    r'(?:primary|main)\s+(?:outcome|endpoint|end\s*point)\s+(?:was|were)\s+(?:not\s+)?'
    r'(?:statistically\s+)?(?:not\s+significant|nonsignificant|non-significant)',
    r'(?:no|did not|failed to)\s+(?:show|demonstrate|reach|achieve|find)\s+'
    r'(?:a\s+)?(?:statistically\s+)?significant\s+(?:difference|improvement|reduction|benefit)',
    r'p\s*=\s*0\.[1-9]\d*',  # p > 0.05 (crude)
    r'(?:primary|main)\s+(?:outcome|endpoint).*?(?:NS|n\.s\.|not significant)',
]

# Patterns suggesting spin in conclusions despite NS primary
BENEFIT_DESPITE_NS_PATTERNS = [
    r'(?:CONCLUSION|INTERPRETATION)[:\s]+.*?(?:effective|beneficial|superior|improved|reduced|'
    r'promising|encouraging|favourable|favorable|well-tolerated|safe and effective)',
    r'(?:suggest|indicate|demonstrate|show|confirm)s?\s+(?:that\s+)?(?:\w+\s+){0,4}'
    r'(?:is|was|were|may be)\s+(?:effective|beneficial|superior)',
]

# "Trend" language (often used to spin NS results)
TREND_PATTERNS = [
    r'\b(?:trend|tendency)\s+(?:toward|towards|to|for)\s+(?:\w+\s+){0,3}'
    r'(?:benefit|improvement|reduction|superiority)',
    r'\b(?:numerically|clinically)\s+(?:meaningful|significant|important|relevant)\b',
    r'\bnon-significant\s+(?:trend|tendency|reduction|improvement)\b',
    r'\bapproaching\s+(?:statistical\s+)?significance\b',
    r'\bmarginal(?:ly)?\s+significant\b',
]

# Subgroup emphasis
SUBGROUP_PATTERNS = [
    r'\b(?:subgroup|sub-group|subset|stratified)\s+(?:analysis|analyses)\b',
    r'\b(?:in|among)\s+(?:the\s+)?(?:subgroup|subset)\s+of\b',
    r'\b(?:exploratory|post[\s-]?hoc|pre-specified subgroup)\b',
]

# Within-group comparison (pre-post) instead of between-group
WITHIN_GROUP_PATTERNS = [
    r'\b(?:compared\s+(?:to|with)\s+baseline|from\s+baseline|change\s+from\s+baseline)\b'
    r'(?!.*(?:between[\s-]group|vs\.?\s+(?:placebo|control|comparator)))',
    r'\bpre-?\s*(?:and|to|vs\.?)\s*post\b',
    r'\b(?:before\s+and\s+after|improvement\s+from\s+baseline)\b',
]

# Inappropriate causal language from observational designs
CAUSAL_LANGUAGE_PATTERNS = [
    r'\b(?:caused?|leads?\s+to|results?\s+in|prevents?|produces?)\b',
    # Only flagged if study is observational:
    r'\b(?:cohort|cross[\s-]sectional|case[\s-]control|retrospective|registry|'
    r'observational|real[\s-]world)\b',
]

# Premature clinical recommendation
RECOMMENDATION_PATTERNS = [
    r'(?:should|recommend|warrant|advocate|suggest)\s+(?:\w+\s+){0,4}'
    r'(?:use|adoption|implementation|prescri|consideration|standard\s+of\s+care)',
    r'\bfirst[\s-]line\s+(?:therapy|treatment|option)\b',
    r'\bstandard\s+of\s+care\b',
    r'\b(?:practice[\s-]changing|paradigm[\s-]shifting)\b',
]

# Title spin patterns
TITLE_BENEFIT_PATTERNS = [
    r'\b(?:effective|superior|beneficial|improves?|reduces?|prevents?|'
    r'safe\s+and\s+effective|promising|breakthrough)\b',
]


from schemas import extract_abstract_sections as _extract_sections


def _has_match(text: str, patterns: list[str]) -> list[str]:
    """Return all regex matches from a list of patterns."""
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(m.group())
    return matches


def screen_for_spin(pmid: str, title: str, abstract: str) -> SpinScreeningResult:
    """
    Heuristic spin screening for a single abstract.

    Returns structured flags with evidence text.
    Designed for pre-screening large batches to identify training candidates.
    """
    result = SpinScreeningResult(pmid=pmid, title=title)
    sections = _extract_sections(abstract)
    result.has_structured_abstract = len(sections) > 1

    full_text = f"{title}\n{abstract}"

    # Get conclusion text
    conclusion_text = ""
    for key in ["CONCLUSION", "CONCLUSIONS", "INTERPRETATION"]:
        if key in sections:
            conclusion_text = sections[key]
            break

    # Get results text
    results_text = ""
    for key in ["RESULTS", "FINDINGS"]:
        if key in sections:
            results_text = sections[key]
            break

    # ---- 1. Check if primary outcome was non-significant ----
    ns_matches = _has_match(full_text, NS_PRIMARY_PATTERNS)
    if ns_matches:
        result.primary_outcome_significant = "no"
    else:
        # Check for strong significance claims
        sig_patterns = [
            r'(?:primary|main)\s+(?:outcome|endpoint).*?(?:significant(?:ly)?|p\s*[<≤]\s*0\.0[0-5])',
        ]
        sig_matches = _has_match(full_text, sig_patterns)
        if sig_matches:
            result.primary_outcome_significant = "yes"
        else:
            result.primary_outcome_significant = "unclear"

    # ---- 2. Check for spin patterns ----

    # Benefit claimed despite NS primary
    if result.primary_outcome_significant in ("no", "unclear"):
        benefit_matches = _has_match(conclusion_text or full_text, BENEFIT_DESPITE_NS_PATTERNS)
        for match in benefit_matches:
            result.flags.append(SpinFlag(
                spin_type=SpinType.BENEFIT_DESPITE_NS,
                description="Conclusion claims benefit despite non-significant or unclear primary outcome",
                evidence=match[:200],
                severity="high",
                section="conclusion",
            ))

    # Trend language
    trend_matches = _has_match(full_text, TREND_PATTERNS)
    for match in trend_matches:
        result.flags.append(SpinFlag(
            spin_type=SpinType.TREND_EMPHASIS,
            description='Use of "trend" or "clinically meaningful" language, '
                        "potentially disguising non-significant results",
            evidence=match[:200],
            severity="moderate",
            section="results" if match.lower() in (results_text or "").lower() else "other",
        ))

    # Subgroup emphasis
    subgroup_in_conclusion = _has_match(conclusion_text, SUBGROUP_PATTERNS)
    if subgroup_in_conclusion:
        result.flags.append(SpinFlag(
            spin_type=SpinType.SUBGROUP_EMPHASIS,
            description="Subgroup or exploratory analysis emphasised in conclusions",
            evidence=subgroup_in_conclusion[0][:200],
            severity="moderate" if result.primary_outcome_significant == "yes" else "high",
            section="conclusion",
        ))

    # Within-group comparisons (common in non-randomized or poorly reported studies)
    within_matches = _has_match(full_text, WITHIN_GROUP_PATTERNS)
    if within_matches and not _has_match(full_text, [r'between[\s-]group']):
        result.flags.append(SpinFlag(
            spin_type=SpinType.WITHIN_GROUP,
            description="Within-group (pre-post) comparisons without between-group analysis",
            evidence=within_matches[0][:200],
            severity="moderate",
            section="results",
        ))

    # Causal language from observational designs
    observational_matches = _has_match(
        full_text,
        [r'\b(?:cohort|cross[\s-]sectional|case[\s-]control|retrospective|'
         r'registry|observational|real[\s-]world)\b']
    )
    if observational_matches:
        causal_matches = _has_match(
            conclusion_text or full_text,
            [r'\b(?:caused?|leads?\s+to|results?\s+in|prevents?|produces?)\b']
        )
        if causal_matches:
            result.flags.append(SpinFlag(
                spin_type=SpinType.CAUSAL_LANGUAGE,
                description="Causal language used in conclusions of observational study",
                evidence=causal_matches[0][:200],
                severity="high",
                section="conclusion",
            ))

    # Premature clinical recommendation
    rec_matches = _has_match(conclusion_text, RECOMMENDATION_PATTERNS)
    for match in rec_matches:
        result.flags.append(SpinFlag(
            spin_type=SpinType.CLINICAL_RECOMMENDATION,
            description="Clinical recommendation or practice change suggested from single study",
            evidence=match[:200],
            severity="moderate",
            section="conclusion",
        ))

    # Title spin
    if result.primary_outcome_significant in ("no", "unclear"):
        title_benefit = _has_match(title, TITLE_BENEFIT_PATTERNS)
        if title_benefit:
            result.flags.append(SpinFlag(
                spin_type=SpinType.TITLE_SPIN,
                description="Title implies benefit not supported by primary outcome results",
                evidence=title_benefit[0][:200],
                severity="high",
                section="title",
            ))

    # ---- 3. Compute overall score ----
    severity_weights = {"low": 0.1, "moderate": 0.2, "high": 0.35}
    score = sum(severity_weights.get(f.severity, 0.15) for f in result.flags)
    result.overall_spin_score = min(score, 1.0)

    # ---- 4. Boutron level classification ----
    if not result.flags:
        result.boutron_level = "none"
    elif result.overall_spin_score < 0.2:
        result.boutron_level = "low"
    elif result.overall_spin_score < 0.5:
        result.boutron_level = "moderate"
    else:
        result.boutron_level = "high"

    return result


def batch_screen(
    abstracts: list[dict],
    min_score: float = 0.2,
) -> list[SpinScreeningResult]:
    """Screen a batch and return those above the spin score threshold."""
    results = []
    for item in abstracts:
        result = screen_for_spin(
            pmid=item.get("pmid", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", ""),
        )
        if result.overall_spin_score >= min_score:
            results.append(result)
    return results


if __name__ == "__main__":
    # Test with synthetic abstracts

    # Example 1: Classic spin - NS primary, benefit in conclusion
    test_spin = screen_for_spin(
        pmid="SPIN001",
        title="Wonderdrug improves outcomes in heart failure patients",
        abstract=(
            "BACKGROUND: Heart failure remains a major cause of morbidity. "
            "METHODS: Randomized, double-blind trial of wonderdrug vs placebo in 500 patients. "
            "Primary endpoint: all-cause mortality at 12 months. "
            "RESULTS: The primary endpoint did not reach statistical significance "
            "(HR 0.82, 95% CI 0.65-1.04, p=0.10). However, there was a trend toward "
            "reduced mortality. In the pre-specified subgroup of patients with EF<30%, "
            "wonderdrug significantly reduced mortality (HR 0.60, p=0.02). "
            "Quality of life scores improved significantly from baseline in the treatment group. "
            "CONCLUSIONS: Wonderdrug shows promising efficacy in heart failure patients, "
            "particularly those with severely reduced ejection fraction. These results "
            "suggest wonderdrug should be considered as an addition to standard therapy."
        ),
    )

    print(f"=== Spin example ===")
    print(f"Boutron level: {test_spin.boutron_level}")
    print(f"Score: {test_spin.overall_spin_score:.2f}")
    print(f"Primary NS: {test_spin.primary_outcome_significant}")
    for flag in test_spin.flags:
        print(f"  [{flag.severity}] {flag.spin_type.value}: {flag.description}")
        print(f"    Evidence: {flag.evidence[:100]}")
    print()

    # Example 2: Clean reporting
    test_clean = screen_for_spin(
        pmid="CLEAN001",
        title="Effect of gooddrug on mortality in heart failure: a randomized trial",
        abstract=(
            "BACKGROUND: We assessed whether gooddrug reduces mortality in heart failure. "
            "METHODS: Multicentre RCT, n=3000, median follow-up 2.5 years. "
            "RESULTS: All-cause mortality occurred in 12.1% of the gooddrug group and "
            "14.8% of the placebo group (HR 0.80, 95% CI 0.68-0.94, p=0.006). "
            "The absolute risk reduction was 2.7% (NNT 37). "
            "CONCLUSIONS: Gooddrug reduced all-cause mortality in heart failure patients. "
            "The modest absolute benefit should be weighed against cost and adverse effects. "
            "Longer-term follow-up data are needed."
        ),
    )

    print(f"=== Clean example ===")
    print(f"Boutron level: {test_clean.boutron_level}")
    print(f"Score: {test_clean.overall_spin_score:.2f}")
    print(f"Flags: {len(test_clean.flags)}")
